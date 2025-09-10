import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import re

from src.utils.embedding import EmbeddingModel
from src.utils.elastic import ElasticSearchClient
from src.utils.milvus import MilvusClient
from src.utils.qdrant import QdrantClientWrapper

st.set_page_config(
    page_title="Chat with Knowledge Base",
    page_icon="💬",
)

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

st.title("Chat with Knowledge Base")

user_input = st.chat_input("Ask a question about the knowledge base:")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    embedding_model = EmbeddingModel(model_name=st.session_state.embedding_model)
    embedding = embedding_model.embed(
        user_input, type="query", prefix=st.session_state.prefix
    )

    template = """
    Jawab pertanyaan berikut berdasarkan konteks yang diberikan. Jika konteks mengandung tabel, pahami isi tabel sebelum menjawab.
    Jika jawaban tidak ditemukan dalam konteks, jawab: "Saya tidak memiliki informasi terkait."

    Konteks:
    {context}

    Pertanyaan:
    {question}

    /no_think
    """
    prompt = PromptTemplate.from_template(template)
    contexts = None

    match st.session_state.vector_backend:
        case "elasticsearch":
            search_client = ElasticSearchClient(
                index=st.session_state.es_index, verbose=True
            )
            retrieved_ctx = search_client.search(
                query_embedding=embedding,
                size=3,
            )

            if (
                retrieved_ctx
                and "hits" in retrieved_ctx
                and retrieved_ctx["hits"]["hits"]
            ):
                contexts = [
                    hit["_source"]["Content"] for hit in retrieved_ctx["hits"]["hits"]
                ]
                context_text = "\n\n".join(contexts)
        case "milvus":
            search_client = MilvusClient(
                collection_name=st.session_state.milvus_collection,
                connection_mode=st.session_state.milvus_connection,
            )

            retrieved_ctx = search_client.retriever(st.session_state.top_docs)
            
        case "qdrant":
            query_client = QdrantClientWrapper(path=st.session_state.local_path_db)
            retrieved_ctx = query_client.search(
                collection_name=st.session_state.collection_name,
                query=embedding,
                top_k=st.session_state.top_docs,
            )

            contexts = [ctx.payload["text"] for ctx in retrieved_ctx]

    context_text = "\n\n".join(contexts)
    prompt = prompt.partial(context=context_text, question=user_input).format()

    llm = OllamaLLM(
        model="qwen3:4b-q4_K_M",
        temperature=0.0,
    )

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            # Streaming response
            response_placeholder = st.empty()
            full_response = ""
            for chunk in llm.stream(prompt):
                # Remove <think>...</think> tags and any leading/trailing whitespace from the chunk if present
                chunk = re.sub(r"<think>.*?</think>", "", chunk, flags=re.DOTALL)
                # Skip empty chunks after cleaning
                if not chunk:
                    continue
                full_response += chunk
                response_placeholder.markdown(full_response)
            st.session_state["message_history"].append(
                {"role": "assistant", "content": full_response}
            )
# else:
#     response_text = "Tidak ada konteks yang ditemukan untuk pertanyaan ini."
#     st.session_state["message_history"].append(
#         {"role": "assistant", "content": response_text}
#     )
#     st.chat_message("assistant").markdown(response_text)
