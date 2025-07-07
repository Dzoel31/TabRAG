import streamlit as st
import os
import sys
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.utils.embedding import EmbeddingModel
from src.utils.elastic import ElasticSearchClient

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

    embedding_model = EmbeddingModel(model_name="nomic-ai/nomic-embed-text-v1.5")
    # embedding_model = EmbeddingModel(
    #     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # )
    
    embedding = embedding_model.embed(user_input)

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

    search_client = ElasticSearchClient(index="rag-faq", verbose=True)
    retrieved_ctx = search_client.search(
        query_embedding=embedding,
        size=3,
    )

    if retrieved_ctx and "hits" in retrieved_ctx and retrieved_ctx["hits"]["hits"]:
        contexts = [hit["_source"]["Content"] for hit in retrieved_ctx["hits"]["hits"]]
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
    else:
        response_text = "Tidak ada konteks yang ditemukan untuk pertanyaan ini."
        st.session_state["message_history"].append(
            {"role": "assistant", "content": response_text}
        )
        st.chat_message("assistant").markdown(response_text)
