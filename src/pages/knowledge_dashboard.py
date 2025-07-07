from venv import logger
import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import json
from uuid import uuid4
import logging

from src.utils.parsing import DocumentParser
from src.utils.embedding import EmbeddingModel
from src.utils.elastic import ElasticSearchClient

temp_dir_pdf = Path("src/data/pdf")
temp_dir_json = Path("src/data/json")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Knowledge Dashboard",
    page_icon="📚",
)

if "pdf_uploader_key" not in st.session_state:
    st.session_state["pdf_uploader_key"] = str(uuid4())
if "embedding_size" not in st.session_state:
    st.session_state["embedding_size"] = 0


def toast_upload_success():
    st.toast(
        "File uploaded successfully!",
        icon=":material/done_outline:",
    )


uploaded_documents = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    key=st.session_state["pdf_uploader_key"],
    accept_multiple_files=True,
    on_change=toast_upload_success,
    help="You can upload multiple PDF files at once. The documents will be parsed and indexed.",
)

# Save uploaded documents to a temporary directory
if uploaded_documents:
    temp_dir_pdf.mkdir(parents=True, exist_ok=True)
    for doc in uploaded_documents:
        file_path = temp_dir_pdf / doc.name
        with open(file_path, "wb") as f:
            f.write(doc.getbuffer())
    # Clear the file_uploader after processing
    st.session_state["pdf_uploader_key"] = str(uuid4())
    st.rerun()

documents = os.listdir(temp_dir_pdf) if temp_dir_pdf.exists() else []

process_button = st.button(
    "Process Documents",
    key="process_button",
    help="Click to parse and index the uploaded documents.",
    disabled=not documents,  # Disable if no documents are uploaded
)

if process_button:
    if not documents:
        st.error("No documents to process. Please upload PDF files first.")
    else:
        with st.status("Processing documents..."):
            st.spinner("Parsing and indexing documents...")
            progress_docs = st.progress(0, "Processing documents...")

            for counter in range(len(documents)):
                doc_name = documents[counter]
                doc_path = temp_dir_pdf / doc_name
                parser = DocumentParser(doc_path)

                try:
                    progress_docs.progress(
                        counter / len(documents),
                        f"{counter}/{len(documents)}: Processing {doc_name}...",
                    )
                    parsed_document = parser.parse_pdf()
                    parser.to_json(parsed_document)
                    # progress_docs.progress((counter + 1) / len(documents), f"{counter+1}/{len(documents)}: Processing {doc_name}...")
                except Exception as e:
                    logging.error(f"Error processing {doc_name}: {e}")
                    st.error(f"Failed to process {doc_name}. Check logs for details.")
            progress_docs.progress(1.0, "All documents processed successfully!")

st.subheader("Preview Parsed Documents")
result_documents = os.listdir(temp_dir_json) if temp_dir_json.exists() else []


def process_json_to_dataframe(json_file):
    """
    Process a JSON file and return its content as a DataFrame.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    pages = data.get("content", [])

    # Create a DataFrame for metadata
    metadata_keys = ["document_id", "document_name", "parse_time", "total_pages"]
    df_temp = pd.DataFrame([{key: data[key] for key in metadata_keys}])

    df_content = pd.DataFrame(pages)
    # Repeat metadata for each page
    df_temp = pd.concat([df_temp] * len(df_content), ignore_index=True)
    df_content = df_content.rename(
        columns={"text": "Content", "page_number": "Page Number"}
    )

    return pd.concat([df_temp, df_content], axis=1)


result = st.selectbox(
    "Select a document to preview",
    options=result_documents,
    help="Choose a document to view its parsed content.",
)

if result_documents:
    st.write("Parsed documents:")
    if result:
        doc_path = temp_dir_json / result
        df = process_json_to_dataframe(doc_path)
        st.dataframe(df, use_container_width=True)

    embedding = st.button(
        "Generate Embeddings",
        key="generate_embeddings",
        help="Click to generate embeddings for the selected document.",
        disabled=not result,  # Enable if a document is selected
    )

    if embedding:
        doc_path = temp_dir_json / result
        df = process_json_to_dataframe(doc_path)
        # embedding_model = EmbeddingModel(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        embedding_model = EmbeddingModel(model_name="nomic-ai/nomic-embed-text-v1.5")
        st.session_state["embedding_size"] = embedding_model.embedding_size
        with st.status("Generating embeddings..."):
            with st.spinner(f"Generating embeddings for {result}..."):
                try:
                    logger.info(f"Dimension of embeddings: {st.session_state['embedding_size']}")
                    if "embedding" in df.columns and df["embedding"].notnull().any():
                        st.warning(
                            "Embeddings already exist for this document. Skipping generation."
                        )
                        st.toast(f"Embeddings already exist for {result}.")
                    else:
                        texts = df["Content"].tolist()
                        embeddings = embedding_model.embed(texts)
                        df["embedding"] = embeddings

                        # Save embeddings back to JSON
                        with open(doc_path, "r") as f:
                            data = json.load(f)
                        for i, page in enumerate(data.get("content", [])):
                            page["embedding"] = embeddings[i]
                        with open(doc_path, "w") as f:
                            json.dump(data, f, indent=2)

                    elastic_client = ElasticSearchClient(
                        index="rag-faq", verbose=True
                    )

                    mapping = elastic_client.elastic_mapping(
                        df, st.session_state["embedding_size"]
                    )  # type: ignore

                    elastic_client.create_index(
                        mapping=mapping,
                    )

                    elastic_client.bulk_index(
                        data=df,
                        refresh=True,
                    )

                except Exception as e:
                    logging.error(f"Error generating embeddings for {result}: {e}")
                    st.error(
                        f"Failed to generate embeddings for {result}. Check logs for details."
                    )

        st.success(f"Embeddings generated and saved for {result}.")
        st.dataframe(df, use_container_width=True)
