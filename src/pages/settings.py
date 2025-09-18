import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="TabRAG - Tanya Bareng RAG",
    page_icon="📚",
    layout="wide",
)

# --- Session State Initialization ---
DEFAULTS = {
    "embedding_model": None,
    "provider": "ollama",
    "llm_model": None,
    "temperature": 0.0,
    "prefix": None,
    # Deprecated: use vector store config below
    "database": None,
    # Vector store configuration
    "vector_backend": None,  # elasticsearch | milvus | qdrant
    "es_host": "https://localhost:9200",
    "milvus_host": "127.0.0.1",
    "milvus_port": "19530",
    "qdrant_host": "127.0.0.1",
    "qdrant_port": "6333",
    "collection_name": "rag_faq",
    "connection_type": "local",  # local | remote
    "local_path_db": None,
    "chunk_strategy": "page",
    "chunk_size": 0,
    "chunk_overlap": 0,
    "top_docs": 3,
    "API_KEY": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- UI: Title & Description ---
st.title("Settings")
st.markdown(
    (
        "This is the settings page where you can configure various parameters "
        "for the TabRAG application."
    )
)

# --- UI: Settings Form ---
col1, col2 = st.columns(2)

with col1:
    st.session_state.embedding_model = st.text_input(
        "Embedding Model",
        value=st.session_state.embedding_model
        or "nomic-ai/nomic-embed-text-v1.5",  # Default value
        placeholder="e.g. nomic-ai/nomic-embed-text-v1.5",
    )
    st.session_state.prefix = st.text_input(
        "Prefix",
        value=st.session_state.prefix or "",
        help=(
            "Optional prefix for the embedding model. Refer to the model docs."
            "if more than one, use comma"
        ),
    )
    st.session_state.llm_model = st.text_input(
        "LLM Model",
        value=st.session_state.llm_model or "",
        placeholder="e.g. llama3.2:latest",
    )
    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(st.session_state.temperature or 0.0),
        step=0.1,
        help="Sampling temperature for the selected LLM.",
    )
    # Deprecated single database field (kept for backward compatibility)
    st.session_state.database = st.text_input(
        "Database (deprecated)",
        value=st.session_state.database or "",
        help="Use Vector Store section instead.",
    )

with col2:
    provider_options = ["ollama", "openai", "gemini"]
    try:
        provider_index = provider_options.index(st.session_state.provider)
    except Exception:
        provider_index = 0
    st.session_state.provider = st.selectbox(
        "Provider",
        options=provider_options,
        index=provider_index,
    )
    st.session_state.chunk_strategy = st.selectbox(
        "Chunk Strategy",
        options=["page", "hybrid"],
        index=(
            ["page", "hybrid"].index(st.session_state.chunk_strategy)
            if st.session_state.chunk_strategy in ["page", "hybrid"]
            else 0
        ),
    )
    st.session_state.chunk_size = st.number_input(
        "Chunk Size",
        value=int(st.session_state.chunk_size or 0),
        min_value=0,
        step=32,
    )
    st.session_state.chunk_overlap = st.number_input(
        "Chunk Overlap",
        value=int(st.session_state.chunk_overlap or 0),
        min_value=0,
        step=8,
    )
    st.session_state.top_docs = st.number_input(
        "Top Docs",
        value=int(st.session_state.top_docs or 3),
        min_value=1,
        max_value=50,
    )

# Vector Store Section
st.subheader("Vector Store")
vs_cols = st.columns(2)
with vs_cols[0]:
    backend_options = {
        "Elasticsearch": "elasticsearch",
        "Milvus": "milvus",
        "Qdrant": "qdrant",
    }
    backend_labels = list(backend_options.keys())
    current_backend_value = st.session_state.vector_backend or "elasticsearch"
    try:
        backend_index = list(backend_options.values()).index(current_backend_value)
    except Exception:
        backend_index = 0
    selected_label = st.selectbox(
        "Backend",
        options=backend_labels,
        index=backend_index,
        help="Choose the vector database backend.",
    )
    st.session_state.vector_backend = backend_options[selected_label]

    # Only show the relevant settings for the selected backend
    if selected_label == "Elasticsearch":
        with vs_cols[1]:
            st.session_state.es_host = st.text_input(
                "Elasticsearch Host",
                value=st.session_state.es_host,
                help="URL to Elasticsearch (with scheme).",
            )
            st.session_state.collection_name = st.text_input(
                "Elasticsearch Index",
                value=st.session_state.collection_name,
            )
    elif selected_label == "Milvus":
        with vs_cols[1]:
            milvus_conn_labels = ["Local file", "Host & Port"]
            milvus_conn_values = {"Local file": "local", "Host & Port": "remote"}
            try:
                milvus_conn_index = list(milvus_conn_values.values()).index(
                    st.session_state.connection_type
                )
            except Exception:
                milvus_conn_index = 0
            milvus_conn_label = st.selectbox(
                "Milvus Connection",
                options=milvus_conn_labels,
                index=milvus_conn_index,
                help=(
                    "Choose how to connect to Milvus: local file (URI/path) "
                    "or remote host:port."
                ),
            )
            st.session_state.connection_type = milvus_conn_values[milvus_conn_label]

            if st.session_state.connection_type == "local":
                st.session_state.milvus_uri = st.text_input(
                    "Milvus URI (file path)",
                    value=str(
                        st.session_state.milvus_uri or "src/data/milvus/vector_store.db"
                    ),
                    help=("Path to a local Milvus/SQLite-style store."),
                    placeholder="e.g. src/data/milvus/vector_store.db",
                )
            else:
                st.session_state.milvus_host = st.text_input(
                    "Milvus Host",
                    value=st.session_state.milvus_host,
                    placeholder="e.g. 127.0.0.1",
                )
                st.session_state.milvus_port = st.text_input(
                    "Milvus Port",
                    value=str(st.session_state.milvus_port),
                    placeholder="e.g. 19530",
                )
            # Collection is required in both modes
            st.session_state.collection_name = st.text_input(
                "Milvus Collection",
                value=st.session_state.collection_name,
            )
    elif selected_label == "Qdrant":
        with vs_cols[1]:
            qdrant_conn_labels = ["Local file", "Host & Port"]
            qdrant_conn_values = {"Local file": "local", "Host & Port": "remote"}
            try:
                qdrant_conn_index = list(qdrant_conn_values.values()).index(
                    st.session_state.connection_type
                )
            except Exception:
                qdrant_conn_index = 0
            qdrant_conn_label = st.selectbox(
                "Qdrant Connection",
                options=qdrant_conn_labels,
                index=qdrant_conn_index,
                help=(
                    "Choose how to connect to Qdrant: local file (URI/path) "
                    "or remote host:port."
                ),
            )
            st.session_state.connection_type = qdrant_conn_values[qdrant_conn_label]

            if st.session_state.connection_type == "local":
                st.session_state.local_path_db = st.text_input(
                    "Qdrant URI (file path)",
                    value=str(
                        st.session_state.local_path_db
                        or "src/data/qdrant/vector_store.db"
                    ),
                    help=("Path to a local Qdrant/SQLite-style store."),
                    placeholder="e.g. src/data/qdrant/vector_store.db",
                )
            else:
                st.session_state.qdrant_host = st.text_input(
                    "Qdrant Host",
                    value=st.session_state.qdrant_host,
                    placeholder="e.g. 127.0.0.1",
                )
                st.session_state.qdrant_port = st.text_input(
                    "Qdrant Port",
                    value=str(st.session_state.qdrant_port),
                    placeholder="e.g. 19530",
                )
            # Collection is required in both modes
            st.session_state.collection_name = st.text_input(
                "Qdrant Collection",
                value=st.session_state.collection_name,
            )

# Provider API key (not needed for Ollama)
needs_key = st.session_state.provider in {"openai", "gemini"}
st.session_state.API_KEY = st.text_input(
    "API Key",
    type="password",
    value=st.session_state.API_KEY or "",
    help=("Enter your API key for the selected provider. Not required for Ollama."),
    disabled=not needs_key,
)

submitted = st.button("Save Settings", help="Click to save the settings.")
if submitted:
    has_error = False

    if st.session_state.chunk_overlap > st.session_state.chunk_size:
        st.session_state.chunk_overlap = st.session_state.chunk_size
        st.info(
            "Chunk Overlap was larger than Chunk Size and has been adjusted.",
        )

    if st.session_state.vector_backend == "milvus":
        if st.session_state.connection_type == "remote":
            # Ensure port is numeric string
            port = str(st.session_state.milvus_port).strip()
            if not port.isdigit():
                st.error("Milvus Port must be a number.")
                has_error = True
            else:
                st.session_state.milvus_port = port
        else:
            # Basic validation for local uri/path
            uri = str(st.session_state.milvus_uri or "").strip()
            if not uri:
                st.error("Milvus URI (file path) cannot be empty for local mode.")
                has_error = True
            else:
                st.session_state.milvus_uri = uri

    if not has_error:
        st.toast("Settings saved successfully!", icon=":material/done_outline:")
