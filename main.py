import streamlit as st

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "llm_model" not in st.session_state:
    st.session_state.llm_model = None
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.0
if "prefix" not in st.session_state:
    st.session_state.prefix = None

st.set_page_config(
    page_title="TabRAG - Tanya Bareng RAG",
    page_icon=":question:",
)

def main():
    st.title("TabRAG - Tanya Bareng RAG")
    st.markdown(
        """
        This is a simple TabRAG - Tanya Bareng RAG application built with Streamlit, LangChain, and Ollama.
        You can upload PDF documents, chat with the knowledge base, and view the knowledge dashboard.
        - **Knowledge Dashboard**: View and manage your knowledge base.
        - **Chat**: Ask questions about the knowledge base and get answers.

        - Next plans:
            - Add more models.
            - Improve the UI/UX of the application.
        """
    )

def config():
    st.dialog("Configuration")
    def menu_config():
        ...


# Example navigation using st.navigation and st.Page
pg = st.navigation([
    st.Page(main, title="Home", icon="🏠"),
    st.Page("./src/pages/knowledge_dashboard.py", title="Knowledge Dashboard", icon="📚"),
    st.Page("./src/pages/chat.py", title="Chat", icon="💬"),
])
pg.run()
