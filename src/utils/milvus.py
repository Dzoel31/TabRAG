from langchain_milvus import Milvus
from langchain_core.embeddings import Embeddings
from langchain.schema import Document
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusClient:
    """Milvus backend vector store

    Args:
        collection_name (str): The name of the collection.
        documents (list): A list of documents to be stored.
        embedding: The embedding model to use for document vectors.
        connection_mode (str): "local" to use a file URI, "remote" for host/port.
        uri (str): Local file path/URI for Milvus Lite style persistence.
        host (str): Milvus server host (for remote mode).
        port (str | int): Milvus server port (for remote mode).
        drop_old (bool): Whether to drop and recreate the collection.
    """

    def __init__(
        self,
        collection_name: str = "rag_faq",
        documents: list[Document] | None = None,
        embedding: Embeddings | None = None,
        connection_mode: str = "local",
        uri: str | None = "src/data/milvus/vector_store.db",
        host: str = "127.0.0.1",
        port: str | int = "19530",
        drop_old: bool = True,
    ):
        self.collection_name = collection_name
        self.connection_mode = connection_mode
        self.uri = Path(uri) if uri else None
        self.host = host
        self.port = str(port)
        self.documents = documents or []

        if embedding is None:
            raise ValueError("An Embeddings instance must be provided for MilvusClient")

        # Ensure local directory exists if using file-based store
        connection_args: dict
        if self.connection_mode == "local":
            if self.uri is None:
                raise ValueError("uri must be provided for local Milvus connection")
            self.uri.parent.mkdir(parents=True, exist_ok=True)
            connection_args = {"uri": str(self.uri)}
        else:
            connection_args = {"host": self.host, "port": self.port}

        try:
            self.client = Milvus.from_documents(
                documents=self.documents,
                embedding=embedding,
                collection_name=collection_name,
                connection_args=connection_args,
                index_params={"metric_type": "COSINE", "index_type": "FLAT"},
                drop_old=drop_old,
            )
        except Exception as e:
            logger.error(f"Error initializing Milvus client: {e}")
            raise

    def retriever(self, top_k: int):
        return self.client.as_retriever(search_kwargs={"k": top_k})
