import logging
import time
from typing import Optional, Dict
from contextlib import suppress
import threading
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from pandas import DataFrame

from uuid import uuid4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Serialize access to embedded storage in-process
_lock = threading.Lock()


class QdrantClientWrapper:
    """Qdrant backend vector store

    Args:
        url (str): The URL of the Qdrant instance. local, memory or <cloud>
    """

    def __init__(
        self,
        url: Optional[str] | None = None,
        path: Optional[str] | None = None,
        location: Optional[str] | None = None,
    ):
        self.url = url
        self.path = path
        self.location = location

    def _make_client(self) -> QdrantClient:
        if self.url:
            return QdrantClient(url=self.url)
        if self.path:
            return QdrantClient(path=self.path)
        if self.location:
            return QdrantClient(location=self.location)
        raise ValueError("One of 'url', 'path', or 'location' must be provided.")

    def create_collection(self, collection_name: str, vector_size: int):
        with _lock:
            client = self._make_client()
            try:
                if client.collection_exists(collection_name):
                    client.delete_collection(collection_name=collection_name)
                    logger.info(f"Collection dropped: {collection_name}")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    ),
                )
                logger.info(f"Collection created: {collection_name}")
            finally:
                with suppress(Exception):
                    client.close()

    def add_document(self, collection_name: str, document: Dict | DataFrame):
        docs = []
        if isinstance(document, DataFrame):
            for _, row in document.iterrows():
                payload = {
                    "text": row["text"],
                    "document_id": row["document_id"],
                    "document_name": row["document_name"],
                    "total_pages": row["total_pages"],
                    "page_number": row["page_number"],
                }
                if "chunk_number" in row:
                    payload["chunk_number"] = row["chunk_number"]
                docs.append(
                    PointStruct(
                        id=str(uuid4()),
                        vector=row["embedding"],
                        payload=payload,
                    )
                )
                time.sleep(0.5)
        else:
            docs.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=document["embedding"],
                    payload={
                        "text": document["text"],
                        "document_id": document["document_id"],
                        "document_name": document["document_name"],
                        "total_pages": document["total_pages"],
                        "page_number": document["page_number"],
                    },
                )
            )

        with _lock:
            client = self._make_client()
            try:
                client.upload_points(collection_name=collection_name, points=docs)
                logger.info(f"Document added to Qdrant: {len(docs)} points")
            finally:
                with suppress(Exception):
                    client.close()

    def search(self, collection_name: str, query: list[float], top_k: int = 3):
        with _lock:
            client = self._make_client()
            try:
                # results = client.search(
                #     collection_name=collection_name, query_vector=query, limit=top_k
                # )
                print(f"top_k: {top_k}")
                results = client.query_points(
                    collection_name=collection_name, query=query, limit=top_k
                ).points
                logger.info(f"Search completed: retrieved {len(results)} results")
                return results
            finally:
                with suppress(Exception):
                    client.close()
