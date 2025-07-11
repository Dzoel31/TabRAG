import os
import time
import logging
from typing import Optional

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, JsonSerializer, helpers
from pandas import DataFrame
from uuid import uuid4

load_dotenv()

# Password for the 'elastic' user generated by Elasticsearch
ELASTIC_PASSWORD = os.environ.get("ES_PASSWORD", None)
HTTP_CA_FINGERPRINT_SHA = os.environ.get("HTTP_CA_FINGERPRINT_SHA", None)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JsonSetSerializer(JsonSerializer):
    """Custom JSON serializer to handle sets by converting them to lists."""

    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


class ElasticSearchClient:
    def __init__(
        self,
        host: str = "https://localhost:9200",
        index: str = "rag_faq",
        verbose: bool = False,
    ):
        self.host = host
        self.index = index
        self.client = self._create_client()
        self.verbose = verbose
        if self.verbose:
            info = self.client.info()
            logger.info("\n" + "=" * 40)
            logger.info("✅ Connected to Elasticsearch!")
            logger.info(f"🔍 Cluster:   {info.get('cluster_name', 'N/A')}")
            logger.info(f"📊 Version:   {info.get('version', {}).get('number', 'N/A')}")
            logger.info(f"🌐 Node Name: {info.get('name', 'N/A')}")
            logger.info("=" * 40 + "\n")

    def _create_client(self):
        try:
            if ELASTIC_PASSWORD and HTTP_CA_FINGERPRINT_SHA:
                return Elasticsearch(
                    hosts=self.host,
                    basic_auth=("elastic", ELASTIC_PASSWORD),
                    ca_certs="http_ca.crt",
                    verify_certs=True,  # Set to True in production with proper CA certs
                    serializer=JsonSetSerializer(),
                )
            else:
                return Elasticsearch([self.host], serializer=JsonSetSerializer())
        except Exception as e:
            logger.error(f"❌ Failed to connect to Elasticsearch: {e}")
            raise

    def create_index(self, index: Optional[str] = None, mapping: Optional[dict] = None):
        """
        Create an index in Elasticsearch with the specified mapping.

        Args:
            index (str): The name of the index to create. Defaults to the instance's index.
            mapping (dict): The mapping for the index. If None, uses default mapping.
        """
        if not index:
            index = self.index

        if self.client.indices.exists(index=index):
            self.client.indices.delete(index=index)
            if self.verbose:
                logger.info(f"🗑️  Deleted existing index: {index}")

        if mapping:
            self.client.indices.create(index=index, mappings={"properties": mapping})
        else:
            self.client.indices.create(index=index)

        if self.verbose:
            logger.info(f"✅ Created index: {index}")

    def index_document(
        self, document: dict, index: Optional[str] = None, doc_id: Optional[str] = None
    ):
        """
        Index a document in Elasticsearch.

        Args:
            document (dict): The document to index.
            index (str): The name of the index. Defaults to the instance's index.
            doc_id (str): Optional document ID. If None, Elasticsearch generates one.

        Returns:
            dict: The response from Elasticsearch after indexing the document.
        """
        if not index:
            index = self.index

        start_time = time.time()
        response = self.client.index(index=index, id=doc_id, document=document)

        if self.verbose:
            elapsed_time = time.time() - start_time
            logger.info(
                f"📄 Indexed document in {elapsed_time:.2f} seconds: {response}"
            )

        return response

    def search(
        self,
        query_embedding: list[float],
        perform_knn: bool = True,
        index: Optional[str] = None,
        size: int = 10,
    ):
        """
        Search for documents in Elasticsearch.

        Args:
            query_embedding (list[float]): The embedding vector for the search query.
            perform_knn (bool): Whether to perform a k-NN search. Defaults to True.
            index (str): The name of the index. Defaults to the instance's index.
            size (int): The number of results to return. Defaults to 10.

        Returns:
            dict: The search results from Elasticsearch.
        """
        if not index:
            index = self.index

        start_time = time.time()

        query = None
        if perform_knn:
            query = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": size,
                }
            }
        else:
            # Default to match_all if no query is specified
            query = {"match_all": {}}

        response = self.client.search(index=index, query=query, size=size)

        if self.verbose:
            elapsed_time = time.time() - start_time
            logger.info(
                f"🔎 Searched {size} documents in {elapsed_time:.2f} seconds"
            )
            logger.info(response)

        return response

    def elastic_mapping(self, data: DataFrame, embedding_size: int):
        """
        Generate Elasticsearch mapping for the given DataFrame.

        Args:
            data (DataFrame): The DataFrame containing the data.
            embedding_size (int): The size of the embedding vector.

        Returns:
            dict: The Elasticsearch mapping.
        """
        mapping = {"mappings": {"properties": {}}}
        logger.info(
            f"Dimension of embeddings: {embedding_size}"
        )

        for col, dtype in data.dtypes.items():
            if isinstance(dtype, object):
                es_type = "text"
            elif "int" in str(dtype):
                es_type = "integer"
            elif "float" in str(dtype):
                es_type = "float"
            else:
                es_type = "text"

            mapping["mappings"]["properties"][col] = {"type": es_type}

        mapping["mappings"]["properties"]["text"] = {"type": "text"}

        mapping["mappings"]["properties"]["embedding"] = {
            "type": "dense_vector",
            "dims": embedding_size,
            "index": True,  # Enable indexing for k-NN search
            "similarity": "cosine",  # Use cosine similarity for k-NN search
        }

        return mapping
    
    # Need attention for implement embedding
    def parse_json(
        self,
        json_data: dict,
        embedding_size: int,
        index: Optional[str] = None,
        doc_id: Optional[str] = None,
    ):
        """
        Parse JSON data (with 'content' as a list of pages) and index each page in Elasticsearch.

        Args:
            json_data (dict): The JSON data to index (with 'content' as a list of pages).
            embedding_size (int): The size of the embedding vector (mandatory).
            index (str): The name of the index. Defaults to the instance's index.
            doc_id (str): The document ID. If not provided, a new ID will be generated for each page.

        Returns:
            list: List of responses from Elasticsearch for each page.
        """
        if not index:
            index = self.index

        content = json_data.get("content", [])
        if not content:
            logger.warning("No content found in JSON data.")
            return []

        # Build DataFrame for mapping
        df = DataFrame(content)

        mapping = self.elastic_mapping(df, embedding_size)
        # Create index with mapping if not exists
        if not self.client.indices.exists(index=index):
            self.create_index(index=index, mapping=mapping["mappings"])

        responses = []
        for page in content:
            page_doc = {
                "document_id": json_data.get("document_id"),
                "document_name": json_data.get("document_name"),
                "source_path": json_data.get("source_path"),
                "parse_time": json_data.get("parse_time"),
                **page,
            }
            page_id = doc_id or str(uuid4())
            response = self.client.index(index=index, id=page_id, document=page_doc)
            if self.verbose:
                logger.info(
                    f"📥 Indexed page {page.get('page_number')} of document {json_data.get('document_name')} as {page_id} in {index}: {response}"
                )
            responses.append(response)
        return responses

    def bulk_index(
        self, data: list | DataFrame, index: Optional[str] = None, refresh: bool = True
    ):
        """
        Bulk index documents in Elasticsearch.

        Args:
            actions (list): A list of actions to perform (e.g., index, update).
            index (str): The name of the index. Defaults to the instance's index.
            refresh (bool): Whether to refresh the index after bulk indexing. Defaults to True.

        Returns:
            dict: The response from Elasticsearch after bulk indexing.
        """
        if not index:
            index = self.index

        def create_actions():
            if isinstance(data, DataFrame):
                for i, row in data.iterrows():
                    yield {
                        "_index": index,
                        "_id": row.get(
                            "id",
                            f"{uuid4()}",  # Generate a UUID if 'id' is not present
                        ),  # Use 'id' column or generate one
                        "_source": row.to_dict(),
                    }
            else:
                for i, doc in enumerate(data):
                    yield {
                        "_index": index,
                        "_id": doc.get(
                            "id",
                            f"{uuid4()}",  # Generate a UUID if 'id' is not present
                        ),  # Use 'id' key or generate one
                        "_source": doc,
                    }

        start_time = time.time()
        response = helpers.bulk(
            self.client, create_actions(), index=index, refresh=refresh
        )
        if isinstance(response, tuple):
            response = {"items": response[0], "errors": response[1]}

        if self.verbose:
            elapsed_time = time.time() - start_time
            logger.info(
                f"📦 Bulk indexed {len(data)} documents in {elapsed_time:.2f} seconds: {response}"
            )

        if "errors" in response and response["errors"]:
            logger.error(f"Bulk indexing encountered errors: {response['errors']}")
        return response

    def delete_index(self, index: Optional[str] = None):
        """
        Delete an index in Elasticsearch.

        Args:
            index (str): The name of the index to delete. Defaults to the instance's index.
        """
        if not index:
            index = self.index

        if self.client.indices.exists(index=index):
            self.client.indices.delete(index=index)
            if self.verbose:
                logger.info(f"🗑️  Deleted index: {index}")
        else:
            if self.verbose:
                logger.warning(f"⚠️ Index '{index}' does not exist.")

    def get_index_info(self, index: Optional[str] = None):
        """
        Get information about an index in Elasticsearch.

        Args:
            index (str): The name of the index. Defaults to the instance's index.

        Returns:
            dict: Information about the index.
        """
        if not index:
            index = self.index

        if self.client.indices.exists(index=index):
            info = self.client.indices.get(index=index)
            if self.verbose:
                logger.info(f"ℹ️ Index info for '{index}':\n{info}")
            return info
        else:
            if self.verbose:
                logger.warning(f"⚠️ Index '{index}' does not exist.")
            return None

    def __str__(self) -> str:
        return f"ElasticSearchClient(host={self.host}, index={self.index})"

    def __repr__(self) -> str:
        return f"ElasticSearchClient(host={self.host}, index={self.index})"


def main():
    client = ElasticSearchClient(
        host="https://localhost:9200", index="test_index", verbose=True
    )
    logger.info(client)


if __name__ == "__main__":
    main()
    # Example usage
    # client.create_index(index="test_index", mapping={"properties": {"text": {"type": "text"}}})
    # client.index_document({"text": "Hello, world!"}, index="test_index")
    # response = client.search(query_embedding=[0.1, 0.2, 0.3], index="test_index")
    # print(response)
