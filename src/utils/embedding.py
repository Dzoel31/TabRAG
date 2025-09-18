from sentence_transformers import SentenceTransformer
from typing import Any, List, Optional
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(
        self,
        model_name: Optional[str] = None,
        prompt: Optional[dict] = None,
        device: Optional[str] = None,
        default_prompt: str = "retrieval",
    ):
        """
        Initializes the embedding model with an English-focused model and prompts.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.
        """
        self.model_name = model_name if model_name else "nomic-ai/nomic-embed-text-v1.5"
        self.prompt = {
            "classification": "Classify the following text: ",
            "retrieval": "Retrieve semantically similar text: ",
            "clustering": "Identify topics or themes based on the text: ",
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.default_prompt = default_prompt
        logger.info(
            (f"Initializing SentenceTransformer with model '{self.model_name}' "
             f"on device '{self.device}'")
        )
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            prompts=self.prompt,
            default_prompt_name=self.default_prompt,
            trust_remote_code=True,
        )

    def embed(
        self,
        texts: str | List[str],
        prefix: Optional[str] = None,
        type: Optional[str] = None,
    ) -> List[Any]:
        """
        Generates embeddings for a list of texts.

        Args:
            texts (List[str]): A list of strings to embed.
            prefix (str, optional): A prefix to prepend to each text before embedding.
        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of 
                floats.
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        prefixes = prefix.split(",") if prefix else []

        if prefixes:
            if type == "document":
                texts = [f"{prefixes[0]}: {text}" for text in texts]
            elif type == "query" and len(prefixes) > 1:
                texts = [f"{prefixes[1]}: {text}" for text in texts]

        return self.model.encode(
            texts, convert_to_tensor=False, show_progress_bar=True
        ).tolist()

    @property
    def embedding_size(self) -> int | None:
        """
        Returns the size of the embedding vector.

        Returns:
            int: The size of the embedding vector.
        """
        return self.model.get_sentence_embedding_dimension()


def main():
    # Example usage
    texts = ["Hello, world!", "This is a test sentence."]
    embedding_model = EmbeddingModel()
    embeddings = embedding_model.embed(texts)

    for text, embedding in zip(texts, embeddings):
        print(f"Text: {text}\nEmbedding: {embedding[:5]}\n")

    print(f"Embedding size: {embedding_model.embedding_size}")


if __name__ == "__main__":
    main()
