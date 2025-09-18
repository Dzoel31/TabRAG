import logging
from pathlib import Path
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.settings import settings
from docling_core.transforms.chunker.page_chunker import PageChunker
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

import time
from uuid import uuid4
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Parses PDF documents using Docling.
    """

    def __init__(self, pdf_path: Path, chunk_strategy: str, embedding_model: str):
        self.pdf_path = pdf_path
        self.document_id = str(uuid4())
        self.document_name = pdf_path.stem
        self._chunk_strategy = chunk_strategy
        self._source_path = str(pdf_path)
        self._embedding_model = embedding_model
        self._total_pages = 0
        self._parse_time = None

    def parse_pdf(self) -> list[dict]:
        """
        Parse a PDF file and return the parsed document.

        Returns:
            list[dict]: The parsed document object.
        Raises:
            ValueError: If the PDF file cannot be parsed.
        """
        easyocr_options = EasyOcrOptions(
            lang=["id", "en"],
            force_full_page_ocr=False,
        )
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            ocr_options=easyocr_options,
            accelerator_options=AcceleratorOptions(
                device=AcceleratorDevice.AUTO,
                num_threads=4,
            ),
        )
        pipeline_options.table_structure_options.do_cell_matching = True

        doc_convert = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
        settings.debug.profile_pipeline_timings = True

        doc = doc_convert.convert(source=self.pdf_path)

        self._parse_time = round(doc.timings["pipeline_total"].times[0], 2)

        # Check if the document was parsed successfully
        if doc is None:
            raise ValueError(f"Failed to parse PDF file: {self.pdf_path}")

        result = []
        contents = []

        if self._chunk_strategy == "page":
            page_chunker = PageChunker()
            chunk_iter = page_chunker.chunk(doc.document)
            contents = list(chunk_iter)

            for i, chunk in enumerate(contents):
                metadata = chunk.meta.model_dump()
                doc_items = metadata["doc_items"]
                halaman = doc_items[0]["prov"][0]["page_no"]

                result.append(
                    {
                        "text": chunk.text,
                        "page_number": halaman,
                    }
                )

        elif self._chunk_strategy == "hybrid":
            from docling_core.transforms.chunker.tokenizer.huggingface import (
                HuggingFaceTokenizer,
            )
            from transformers import AutoTokenizer

            tokenizer = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(self._embedding_model),
                max_tokens=512,
            )
            chunker = HybridChunker(
                tokenizer=tokenizer,
            )
            chunk_iter = chunker.chunk(dl_doc=doc.document)
            contents = list(chunk_iter)

            for i, chunk in enumerate(contents):
                metadata = chunk.meta.model_dump()
                doc_items = metadata["doc_items"]
                halaman = doc_items[0]["prov"][0]["page_no"]
                contextual_text = chunker.contextualize(chunk=chunk)
                length_token = chunker.tokenizer.count_tokens(contextual_text)

                result.append(
                    {
                        "text": contextual_text,
                        "page_number": halaman,
                        "chunk_number": i,
                        "token_length": length_token,
                    }
                )

        # Perform a check for empty content and force full page OCR
        if not contents:
            raise ValueError(f"No content found in PDF file: {self.pdf_path}")

        self._total_pages = doc.input.page_count

        return result

    def is_content_empty(self, parsed_doc: list[dict]) -> bool:
        """
        Check if the parsed document content is empty.

        Args:
            parsed_doc (list[dict]): The parsed document object.
        Returns:
            bool: True if the content is empty, False otherwise.
        Raises:
            ValueError: If the parsed document is None or empty.
        """
        if not parsed_doc:
            raise ValueError("Parsed document is None or empty.")

        # Check all pages and store if any page is empty
        if not isinstance(parsed_doc, list):
            raise ValueError("Parsed document should be a list of pages.")

        page_no = [
            i + 1
            for i, page in enumerate(parsed_doc)
            if not page.get("text", "").strip()
        ]

        if page_no:
            logger.warning(f"Empty pages found: {page_no}")
            return True
        return False

    def to_dict(self, parsed_doc: list[dict]) -> dict:
        """
        Convert the parsed document to a dictionary format.

        Args:
            parsed_doc (list[dict]): The parsed document object.
        Returns:
            dict: A dictionary representation of the parsed document.
        Raises:
            ValueError: If the parsed document is None or empty.
        """
        parsed_content = {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "parse_time": self._parse_time,
            "total_pages": self._total_pages,
            "content": parsed_doc,
        }

        return parsed_content

    def to_json(self, parsed_doc: list[dict]) -> None:
        """
        Convert the parsed document to a JSON file format.

        Args:
            parsed_doc (list[dict]): The parsed document object.
        Returns:
            str: A JSON string representation of the parsed document.
        Raises:
            ValueError: If the parsed document is None or empty.
        """
        if not parsed_doc:
            raise ValueError("Parsed document is None or empty.")

        # Create temporary directory if it doesn't exist
        json_dir = Path("./src/data/json")
        json_dir.mkdir(parents=True, exist_ok=True)

        json_file_path = Path(f"./src/data/json/{self.document_name}.json")
        with open(json_file_path, "w+") as json_file:
            json.dump(self.to_dict(parsed_doc), json_file, indent=2)

    @property
    def metadata(self) -> dict:
        """
        Get the metadata of the parsed document.

        Returns:
            dict: The metadata of the parsed document.
        """
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "parse_time": self._parse_time,
        }

    def __str__(self):
        return (f"DocumentParser(pdf_path={self.pdf_path}, "
                f"document_name={self.document_name})")

    def __repr__(self):
        return (f"DocumentParser(pdf_path={self.pdf_path}, "
                f"document_name={self.document_name})")


def main():
    pdf_path = Path("./src/data/1908.10084v1.pdf")
    start = time.time()
    logger.info(f"Starting to parse PDF document: {pdf_path.stem}")
    result = DocumentParser(
        pdf_path,
        chunk_strategy="hybrid",
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
    )
    parsed_document = result.parse_pdf()
    end = time.time()
    print("Metadata: \n", json.dumps(result.metadata, indent=2))
    logger.info(f"Parsed PDF document with {len(parsed_document)} pages.")
    print("Parsed Document: \n", json.dumps(parsed_document, indent=2))
    print(
        "Parsed Document Full: \n",
        json.dumps(result.to_dict(parsed_document), indent=2),
    )
    logger.info(f"PDF parsing took {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()
