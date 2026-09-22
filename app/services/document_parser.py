import io
import re
import uuid
from typing import List
from pydantic import BaseModel, Field
from pypdf import PdfReader


class DocumentChunk(BaseModel):
    """
    Represents a discrete semantic chunk extracted from a document.
    """
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="Text content of the chunk")
    document_id: str = Field(..., description="Parent document identifier")
    filename: str = Field(..., description="Original filename")
    page_number: int = Field(..., description="1-indexed page number in the PDF")
    chunk_index: int = Field(..., description="Sequential index of the chunk within the document")


class DocumentParser:
    """
    Service responsible for reading PDF documents, cleaning text,
    and partitioning it into overlapping chunks for embedding.
    """

    def __init__(self, default_chunk_size: int = 1000, default_chunk_overlap: int = 200) -> None:
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap

    @staticmethod
    def _clean_text(raw_text: str) -> str:
        """
        Normalizes whitespace and removes unprintable artifacts.
        """
        if not raw_text:
            return ""
        # Replace multiple whitespace/newlines with single space/linebreaks
        cleaned = re.sub(r"[ \t]+", " ", raw_text)
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
        return cleaned.strip()

    def parse_pdf(
        self,
        file_bytes: bytes,
        filename: str,
        document_id: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None
    ) -> List[DocumentChunk]:
        """
        Extracts text from each PDF page and generates overlapping semantic chunks.

        :param file_bytes: Raw binary content of the PDF file.
        :param filename: Original name of the uploaded file.
        :param document_id: Unique UUID associated with the document upload.
        :param chunk_size: Maximum character length per chunk (defaults to 1000).
        :param chunk_overlap: Character overlap between consecutive chunks (defaults to 200).
        :return: List of DocumentChunk objects.
        """
        size = chunk_size or self.default_chunk_size
        overlap = chunk_overlap or self.default_chunk_overlap

        if overlap >= size:
            overlap = size // 4

        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as exc:
            raise ValueError(f"Invalid or corrupted PDF file: {str(exc)}") from exc

        chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page_idx, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            page_text = self._clean_text(raw_text)

            if not page_text:
                continue

            page_num = page_idx + 1
            start = 0
            text_len = len(page_text)

            while start < text_len:
                end = min(start + size, text_len)
                chunk_str = page_text[start:end].strip()

                if chunk_str:
                    chunk_id = f"{document_id}_p{page_num}_c{global_chunk_idx}_{uuid.uuid4().hex[:6]}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            text=chunk_str,
                            document_id=document_id,
                            filename=filename,
                            page_number=page_num,
                            chunk_index=global_chunk_idx
                        )
                    )
                    global_chunk_idx += 1

                if end >= text_len:
                    break
                start += size - overlap

        return chunks
