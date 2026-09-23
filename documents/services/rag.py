import io
import os
import uuid
import logging
from typing import List, Dict, Any, Iterator
from pypdf import PdfReader
import chromadb
from chromadb.api.models.Collection import Collection
import google.generativeai as genai
from django.conf import settings

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG service managing document chunking, ChromaDB vector indexing,
    and grounded answer generation.
    """

    SIMILARITY_THRESHOLD = 0.75  # Cosine distance cutoff to discard noisy chunks

    STRICT_SYSTEM_PROMPT = (
        "You are a helpful document assistant. "
        "Answer the user query based strictly on the provided context snippets. "
        "Always cite the relevant page numbers (e.g. [Page 2]) for the facts you provide. "
        "If the information is not available in the context snippets, state clearly: "
        "'I cannot find this information in the document.'"
    )

    def __init__(self) -> None:
        self.persist_dir = settings.CHROMA_PERSIST_DIR
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        os.makedirs(self.persist_dir, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection: Collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)

    def _find_sentence_boundary(self, text: str, target: int, window: int = 120) -> int:
        """Finds the nearest sentence or paragraph boundary near target index."""
        text_len = len(text)
        if target >= text_len:
            return text_len

        search_start = max(0, target - window)
        search_end = min(text_len, target + window)
        slice_text = text[search_start:search_end]

        for delim in ["\n\n", ".\n", ". ", "? ", "! ", "\n"]:
            idx = slice_text.rfind(delim)
            if idx != -1:
                return search_start + idx + len(delim)

        # Fallback to word boundary
        idx = slice_text.rfind(" ")
        if idx != -1:
            return search_start + idx + 1

        return target

    def extract_and_chunk_pdf(
        self,
        file_bytes: bytes,
        user_id: int,
        document_id: int,
        filename: str,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP
    ) -> List[Dict[str, Any]]:
        """
        Parses a PDF buffer and creates sentence-aware chunks with metadata.
        """
        reader = PdfReader(io.BytesIO(file_bytes))
        chunks: List[Dict[str, Any]] = []
        global_idx = 0

        for page_idx, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            page_num = page_idx + 1
            start = 0
            text_len = len(text)

            while start < text_len:
                raw_end = start + chunk_size
                end = self._find_sentence_boundary(text, raw_end)
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunk_id = f"usr_{user_id}_doc_{document_id}_p{page_num}_c{global_idx}_{uuid.uuid4().hex[:6]}"
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": {
                            "user_id": user_id,
                            "document_id": document_id,
                            "filename": filename,
                            "page_number": page_num,
                            "chunk_index": global_idx,
                        }
                    })
                    global_idx += 1

                if end >= text_len:
                    break

                # Advance with overlap
                start = max(start + 1, end - chunk_overlap)

        return chunks

    def ingest_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Inserts chunks into ChromaDB with tenant metadata."""
        if not chunks:
            return 0

        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks]
        )
        return len(chunks)

    def query_context(
        self,
        user_id: int,
        document_id: int,
        query: str,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top relevant chunks filtered strictly by user_id and document_id.
        Discards chunks exceeding the distance threshold.
        """
        where_filter = {
            "$and": [
                {"user_id": user_id},
                {"document_id": document_id}
            ]
        }

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        contexts: List[Dict[str, Any]] = []
        for i, doc in enumerate(documents):
            dist = distances[i] if i < len(distances) else 0.0
            if dist <= self.SIMILARITY_THRESHOLD:
                meta = metadatas[i] if i < len(metadatas) else {}
                contexts.append({
                    "text": doc,
                    "page_number": meta.get("page_number", 1),
                    "distance": dist,
                })

        return contexts

    def stream_answer(self, user_id: int, document_id: int, query: str) -> Iterator[str]:
        """
        Streams grounded answer tokens from Gemini based on retrieved snippets.
        """
        contexts = self.query_context(
            user_id=user_id,
            document_id=document_id,
            query=query,
            top_k=settings.TOP_K
        )

        if not contexts:
            yield "I cannot find this information in the document."
            return

        formatted_context = "\n\n".join([
            f"--- Snippet {i+1} [Page {c['page_number']}] ---\n{c['text']}"
            for i, c in enumerate(contexts)
        ])

        full_prompt = (
            f"SYSTEM INSTRUCTION:\n{self.STRICT_SYSTEM_PROMPT}\n\n"
            f"CONTEXT:\n{formatted_context}\n\n"
            f"USER QUERY:\n{query}\n\n"
            f"ANSWER:"
        )

        if not settings.GEMINI_API_KEY:
            yield "Service is currently operating in offline mode. Please configure GEMINI_API_KEY."
            return

        try:
            model = genai.GenerativeModel(
                model_name=settings.GEMINI_MODEL,
                generation_config={"temperature": 0.2}
            )
            response = model.generate_content(full_prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception:
            logger.exception("Error during LLM stream generation")
            yield "\n\n[An error occurred while generating the answer. Please try again later.]"

    def reset_vector_store(self) -> None:
        """Deletes and recreates the ChromaDB collection (staff only)."""
        self.chroma_client.delete_collection(name=self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
