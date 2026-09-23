import os
import io
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
    Enterprise RAG Service managing:
    1. Document parsing & text chunking.
    2. Multi-tenant ChromaDB vector persistence with strict user_id filtering.
    3. Gemini 1.5 Flash grounded prompt construction & real-time token streaming.
    """

    STRICT_SYSTEM_PROMPT = (
        "You are a professional Document AI. "
        "Answer ONLY based on the context. "
        "If the answer is not in the context, say 'I cannot find this in the document.'"
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
        Parses a PDF from memory and creates semantic overlapping chunks.
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
                end = min(start + chunk_size, text_len)
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
                start += chunk_size - chunk_overlap

        return chunks

    def ingest_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Inserts chunks into ChromaDB with explicit tenant metadata.
        """
        if not chunks:
            return 0

        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks]
        )
        return len(chunks)

    def query_context(self, user_id: int, document_id: int, query: str, top_k: int = 4) -> List[str]:
        """
        Retrieves context chunks with strict multi-tenant isolation.
        Ensures a user CANNOT retrieve data belonging to another tenant.
        """
        # Strict tenant isolation filter
        where_filter = {
            "$and": [
                {"user_id": user_id},
                {"document_id": document_id}
            ]
        }

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter
        )

        documents = results.get("documents", [])
        if documents and documents[0]:
            return documents[0]
        return []

    def stream_answer(self, user_id: int, document_id: int, query: str) -> Iterator[str]:
        """
        Performs tenant-filtered vector search, constructs grounded prompt from retrieved contexts,
        and streams tokens from Gemini 1.5 Flash.
        """
        contexts = self.query_context(user_id=user_id, document_id=document_id, query=query, top_k=settings.TOP_K)

        if not contexts:
            yield "I cannot find this in the document."
            return

        formatted_context = "\n\n".join([f"--- Context Snippet {i+1} ---\n{c}" for i, c in enumerate(contexts)])
        full_prompt = (
            f"SYSTEM INSTRUCTION:\n{self.STRICT_SYSTEM_PROMPT}\n\n"
            f"CONTEXT:\n{formatted_context}\n\n"
            f"USER QUERY:\n{query}\n\n"
            f"ANSWER:"
        )

        if not settings.GEMINI_API_KEY:
            yield "⚠️ Server Error: GEMINI_API_KEY is not configured on the backend."
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
        except Exception as exc:
            logger.exception("Error during LLM stream generation")
            yield f"\n\n[Generation Error: {str(exc)}]"

    def reset_vector_store(self) -> None:
        """
        Admin destructive action: deletes the collection.
        """
        self.chroma_client.delete_collection(name=self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )