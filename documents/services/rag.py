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

    SIMILARITY_THRESHOLD = 1.2  # Generous threshold to support multilingual queries

    SYSTEM_PROMPT = (
        "You are an intelligent, helpful, and highly capable AI assistant named DocuMind AI, powered by Gemini 3.6 Flash. "
        "Your goal is to answer the user's questions clearly, accurately, and thoroughly. "
        "When context from an uploaded document is provided, prioritize it and cite the page numbers (e.g. [1-sahifa] or [Page 1]). "
        "When context is not directly applicable or the user asks general questions (greetings, analysis, questions about people, code, etc.), answer helpfully and intelligently. "
        "CRITICAL LANGUAGE RULE: You MUST always respond in the EXACT same language that the user asks in (e.g., if the user asks in Uzbek, respond entirely in polite and professional Uzbek; if in Russian, respond in Russian; if in English, respond in English)."
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
        top_k: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top relevant chunks filtered strictly by user_id and document_id.
        Orders chunks by semantic similarity (ascending distance).
        """
        try:
            total_elements = self.collection.count()
            if total_elements == 0:
                logger.warning("Query attempted but ChromaDB collection is empty.")
                return []

            where_filter = {
                "$and": [
                    {"user_id": {"$eq": int(user_id)}},
                    {"document_id": {"$eq": int(document_id)}}
                ]
            }

            actual_k = max(1, min(top_k, total_elements))

            results = self.collection.query(
                query_texts=[query],
                n_results=actual_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            documents = results.get("documents", [[]])[0] if results.get("documents") else []
            metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
            distances = results.get("distances", [[]])[0] if results.get("distances") else []

            contexts: List[Dict[str, Any]] = []
            for i, doc in enumerate(documents):
                dist = distances[i] if i < len(distances) else 0.0
                if dist <= self.SIMILARITY_THRESHOLD:
                    meta = metadatas[i] if i < len(metadatas) else {}
                    contexts.append({
                        "text": doc,
                        "page_number": meta.get("page_number", 1),
                        "chunk_index": meta.get("chunk_index", 0),
                        "distance": dist,
                    })

            # If strict threshold filtered all out, fallback to all retrieved document chunks
            if not contexts and documents:
                for i, doc in enumerate(documents):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    contexts.append({
                        "text": doc,
                        "page_number": meta.get("page_number", 1),
                        "chunk_index": meta.get("chunk_index", 0),
                        "distance": distances[i] if i < len(distances) else 0.0,
                    })

            return contexts
        except Exception:
            logger.exception("Error querying context from ChromaDB")
            return []

    def stream_answer(self, user_id: int, document_id: int, query: str) -> Iterator[str]:
        """
        Streams grounded answer tokens from Gemini 3.6 Flash based on retrieved snippets
        or full document context.
        """
        try:
            where_filter = {
                "$and": [
                    {"user_id": {"$eq": int(user_id)}},
                    {"document_id": {"$eq": int(document_id)}}
                ]
            }

            # Check total chunks for this document
            doc_data = self.collection.get(
                where=where_filter,
                include=["documents", "metadatas"]
            )
            doc_ids = doc_data.get("ids", []) if doc_data else []
            total_doc_chunks = len(doc_ids)

            # Self-healing: if document exists in DB but chunks are missing in ChromaDB, re-index on the fly
            if total_doc_chunks == 0:
                try:
                    from documents.models import Document
                    doc_obj = Document.objects.filter(id=document_id, user_id=user_id).first()
                    if doc_obj and doc_obj.file and os.path.exists(doc_obj.file.path):
                        logger.info("Self-healing: Re-indexing chunks for document %s", document_id)
                        with open(doc_obj.file.path, "rb") as f:
                            pdf_bytes = f.read()
                        chunks = self.extract_and_chunk_pdf(
                            file_bytes=pdf_bytes,
                            user_id=user_id,
                            document_id=document_id,
                            filename=doc_obj.filename
                        )
                        if chunks:
                            self.ingest_chunks(chunks)
                            doc_data = self.collection.get(
                                where=where_filter,
                                include=["documents", "metadatas"]
                            )
                            doc_ids = doc_data.get("ids", []) if doc_data else []
                            total_doc_chunks = len(doc_ids)
                except Exception:
                    logger.exception("Failed to auto-index missing document chunks")

            # If document is small/medium (<= 25 chunks, up to ~25 pages),
            # Gemini 3.6 Flash easily ingests the entire document in natural reading order.
            # This provides 100% full context and answers both general and specific questions.
            if 0 < total_doc_chunks <= 25:
                all_chunks: List[Dict[str, Any]] = []
                for i in range(total_doc_chunks):
                    meta = doc_data["metadatas"][i] if doc_data.get("metadatas") else {}
                    all_chunks.append({
                        "text": doc_data["documents"][i],
                        "page_number": meta.get("page_number", 1),
                        "chunk_index": meta.get("chunk_index", 0),
                    })
                all_chunks.sort(key=lambda x: (x["page_number"], x["chunk_index"]))
                contexts = all_chunks
            else:
                contexts = self.query_context(
                    user_id=user_id,
                    document_id=document_id,
                    query=query,
                    top_k=getattr(settings, "TOP_K", 10)
                )

            if contexts:
                formatted_context = "\n\n".join([
                    f"--- [Sahifa {c['page_number']}, Qism {i+1}] ---\n{c['text']}"
                    for i, c in enumerate(contexts)
                ])
                context_instruction = (
                    f"HUJJAT MATNI (KONTEKST):\n{formatted_context}\n\n"
                    "KO'RSATMA: Yuqoridagi hujjat matnidan foydalanib, foydalanuvchining savoliga batafsil, "
                    "aniq va to'liq javob bering. Hujjatdagi ma'lumotlarga qat'iy tayaning va tegishli sahifa raqamlarini "
                    "(masalan, [1-sahifa] yoki [Sahifa 1]) ko'rsating. Agar foydalanuvchi umumiy savol bersa (masalan, 'PDF nima haqida?'), "
                    "hujjatning to'liq mazmuni, unda keltirilgan shaxs/mavzu, asosiy bo'limlar va faktlarni to'liq ochib bering."
                )
            else:
                context_instruction = (
                    "Eslatma: Hujjatdan matn topilmadi. Foydalanuvchi savoliga umumiy bilimlaringiz asosida yordam bering."
                )

            full_prompt = (
                f"TIZIM KO'RSATMASI:\n{self.SYSTEM_PROMPT}\n\n"
                f"{context_instruction}\n\n"
                f"FOYDALANUVCHI SAVOLI:\n{query}\n\n"
                f"JAVOB:"
            )

            api_key = getattr(settings, "GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                yield "Gemini API kaliti (GEMINI_API_KEY) ko'rsatilmagan. Iltimos, sozlamalarda GEMINI_API_KEY ni sozlang."
                return

            primary_model = getattr(settings, "GEMINI_MODEL", "gemini-3.6-flash") or "gemini-3.6-flash"
            if "1.5" in primary_model:
                primary_model = "gemini-3.6-flash"

            # Prioritize gemini-3.6-flash as requested by user
            candidate_models = [primary_model]
            for fallback in ["gemini-2.5-flash", "gemini-flash-latest"]:
                if fallback not in candidate_models:
                    candidate_models.append(fallback)

            last_error = None
            success = False

            for model_candidate in candidate_models:
                try:
                    model = genai.GenerativeModel(
                        model_name=model_candidate,
                        generation_config={"temperature": 0.3}
                    )
                    response = model.generate_content(full_prompt, stream=True)
                    has_tokens = False
                    for chunk in response:
                        try:
                            if chunk.text:
                                has_tokens = True
                                yield chunk.text
                        except Exception:
                            pass
                    if has_tokens:
                        success = True
                        break
                except Exception as exc:
                    err_str = str(exc)
                    last_error = exc
                    logger.warning("Model %s generation failed: %s", model_candidate, err_str[:120])
                    # If 429 Quota or ResourceExhausted, automatically try next fallback candidate
                    if "429" in err_str or "quota" in err_str.lower() or "resourceexhausted" in err_str.lower():
                        continue
                    else:
                        break

            if not success and last_error:
                err_str = str(last_error)
                if "429" in err_str or "quota" in err_str.lower():
                    yield (
                        "\n\n⚠️ Google Gemini bepul tarif limiti (429 Quota Exceeded) vaqtinchalik to'ldi. "
                        "Iltimos, 1-2 daqiqadan so'ng qayta urinib ko'ring yoki .env faylida yangi GEMINI_API_KEY o'rnating."
                    )
                else:
                    yield f"\n\n[Javob yaratishda xatolik: {err_str}]"
        except Exception as exc:
            logger.exception("Error during LLM stream generation")
            yield f"\n\n[Javob yaratishda xatolik: {str(exc)}]"

    def delete_document_chunks(self, user_id: int, document_id: int) -> None:
        """Removes all indexed chunks for a specific document and tenant."""
        try:
            where_filter = {
                "$and": [
                    {"user_id": {"$eq": int(user_id)}},
                    {"document_id": {"$eq": int(document_id)}}
                ]
            }
            self.collection.delete(where=where_filter)
        except Exception:
            logger.exception("Failed to delete document chunks from ChromaDB")

    def reset_vector_store(self) -> None:
        """Deletes and recreates the ChromaDB collection (staff only)."""
        self.chroma_client.delete_collection(name=self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
