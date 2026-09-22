import os
from typing import List, Dict, Any, Iterator
import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import Settings
from app.services.document_parser import DocumentChunk


class RAGEngine:
    """
    Enterprise RAG Service managing ChromaDB vector storage and multi-provider
    LLM streaming (Google Gemini and OpenAI) with strict zero-hallucination grounding.
    """

    STRICT_SYSTEM_PROMPT = (
        "You are a professional Document AI. "
        "Answer ONLY based on the context. "
        "If the answer is not in the context, say 'I cannot find this in the document.'"
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.persist_directory = settings.CHROMA_PERSIST_DIR
        self.collection_name = settings.CHROMA_COLLECTION_NAME

        # Ensure vector store persistence directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize persistent ChromaDB vector store
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Inserts document chunks and associated metadata into the ChromaDB vector collection.

        :param chunks: List of DocumentChunk models.
        :return: Count of successfully indexed chunks.
        """
        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page": chunk.page_number,
                "chunk_index": chunk.chunk_index
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        return len(chunks)

    def retrieve_context(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search against the indexed collection.

        :param query: Natural language user query.
        :param top_k: Number of nearest semantic chunks to return.
        :return: List of retrieved context snippets.
        """
        total_docs = self.collection.count()
        if total_docs == 0:
            return []

        limit = min(top_k, total_docs)
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )

        contexts: List[Dict[str, Any]] = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                contexts.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                })

        return contexts

    def _stream_openai(self, system_instruction: str, user_prompt: str) -> Iterator[str]:
        """
        Streams completions token-by-token from OpenAI.
        """
        from openai import OpenAI

        api_key = self.settings.OPENAI_API_KEY
        if not api_key or api_key == "your_openai_api_key_here":
            yield "⚠️ OPENAI_API_KEY is not configured in backend/.env. Please configure your API key."
            return

        client = OpenAI(api_key=api_key)
        stream = client.chat.completions.create(
            model=self.settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            stream=True
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _stream_gemini(self, system_instruction: str, user_prompt: str) -> Iterator[str]:
        """
        Streams completions token-by-token from Google Gemini.
        """
        import google.generativeai as genai

        api_key = self.settings.GEMINI_API_KEY
        if not api_key or api_key == "your_gemini_api_key_here":
            yield "⚠️ GEMINI_API_KEY is not configured in backend/.env. Please configure your API key."
            return

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=self.settings.GEMINI_MODEL,
            generation_config={"temperature": 0.2}
        )

        full_prompt = (
            f"SYSTEM INSTRUCTION:\n{system_instruction}\n\n"
            f"USER QUERY:\n{user_prompt}\n\n"
            "ANSWER:"
        )

        response = model.generate_content(full_prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def stream_rag_answer(self, query: str, top_k: int = 4) -> Iterator[str]:
        """
        Main RAG pipeline: Retrieves context from ChromaDB, builds strict prompt,
        and streams token-by-token via the configured LLM provider (Gemini or OpenAI).

        :param query: User question.
        :param top_k: Number of relevant chunks to retrieve.
        :yield: Generated text tokens.
        """
        contexts = self.retrieve_context(query, top_k=top_k)

        if not contexts:
            yield "I cannot find this in the document."
            return

        # Prepare context blocks
        context_blocks = []
        for i, item in enumerate(contexts, 1):
            meta = item.get("metadata", {})
            source_tag = f"[Doc: {meta.get('filename', 'Unknown')}, Page {meta.get('page', '?')}]"
            context_blocks.append(f"--- Context {i} {source_tag} ---\n{item['text']}")

        formatted_context = "\n\n".join(context_blocks)
        system_instruction = (
            f"{self.STRICT_SYSTEM_PROMPT}\n\n"
            f"DOCUMENT CONTEXT:\n{formatted_context}"
        )

        provider = self.settings.LLM_PROVIDER.lower().strip()

        try:
            if provider == "openai":
                yield from self._stream_openai(system_instruction=system_instruction, user_prompt=query)
            else:
                # Default to Gemini
                yield from self._stream_gemini(system_instruction=system_instruction, user_prompt=query)
        except Exception as exc:
            yield f"\n\n[Error generating response: {str(exc)}]"

    def count_chunks(self) -> int:
        """
        Returns the total number of indexed chunks.
        """
        return self.collection.count()

    def reset_store(self) -> None:
        """
        Purges all records in the ChromaDB collection.
        """
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
