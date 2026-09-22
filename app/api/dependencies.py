from typing import Optional
from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.document_parser import DocumentParser
from app.services.rag_engine import RAGEngine

_document_parser_instance: Optional[DocumentParser] = None
_rag_engine_instance: Optional[RAGEngine] = None


def get_document_parser_service(
    settings: Settings = Depends(get_settings)
) -> DocumentParser:
    """
    Provides a singleton instance of DocumentParser configured with application defaults.
    """
    global _document_parser_instance
    if _document_parser_instance is None:
        _document_parser_instance = DocumentParser(
            default_chunk_size=settings.CHUNK_SIZE,
            default_chunk_overlap=settings.CHUNK_OVERLAP
        )
    return _document_parser_instance


def get_rag_engine_service(
    settings: Settings = Depends(get_settings)
) -> RAGEngine:
    """
    Provides a singleton instance of RAGEngine connected to the persistent ChromaDB store.
    """
    global _rag_engine_instance
    if _rag_engine_instance is None:
        _rag_engine_instance = RAGEngine(settings=settings)
    return _rag_engine_instance
