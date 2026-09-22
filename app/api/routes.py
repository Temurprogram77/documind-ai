import json
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.dependencies import get_document_parser_service, get_rag_engine_service
from app.services.document_parser import DocumentParser
from app.services.rag_engine import RAGEngine

router = APIRouter()


class UploadSuccessResponse(BaseModel):
    status: str = Field(default="success", description="Status string")
    chunks_processed: int = Field(..., description="Total number of chunks extracted and indexed")
    document_id: Optional[str] = Field(default=None, description="UUID of the processed document")
    filename: Optional[str] = Field(default=None, description="Original filename")


class ChatQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question asked by the user")
    top_k: Optional[int] = Field(default=4, ge=1, le=10, description="Context retrieval limit")


@router.get("/health", summary="Health check endpoint")
def health_check():
    """
    Verifies service vitality.
    """
    return {"status": "healthy", "service": "DocuMind AI Backend"}


@router.post(
    "/upload",
    response_model=UploadSuccessResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload PDF and index into ChromaDB"
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF document to parse and index"),
    parser: DocumentParser = Depends(get_document_parser_service),
    rag_engine: RAGEngine = Depends(get_rag_engine_service),
) -> UploadSuccessResponse:
    """
    Accepts multipart/form-data (PDF), extracts text into overlapping chunks,
    stores vectors in ChromaDB, and returns processing count.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files are supported."
        )

    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not read upload buffer: {str(exc)}"
        )

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded PDF file is empty."
        )

    doc_id = str(uuid.uuid4())

    try:
        chunks = parser.parse_pdf(
            file_bytes=file_bytes,
            filename=file.filename,
            document_id=doc_id
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse PDF document: {str(exc)}"
        )

    if not chunks:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail="Could not extract readable text from PDF. The document may be scanned or image-based."
        )

    try:
        inserted_count = rag_engine.add_chunks(chunks)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database vector ingestion failed: {str(exc)}"
        )

    return UploadSuccessResponse(
        status="success",
        chunks_processed=inserted_count,
        document_id=doc_id,
        filename=file.filename
    )


@router.post(
    "/chat",
    summary="Query document intelligence with real-time SSE streaming"
)
async def chat_endpoint(
    payload: ChatQueryRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine_service),
) -> StreamingResponse:
    """
    Accepts query JSON, performs vector similarity search in ChromaDB (top_k=4),
    injects context into strict zero-hallucination prompt, and streams LLM tokens.
    """
    query_text = payload.query.strip()
    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter cannot be empty."
        )

    def sse_event_stream():
        try:
            for token in rag_engine.stream_rag_answer(query=query_text, top_k=payload.top_k or 4):
                # Standard Server-Sent Events structure
                payload_json = json.dumps({"token": token})
                yield f"data: {payload_json}\n\n"

            # SSE Completion sentinel
            yield "data: [DONE]\n\n"

        except Exception as exc:
            err_json = json.dumps({"error": str(exc)})
            yield f"data: {err_json}\n\n"

    return StreamingResponse(
        sse_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disables proxy buffering (e.g. Nginx on VPS)
            "Content-Type": "text/event-stream; charset=utf-8",
        }
    )


@router.get("/stats", summary="Vector Store Statistics")
def get_stats(rag_engine: RAGEngine = Depends(get_rag_engine_service)):
    return {"total_chunks": rag_engine.count_chunks()}


@router.delete("/reset", summary="Purge Vector Store")
def reset_store(rag_engine: RAGEngine = Depends(get_rag_engine_service)):
    rag_engine.reset_store()
    return {"status": "success", "message": "Vector store reset successfully."}
