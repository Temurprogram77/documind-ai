import json
import logging
from typing import Generator
from django.conf import settings
from django.http import StreamingHttpResponse
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework.permissions import IsAuthenticated, IsAdminUser, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.renderers import BaseRenderer, JSONRenderer
from rest_framework.negotiation import DefaultContentNegotiation

from documents.models import Document
from documents.serializers import DocumentSerializer, ChatQuerySerializer
from documents.services.rag import RAGService

logger = logging.getLogger(__name__)
rag_service = RAGService()


class DocumentUploadView(APIView):
    """
    POST /api/documents/upload/
    Upload endpoint with strict file validation and rate limiting.
    1. Validates MIME type == application/pdf.
    2. Enforces hard 25MB file size limit.
    3. Saves file, chunks text, and vectorizes into ChromaDB with tenant metadata.
    """
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "uploads"

    def post(self, request: Request) -> Response:
        uploaded_file = request.FILES.get("file")

        if not uploaded_file:
            return Response(
                {"error": "Bad Request: No file provided under key 'file'."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 1. Strict MIME type validation BEFORE reading into memory
        if uploaded_file.content_type != "application/pdf":
            return Response(
                {"error": f"Unsupported Media Type: Expected 'application/pdf', received '{uploaded_file.content_type}'."},
                status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
            )

        if not uploaded_file.name.lower().endswith(".pdf"):
            return Response(
                {"error": "Invalid file extension. Only .pdf files are accepted."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 2. Strict file size validation (< 25MB)
        if uploaded_file.size > settings.MAX_FILE_SIZE:
            return Response(
                {"error": f"Payload Too Large: File size ({uploaded_file.size} bytes) exceeds the 25MB limit."},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            )

        if uploaded_file.size == 0:
            return Response(
                {"error": "Bad Request: Uploaded file is empty."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 3. Read and verify content
        try:
            file_bytes = uploaded_file.read()
        except Exception:
            logger.exception("Error reading file upload buffer")
            return Response(
                {"error": "Internal server error reading upload buffer."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 4. Save to Database with tenant ownership
        document = Document.objects.create(
            user=request.user,
            file=uploaded_file,
            filename=uploaded_file.name
        )

        # 5. Extract, chunk, and index into ChromaDB
        try:
            chunks = rag_service.extract_and_chunk_pdf(
                file_bytes=file_bytes,
                user_id=request.user.id,
                document_id=document.id,
                filename=document.filename
            )

            if not chunks:
                document.delete()
                return Response(
                    {"error": "Unprocessable Entity: No extractable text found in PDF (scanned or protected)."},
                    status=status.HTTP_422_UNPROCESSABLE_ENTITY
                )

            inserted_count = rag_service.ingest_chunks(chunks)
        except Exception:
            logger.exception("Error ingesting document into vector store")
            document.delete()
            return Response(
                {"error": "Failed to index document into vector store. Please try again."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response(
            {
                "status": "success",
                "document_id": document.id,
                "filename": document.filename,
                "chunks_processed": inserted_count
            },
            status=status.HTTP_201_CREATED
        )


class DocumentListView(APIView):
    """
    GET /api/documents/
    Returns documents owned exclusively by the authenticated user.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request: Request) -> Response:
        docs = Document.objects.filter(user=request.user)
        serializer = DocumentSerializer(docs, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ServerSentEventRenderer(BaseRenderer):
    media_type = "text/event-stream"
    format = "text"
    charset = "utf-8"

    def render(self, data, accepted_media_type=None, renderer_context=None):
        if isinstance(data, (dict, list)):
            return json.dumps(data)
        return data


class OpenContentNegotiation(DefaultContentNegotiation):
    """
    Permissive content negotiator for SSE streaming endpoints.
    Allows text/event-stream, application/json, and wildcards without 406 NotAcceptable.
    """
    def select_renderer(self, request, renderers, format_suffix=None):
        try:
            return super().select_renderer(request, renderers, format_suffix)
        except Exception:
            return (renderers[0], renderers[0].media_type)


class ChatSSEView(APIView):
    """
    POST /api/chat/
    Multi-tenant Server-Sent Events (SSE) chat endpoint.
    Streams token-by-token using Django's StreamingHttpResponse.
    """
    permission_classes = [IsAuthenticated]
    renderer_classes = [ServerSentEventRenderer, JSONRenderer]
    content_negotiation_class = OpenContentNegotiation

    def post(self, request: Request) -> Response:
        serializer = ChatQuerySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        doc_id = serializer.validated_data["document_id"]
        query = serializer.validated_data["query"]

        # Strict Tenant Isolation: Ensure document exists and belongs to request.user
        document = get_object_or_404(Document, id=doc_id, user=request.user)

        def event_stream_generator() -> Generator[str, None, None]:
            try:
                for token in rag_service.stream_answer(
                    user_id=request.user.id,
                    document_id=document.id,
                    query=query
                ):
                    data = json.dumps({"token": token})
                    yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"
            except Exception:
                logger.exception("Error during SSE stream response generation")
                err = json.dumps({"error": "An error occurred while generating the answer."})
                yield f"data: {err}\n\n"

        response = StreamingHttpResponse(
            event_stream_generator(),
            content_type="text/event-stream; charset=utf-8"
        )
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"  # Disable Nginx proxy buffering
        return response


class AdminResetStoreView(APIView):
    """
    DELETE /api/admin/reset/
    Admin-only endpoint to purge the vector store.
    Strictly protected by IsAdminUser.
    """
    permission_classes = [IsAdminUser]

    def delete(self, request: Request) -> Response:
        logger.warning(
            f"ADMIN ACTION: ChromaDB collection purge initiated by staff user: "
            f"'{request.user.username}' (ID: {request.user.id})"
        )
        try:
            rag_service.reset_vector_store()
            return Response(
                {"status": "success", "message": "ChromaDB collection successfully purged."},
                status=status.HTTP_200_OK
            )
        except Exception:
            logger.exception("Failed to purge ChromaDB collection")
            return Response(
                {"error": "Failed to reset store. An internal error occurred."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class StatsView(APIView):
    """
    GET /api/stats/
    Returns vector store statistics. Publicly accessible.
    """
    permission_classes = [AllowAny]

    def get(self, request: Request) -> Response:
        try:
            total_chunks = rag_service.collection.count()
        except Exception:
            logger.exception("Failed to retrieve ChromaDB collection count")
            total_chunks = 0

        return Response(
            {
                "total_chunks": total_chunks,
                "indexed_chunks": total_chunks,
                "status": "operational",
            },
            status=status.HTTP_200_OK,
        )

