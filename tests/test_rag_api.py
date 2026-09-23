import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from documents.models import Document

# Minimal valid PDF binary
VALID_PDF_BYTES = (
    b"%PDF-1.4\n1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
    b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n"
    b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
    b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>endobj\n"
    b"4 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n"
    b"5 0 obj<< /Length 44 >>stream\nBT /F1 12 Tf 100 700 Td (Confidential Tenant A Document) Tj ET\nendstream\nendobj\n"
    b"xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n00000000117 00000 n \n0000000244 00000 n \n0000000319 00000 n \n"
    b"trailer<< /Root 1 0 R /Size 6 >>\nstartxref\n412\n%%EOF"
)


@pytest.mark.django_db
class TestRAGSecurity:

    def test_unauthenticated_requests_fail_with_401(self, api_client):
        """All protected endpoints require valid JWT authentication."""
        upload_url = reverse("document_upload")
        chat_url = reverse("chat_sse")
        docs_url = reverse("document_list")
        reset_url = reverse("admin_reset_store")

        assert api_client.post(upload_url).status_code == 401
        assert api_client.get(docs_url).status_code == 401
        assert api_client.post(chat_url, {"document_id": 1, "query": "test"}).status_code == 401
        assert api_client.delete(reset_url).status_code == 401

    def test_tenant_isolation_cross_user_access_blocked(self, auth_client_a, auth_client_b, user_a):
        """Tenant isolation prevents cross-user document chat queries."""
        doc_a = Document.objects.create(
            user=user_a,
            filename="tenant_a_confidential.pdf",
            file=SimpleUploadedFile("tenant_a.pdf", VALID_PDF_BYTES, content_type="application/pdf")
        )

        chat_url = reverse("chat_sse")
        payload = {"document_id": doc_a.id, "query": "What are the secrets?"}

        # Tenant B attempts to query Tenant A's document -> must receive 404
        response = auth_client_b.post(chat_url, payload, format="json")
        assert response.status_code == 404, "Tenant B must not access Tenant A's document."

    def test_document_list_isolated_between_tenants(self, auth_client_a, auth_client_b, user_a, user_b):
        """User A only sees their own documents in GET /api/documents/."""
        Document.objects.create(
            user=user_a,
            filename="user_a_file.pdf",
            file=SimpleUploadedFile("user_a.pdf", VALID_PDF_BYTES, content_type="application/pdf")
        )
        Document.objects.create(
            user=user_b,
            filename="user_b_file.pdf",
            file=SimpleUploadedFile("user_b.pdf", VALID_PDF_BYTES, content_type="application/pdf")
        )

        docs_url = reverse("document_list")

        res_a = auth_client_a.get(docs_url)
        assert res_a.status_code == 200
        filenames_a = [d["filename"] for d in res_a.data]
        assert "user_a_file.pdf" in filenames_a
        assert "user_b_file.pdf" not in filenames_a

        res_b = auth_client_b.get(docs_url)
        assert res_b.status_code == 200
        filenames_b = [d["filename"] for d in res_b.data]
        assert "user_b_file.pdf" in filenames_b
        assert "user_a_file.pdf" not in filenames_b

    def test_upload_file_size_exceeding_25mb_fails_with_413(self, auth_client_a):
        """Enforce hard 25MB file size limit on uploads."""
        upload_url = reverse("document_upload")
        large_content = b"0" * (25 * 1024 * 1024 + 10)  # 25MB + 10 bytes
        file_obj = SimpleUploadedFile("oversized.pdf", large_content, content_type="application/pdf")

        response = auth_client_a.post(upload_url, {"file": file_obj}, format="multipart")
        assert response.status_code == 413, "Files > 25MB must be rejected with 413."

    def test_upload_invalid_mime_type_fails_with_415(self, auth_client_a):
        """Enforce application/pdf MIME type check."""
        upload_url = reverse("document_upload")
        fake_file = SimpleUploadedFile("test.pdf", b"fake binary", content_type="text/plain")

        response = auth_client_a.post(upload_url, {"file": fake_file}, format="multipart")
        assert response.status_code == 415

    def test_admin_reset_endpoint_security(self, auth_client_a, auth_admin_client):
        """Non-admin users cannot trigger vector store reset."""
        reset_url = reverse("admin_reset_store")

        # Regular authenticated user -> 403 Forbidden
        assert auth_client_a.delete(reset_url).status_code == 403

        # Superuser / Staff member -> 200 OK
        assert auth_admin_client.delete(reset_url).status_code == 200


@pytest.mark.django_db
class TestAuthenticationEndpoints:

    def test_register_user_success(self, api_client):
        """Valid registration creates user and returns 201."""
        url = reverse("auth_register")
        payload = {
            "username": "new_engineer",
            "email": "engineer@documind.ai",
            "password": "SecurePassword123!",
            "password_confirm": "SecurePassword123!"
        }
        res = api_client.post(url, payload, format="json")
        assert res.status_code == 201
        assert res.data["username"] == "new_engineer"

    def test_register_duplicate_username_fails(self, api_client, user_a):
        """Registration with existing username returns 400."""
        url = reverse("auth_register")
        payload = {
            "username": user_a.username,
            "password": "SecurePassword123!",
            "password_confirm": "SecurePassword123!"
        }
        res = api_client.post(url, payload, format="json")
        assert res.status_code == 400

    def test_login_success(self, api_client, user_a):
        """Valid login returns JWT access and refresh tokens."""
        url = reverse("token_obtain_pair")
        payload = {
            "username": user_a.username,
            "password": "password123A!"
        }
        res = api_client.post(url, payload, format="json")
        assert res.status_code == 200
        assert "access" in res.data
        assert "refresh" in res.data

    def test_login_invalid_password_fails(self, api_client, user_a):
        """Invalid credentials return 401 Unauthorized."""
        url = reverse("token_obtain_pair")
        payload = {
            "username": user_a.username,
            "password": "wrong_password!"
        }
        res = api_client.post(url, payload, format="json")
        assert res.status_code == 401
