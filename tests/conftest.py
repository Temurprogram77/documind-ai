import pytest
from django.contrib.auth.models import User
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken


@pytest.fixture
def api_client():
    return APIClient()


@pytest.fixture
def user_a(db):
    return User.objects.create_user(username="tenant_a", password="password123A!")


@pytest.fixture
def user_b(db):
    return User.objects.create_user(username="tenant_b", password="password123B!")


@pytest.fixture
def admin_user(db):
    return User.objects.create_superuser(username="admin_staff", password="adminpassword123!", email="admin@example.com")


@pytest.fixture
def auth_client_a(user_a):
    client = APIClient()
    token = str(RefreshToken.for_user(user_a).access_token)
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")
    return client


@pytest.fixture
def auth_client_b(user_b):
    client = APIClient()
    token = str(RefreshToken.for_user(user_b).access_token)
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")
    return client


@pytest.fixture
def auth_admin_client(admin_user):
    client = APIClient()
    token = str(RefreshToken.for_user(admin_user).access_token)
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")
    return client


@pytest.fixture(autouse=True)
def isolate_chroma_for_tests(tmp_path, settings):
    """
    Ensures tests use an isolated temporary directory and collection for ChromaDB,
    preventing any test resets from wiping actual user or development vectors.
    """
    import chromadb
    from documents.views import rag_service

    test_chroma_dir = str(tmp_path / "test_chroma")
    settings.CHROMA_PERSIST_DIR = test_chroma_dir
    settings.CHROMA_COLLECTION_NAME = "test_documind_vectors"

    old_client = rag_service.chroma_client
    old_coll = rag_service.collection
    old_dir = rag_service.persist_dir
    old_name = rag_service.collection_name

    rag_service.persist_dir = test_chroma_dir
    rag_service.collection_name = "test_documind_vectors"
    rag_service.chroma_client = chromadb.PersistentClient(path=test_chroma_dir)
    rag_service.collection = rag_service.chroma_client.get_or_create_collection(
        name="test_documind_vectors",
        metadata={"hnsw:space": "cosine"}
    )

    yield

    rag_service.persist_dir = old_dir
    rag_service.collection_name = old_name
    rag_service.chroma_client = old_client
    rag_service.collection = old_coll
