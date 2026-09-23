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
