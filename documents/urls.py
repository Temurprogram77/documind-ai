from django.urls import path
from documents.views import (
    DocumentUploadView,
    DocumentListView,
    DocumentDetailView,
    ChatSSEView,
    AdminResetStoreView,
    StatsView
)

urlpatterns = [
    path("documents/upload/", DocumentUploadView.as_view(), name="document_upload"),
    path("documents/", DocumentListView.as_view(), name="document_list"),
    path("documents/<int:pk>/", DocumentDetailView.as_view(), name="document_detail"),
    path("chat/", ChatSSEView.as_view(), name="chat_sse"),
    path("admin/reset/", AdminResetStoreView.as_view(), name="admin_reset_store"),
    path("stats/", StatsView.as_view(), name="document_stats"),
]