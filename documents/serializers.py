from rest_framework import serializers
from documents.models import Document


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ("id", "filename", "uploaded_at")
        read_only_fields = ("id", "filename", "uploaded_at")


class ChatQuerySerializer(serializers.Serializer):
    document_id = serializers.IntegerField(required=True, min_value=1)
    query = serializers.CharField(required=True, min_length=1, max_length=2000)