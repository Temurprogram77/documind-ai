from django.db import models
from django.conf import settings


class Document(models.Model):
    """
    Tracks uploaded documents partitioned strictly per tenant (user).
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="documents"
    )
    file = models.FileField(upload_to="documents/%Y/%m/%d/")
    filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]
        indexes = [
            models.Index(fields=["user", "id"]),
        ]

    def __str__(self) -> str:
        return f"{self.filename} (Owner: {self.user.username})"