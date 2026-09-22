# =========================================================================
# DocuMind AI Backend - Production Dockerfile (Ubuntu VPS Optimized)
# =========================================================================
FROM python:3.11-slim

# Prevent bytecode caching and enable immediate unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install minimal OS dependencies for healthchecks and builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Optimize layer caching: Install Python packages before copying code
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend application source
COPY app ./app

# Create ChromaDB persistent vector storage directory
RUN mkdir -p /app/chroma_data

# Expose FastAPI port
EXPOSE 8000

# Docker healthcheck for reverse proxies and orchestrators
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Production uvicorn server with proxy headers for Nginx/domain compatibility
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
