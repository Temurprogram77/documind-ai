# DocuMind AI - Backend Service

High-performance, multi-tenant RAG (Retrieval-Augmented Generation) document intelligence backend built with Django 5, Django REST Framework (DRF), SimpleJWT, ChromaDB, and Google Gemini.

---

## 🏛 Architecture & Key Features

* **Multi-Tenant Isolation**: Every uploaded document and indexed chunk in ChromaDB is tagged with `user_id` and `document_id`. Vector retrieval uses strict `$and: [{"user_id": {"$eq": user_id}}, {"document_id": {"$eq": document_id}}]` filters, preventing cross-tenant data leakage.
* **JWT Authentication**: Secured with `djangorestframework-simplejwt` with rotating refresh tokens and token blacklisting.
* **Upload Security**: Hard 25MB file size limit enforced before loading into memory; PDF MIME validation (`application/pdf`) and rate throttling.
* **Server-Sent Events (SSE) Streaming**: Token-by-token streaming chat via `StreamingHttpResponse` with custom `ServerSentEventRenderer` and open content negotiation.
* **Smart Chunking**: Sentence-boundary aware text splitting preserving semantic completeness with overlap.
* **Role-Based Access Control**: Sensitive actions like collection reset (`/api/admin/reset/`) are strictly restricted to staff/superusers (`IsAdminUser`).

---

## 🛠 Tech Stack

* **Framework**: Django 5.1, Django REST Framework 3.17
* **Authentication**: SimpleJWT 5.5
* **Vector Store**: ChromaDB 1.5
* **PDF Parser**: PyPDF 6.19
* **LLM Engine**: Google Generative AI (Gemini 1.5 Flash)
* **WSGI Server**: Gunicorn 26.2
* **Testing**: Pytest, Pytest-Django

---

## 🚀 Getting Started

### 1. Local Development Setup

```bash
# 1. Clone repository
git clone https://github.com/Temurprogram77/documind-ai.git
cd documind-ai

# 2. Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and supply your GEMINI_API_KEY

# 5. Run database migrations
python manage.py migrate

# 6. Start development server
python manage.py runserver 0.0.0.0:8000
```

---

## 🧪 Running the Test Suite

```bash
# Run all tests with pytest
pytest -v tests/

# Run with verbose output and coverage
pytest -q tests/
```

### Test Coverage includes:
- **Authentication**: Registration, Login, Token Refresh, Me endpoint, Duplicate prevention.
- **Tenant Isolation**: Verifying User A cannot access or delete User B's documents or vectors.
- **File Upload Limits**: Enforcing 25MB size ceiling and PDF MIME checks.
- **Admin Security**: Ensuring non-admins receive 403 Forbidden on store reset.
- **RAG Evaluation**: Sentence-boundary chunking, golden Q&A dataset retrieval hit-rate (>= 80%), and page citations.

---

## 📡 API Specification

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/api/auth/register/` | Public | Register a new user |
| `POST` | `/api/auth/login/` | Public | Login and obtain JWT access & refresh tokens |
| `POST` | `/api/auth/refresh/` | Public | Refresh expired access token |
| `GET` | `/api/auth/me/` | Bearer | Get current user profile |
| `GET` | `/api/documents/` | Bearer | List documents owned by authenticated user |
| `POST` | `/api/documents/upload/` | Bearer | Upload and index PDF (max 25MB, rate limited) |
| `DELETE` | `/api/documents/<id>/` | Bearer | Delete document and purge its vector chunks |
| `POST` | `/api/chat/` | Bearer | Query document via real-time SSE stream |
| `GET` | `/api/stats/` | Public | View vector store status and indexed chunk count |
| `DELETE` | `/api/admin/reset/` | Staff Only | Purge vector store collection |
| `GET` | `/api/health/` | Public | Health check endpoint |

---

## 🐳 Docker Deployment & Persistent Volumes

```bash
# Build Docker image
docker build -t documind-ai .

# Run container with persistent ChromaDB volume
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/chroma_data:/app/chroma_data \
  -e GEMINI_API_KEY="your-gemini-key" \
  --name documind-backend \
  documind-ai
```
