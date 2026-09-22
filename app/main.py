from contextlib import asynccontextmanager
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.api.routes import router as api_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager for startup and shutdown routines.
    """
    # Ensure ChromaDB persistent directory exists on startup
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise Document Intelligence SaaS Backend (RAG System with FastAPI & ChromaDB).",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Robust CORS middleware configuration allowing Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "X-Accel-Buffering"],
)

# Mount API endpoints under /api
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", summary="Root Status", tags=["General"])
def root():
    """
    Root status endpoint verifying service health.
    """
    return JSONResponse(
        content={
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "status": "operational",
            "documentation": "/docs"
        }
    )
