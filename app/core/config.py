import os
from pathlib import Path
from functools import lru_cache
from typing import List
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Automatically locate and load .env from backend/ or root
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend dir
ROOT_DIR = BASE_DIR.parent

# Pre-load environment variables if .env exists
if (BASE_DIR / ".env").exists():
    load_dotenv(BASE_DIR / ".env", override=False)
elif (ROOT_DIR / ".env").exists():
    load_dotenv(ROOT_DIR / ".env", override=False)
elif Path(".env").exists():
    load_dotenv(".env", override=False)


class Settings(BaseSettings):
    """
    Enterprise Application Settings for DocuMind AI Backend.
    Loads configurations seamlessly from environment variables and .env files.
    """
    model_config = SettingsConfigDict(
        env_file=(
            str(BASE_DIR / ".env"),
            str(ROOT_DIR / ".env"),
            ".env",
            "backend/.env",
        ),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    PROJECT_NAME: str = "DocuMind AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"

    # CORS configuration allowing frontend origin
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://frontend:3000",
        "*"
    ]

    # LLM Provider selection: "gemini" or "openai"
    LLM_PROVIDER: str = "gemini"

    # Google Gemini Settings
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # OpenAI Settings
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # ChromaDB Vector Store Persistence
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    CHROMA_COLLECTION_NAME: str = "documind_collection"

    # Document Chunking Parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 4


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings singleton provider.
    """
    return Settings()
