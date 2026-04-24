"""
config/settings.py
------------------
Central settings loaded from environment variables / .env file.
Import `settings` everywhere instead of reading os.getenv directly.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent


class _Settings:
    # LLM
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db")
    )
    CHROMA_COLLECTION_NAME: str = "tickets"

    # Embeddings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Retrieval
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))

    # Data paths
    RAW_DATA_PATH: str = str(BASE_DIR / "data" / "raw" / "tickets.csv")
    PROCESSED_DATA_PATH: str = str(
        BASE_DIR / "data" / "processed" / "tickets_processed.csv"
    )


settings = _Settings()
