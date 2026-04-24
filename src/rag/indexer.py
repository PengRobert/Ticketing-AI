"""
src/rag/indexer.py
------------------
Loads the processed ticket CSV into ChromaDB for semantic search.

Each ticket is stored as a rich document combining description, problem,
worknotes, and metadata so that similarity search matches on issue semantics.

Usage:
    from src.rag.indexer import index_tickets
    index_tickets()                          # uses settings defaults
    index_tickets(force_reindex=True)        # rebuild from scratch
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from config.settings import settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------

_NA = {"nan", "none", "n/a", ""}


def _clean(val) -> str:
    s = str(val).strip() if pd.notna(val) else ""
    return "" if s.lower() in _NA else s


def _build_document(row: pd.Series) -> str:
    """
    Construct the text blob that gets embedded for a single ticket.
    More semantic signal = better similarity matching.
    """
    parts: list[str] = []

    brief = _clean(row.get("BRIEF_DESCRIPTION", ""))
    if brief:
        parts.append(f"Issue: {brief}")

    problem = _clean(row.get("PROBLEM", ""))
    if problem:
        parts.append(f"Problem: {problem}")

    rca = _clean(row.get("RCA", ""))
    if rca:
        parts.append(f"Root Cause: {rca}")

    solution = _clean(row.get("SOLUTION", ""))
    if solution:
        parts.append(f"Solution: {solution}")

    app = _clean(row.get("APPLICATION_NAME", ""))
    if app:
        parts.append(f"Application: {app}")

    product = _clean(row.get("PRODUCT_TYPE", ""))
    if product:
        parts.append(f"Product: {product}")

    ticket_type = _clean(row.get("TYPE", ""))
    if ticket_type:
        parts.append(f"Type: {ticket_type}")

    # Include first 400 chars of concatenated worknotes for extra signal
    notes = _clean(row.get("WORKNOTE", ""))
    if notes:
        parts.append(f"Notes: {notes[:400]}")

    return "\n".join(parts)


def _build_metadata(row: pd.Series) -> dict:
    """
    Metadata stored alongside each document — returned in query results
    so the retriever can build structured similar-case cards.
    """
    def safe(col: str) -> str:
        return _clean(row.get(col, "")) or "Unknown"

    return {
        "ticket_no": safe("TICKET_NO"),
        "resolution_code": safe("RESOLUTION_CODE"),
        "priority": safe("PRIORITY"),
        "sla": safe("RESPOND_SLA"),
        "application_name": safe("APPLICATION_NAME"),
        "product_type": safe("PRODUCT_TYPE"),
        "ticket_type": safe("TYPE"),
        "brief_description": safe("BRIEF_DESCRIPTION"),
        "problem": safe("PROBLEM"),
        "rca": safe("RCA"),
        "solution": safe("SOLUTION"),
    }


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

def _get_collection(client: chromadb.PersistentClient):
    # DefaultEmbeddingFunction uses all-MiniLM-L6-v2 via ONNX runtime —
    # no PyTorch dependency, works across environments.
    ef = DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def index_tickets(
    csv_path: Optional[str] = None,
    force_reindex: bool = False,
    batch_size: int = 100,
) -> int:
    """
    Index processed ticket CSV into ChromaDB.

    Args:
        csv_path:       Path to processed CSV. Defaults to settings.PROCESSED_DATA_PATH.
        force_reindex:  If True, deletes existing collection and rebuilds.
        batch_size:     Number of documents to upsert per batch.

    Returns:
        Number of documents indexed.
    """
    path = Path(csv_path or settings.PROCESSED_DATA_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed CSV not found: {path}\n"
            "Run: python data_processor.py --input data/raw/tickets.csv"
        )

    log.info("Loading processed CSV: %s", path)
    df = pd.read_csv(path, dtype=str).fillna("")

    if "TICKET_NO" not in df.columns:
        raise ValueError("CSV missing TICKET_NO column — is this the processed file?")

    # Initialise persistent ChromaDB
    persist_dir = Path(settings.CHROMA_PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    if force_reindex:
        try:
            client.delete_collection(settings.CHROMA_COLLECTION_NAME)
            log.info("Deleted existing collection for reindex.")
        except Exception:
            pass

    collection = _get_collection(client)
    existing_count = collection.count()

    if existing_count > 0 and not force_reindex:
        log.info(
            "Collection already has %d documents. Use force_reindex=True to rebuild.",
            existing_count,
        )
        return existing_count

    log.info("Building index for %d tickets …", len(df))

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for _, row in df.iterrows():
        doc = _build_document(row)
        if not doc.strip():
            continue
        documents.append(doc)
        metadatas.append(_build_metadata(row))
        ids.append(str(row.get("TICKET_NO", f"row_{_}")))

    # Upsert in batches to avoid memory spikes
    total = 0
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        collection.upsert(documents=batch_docs, metadatas=batch_meta, ids=batch_ids)
        total += len(batch_docs)
        log.info("  Indexed %d / %d …", total, len(documents))

    log.info("Indexing complete. Total documents in collection: %d", collection.count())
    return collection.count()
