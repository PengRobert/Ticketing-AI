"""
src/rag/retriever.py
--------------------
Query ChromaDB to find historically similar tickets given a free-text description.

Returns structured dicts that are used both as context for the CrewAI agents
and as the "Top 3 Similar Cases" displayed in the Streamlit UI.
"""

import logging
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from config.settings import settings

log = logging.getLogger(__name__)


class TicketRetriever:
    """Thin wrapper around a ChromaDB collection for semantic ticket retrieval."""

    def __init__(self):
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None

    def _ensure_connected(self):
        if self._collection is not None:
            return
        ef = DefaultEmbeddingFunction()
        self._client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Return number of documents in the collection."""
        self._ensure_connected()
        return self._collection.count()

    def query(self, description: str, n_results: int = 5) -> list[dict]:
        """
        Find the most semantically similar historical tickets.

        Args:
            description: Free-text ticket description from the user.
            n_results:   Number of similar tickets to return.

        Returns:
            List of dicts, each containing:
                ticket_no, resolution_code, priority, sla,
                application_name, product_type, brief_description,
                problem, rca, solution, similarity_score (0–1, higher = more similar)
        """
        self._ensure_connected()

        total = self._collection.count()
        if total == 0:
            log.warning("ChromaDB collection is empty. Run build_index.py first.")
            return []

        n = min(n_results, total)
        results = self._collection.query(
            query_texts=[description],
            n_results=n,
            include=["metadatas", "distances"],
        )

        similar: list[dict] = []
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for meta, dist in zip(metadatas, distances):
            # ChromaDB cosine distance → similarity score (0–1)
            score = round(1.0 - float(dist), 4)
            similar.append(
                {
                    "ticket_no": meta.get("ticket_no") or meta.get("TICKET_NO") or "N/A",
                    "resolution_code": (meta.get("resolution_code") or meta.get("RESOLUTION_CODE") or "Unknown"),
                    "priority": meta.get("priority") or meta.get("PRIORITY") or "Unknown",
                    "sla": meta.get("sla") or meta.get("RESPOND_SLA") or "Unknown",
                    "application_name": meta.get("application_name") or meta.get("APPLICATION_NAME") or "Unknown",
                    "product_type": meta.get("product_type") or meta.get("PRODUCT_TYPE") or "Unknown",
                    "brief_description": meta.get("brief_description", ""),
                    "problem": meta.get("problem", ""),
                    "rca": meta.get("rca", ""),
                    "solution": meta.get("solution", ""),
                    "similarity_score": score,
                }
            )

        return similar


def format_similar_cases_for_prompt(cases: list[dict]) -> str:
    """
    Format retrieved similar cases as a readable text block for injection
    into CrewAI task descriptions.
    """
    if not cases:
        return "No similar historical cases found."

    lines: list[str] = []
    for i, case in enumerate(cases, start=1):
        score_pct = int(case["similarity_score"] * 100)
        lines.append(f"--- CASE {i} (Similarity: {score_pct}%) ---")
        lines.append(f"Ticket:          {case['ticket_no']}")
        lines.append(f"Description:     {case['brief_description']}")
        lines.append(f"Resolution Code: {case['resolution_code']}")
        lines.append(f"Priority:        {case['priority']}")
        lines.append(f"Problem:         {case['problem']}")
        lines.append(f"Root Cause:      {case['rca']}")
        lines.append(f"Solution:        {case['solution']}")
        lines.append(f"SLA Result:      {case['sla']}")
        lines.append("")

    return "\n".join(lines).strip()
