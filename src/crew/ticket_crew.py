"""
src/crew/ticket_crew.py
------------------------
Main entry point for ticket analysis.

Workflow:
    1. Retrieve top-K similar historical tickets from ChromaDB (RAG).
    2. Format them as a text context block injected into every task.
    3. Run the 4-agent CrewAI pipeline sequentially.
    4. Parse the Response Agent's JSON output into a clean Python dict.
    5. Attach the top-3 similar cases from step 1 to the result.

Public API:
    result = analyze_ticket("Users cannot log in after the weekend patch rollout")
    # result is an AnalysisResult dataclass
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from crewai import Crew, Process

from config.settings import settings
from src.agents import (
    build_classifier_agent, build_classifier_task,
    build_rca_agent, build_rca_task,
    build_solution_agent, build_solution_task,
    build_response_agent, build_response_task,
)
from src.rag.retriever import TicketRetriever, format_similar_cases_for_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# Singleton retriever — reuse the ChromaDB connection across calls
_retriever: Optional[TicketRetriever] = None


def _get_retriever() -> TicketRetriever:
    global _retriever
    if _retriever is None:
        _retriever = TicketRetriever()
    return _retriever


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimilarCase:
    ticket_no: str
    resolution_code: str
    priority: str
    sla: str
    brief_description: str
    problem: str
    solution: str
    similarity_score: float


@dataclass
class AnalysisResult:
    resolution_code: str
    priority: str
    problem: str
    rca: str
    solution: str
    similar_cases: list[SimilarCase] = field(default_factory=list)
    raw_output: str = ""
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# JSON parsing (robust — handles LLM formatting quirks)
# ---------------------------------------------------------------------------

def _fix_json_newlines(text: str) -> str:
    """
    Replace literal newlines/tabs inside JSON string values with their escape
    sequences so that json.loads() can handle them.  Uses a simple state machine
    to avoid touching structural whitespace outside of strings.
    """
    result: list[str] = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
        elif ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            result.append("\\r")
        elif in_string and ch == "\t":
            result.append("\\t")
        else:
            result.append(ch)
    return "".join(result)


def _parse_json_output(raw: str) -> dict:
    """
    Extract and parse the JSON object from the Response Agent's raw output.
    Handles common LLM formatting issues:
      - Markdown code fences anywhere in the text (not only at the very start)
      - Leading/trailing prose
      - Literal newlines inside JSON string values
      - Escaped quotes inside string values
    """
    text = raw.strip()
    log.info("Raw agent output (%d chars):\n%.3000s", len(text), text)

    # Strip markdown code fences wherever they appear (not just at position 0)
    text = re.sub(r"```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```", "", text)
    text = text.strip()

    # Attempt direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix literal newlines inside string values and retry
    try:
        return json.loads(_fix_json_newlines(text))
    except json.JSONDecodeError:
        pass

    # Extract first {...} block (handles preamble/postamble prose)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Fix literal newlines in the extracted block and retry
        try:
            return json.loads(_fix_json_newlines(candidate))
        except json.JSONDecodeError:
            pass

    # Last resort: regex extraction of individual fields.
    # Pattern ((?:[^"\\]|\\.)*) matches any char except unescaped quote,
    # so it correctly handles escaped quotes AND multi-line values.
    log.warning("JSON parse failed; falling back to regex field extraction.")
    result: dict = {}
    patterns = {
        "resolution_code": r'"resolution_code"\s*:\s*"((?:[^"\\]|\\.)*)"',
        "priority":        r'"priority"\s*:\s*"((?:[^"\\]|\\.)*)"',
        "problem":         r'"problem"\s*:\s*"((?:[^"\\]|\\.)*)"',
        "rca":             r'"rca"\s*:\s*"((?:[^"\\]|\\.)*)"',
        "solution":        r'"solution"\s*:\s*"((?:[^"\\]|\\.)*)"',
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            # Decode any \\n escape sequences from the captured value
            result[key] = m.group(1).replace("\\n", "\n").replace("\\t", "\t").strip()
        else:
            result[key] = ""

    return result


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def analyze_ticket(description: str) -> AnalysisResult:
    """
    Run the full 4-agent analysis pipeline for a ticket description.

    Args:
        description: Free-text ticket description provided by the user.

    Returns:
        AnalysisResult with all fields populated.
    """
    description = description.strip()
    if not description:
        return AnalysisResult(
            resolution_code="", priority="", problem="", rca="", solution="",
            success=False, error="Ticket description cannot be empty.",
        )

    # ── Step 1: RAG retrieval ────────────────────────────────────────────────
    log.info("Retrieving similar historical tickets …")
    retriever = _get_retriever()
    similar_raw = retriever.query(description, n_results=settings.RAG_TOP_K)
    similar_context = format_similar_cases_for_prompt(similar_raw)

    inputs = {
        "description": description,
        "similar_cases_context": similar_context,
    }

    # ── Step 2: Build agents ─────────────────────────────────────────────────
    classifier_agent  = build_classifier_agent()
    rca_agent         = build_rca_agent()
    solution_agent    = build_solution_agent()
    response_agent    = build_response_agent()

    # ── Step 3: Build tasks (chain context) ──────────────────────────────────
    classifier_task = build_classifier_task(classifier_agent)
    rca_task        = build_rca_task(rca_agent, classifier_task)
    solution_task   = build_solution_task(solution_agent, classifier_task, rca_task)
    response_task   = build_response_task(
        response_agent, classifier_task, rca_task, solution_task
    )

    # ── Step 4: Run crew ─────────────────────────────────────────────────────
    crew = Crew(
        agents=[classifier_agent, rca_agent, solution_agent, response_agent],
        tasks=[classifier_task, rca_task, solution_task, response_task],
        process=Process.sequential,
        verbose=False,
    )

    log.info("Starting CrewAI pipeline …")
    crew_output = crew.kickoff(inputs=inputs)
    log.info("Crew pipeline complete.")

    # Prefer the response agent's task output directly — more reliable than
    # crew_output.raw which can contain concatenated output from all tasks.
    if hasattr(crew_output, "tasks_output") and crew_output.tasks_output:
        raw = str(crew_output.tasks_output[-1].raw)
        log.debug("Parsing from tasks_output[-1].raw")
    elif hasattr(crew_output, "raw"):
        raw = str(crew_output.raw)
        log.debug("Parsing from crew_output.raw")
    else:
        raw = str(crew_output)
        log.debug("Parsing from str(crew_output)")
    log.debug("Source length: %d chars", len(raw))

    # ── Step 5: Parse output ─────────────────────────────────────────────────
    parsed = _parse_json_output(raw)

    # ── Step 6: Build similar case objects (top 3) ───────────────────────────
    top3 = [
        SimilarCase(
            ticket_no=c["ticket_no"],
            resolution_code=c["resolution_code"],
            priority=c["priority"],
            sla=c["sla"],
            brief_description=c["brief_description"],
            problem=c["problem"],
            solution=c["solution"],
            similarity_score=c["similarity_score"],
        )
        for c in similar_raw[:3]
    ]

    return AnalysisResult(
        resolution_code=parsed.get("resolution_code", "Unknown"),
        priority=parsed.get("priority", "Unknown"),
        problem=parsed.get("problem", ""),
        rca=parsed.get("rca", ""),
        solution=parsed.get("solution", ""),
        similar_cases=top3,
        raw_output=raw,
        success=True,
    )
