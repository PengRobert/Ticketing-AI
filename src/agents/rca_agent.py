"""
src/agents/rca_agent.py
-----------------------
RCA Agent: derives a crisp Problem Statement and Root Cause Analysis
from the ticket description and similar historical incidents.
"""

from crewai import Agent, Task, LLM

from config.settings import settings


def build_rca_agent() -> Agent:
    llm = LLM(
        model=settings.CLAUDE_MODEL,
        api_key=settings.ANTHROPIC_API_KEY,
    )
    return Agent(
        role="Root Cause Analysis Specialist",
        goal=(
            "Identify the precise problem statement and underlying root cause "
            "for any IT incident or service request."
        ),
        backstory=(
            "You are a principal systems engineer who has led post-incident reviews "
            "at multiple large enterprises. You are rigorous, precise, and trained to "
            "distinguish symptoms from root causes. You use the 5-Whys method and "
            "draw on historical incident patterns to pinpoint what actually went wrong "
            "and why — not just what the user reported."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def build_rca_task(agent: Agent, classifier_task: Task) -> Task:
    return Task(
        description=(
            "Perform root cause analysis for the IT support ticket below.\n\n"
            "TICKET DESCRIPTION:\n{description}\n\n"
            "SIMILAR HISTORICAL INCIDENTS (for pattern matching):\n"
            "{similar_cases_context}\n\n"
            "Your output MUST contain:\n"
            "1. PROBLEM STATEMENT — a single, precise sentence describing WHAT is "
            "broken or failing (the symptom from the user's perspective).\n"
            "2. ROOT CAUSE (RCA) — the underlying technical reason WHY the problem "
            "occurred. Be specific. Reference the historical cases if they reveal "
            "a recurring root cause pattern.\n"
            "Do NOT propose a solution here — that comes next."
        ),
        expected_output=(
            "A plain-text block with exactly two labelled fields:\n"
            "Problem: <precise one-sentence problem statement>\n"
            "RCA: <specific root cause explanation, 2-4 sentences>"
        ),
        agent=agent,
        context=[classifier_task],
    )
