"""
src/agents/classifier_agent.py
-------------------------------
Classifier Agent: determines Resolution Code and Priority from a ticket description
and similar historical cases.
"""

from crewai import Agent, Task, LLM

from config.settings import settings


def build_classifier_agent() -> Agent:
    llm = LLM(
        model=settings.CLAUDE_MODEL,
        api_key=settings.ANTHROPIC_API_KEY,
    )
    return Agent(
        role="Senior IT Ticket Classifier",
        goal=(
            "Accurately classify IT support tickets by assigning the correct "
            "Resolution Code category and the appropriate Priority level."
        ),
        backstory=(
            "You are a seasoned IT support analyst with 10+ years of experience "
            "triaging and classifying thousands of support tickets across enterprise "
            "systems. You can instantly recognise issue patterns, match them to the "
            "correct resolution category, and assess business impact to set the right "
            "priority. You always base your classification on evidence from the ticket "
            "description and cross-reference with historical cases."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def build_classifier_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Analyze the IT support ticket below and classify it.\n\n"
            "TICKET DESCRIPTION:\n{description}\n\n"
            "SIMILAR HISTORICAL TICKETS (for reference):\n{similar_cases_context}\n\n"
            "Your output MUST contain:\n"
            "1. RESOLUTION CODE — pick the single most appropriate category based on "
            "the historical examples above. Use the exact wording from the examples.\n"
            "2. PRIORITY — one of: P1 (Critical), P2 (High), P3 (Medium), P4 (Low). "
            "Base this on business impact and urgency described in the ticket.\n"
            "3. CLASSIFICATION RATIONALE — 1-2 sentences explaining why you chose "
            "this code and priority."
        ),
        expected_output=(
            "A plain-text block with exactly three labelled fields:\n"
            "Resolution Code: <value>\n"
            "Priority: <P1|P2|P3|P4>\n"
            "Rationale: <1-2 sentence justification>"
        ),
        agent=agent,
    )
