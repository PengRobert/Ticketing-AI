"""
src/agents/solution_agent.py
-----------------------------
Solution Agent: recommends a concrete, actionable resolution
based on the identified root cause and historical solutions.
"""

from crewai import Agent, Task, LLM

from config.settings import settings


def build_solution_agent() -> Agent:
    llm = LLM(
        model=settings.CLAUDE_MODEL,
        api_key=settings.ANTHROPIC_API_KEY,
    )
    return Agent(
        role="IT Solutions Architect",
        goal=(
            "Recommend a clear, step-by-step solution that resolves the root cause "
            "and prevents recurrence, drawing on proven historical resolutions."
        ),
        backstory=(
            "You are a senior IT solutions architect who has resolved thousands of "
            "incidents across infrastructure, applications, and service management. "
            "You favour solutions that are direct and actionable. You always look at "
            "what worked in similar past cases and adapt those approaches to the "
            "current context. Your recommendations are specific enough that a "
            "technician can execute them without ambiguity."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def build_solution_task(agent: Agent, classifier_task: Task, rca_task: Task) -> Task:
    return Task(
        description=(
            "Recommend a resolution for the IT support ticket below.\n\n"
            "TICKET DESCRIPTION:\n{description}\n\n"
            "SIMILAR HISTORICAL SOLUTIONS (for reference):\n{similar_cases_context}\n\n"
            "Use the Problem Statement and Root Cause identified in the previous "
            "analysis steps to guide your recommendation.\n\n"
            "Your output MUST contain:\n"
            "SOLUTION — numbered, step-by-step resolution actions. Each step must be "
            "specific and actionable. Reference historical solutions where they apply. "
            "Include a preventive measure as the final step if applicable."
        ),
        expected_output=(
            "A plain-text block starting with:\n"
            "Solution:\n"
            "1. <action step>\n"
            "2. <action step>\n"
            "... (3-6 steps total)"
        ),
        agent=agent,
        context=[classifier_task, rca_task],
    )
