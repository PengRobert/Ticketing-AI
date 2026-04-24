"""
src/agents/response_agent.py
-----------------------------
Response Agent: compiles all analysis outputs into a single clean JSON object.
This is the final agent in the pipeline — its output is parsed by ticket_crew.py
and returned to the Streamlit UI.
"""

from crewai import Agent, Task, LLM

from config.settings import settings


def build_response_agent() -> Agent:
    llm = LLM(
        model=settings.CLAUDE_MODEL,
        api_key=settings.ANTHROPIC_API_KEY,
    )
    return Agent(
        role="Incident Report Compiler",
        goal=(
            "Consolidate all analytical findings from the classifier, RCA specialist, "
            "and solutions architect into a single, well-structured JSON report."
        ),
        backstory=(
            "You are a meticulous technical writer who specialises in producing "
            "structured incident reports consumed by both engineering teams and "
            "management. You extract the key findings from analyst notes and render "
            "them as clean, valid JSON — never adding fields that weren't determined "
            "by the analysis, and never omitting required fields."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def build_response_task(
    agent: Agent,
    classifier_task: Task,
    rca_task: Task,
    solution_task: Task,
) -> Task:
    return Task(
        description=(
            "Compile all prior analysis into a single JSON object.\n\n"
            "TICKET DESCRIPTION:\n{description}\n\n"
            "Extract each field using the exact labels shown in the agent outputs:\n\n"
            "  resolution_code — copy the value after 'Resolution Code:' in the "
            "Classifier output. NEVER output 'Unknown': if the classifier did not "
            "produce an explicit code, use the resolution code from the closest "
            "historical case provided in context.\n\n"
            "  priority — copy the value after 'Priority:' in the Classifier output "
            "(must be P1, P2, P3, or P4).\n\n"
            "  problem — copy the full text after 'Problem:' in the RCA output.\n\n"
            "  rca — copy the full text after 'RCA:' in the RCA output.\n\n"
            "  solution — copy ALL numbered steps from the 'Solution:' section of "
            "the Solutions Architect output. Do not truncate.\n\n"
            "Output ONLY a valid JSON object — no markdown fences, no explanation.\n\n"
            "Required schema:\n"
            "{{\n"
            '  "resolution_code": "<value from Classifier>",\n'
            '  "priority": "<P1|P2|P3|P4>",\n'
            '  "problem": "<full text from RCA Problem label>",\n'
            '  "rca": "<full text from RCA label>",\n'
            '  "solution": "<all numbered steps from Solution label>"\n'
            "}}\n\n"
            "Rules:\n"
            "- All 5 fields are required — never omit any.\n"
            "- Start the response with {{ and end with }}.\n"
            "- Use double-quoted strings; escape internal double quotes with \\."
        ),
        expected_output=(
            'A valid JSON object with exactly 5 keys: resolution_code, priority, '
            'problem, rca, solution. No markdown, no extra text.'
        ),
        agent=agent,
        context=[classifier_task, rca_task, solution_task],
    )
