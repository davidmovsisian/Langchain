# extends the handsoff_with_middleware.py example to include sub-agents inside tools

from langchain_core.utils.uuid import uuid7
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typing import Literal, Callable
from typing_extensions import NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage, SystemMessage
from langchain.tools import tool, ToolRuntime

# handsoff multiagent with middleware + sub-agents inside tools

model = init_chat_model(model="openai:gpt-4o", temperature=0)

# workflow steps
SupportStep = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]

class SupportState(AgentState):
    current_step: NotRequired[SupportStep]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]


# ---------------------------------------------------------------------------
# Sub-agent helpers
# Each sub-agent is a single LLM call with a focused system prompt.
# They return a plain string (the AI's message content).
# ---------------------------------------------------------------------------

warranty_recorder_subagent = create_agent(
    model,
    system_prompt=SystemMessage(content=(
        "You are a customer support assistant. "
        "The customer's warranty status has just been recorded. "
        "Compose a brief, warm acknowledgment (1-2 sentences) confirming "
        "the status and letting the customer know you will now ask about their issue. "
        "Do NOT ask any questions yet."
    )),
    # checkpointer=InMemorySaver(),
)

issue_recorder_subagent = create_agent(
    model,
    system_prompt=SystemMessage(content=(
        "You are a customer support assistant. "
        "The customer's warranty status has just been recorded. "
        "Compose a brief, warm acknowledgment (1-2 sentences) confirming "
        "the status and letting the customer know you will now ask about their issue. "
        "Do NOT ask any questions yet."
    )),
    # checkpointer=InMemorySaver(),
)

solution_subagent = create_agent(
    model,
    system_prompt=SystemMessage(content=(
        "You are an expert customer support specialist. "
        "Given the raw solution outline, expand it into a clear, empathetic, "
        "step-by-step response for the customer. "
        "Be concise but thorough. Tailor your tone to the warranty/issue context."
    )),
    # checkpointer=InMemorySaver(),
)

escalation_subagent = create_agent(
    model,
    system_prompt=SystemMessage(content=(
        "You are a customer support specialist handling escalations. "
        "A case is being escalated to a human specialist. "
        "Write a professional, empathetic message (2-3 sentences) for the customer "
        "explaining what will happen next and setting expectations. "
        "Reassure them their issue will be resolved."
    )),
    # checkpointer=InMemorySaver(),
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the customer's warranty status and transition to issue classification."""
    result = warranty_recorder_subagent.invoke(
            {"messages": [HumanMessage(f"Warranty status recorded: {status}")]},
            config,
        )

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=result.content,          # sub-agent response instead of plain string
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "warranty_status": status,
            "current_step": "issue_classifier",
        }
    )


@tool
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the type of issue and transition to resolution specialist."""
    warranty_status = runtime.state.get("warranty_status", "unknown")

    # Sub-agent composes the user-facing acknowledgment
    result = issue_recorder_subagent.invoke(
        {"messages": [HumanMessage(f"Issue type recorded: {issue_type}. "
                    f"Customer warranty status: {warranty_status}.")]},
        config,
    )

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=result.content,          # sub-agent response instead of plain string
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",
        }
    )


@tool
def escalate_to_human(
    reason: str,
    runtime: ToolRuntime[None, SupportState],
) -> str:
    """Escalate the case to a human support specialist."""
    warranty_status = runtime.state.get("warranty_status", "unknown")
    issue_type = runtime.state.get("issue_type", "unknown")

    # Sub-agent drafts the escalation message for the customer
    result = escalation_subagent.invoke(
        {"messages": [HumanMessage(f"Escalation reason: {reason}\n"
                    f"Warranty status: {warranty_status}\n"
                    f"Issue type: {issue_type}")]},
        config,
    )
    return result.content
    # Automatically converted to ToolMessage by the calling agent

@tool
def provide_solution(
    solution: str,
    runtime: ToolRuntime[None, SupportState],
) -> str:
    """Provide a solution to the customer's issue."""
    warranty_status = runtime.state.get("warranty_status", "unknown")
    issue_type = runtime.state.get("issue_type", "unknown")

    # Sub-agent expands and personalises the solution
    result = solution_subagent.invoke(
        {"messages": [HumanMessage(f"Warranty status: {warranty_status}\n"
                    f"Issue type: {issue_type}\n"
                    f"Solution outline: {solution}")]},
        config,
    )
    return result.content
    # Automatically converted to ToolMessage by the calling agent


# ---------------------------------------------------------------------------
# Go-back tools — pure state transitions, no sub-agent needed
# ---------------------------------------------------------------------------

@tool
def go_back_to_warranty(runtime: ToolRuntime[None, SupportState]) -> Command:
    """Go back to warranty verification step."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="Going back to warranty verification step.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "current_step": "warranty_collector",
        }
    )


@tool
def go_back_to_classification(runtime: ToolRuntime[None, SupportState]) -> Command:
    """Go back to issue classification step."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="Going back to issue classification step.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "current_step": "issue_classifier",
        }
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

WARRANTY_COLLECTOR_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Warranty verification

At this step, you need to:
1. Greet the customer warmly
2. Ask if their device is under warranty
3. Use record_warranty_status to record their response and move to the next step

Be conversational and friendly. Don't ask multiple questions at once."""

ISSUE_CLASSIFIER_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Issue classification
CUSTOMER INFO: Warranty status is {warranty_status}

At this step, you need to:
1. Ask the customer to describe their issue
2. Determine if it's a hardware issue (physical damage, broken parts) or software issue (app crashes, performance)
3. Use record_issue_type to record the classification and move to the next step

If unclear, ask clarifying questions before classifying."""

RESOLUTION_SPECIALIST_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Resolution
CUSTOMER INFO: Warranty status is {warranty_status}, issue type is {issue_type}

At this step, you need to:
1. For SOFTWARE issues: provide troubleshooting steps using provide_solution
2. For HARDWARE issues:
   - If IN WARRANTY: explain warranty repair process using provide_solution
   - If OUT OF WARRANTY: escalate_to_human for paid repair options

If the customer indicates any information was wrong, use:
- go_back_to_warranty to correct warranty status
- go_back_to_classification to correct issue type

Be specific and helpful in your solutions."""

# ---------------------------------------------------------------------------
# Step configuration
# ---------------------------------------------------------------------------

STEP_CONFIG = {
    "warranty_collector": {
        "prompt": WARRANTY_COLLECTOR_PROMPT,
        "tools": [record_warranty_status],
        "requires": [],
    },
    "issue_classifier": {
        "prompt": ISSUE_CLASSIFIER_PROMPT,
        "tools": [record_issue_type],
        "requires": ["warranty_status"],
    },
    "resolution_specialist": {
        "prompt": RESOLUTION_SPECIALIST_PROMPT,
        "tools": [provide_solution, escalate_to_human, go_back_to_warranty, go_back_to_classification],
        "requires": ["warranty_status", "issue_type"],
    },
}

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@wrap_model_call
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Configure agent behaviour based on the current step."""
    current_step = request.state.get("current_step", "warranty_collector")
    stage_config = STEP_CONFIG[current_step]

    for key in stage_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"{key} must be set before reaching {current_step}")

    system_prompt = stage_config["prompt"].format(**request.state)

    request = request.override(
        system_prompt=system_prompt,
        tools=stage_config["tools"],
    )

    return handler(request)


# ---------------------------------------------------------------------------
# Agent assembly
# ---------------------------------------------------------------------------

all_tools = [
    record_warranty_status,
    record_issue_type,
    provide_solution,
    escalate_to_human,
    go_back_to_warranty,
    go_back_to_classification,
]

agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[
        apply_step_config,
        SummarizationMiddleware(
            model="openai:gpt-4o",
            trigger=("messages", 4),
            keep=("messages", 1),
        ),
    ],
    # checkpointer=InMemorySaver(),
)

thread_id = str(uuid7())
config = {"configurable": {"thread_id": thread_id}}

graph = agent  # for LangGraph dev server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_last_ai_response(result):
    """Print only the most recent AI message from the agent result."""
    for msg in reversed(result.get("messages", [])):
        if getattr(msg, "type", None) == "ai":
            msg.pretty_print()
            return
    print("No AI response found.")


# ---------------------------------------------------------------------------
# Manual test run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Turn 1: Warranty Collection ===")
    result = agent.invoke(
        {"messages": [HumanMessage("Hi, my phone is not working properly")]},
        config,
    )
    print_last_ai_response(result)

    print("\n=== Turn 2: Warranty Response ===")
    result = agent.invoke(
        {"messages": [HumanMessage("Yes, it's still under warranty")]},
        config,
    )
    print_last_ai_response(result)
    print(f"Current step: {result.get('current_step')}")

    print("\n=== Turn 3: Issue Description ===")
    result = agent.invoke(
        {"messages": [HumanMessage("The screen is physically cracked from dropping it")]},
        config,
    )
    print_last_ai_response(result)
    print(f"Current step: {result.get('current_step')}")

    print("\n=== Turn 4: Resolution ===")
    result = agent.invoke(
        {"messages": [HumanMessage("Actually, I made a mistake - my device is out of warranty. What should I do?")]},
        config,
    )
    print_last_ai_response(result)