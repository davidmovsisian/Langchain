from fastapi import logger
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from typing_extensions import NotRequired

from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import tool, ToolRuntime

# handsoff multiagent with multiple agent subgraphs

model = init_chat_model(model="openai:gpt-4o", temperature=0)

# Workflow steps (used as graph node names)
SupportStep = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]


class SupportState(AgentState):
    active_agent: NotRequired[SupportStep]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]


# ---------------------------------------------------------------------------
# Handoff tools
# Each tool grabs the triggering AIMessage + creates a ToolMessage to keep
# conversation history valid in the parent graph, then routes via Command.PARENT.
# ---------------------------------------------------------------------------

@tool#(return_direct=True)
def transfer_to_issue_classifier(
    warranty_status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState]
) -> Command:
    """Record the customer's warranty status and hand off to the issue classifier."""
    last_ai_message = next(
        msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)
    )
    transfer_message = ToolMessage(
        content=f"Warranty status recorded as: {warranty_status}. Transferring to issue classifier.",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        goto="issue_classifier",
        update={
            "active_agent": "issue_classifier",
            "warranty_status": warranty_status,
            "messages": [last_ai_message, transfer_message],
        },
        graph=Command.PARENT,
    )


@tool#(return_direct=True)
def transfer_to_resolution_specialist(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime[None, SupportState]
) -> Command:
    """Record the issue type and hand off to the resolution specialist."""
    last_ai_message = next(
        msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)
    )
    transfer_message = ToolMessage(
        content=f"Issue type recorded as: {issue_type}. Transferring to resolution specialist.",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        goto="resolution_specialist",
        update={
            "active_agent": "resolution_specialist",
            "issue_type": issue_type,
            "messages": [last_ai_message, transfer_message],
        },
        graph=Command.PARENT,
    )


@tool#(return_direct=True)
def transfer_to_warranty_collector(
    runtime: ToolRuntime[None, SupportState]
) -> Command:
    """Go back to warranty verification if the customer provided incorrect warranty information."""
    last_ai_message = next(
        msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)
    )
    transfer_message = ToolMessage(
        content="Going back to warranty verification step.",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        goto="warranty_collector",
        update={
            "active_agent": "warranty_collector",
            "warranty_status": None,
            "issue_type": None,
            "messages": [last_ai_message, transfer_message],
        },
        graph=Command.PARENT,
    )


@tool#(return_direct=True)
def transfer_back_to_issue_classifier(
    runtime: ToolRuntime[None, SupportState]
) -> Command:
    """Go back to issue classification if the customer provided the wrong issue type."""
    last_ai_message = next(
        msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)
    )
    transfer_message = ToolMessage(
        content="Going back to issue classification step.",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        goto="issue_classifier",
        update={
            "active_agent": "issue_classifier",
            "issue_type": None,
            "messages": [last_ai_message, transfer_message],
        },
        graph=Command.PARENT,
    )

@tool
def update_warranty_status(
    warranty_status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState]
) -> Command:
    """Correct the warranty status without leaving the resolution step."""
    return Command(
        update={
            "warranty_status": warranty_status,
            "messages": [ToolMessage(
                content=f"Warranty status updated to: {warranty_status}.",
                tool_call_id=runtime.tool_call_id,
            )],
        },
        # graph=Command.PARENT, → stays inside the subagent
    )

# ---------------------------------------------------------------------------
# Action tools (no routing – these stay within the current agent)
# ---------------------------------------------------------------------------

@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the case to a human support specialist."""
    return f"Escalating to human support. Reason: {reason}"


@tool
def provide_solution(solution: str) -> str:
    """Provide a solution to the customer's issue."""
    return f"Solution saved. Now present this to the customer exactly as written:\n\n{solution}"


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

WARRANTY_COLLECTOR_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Warranty verification

At this step, you need to:
1. Greet the customer warmly
2. Ask if their device is under warranty
3. Use transfer_to_issue_classifier (passing the warranty status) to record their
   response and move to the next step

Be conversational and friendly. Don't ask multiple questions at once."""

ISSUE_CLASSIFIER_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Issue classification

At this step, you need to:
1. Ask the customer to describe their issue
2. Determine whether it is a hardware issue (physical damage, broken parts) or a
   software issue (app crashes, performance problems)
3. Use transfer_to_resolution_specialist (passing the issue type) to record the
   classification and move to the next step

If the customer corrects their warranty status:
- Use update_warranty_status to correct it, then proceed with the issue classification

If the situation is unclear, ask clarifying questions before classifying."""

RESOLUTION_SPECIALIST_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Resolution
CUSTOMER INFO: Warranty status is {warranty_status}, issue type is {issue_type}

At this step, you need to:
1. For SOFTWARE issues: provide troubleshooting steps using provide_solution
2. For HARDWARE issues:
   - If IN WARRANTY: explain the warranty repair process using provide_solution
   - If OUT OF WARRANTY: use escalate_to_human for paid repair options

If the customer corrects their warranty status:
- Use update_warranty_status to correct it, then proceed with the resolution

If the customer corrects their issue type:
- Use transfer_back_to_issue_classifier to re-classify

Be specific and helpful in your solutions."""


warranty_collector_agent = create_agent(
    model,
    tools=[transfer_to_issue_classifier],
    system_prompt=WARRANTY_COLLECTOR_PROMPT,
)

issue_classifier_agent = create_agent(
    model = model.bind(parallel_tool_calls=False), #prevent multiple tool calls in the same turn, which can cause confusion
    tools=[transfer_to_resolution_specialist,
           update_warranty_status],
    system_prompt=ISSUE_CLASSIFIER_PROMPT,
)


resolution_specialist_agent = create_agent(
    model = model.bind(parallel_tool_calls=False), #prevent multiple tool calls in the same turn, which can cause confusion
    tools=[
        provide_solution,
        escalate_to_human,
        update_warranty_status,
        transfer_back_to_issue_classifier,
    ],
    system_prompt=RESOLUTION_SPECIALIST_PROMPT,
)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def call_warranty_collector(state: SupportState) -> Command:
    try:
        return warranty_collector_agent.invoke(state)
    except Exception as e:
        logger.error(f"warranty_collector failed: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content="I'm sorry, something went wrong. Please try again.")]
        }

def call_issue_classifier(state: SupportState) -> Command:
    try:
        return issue_classifier_agent.invoke(state)
    except Exception as e:
        logger.error(f"issue_classifier failed: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content="I'm sorry, something went wrong. Please try again.")]
        }

def call_resolution_specialist(state: SupportState) -> Command:
    try:
        return resolution_specialist_agent.invoke(state)
    except Exception as e:
        logger.error(f"resolution_specialist failed: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content="I'm sorry, something went wrong. Please try again.")]
        }


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _last_message_is_final_ai(state: SupportState) -> bool:
    """Return True when the last message is an AI message with no pending tool calls."""
    messages = state.get("messages", [])
    if not messages:
        return False
    last = messages[-1]
    return isinstance(last, AIMessage) and not getattr(last, "tool_calls", [])

def route_from_start(state: SupportState) -> str:
    return state.get("active_agent") or "warranty_collector"

def route_after_agent(
    state: SupportState,
) -> Literal["warranty_collector", "issue_classifier", "resolution_specialist", "__end__"]:
    """After each agent node, end the turn or route to the next active agent."""
    if _last_message_is_final_ai(state):
        return "__end__"
    return state.get("active_agent", "warranty_collector")


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

builder = StateGraph(SupportState)

builder.add_node("warranty_collector", call_warranty_collector)
builder.add_node("issue_classifier", call_issue_classifier)
builder.add_node("resolution_specialist", call_resolution_specialist)

builder.add_conditional_edges(
    START,
    route_from_start,
    ["warranty_collector", "issue_classifier", "resolution_specialist"],
)

builder.add_conditional_edges(
    "warranty_collector",
    route_after_agent,
    [ "issue_classifier", END],
)
builder.add_conditional_edges(
    "issue_classifier",
    route_after_agent,
    ["warranty_collector", "resolution_specialist", END],
)
builder.add_conditional_edges(
    "resolution_specialist",
    route_after_agent,
    ["warranty_collector", "issue_classifier", END],
)

graph = builder.compile(checkpointer=InMemorySaver()) #checkpointer=InMemorySaver()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

thread_id = str(uuid7())
config = {"configurable": {"thread_id": thread_id}}


def print_last_ai_response(result):
    """Print only the most recent AI message from the agent result."""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            msg.pretty_print()
            return
    print("No AI response found.")


# ---------------------------------------------------------------------------
# Simulated conversation
# ---------------------------------------------------------------------------

print("=== Turn 1: Warranty Collection ===")
result = graph.invoke(
    {"messages": [HumanMessage("Hi, my phone is not working properly")]},
    config,
)
print_last_ai_response(result)

print("\n=== Turn 2: Warranty Response ===")
result = graph.invoke(
    {"messages": [HumanMessage("Yes, it's still under warranty")]},
    config,
)
print_last_ai_response(result)
print(f"Active agent: {result.get('active_agent')}")

print("\n=== Turn 3: Issue Description ===")
result = graph.invoke(
    {"messages": [HumanMessage("The screen is physically cracked from dropping it")]},
    config,
)
print_last_ai_response(result)
print(f"Active agent: {result.get('active_agent')}")

print("\n=== Turn 4: Correction + Resolution ===")
result = graph.invoke(
    {"messages": [HumanMessage("Actually, I made a mistake - my device is out of warranty. What should I do?")]},
    config,
)
print_last_ai_response(result)