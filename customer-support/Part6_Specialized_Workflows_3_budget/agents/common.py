from __future__ import annotations
from datetime import datetime
import logging
from typing_extensions import NotRequired
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage, AIMessage, RemoveMessage
from typing import Any, Callable, Optional, Literal
from langgraph.types import interrupt, Command
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import (
    wrap_model_call,
    before_model,
    ModelRequest, 
    ModelResponse,
    SummarizationMiddleware,
    ContextEditingMiddleware,
    ClearToolUsesEdit
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class TravelState(AgentState):
    active_agent: Optional[
        Literal[
            "primary_agent",
            "flight_agent",
            "car_rental_agent",
            "hotel_agent",
            "excursion_agent",
        ]
    ]
    user_info: NotRequired[str]
    handoff_data: NotRequired[dict]

# ---------------------------------------------------------------------------
# Sensitive-tool & prompt middleware
# ---------------------------------------------------------------------------
def _parse_decision(response: Any) -> tuple[bool, Optional[str]]:
    """Normalise a resume value into (approved, reason).
 
    Preferred shape: {"approved": bool, "reason": str | None}.
    Also tolerates a bare bool or a "y"/"yes" string.
    """
    if isinstance(response, dict):
        return bool(response.get("approved", False)), response.get("reason")
    if isinstance(response, bool):
        return response, None
    if isinstance(response, str):
        approved = response.strip().lower() in {"y", "yes", "approve", "approved"}
        return approved, (None if approved else response)
    return False, None

def sensitive_tools_middleware(sensitive_tools_names: list[str]):
    @wrap_tool_call
    def _(request: Any, handler: Any) -> Any:
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call["args"]
        tool_call_id = request.tool_call["id"]

        if tool_name in sensitive_tools_names:
             # Pauses the graph. Nothing before this line has side effects, so it is
            # safe that the node re-runs from the top when the graph is resumed.
            human_response = interrupt({
                "question": f"Approval required to execute: '{tool_name}'",
                "action": tool_name,
                "args": tool_args,
            })
            approved, reason = _parse_decision(human_response)
            if not approved:
                detail = f" Reason given: '{reason}'." if reason else ""
                return ToolMessage(
                    content=f"User rejected execution of tool '{tool_name}'.{detail} "
                            "Do not retry the same call. Continue assisting, "
                            "accounting for the user's feedback.",
                    tool_call_id=tool_call_id,
                    status="error"
                )
        return handler(request)

    return _

def _format_handoff(data: dict | None) -> str:
    if not data:
        return "None."
    return "\n".join(f"- {k}: {v}" for k, v in data.items())

def format_prompt_middleware(prompt: str):
    @wrap_model_call
    def _(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        
        state = request.state
        formatted_prompt = prompt.format(
            user_info=state.get("user_info", "No user information available."),
            handoff = _format_handoff(request.state.get("handoff_data")),
            time=datetime.now().isoformat(),
        )
        request = request.override(system_prompt=formatted_prompt)
        return handler(request)

    return _

# ---------------------------------------------------------------------------
# Context management: pruning, summarization, tool-result clearing
# ---------------------------------------------------------------------------

def prune_read_only_tool_traffic(read_only: set[str], keep_last: int = 6):
    """Remove old (AIMessage tool call, ToolMessage result) pairs of read-only tools.
 
    Persistent: rewrites the stored ``messages`` state (before_model hook).
    Only tools named in ``read_only`` are touched, so booking/handoff traffic survives.
    Run this BEFORE the summarizer so it summarizes the already-pruned history.
    """
    keep_last = max(keep_last, 1)  # msgs[:-0] would be empty and prune nothing
 
    @before_model
    def _(state: Any, runtime: Any) -> Optional[dict]:
        msgs = state["messages"]
 
        # 1) Ids of read-only calls made by AI messages OUTSIDE the protected window.
        drop = {
            tc["id"]
            for m in msgs[:-keep_last]
            if isinstance(m, AIMessage)
            for tc in m.tool_calls
            if tc["name"] in read_only
        }
        if not drop:
            return None
 
        # 2) Scan ALL messages so a pair straddling the cutoff never leaves an orphan.
        updates: list = []
        for m in msgs:
            if not m.id:
                continue
            if isinstance(m, ToolMessage) and m.tool_call_id in drop:
                updates.append(RemoveMessage(id=m.id))
            elif isinstance(m, AIMessage) and any(tc["id"] in drop for tc in m.tool_calls):
                kept = [tc for tc in m.tool_calls if tc["id"] not in drop]
                if not kept and not m.content:
                    updates.append(RemoveMessage(id=m.id))
                else:
                    # Same id -> add_messages replaces the message in place.
                    extra = m.additional_kwargs
                    if not kept:
                        # OpenAI re-serialises raw tool_calls from additional_kwargs
                        # when .tool_calls is empty, so strip them as well.
                        extra = {
                            k: v for k, v in extra.items()
                            if k not in ("tool_calls", "function_call")
                        }
                    updates.append(
                        m.model_copy(update={"tool_calls": kept, "additional_kwargs": extra})
                    )
 
        return {"messages": updates} if updates else None
 
    return _
 
 
TRAVEL_SUMMARY_PROMPT = """Summarize the conversation so far for a travel-booking assistant.
Preserve exactly: user preferences (budget, dates, locations), every booking or
reservation ID, what was confirmed vs. still pending, options the user rejected,
and each delegation to a specialist (which agent, location, dates, request, and
whether it finished). Be concise.
 
{messages}"""
 
 
def summarization_middleware(trigger: int=3000, keep_last: int=6) -> SummarizationMiddleware:
    """LLM summarization of old history. Attach to the primary agent only."""
    return SummarizationMiddleware(
        model="openai:gpt-4o-mini",
        trigger=("tokens", trigger),
        keep=("messages", keep_last),
        summary_prompt=TRAVEL_SUMMARY_PROMPT,
    )
 
 
def clear_old_search_results_middleware(sensitive_tools: list[str]) -> ContextEditingMiddleware:
    """Request-only clearing of old tool results (no LLM call, state untouched).
 
    Intended for sub-agents; results of ``sensitive_tools`` are never cleared.
    Place it before budget_middleware so the budget sees the cleared request.
    """
    return ContextEditingMiddleware(edits=[
        ClearToolUsesEdit(
            trigger=2000,
            keep=2,
            exclude_tools=sensitive_tools,
            placeholder="[old search results cleared; re-run the search if needed]",
        )
    ])
 
 
# ---------------------------------------------------------------------------
# Shared helper for tools that return a Command
# ---------------------------------------------------------------------------
 
def find_calling_ai_message(runtime: ToolRuntime[None, TravelState]) -> Optional[AIMessage]:
    """The AIMessage that issued the current tool call, or None if not found."""
    return next(
        (
            msg
            for msg in reversed(runtime.state["messages"])
            if isinstance(msg, AIMessage)
            and any(tc["id"] == runtime.tool_call_id for tc in msg.tool_calls)
        ),
        None,
    )


# ---------------------------------------------------------------------------
# complete_or_escalate tool
# ---------------------------------------------------------------------------
@tool
def complete_or_escalate(
    reason: str,
    runtime: ToolRuntime[None, TravelState],
) -> Command:
    """Escalate back to primary assistant.

    Args:
        reason: Reason why the task is complete or why escalation is required.
    """

    last_ai_message = find_calling_ai_message(runtime)

    transfer_message = ToolMessage(
        content=f"Resuming dialog with the host assistant. Reason: {reason}",
        tool_call_id=runtime.tool_call_id,
    )

    messages = [transfer_message]
    if last_ai_message is not None:
        messages.insert(0, last_ai_message)

    return Command(
        goto="primary_agent",
        update={
            "active_agent": "primary_agent",
            "messages": messages,
            "handoff_data": {}
        },
        graph=Command.PARENT,
    )
