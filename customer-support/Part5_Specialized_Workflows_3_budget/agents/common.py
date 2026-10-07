from __future__ import annotations

import logging
from typing_extensions import NotRequired
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage, AIMessage
from typing import Any, Callable, Optional, Literal
from langgraph.types import interrupt, Command
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

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

def sensitive_tools_middleware(sensitive_tools_names: list[str]):
    @wrap_tool_call
    def _(request: Any, handler: Any) -> Any:
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call["args"]
        tool_call_id = request.tool_call["id"]

        if tool_name in sensitive_tools_names:
            human_response = interrupt({
                "question": f"Approval required to execute: '{tool_name}'",
                "action": tool_name,
                "args": tool_args,
            })
            if not human_response.get("approved", False):
                return ToolMessage(
                    content=f"User rejected execution of tool '{tool_name}'. "
                            "Please ask how else you can assist.",
                    tool_call_id=tool_call_id,
                )
        return handler(request)

    return _


def format_prompt_middleware(prompt: str):
    @wrap_model_call
    def _(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        from datetime import datetime

        user_info = request.runtime.state.get(
            "user_info", "No user information available."
        )
        formatted_prompt = prompt.format(
            user_info=user_info,
            time=datetime.now(),
        )
        request = request.override(system_prompt=formatted_prompt)
        return handler(request)

    return _

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
    last_ai_message = next(
        (
            msg
            for msg in reversed(runtime.state["messages"])
            if isinstance(msg, AIMessage)
            and any(
                tc["id"] == runtime.tool_call_id for tc in msg.tool_calls
            )
        ),
        None,
    )

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
        },
        graph=Command.PARENT,
    )
