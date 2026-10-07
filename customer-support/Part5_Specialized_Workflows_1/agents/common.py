from typing import Annotated, Optional, Literal, Callable
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END
from langgraph.prebuilt import tools_condition
from pydantic import BaseModel

def update_dialog_stack(left: list[str], right: Optional[str]) -> list:
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left +[right]

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "update_flight",
                "book_car_rental",
                "book_hotel",
                "book_excursion",
            ]
        ], 
        update_dialog_stack
    ]

class Assistant:
    """Wraps a runnable so it can be used as a LangGraph node.
 
    Keeps re-invoking the runnable until the LLM returns a non-empty response,
    which guards against the model occasionally producing blank outputs.
    """
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    # Defining __call__ makes instances directly callable, matching the
    # signature LangGraph expects for node functions: (state, config) -> dict.
    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


class CompleteOrEscalate(BaseModel):
    """Marks the current task as completed and/or escalates control back to the
    primary assistant, which can re-route based on the user's needs."""
 
    cancel: bool = True
    reason: str
 
    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }

 
def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    """Factory that returns an entry-node callable for a specialized assistant.
 
    When a handoff tool is invoked the primary assistant emits an AI message
    with a tool_call. The entry node converts that into a ToolMessage so the
    sub-agent's LLM sees the handoff context and knows it is now in charge.
    The args passed to the handoff tool are stored in tool_calls[0]["args"],
    so the sub-agent has full access to them.
    """
    def entry_node(state: State):
        tool_call_id = state["messages"][-1].tool_calls[0]["id"] # tool call id of handoff tool
        return{
            "messages": 
            [
                ToolMessage(
                    content=(
                        f"The assistant is now the {assistant_name}. "
                        "Reflect on the above conversation between the host assistant and the user. "
                        f"The user's intent is unsatisfied. Use the provided tools to assist the user. "
                        f"Remember, you are {assistant_name}, and the booking, update, or other action "
                        "is not complete until after you have successfully invoked the appropriate tool. "
                        "If the user changes their mind or needs help for other tasks, call the "
                        "CompleteOrEscalate function to let the primary host assistant take control. "
                        "Do not mention who you are - just act as the proxy for the assistant."
                        ),
                    tool_call_id=tool_call_id
                )
            ],
            "dialog_state": new_dialog_state #push current state to the stack 
        }

    return entry_node

def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return control to the primary assistant.
 
    This lets the full graph explicitly track dialog flow and delegate control
    to specific sub-graphs.
    """
    tool_call_id = state["messages"][-1].tool_calls[0]["id"] # tool call of CompleteOrEscalate
    return{
        "messages":
        [
            ToolMessage(
                content=(
                    "Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed." 
                ), tool_call_id = tool_call_id
            )
        ],
        "dialog_state": "pop"
    }

def make_skill_router(sensitive_tools: list, node_prefix: str):
    """Return a routing function for a specialized assistant subgraph.

    Routes to the safe-tool node, sensitive-tool node, leave_skill, or END
    depending on the last message's tool calls.
    """
    def router(state: State):
        route = tools_condition(state)
        if route == END: # if llm returns plain text. Should not happen, because sub assistand should return tool call
            return END

        tool_calls = state["messages"][-1].tool_calls
        # CompleteOrEscalate tool call means sub assistant finished or cancelled the task
        if any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls):
            return "leave_skill"
        # else sub assistant call the tool to response to user request
        sensitive_names = [t.name for t in sensitive_tools]
        if any(tc["name"] in sensitive_names for tc in tool_calls):
            return f"{node_prefix}_sensitive_tools"
        return f"{node_prefix}_safe_tools"

    return router


   