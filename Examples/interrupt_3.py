"""
Demonstration of interrupt() and resuming with a tool call in a LangGraph graph.
The graph has two nodes:
1. agent_node: Calls an LLM that may decide to call a tool (send_email).
2. tool_node: Performs the tool call and returns the result to the agent.
3. tool interrupt: Pauses the execution before sending the email and allows for user approval.
"""
import operator
from typing import TypedDict, Annotated, Literal
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langchain.messages import AnyMessage, SystemMessage, ToolMessage


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

@tool
def send_email(to: str, subject: str, body: str):
    """Send an email to a recipient."""

    # Pause before sending; payload surfaces on stream.interrupts when using event streaming
    response = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "Approve sending this email?",
    })

    if response.get("action") == "approve":
        final_to = response.get("to", to)
        final_subject = response.get("subject", subject)
        final_body = response.get("body", body)

        # Actually send the email (your implementation here)
        print(f"[send_email] to={final_to} subject={final_subject} body={final_body}")
        return f"Email sent to {final_to}"

    return "Email cancelled by user"


model = ChatOpenAI(model="gpt-5.5").bind_tools([send_email])
tools_by_name = {"send_email": send_email}


def agent_node(state: AgentState):
    # LLM may decide to call the tool; interrupt pauses before sending
    result = model.invoke(state["messages"])
    return {"messages": [result]}

def tool_node(state: AgentState):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: AgentState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"
    return END

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tool_node", END])  # Routes to "tools" or END
builder.add_edge("tool_node", "agent")  # Loop back after tools

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "email-workflow"}}
initial = graph.stream_events(
    {
        "messages": [
            {"role": "user", "content": "Send an email to alice@example.com about the meeting"}
        ]
    },
    config=config,
    version="v3",
)
initial.output  # drive the stream to completion
print(initial.interrupts)  # -> (Interrupt(value={'action': 'send_email', ...}),)

# Resume with approval and optionally edited arguments
resumed = graph.stream_events(
    Command(resume={"action": "approve", "subject": "Updated subject"}),
    config=config,
    version="v3",
)
print(resumed.output["messages"][-1])  # -> Tool result returned by send_email