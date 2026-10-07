from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import Command
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode, tools_condition, InjectedState
from pathlib import Path

img_path = Path(__file__).resolve().parent / "graph.png"

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    preferred_language: str

@tool
def set_preferred_language(language: str, tool_call_id : Annotated[str, InjectedToolCallId]) -> Command:
    """Set the user's preferred response language (e.g., 'es' for Spanish, 'en' for English)."""
    return Command(
        update ={
            "preferred_language": language,
            "messages": [ToolMessage(content=f"Preferred language set to {language}.", tool_call_id=tool_call_id)],
        }
    )

@tool
def get_preferred_language(state: Annotated[GraphState, InjectedState]) -> str:
    """Get the user's current preferred response language."""
    return state.get("preferred_language", "en")

tools = [set_preferred_language, get_preferred_language]
llm = ChatOllama(model="llama3.1", temperature=0.0, max_tokens=300, keep_alive="30m")
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: GraphState):
    """LLM node that inspects the conversation and decides whether to call a tool."""
    preferred_language = state.get("preferred_language", "en")
    system = SystemMessage(
            content=f"""
        preferred language = {preferred_language}
        Rule:
        - The response MUST BE in the user's preferred language.
        - The user's preferred language remains unchanged until they explicitly request a change.
        """
        )

    messages = [system] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(GraphState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools=tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: END
    }
)

builder.add_edge("tools", "agent")

graph = builder.compile()

if __name__ == "__main__":
    graph.get_graph().draw_mermaid_png(output_file_path=img_path)
    init_state = GraphState(messages=[HumanMessage(content="Please change my preferred language to Spanish!")], preferred_language="en")

    print("--- STARTING GRAPH EXECUTION ---")
    for event in graph.stream(init_state):
        for node_name, node_output in event.items():
            print(f"\n[Node Executed]: {node_name}")
            print(f"[Output State]: {node_output}")

    first_result = graph.invoke(init_state)
    follow_up_state = GraphState(
        messages=first_result["messages"] + [HumanMessage(content="What is my preferred language? Use get_preferred_language tool.")],
        preferred_language=first_result.get("preferred_language", "en"),
    )

    print("\n--- FOLLOW-UP GRAPH EXECUTION ---")
    for event in graph.stream(follow_up_state):
        for node_name, node_output in event.items():
            print(f"\n[Node Executed]: {node_name}")
            print(f"[Output State]: {node_output}")