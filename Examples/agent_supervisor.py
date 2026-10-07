from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import convert_to_messages
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model

# Helpers
# --------
def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")
# End helpers-----

web_search = TavilySearch(max_results=3)

research_agent = create_react_agent(
    model="gpt-4o-mini",
    tools=[web_search],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)

def add(a: float, b: float):
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b


math_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[add, multiply, divide],
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
)

supervisor = create_supervisor(
    model=init_chat_model("gpt-4o-mini"),
    agents=[research_agent, math_agent],
    prompt=(
        "You are a supervisor agent.\n\n"
        "INSTRUCTIONS:\n"
        "- You are in charge of the research and math agents.\n"
        "- You will route messages to the appropriate agent based on the user's request.\n"
        "- You will respond to the user with the results of the agents' work.\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
        "Assign work to one agent at a time, do not call agents in parallel.\n"
    ),
    add_handoff_tool=True,
    output_mode="full_history",
    name="supervisor",
).compile()

# for chunk in supervisor.stream(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "find US and New York state GDP in 2024. what % of US GDP was New York state?",
#             }
#         ]
#     },
# ):
#     pretty_print_messages(chunk, last_message=True)

# final_message_history = chunk["supervisor"]["messages"]
# print("Final message history:")
# for m in final_message_history:
#     pretty_print_message(m)


# Create supervisor from scratch
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command

def create_handoff_tool(agent_name: str, description: str = None):
    name = f"handoff_to_{agent_name}"
    description = description or f"Hand off the conversation to {agent_name}."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],):
        """Hand off the conversation to another agent."""
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "tool_call_id": tool_call_id,
            "name": name,
        }
        return Command(
            goto=agent_name,
            update ={
                **state,
                "messages": state["messages"] + [tool_message],
            },
            graph = Command.PARENT
        )

    return handoff_tool

assign_to_research_agent = create_handoff_tool("research_agent", "Assign task to a researcher agent.")
assign_to_math_agent = create_handoff_tool("math_agent", "Assign task to a math agent.")

supervisor_agent = create_react_agent(
    model="gpt-4o-mini",
    tools=[assign_to_research_agent, assign_to_math_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    name="supervisor",
)

graph = (
    StateGraph(MessagesState)
    .add_node(supervisor_agent)
    .add_node(research_agent)
    .add_node(math_agent)
    .add_edge(START, "supervisor")
    .add_edge("research_agent", "supervisor")
    .add_edge("math_agent", "supervisor")
    .compile()
)

for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "find US and New York state GDP in 2024. what % of US GDP was New York state?",
            }
        ]
    },
):
    None
    # pretty_print_messages(chunk, last_message=True)

final_message_history = chunk["supervisor"]["messages"]
for m in final_message_history:
    pretty_print_message(m)