from typing import List, Annotated, Optional, Literal
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage, ToolMessage
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

class PromptInstructions(TypedDict):
    objective: str
    variables: List[str]
    constraints: List[str]
    requirements: List[str]

#stores messages and instructions to create a prompt
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    instructions: Optional[PromptInstructions]

@tool
def save_instructions(
    objective: str,
    variables: List[str],
    constraints: List[str],
    requirements: List[str],
) -> PromptInstructions:
    """
    Call this once, all instructions for prompt are gathered from the user
    Args:
        objective: What the prompt is trying to achieve.
        variables: Template variables that will be passed in (e.g. context, question).
        constraints: Things the output must NOT do.
        requirements: Things the output MUST do.
    """
    return PromptInstructions(
        objective = objective,
        variables = variables,
        constraints=constraints,
        requirements=requirements
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([save_instructions])

INFO_SYSTEM = """\
Your job is to gather requirements for a prompt template.

Collect the following from the user:
- Objective of the prompt
- Variables that will be passed in (e.g. {context}, {question})
- Constraints (what the output must NOT do)
- Requirements (what the output MUST do)

Ask clarifying questions if anything is unclear.
Once you have everything, call the save_instructions tool."""

PROMPT_SYSTEM = """\
Based on the following instructions, write a good prompt template:

Objective:    {objective}
Variables:    {variables}
Constraints:  {constraints}
Requirements: {requirements}"""

#collect instructions from the user
def info_node(state: State):
    messages = [SystemMessage(content=INFO_SYSTEM)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

#create prompt from instructions
def prompt_node(state: State):
    instructions = state["instructions"]
    system = [SystemMessage(content=PROMPT_SYSTEM.format(
        objective = instructions["objective"],
        variables = instructions["variables"],
        constraints = instructions["constraints"],
        requirements = instructions["requirements"]
    ))]
    response = llm.invoke(system)
    return {"messages": [response]}

#tool to save the collected instructions
def tool_node(state: State):
    last = state["messages"][-1]
    tool_call = last.tool_calls[0]

    instructions = save_instructions.invoke(tool_call["args"])
    return{
        "messages": [ToolMessage(content="Instructions saves", tool_call_id=tool_call["id"])],
        "instructions": instructions
    }

def router(state: State) -> Literal["tool_node", "__end__"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_node"
    return END
    
builder = StateGraph(State)
builder.add_node("info_node", info_node)
builder.add_node("prompt_node", prompt_node)
builder.add_node("tool_node", tool_node)
builder.add_edge("tool_node", "prompt_node")
builder.add_edge(START, "info_node")
builder.add_conditional_edges("info_node", router, ["tool_node", END])
builder.add_edge("prompt_node", END)


graph = builder.compile(checkpointer=InMemorySaver())

#run application
import uuid

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

print("Prompt Generator — type 'q' to quit\n")
while True:
    user = input("You: ").strip()
    if user.lower() == "q":
        print("Bye!")
        break

    for chunk in graph.stream(
        {"messages": [HumanMessage(user)]}, config=config, stream_mode="updates"
    ):
        for node_output in chunk.values():
            last = node_output["messages"][-1]
            last.pretty_print()