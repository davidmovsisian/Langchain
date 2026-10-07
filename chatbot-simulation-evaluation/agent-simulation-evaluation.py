from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

chat_bot_llm = ChatOpenAI(model="gpt-5.5", temperature=0.7, max_tokens=300)
system = SystemMessage(
        content="You are a customer support agent for an airline."
    )

def chat_bot_node(state: State):
    messages = [system] + state["messages"]
    response = chat_bot_llm.invoke(messages)
    return {"messages": [response]}

simulated_user_llm = ChatOpenAI(model="gpt-5.5", temperature=0.7, max_tokens=300)
system_prompt_template = """You are a customer of an airline company. \
You are interacting with a user who is a customer support person. \

{instructions}

When you are finished with the conversation, respond with a single word 'FINISHED'"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
prompt = ChatPromptTemplate.from_messages(
    [
        system_message_prompt, 
        MessagesPlaceholder(variable_name="messages")
    ])

instructions = """Your name is Harrison. You are trying to get a refund for the trip you took to Alaska. \
You want them to give you ALL the money back. \
This trip happened 5 years ago."""

    # Apply partial formatting
prompt = prompt.partial(instructions=instructions)
simulated_user = prompt | simulated_user_llm

def simulated_user_node(state: State):
    messages = _swap_roles(state["messages"])
    response = simulated_user.invoke({"messages": messages})
    # This response is an AI message - we need to flip this to be a human message
    return {"messages": [HumanMessage(content=response.content)]} 

def _swap_roles(messages):
    new_messages = []
    for m in messages:
        if isinstance(m, AIMessage):
            new_messages.append(HumanMessage(content=m.content))
        else:
            new_messages.append(AIMessage(content=m.content))
    return new_messages

def should_continue(state: State):
    messages = state["messages"]

    if len(messages) > 6 or messages[-1].content == "FINISHED":
        return "end"
    
    return "continue"

graph_builder = StateGraph(State)
graph_builder.add_node("chat_bot", chat_bot_node)
graph_builder.add_node("user", simulated_user_node)
graph_builder.add_edge("chat_bot", "user")
graph_builder.add_conditional_edges("user", should_continue, 
    {
        "end": END,
        "continue": "chat_bot",
    })

graph_builder.add_edge(START, "chat_bot")
simulation = graph_builder.compile()

for chunk in simulation.stream({"messages": []}, version="v3"):
    # Print out all events aside from the final end chunk
    if END not in chunk:
        print(chunk)
        print("----")
