from typing import Annotated, Any, Optional
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START, add_messages
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict

class SimulationState(TypedDict):
	messages: Annotated[list[AnyMessage], add_messages]    
	inputs: Optional[dict[str, Any]] # carries "instructions" and any other dataset fields

def swap_roles(messages : list[AnyMessage]):
    swapped = []
    for m in messages:
        if isinstance(m, AIMessage):
            swapped.append(HumanMessage(content=m.content))
        else:
            swapped.append(AIMessage(content=m.content))
    
    return swapped

def make_chat_bot_node(llm :ChatOpenAI):
    system = SystemMessage(content="You are a customer support agent for an airline.")
    
    def chat_bot_node(state: SimulationState):
        messages = system + state["messages"]
        response = llm.invoke(messages)
        return {"messages" : [response.content]}
    
    return chat_bot_node

def make_simulated_user_node(system_prompt_template: str, llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        MessagesPlaceholder("messages")
        ])
    simulated_user = prompt|llm
    
    def simulated_user_node(state: SimulationState):
        messages = swap_roles(state["messages"])
        instructions = state.get("inputs", {}).get("instructions", "")
        response = simulated_user.invoke({
            "instructions": instructions,
            "messages": messages
        })
        return {"messages": [HumanMessage(content=response.content)]}
    
    return simulated_user_node

def make_should_continue(max_turns: int=10):
    def should_continue(state: SimulationState):
        messages = state["messages"]
        if len(messages) > max_turns or messages[-1].content =="FINISHED":
            return "end"
        return "continue"
    
    return should_continue
    
def _prepare_example(inputs: dict[str, Any], input_key: Optional[str] = None):
    """Convert raw dataset dict → SimulationState"""
    opening_message = inputs[input_key]
    extra_inputs = {k: v for k, v in inputs.items() if k != input_key}
    initial_state: SimulationState = {
        "messages": [HumanMessage(content=opening_message)],
        "inputs": extra_inputs,
    }

    return initial_state
    
def create_chat_simulator(chat_bot_node, simulated_user_node, should_continue, input_key: str = "input"):
    graph_builder = StateGraph(SimulationState)
    graph_builder.add_node("chat_bot", chat_bot_node)
    graph_builder.add_node("simulated_user", simulated_user_node)
    graph_builder.add_conditional_edges("simulated_user", should_continue,
        {
            "end": END,
            "continue": "chat_bot"
        })
    graph_builder.add_edge(START, "chat_bot")
    graph_builder.add_edge("chat_bot","simulated_user")
    
    graph = graph_builder.compile()
    
    return RunnableLambda(_prepare_example).bind(input_key=input_key) | graph

#Create nodes and simulator
assistant_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=300)
chat_bot_node = make_chat_bot_node(assistant_llm)

system_prompt_template = """\
You are a customer of an airline company interacting with a customer support agent.

{instructions}

When you are finished with the conversation, respond with a single word 'FINISHED'."""

customer_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=300)
simulated_user_node = make_simulated_user_node(system_prompt_template, customer_llm)

should_continue = make_should_continue(max_turns=6)

simulator = create_chat_simulator(
    chat_bot_node=chat_bot_node,
    simulated_user_node=simulated_user_node,
    should_continue=should_continue,
    input_key="input",
)

for event in simulator.stream(
    {
        "input": "I need a discount.",
        "instructions": "You are extremely disgruntled and will cuss and swear to get your way. "
                    "Try to get a discount by any means necessary."
    }):
        
        if "__end__" in event:
            break
        role, state = next(iter(event.items()))
        print(f"\033[1m{role}\033[0m: {state['messages'][-1].content}")
        print("----")