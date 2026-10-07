from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from datetime import datetime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from pathlib import Path

from tools.car_rental import (
    book_car_rental,
    cancel_car_rental,
    search_car_rentals,
    update_car_rental
)
from tools.flights import(
    cancel_ticket,
    fetch_user_flight_information,
    search_flights,
    update_ticket_to_new_flight
)
from tools.lookup_company_policies import(
    lookup_policy
)

from tools.hotels import (
    book_hotel,
    cancel_hotel,
    search_hotels,
    update_hotel
)

from tools.excursions import (
    book_excursion,
    cancel_excursion,
    search_trip_recommendations,
    update_excursion
)

from utils.utils import create_tool_node_with_fallback

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable
    # In Python, defining a __call__ method makes an object instance callable like a standard function (e.g., assistant_instance(state, config)). 
    # LangGraph treats any node function as a callable that takes the graph state and configuration.
    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
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

llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

# create prompt template with partial update of the time. user_info and messages will be passed on runnable invokation
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "You are a helpful customer support assistant for Swiss Airlines. "
            " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}."
        ),
        (
            "placeholder", "{messages}"
        )
    ]
).partial(time=datetime.now)

# "Read"-only tools (such as retrievers) don't need a user confirmation to use
part_3_safe_tools = [
    TavilySearchResults(max_results=1),
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    search_car_rentals,
    search_hotels,
    search_trip_recommendations,
]

# These tools all change the user's reservations.
# The user has the right to control what decisions are made
part_3_sensitive_tools = [
    update_ticket_to_new_flight,
    cancel_ticket,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    book_hotel,
    update_hotel,
    cancel_hotel,
    book_excursion,
    update_excursion,
    cancel_excursion,
]

sensitive_tools_names = [t.name for t in part_3_sensitive_tools]

part_3_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_3_safe_tools + part_3_sensitive_tools)

builder = StateGraph(State)

def user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}

# NEW: The fetch_user_info node runs first, meaning our assistant can see the user's flight information without
# having to take an action
builder.add_node("fetch_user_info", user_info)
builder.add_node("assistant", Assistant(part_3_assistant_runnable))
builder.add_node("safe_tools", create_tool_node_with_fallback(part_3_safe_tools))
builder.add_node("sensitive_tools", create_tool_node_with_fallback(part_3_sensitive_tools))

builder.add_edge(START, "fetch_user_info")
builder.add_edge("fetch_user_info", "assistant")

def root_tools(state: State):
    next_node = tools_condition(state, messages_key="messages")
    #if no tools invokes return to the user
    if next_node == END:
        return END
    ai_message = state["messages"][-1]
    # This assumes single tool calls. To handle parallel tool calling, you'd want to
    # use an ANY condition
    first_tool_call = ai_message.tool_calls[0]
    if first_tool_call["name"] in sensitive_tools_names:
        return "sensitive_tools"
    return "safe_tools"

builder.add_conditional_edges(
    "assistant",
    root_tools,
    ["safe_tools", "sensitive_tools", END]
)

builder.add_edge("safe_tools", "assistant")
builder.add_edge("sensitive_tools", "assistant")

memory = InMemorySaver()
part_3_graph = builder.compile(
    checkpointer=memory,
    # halt before executing the "sensitive_tools" node.
    interrupt_before=["sensitive_tools"]
    )

img_path = Path(__file__).resolve().parent / "graph.png"
part_3_graph.get_graph().draw_mermaid_png(output_file_path = img_path)