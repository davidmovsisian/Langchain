from pathlib import Path
from typing import Literal
 
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition
 
from tools.flights import fetch_user_flight_information
 
from .common import (
    Assistant,
    State,
    create_entry_node,
    pop_dialog_state,
    create_tool_node_with_fallback
)
from . import (
    car_rental_assistant,
    excursion_assistant,
    flight_assistant,
    hotel_assistant,
    primary_assistant,
)
from .flight_assistant import ToFlightBookingAssistant
from .car_rental_assistant import ToBookCarRental
from .hotel_assistant import ToHotelBookingAssistant
from .excursion_assistant import ToBookExcursion

llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

flight_subgraph = flight_assistant.build_graph(llm)
car_rental_subgraph = car_rental_assistant.build_graph(llm)
hotel_subgraph = hotel_assistant.build_graph(llm)
excursion_subgraph = excursion_assistant.build_graph(llm)

def fetch_user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}

builder = StateGraph(State)

builder.add_node("fetch_user_info", fetch_user_info)
builder.add_edge(START, "fetch_user_info")

# ── Flight sub-graph ───────────────────────────────────────────────────────
builder.add_node("enter_update_flight", create_entry_node("Flight Updates & Booking Assistant", "update_flight")) # entry node
builder.add_node("update_flight", flight_subgraph) 
builder.add_edge("enter_update_flight", "update_flight")
builder.add_edge("update_flight", "primary_assistant") # unconditional return from sub-graph to parent graph

# ── Car rental sub-graph ───────────────────────────────────────────────────
builder.add_node("enter_book_car_rental", create_entry_node("Car Rental Assistant", "book_car_rental")) # entry node
builder.add_node("book_car_rental", Assistant(car_rental_subgraph)) 
builder.add_edge("enter_book_car_rental", "book_car_rental")
builder.add_edge("book_car_rental", "primary_assistant")

# ── Hotel sub-graph ────────────────────────────────────────────────────────
builder.add_node("enter_book_hotel", create_entry_node("Hotel Booking Assistant", "book_hotel")) # entry node
builder.add_node("book_hotel", Assistant(hotel_subgraph))
builder.add_edge("enter_book_hotel", "book_hotel")
builder.add_edge("book_hotel", "primary_assistant")

# ── Excursion sub-graph ────────────────────────────────────────────────────
builder.add_node("enter_book_excursion", create_entry_node("Trip Recommendation Assistant", "book_excursion")) # entry node
builder.add_node("book_excursion", Assistant(excursion_subgraph))
builder.add_edge("enter_book_excursion", "book_excursion")
builder.add_edge("book_excursion", "primary_assistant")

# ── Primary assistant ──────────────────────────────────────────────────────

# flow example of what happens after return from sub-graph
# subgraph: update_flight (LLM)
#     → AIMessage(tool_calls=[CompleteOrEscalate])

# subgraph: leave_skill (pop_dialog_state)
#     → appends ToolMessage("Resuming dialog...")
#     → dialog_state: "pop"
#     → edges to END

# subgraph exits, parent merges state

# parent: unconditional edge → primary_assistant
#     → LLM sees ToolMessage, generates reply
#     → AIMessage("Flight updated! Anything else?")   ← no tool calls

# parent: route_primary_assistant
#     → tools_condition returns END (no tool calls)

# parent: END
#     → waits for next user message

assistant_runnable = primary_assistant.build_runnable(llm)

builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_node(
    "primary_assistant_tools",
    create_tool_node_with_fallback(primary_assistant.primary_assistant_tools)
)

def route_primary_assistant(state: State):
    route = tools_condition(state)
    if route == END: 
        return END

    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        name = tool_calls[0]["name"]
        if name == ToFlightBookingAssistant.__name__:
            return "enter_update_flight"
        if name == ToBookCarRental.__name__:
            return "enter_book_car_rental"
        if name == ToHotelBookingAssistant.__name__:
            return "enter_book_hotel"
        if name == ToBookExcursion.__name__:
            return "enter_book_excursion"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")

builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    [
        "enter_update_flight",
        "enter_book_car_rental",
        "enter_book_hotel",
        "enter_book_excursion",
        "primary_assistant_tools",
        END,
    ],
)

builder.add_edge("primary_assistant_tools", "primary_assistant")

def route_to_workflow(
    state: State,
) -> Literal[
    "primary_assistant",
    "update_flight",
    "book_car_rental",
    "book_hotel",
    "book_excursion",
]:
    """After fetching user info, route to whichever assistant is currently active."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]

builder.add_conditional_edges("fetch_user_info", route_to_workflow)

memory = InMemorySaver()
part_5_graph = builder.compile(checkpointer=memory)

# Persist a visual of the compiled graph next to this file.
img_path = Path(__file__).resolve().parent / "graph.png"
part_5_graph.get_graph(xray=True).draw_mermaid_png(output_file_path=img_path)