from pathlib import Path
from typing import Literal
 
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition
 
from tools.flights import fetch_user_flight_information
from utils.utils import create_tool_node_with_fallback
 
from .common import (
    Assistant,
    CompleteOrEscalate,
    State,
    create_entry_node,
    make_skill_router,
    pop_dialog_state,
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

update_flight_runnable = flight_assistant.build_runnable(llm)
book_car_rental_runnable = car_rental_assistant.build_runnable(llm)
book_hotel_runnable = hotel_assistant.build_runnable(llm)
book_excursion_runnable = excursion_assistant.build_runnable(llm)
assistant_runnable = primary_assistant.build_runnable(llm)

def fetch_user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}

builder = StateGraph(State)

builder.add_node("fetch_user_info", fetch_user_info)
builder.add_edge(START, "fetch_user_info")

# Shared leave node
builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "primary_assistant")

# ── Flight sub-graph ───────────────────────────────────────────────────────
builder.add_node("enter_update_flight", create_entry_node("Flight Updates & Booking Assistant", "update_flight")) # entry node
builder.add_node("update_flight", Assistant(update_flight_runnable)) # llm node, which decides to call to the tool or return response
builder.add_edge("enter_update_flight", "update_flight")
builder.add_node( # safe tools node without interrupt
    "update_flight_safe_tools", 
    create_tool_node_with_fallback(flight_assistant.update_flight_safe_tools)
)
builder.add_node( # sensitive tools node with before interrupt
    "update_flight_sensitive_tools",
    create_tool_node_with_fallback(flight_assistant.update_flight_sensitive_tools)
)

builder.add_edge("update_flight_safe_tools", "update_flight") # after tool finish execution return to the sub-agent for next reasoning
builder.add_edge("update_flight_sensitive_tools", "update_flight")

route_update_flight = make_skill_router(flight_assistant.update_flight_sensitive_tools, "update_flight") #routing function

builder.add_conditional_edges(
    "update_flight",
    route_update_flight,
    ["update_flight_safe_tools", "update_flight_sensitive_tools", "leave_skill", END]
)

# ── Car rental sub-graph ───────────────────────────────────────────────────
builder.add_node("enter_book_car_rental", create_entry_node("Car Rental Assistant", "book_car_rental")) # entry node
builder.add_node("book_car_rental", Assistant(book_car_rental_runnable)) # sub-agent node
builder.add_edge("enter_book_car_rental", "book_car_rental")
builder.add_node( # safe tools node without interrupt
    "book_car_rental_safe_tools", 
    create_tool_node_with_fallback(car_rental_assistant.book_car_rental_safe_tools)
)
builder.add_node( # sensitive tools node with before interrupt
    "book_car_rental_sensitive_tools",
    create_tool_node_with_fallback(car_rental_assistant.book_car_rental_sensitive_tools)
)

builder.add_edge("book_car_rental_safe_tools", "book_car_rental") # after tool finish execution return to the sub-agent for next reasoning
builder.add_edge("book_car_rental_sensitive_tools", "book_car_rental")

route_book_car_rental = make_skill_router(car_rental_assistant.book_car_rental_sensitive_tools, "book_car_rental") #routing function

builder.add_conditional_edges(
    "book_car_rental",
    route_book_car_rental,
    ["book_car_rental_sensitive_tools", "book_car_rental_safe_tools", "leave_skill", END]
)

# ── Hotel sub-graph ────────────────────────────────────────────────────────
builder.add_node("enter_book_hotel", create_entry_node("Hotel Booking Assistant", "book_hotel")) # entry node
builder.add_node("book_hotel", Assistant(book_hotel_runnable)) # sub-agent node
builder.add_edge("enter_book_hotel", "book_hotel")
builder.add_node( # safe tools node without interrupt
    "book_hotel_safe_tools", 
    create_tool_node_with_fallback(hotel_assistant.book_hotel_safe_tools)
)
builder.add_node( # sensitive tools node with before interrupt
    "book_hotel_sensitive_tools",
    create_tool_node_with_fallback(hotel_assistant.book_hotel_sensitive_tools)
)

builder.add_edge("book_hotel_safe_tools", "book_hotel") # after tool finish execution return to the sub-agent for next reasoning
builder.add_edge("book_hotel_sensitive_tools", "book_hotel")

route_book_hotel = make_skill_router(hotel_assistant.book_hotel_sensitive_tools, "book_hotel") #routing function

builder.add_conditional_edges(
    "book_hotel",
    route_book_hotel,
    ["book_hotel_safe_tools", "book_hotel_sensitive_tools", "leave_skill", END]
)

# ── Excursion sub-graph ────────────────────────────────────────────────────
builder.add_node("enter_book_excursion", create_entry_node("Trip Recommendation Assistant", "book_excursion")) # entry node
builder.add_node("book_excursion", Assistant(book_excursion_runnable)) # sub-agent node
builder.add_edge("enter_book_excursion", "book_excursion")
builder.add_node( # safe tools node without interrupt
    "book_excursion_safe_tools", 
    create_tool_node_with_fallback(excursion_assistant.book_excursion_safe_tools)
)
builder.add_node( # sensitive tools node with before interrupt
    "book_excursion_sensitive_tools",
    create_tool_node_with_fallback(excursion_assistant.book_excursion_sensitive_tools)
)

builder.add_edge("book_excursion_safe_tools", "book_excursion") # after tool finish execution return to the sub-agent for next reasoning
builder.add_edge("book_excursion_sensitive_tools", "book_excursion")

route_book_excursion  = make_skill_router(excursion_assistant.book_excursion_sensitive_tools, "book_excursion") #routing function

builder.add_conditional_edges(
    "book_excursion",
    route_book_excursion,
    ["book_excursion_safe_tools", "book_excursion_sensitive_tools", "leave_skill", END]
)

# ── Primary assistant ──────────────────────────────────────────────────────
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
part_5_graph = builder.compile(
    checkpointer=memory,
    # Pause before any sensitive (write) tool so the user can approve/deny.
    interrupt_before=[
        "update_flight_sensitive_tools",
        "book_car_rental_sensitive_tools",
        "book_hotel_sensitive_tools",
        "book_excursion_sensitive_tools",
    ],
)

# Persist a visual of the compiled graph next to this file.
img_path = Path(__file__).resolve().parent / "graph.png"
part_5_graph.get_graph(xray=True).draw_mermaid_png(output_file_path=img_path)