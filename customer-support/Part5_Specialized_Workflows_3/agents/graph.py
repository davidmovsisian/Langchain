from pathlib import Path
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from tools.flights import fetch_user_flight_information
from .common import TravelState

from .car_rental_assistant import car_rental_agent
from .excursion_assistant import excursion_agent
from .flight_assistant import flight_agent
from .hotel_assistant import hotel_agent
from .primary_assistant import primary_agent

def fetch_user_info(state: TravelState):
    return {"user_info": fetch_user_flight_information.invoke({})}

def route_to_active_agent(state: TravelState):
    active_agent = state.get("active_agent") or "primary_agent"
    return active_agent

builder = StateGraph(TravelState)

builder.add_node("fetch_user_info", fetch_user_info)
builder.add_edge(START, "fetch_user_info")
builder.add_conditional_edges("fetch_user_info", route_to_active_agent)

# ── Flight sub-graph ───────────────────────────────────────────────────────
builder.add_node("flight_agent", flight_agent) 

# ── Car rental sub-graph ───────────────────────────────────────────────────
builder.add_node("car_rental_agent", car_rental_agent) 

# ── Hotel sub-graph ────────────────────────────────────────────────────────
builder.add_node("hotel_agent", hotel_agent)

# ── Excursion sub-graph ────────────────────────────────────────────────────
builder.add_node("excursion_agent", excursion_agent)

# ── Primary agent ──────────────────────────────────────────────────────
builder.add_node("primary_agent", primary_agent)

memory = InMemorySaver()
part_5_graph = builder.compile(checkpointer=memory)

# Persist a visual of the compiled graph next to this file.
img_path = Path(__file__).resolve().parent / "graph.png"
part_5_graph.get_graph(xray=True).draw_mermaid_png(output_file_path=img_path)