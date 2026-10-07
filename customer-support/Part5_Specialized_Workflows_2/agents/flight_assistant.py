from datetime import datetime
 
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
 
from tools.flights import (
    cancel_ticket,
    search_flights,
    update_ticket_to_new_flight,
)
from .common import (
    Assistant,
    CompleteOrEscalate,
    State,
    make_skill_router,
    pop_dialog_state,
    create_tool_node_with_fallback
)

# Handoff tool, used for transfer from prime assistant to specialized assistant
class ToFlightBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight updates and
    cancellations."""
 
    request: str = Field(
        description="Any necessary follow-up questions the flight assistant should clarify before proceeding."
    )

#  Tools
update_flight_safe_tools = [search_flights]
update_flight_sensitive_tools = [update_ticket_to_new_flight, cancel_ticket]
update_flight_tools = update_flight_safe_tools + update_flight_sensitive_tools

flight_booking_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling flight updates. "
            "The primary assistant delegates work to you whenever the user needs help updating their bookings. "
            "Confirm the updated flight details with the customer and inform them of any additional fees. "
            "When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant. "
            "Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>"
            "\nCurrent time: {time}."
            "\n\nIf the user needs help, and none of your tools are appropriate for it, then "
            '"CompleteOrEscalate" the dialog to the host assistant. '
            "Do not waste the user's time. Do not make up invalid tools or functions.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

def build_graph(llm): #-> CompiledStateGraph
    runnable = flight_booking_prompt | llm.bind_tools(update_flight_tools + [CompleteOrEscalate])
    router = make_skill_router(update_flight_sensitive_tools, "update_flight")
    sg = StateGraph(State)

    #  Nodes
    sg.add_node("update_flight", Assistant(runnable))
    sg.add_node("update_flight_safe_tools", create_tool_node_with_fallback(update_flight_safe_tools))
    sg.add_node("update_flight_sensitive_tools", create_tool_node_with_fallback(update_flight_sensitive_tools))
    sg.add_node("leave_skill", pop_dialog_state) # internal — answers CompleteOrEscalate

    # Edges
    sg.add_edge(START, "update_flight")
    sg.add_edge("update_flight_safe_tools", "update_flight")
    sg.add_edge("update_flight_sensitive_tools", "update_flight")
    sg.add_edge("leave_skill", END) # leave sub-graph after updating the states stack

    sg.add_conditional_edges(
        "update_flight",
        router,
        ["update_flight_safe_tools", "update_flight_sensitive_tools", "leave_skill", END]
    )

    return sg.compile(interrupt_before = ["update_flight_sensitive_tools"])
