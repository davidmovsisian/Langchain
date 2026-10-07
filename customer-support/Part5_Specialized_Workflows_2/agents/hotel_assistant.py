from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from tools.hotels import (
    book_hotel,
    cancel_hotel,
    search_hotels,
    update_hotel,
)
from .common import (
    Assistant,
    CompleteOrEscalate,
    State,
    make_skill_router,
    pop_dialog_state,
    create_tool_node_with_fallback
)


# ---------------------------------------------------------------------------
# Handoff tool
# ---------------------------------------------------------------------------

class ToHotelBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle hotel bookings."""

    location: str = Field(
        description="The location where the user wants to book a hotel."
    )
    checkin_date: str = Field(description="The check-in date for the hotel.")
    checkout_date: str = Field(description="The check-out date for the hotel.")
    request: str = Field(
        description="Any additional information or requests from the user regarding the hotel booking."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Zurich",
                "checkin_date": "2023-08-15",
                "checkout_date": "2023-08-20",
                "request": "I prefer a hotel near the city center with a room that has a view.",
            }
        }


# ---------------------------------------------------------------------------
# Prompt & runnable
# ---------------------------------------------------------------------------

book_hotel_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling hotel bookings. "
            "The primary assistant delegates work to you whenever the user needs help booking a hotel. "
            "Search for available hotels based on the user's preferences and confirm the booking details with the customer. "
            "When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant. "
            "Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\nCurrent time: {time}."
            '\n\nIf the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant. '
            "Do not waste the user's time. Do not make up invalid tools or functions."
            "\n\nSome examples for which you should CompleteOrEscalate:\n"
            " - 'what's the weather like this time of year?'\n"
            " - 'nevermind i think I'll book separately'\n"
            " - 'i need to figure out transportation while i'm there'\n"
            " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
            " - 'Hotel booking confirmed'",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

book_hotel_safe_tools = [search_hotels]
book_hotel_sensitive_tools = [book_hotel, update_hotel, cancel_hotel]
book_hotel_tools = book_hotel_safe_tools + book_hotel_sensitive_tools


def build_graph(llm): #-> CompiledStateGraph:
    runnable = book_hotel_prompt | llm.bind_tools(book_hotel_tools + [CompleteOrEscalate])
    router = make_skill_router(book_hotel_sensitive_tools, "book_hotel")
    sg = StateGraph(State)

    #  Nodes
    sg.add_node("book_hotel", Assistant(runnable))
    sg.add_node("book_hotel_safe_tools", create_tool_node_with_fallback(book_hotel_safe_tools))
    sg.add_node("book_hotel_sensitive_tools", create_tool_node_with_fallback(book_hotel_sensitive_tools))
    sg.add_node("leave_skill", pop_dialog_state) # internal — answers CompleteOrEscalate

    # Edges
    sg.add_edge(START, "book_hotel")
    sg.add_edge("book_hotel_safe_tools", "book_hotel")
    sg.add_edge("book_hotel_sensitive_tools", "book_hotel")
    sg.add_edge("leave_skill", END) # leave sub-graph after updating the states stack

    sg.add_conditional_edges(
        "book_hotel",
        router,
        [
            "book_hotel_safe_tools", 
            "book_hotel_sensitive_tools", 
            "leave_skill", 
            END
        ]
    )

    return sg.compile(interrupt_before = ["book_hotel_sensitive_tools"])