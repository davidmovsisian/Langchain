from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from tools.car_rental import (
    book_car_rental,
    cancel_car_rental,
    search_car_rentals,
    update_car_rental,
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

class ToBookCarRental(BaseModel):
    """Transfers work to a specialized assistant to handle car rental bookings."""

    location: str = Field(
        description="The location where the user wants to rent a car."
    )
    start_date: str = Field(description="The start date of the car rental.")
    end_date: str = Field(description="The end date of the car rental.")
    request: str = Field(
        description="Any additional information or requests from the user regarding the car rental."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Basel",
                "start_date": "2023-07-01",
                "end_date": "2023-07-05",
                "request": "I need a compact car with automatic transmission.",
            }
        }


# ---------------------------------------------------------------------------
# Prompt & runnable
# ---------------------------------------------------------------------------

book_car_rental_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling car rental bookings. "
            "The primary assistant delegates work to you whenever the user needs help booking a car rental. "
            "Search for available car rentals based on the user's preferences and confirm the booking details with the customer. "
            "When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant. "
            "Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\nCurrent time: {time}."
            "\n\nIf the user needs help, and none of your tools are appropriate for it, then "
            '"CompleteOrEscalate" the dialog to the host assistant. '
            "Do not waste the user's time. Do not make up invalid tools or functions."
            "\n\nSome examples for which you should CompleteOrEscalate:\n"
            " - 'what's the weather like this time of year?'\n"
            " - 'What flights are available?'\n"
            " - 'nevermind i think I'll book separately'\n"
            " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
            " - 'Car rental booking confirmed'",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

book_car_rental_safe_tools = [search_car_rentals]
book_car_rental_sensitive_tools = [book_car_rental, update_car_rental, cancel_car_rental]
book_car_rental_tools = book_car_rental_safe_tools + book_car_rental_sensitive_tools


def build_graph(llm): # -> CompiledStateGraph
    runnable = book_car_rental_prompt | llm.bind_tools(book_car_rental_tools + [CompleteOrEscalate])
    router = make_skill_router(book_car_rental_sensitive_tools, "book_car_rental")
    sg = StateGraph(State)

    #  Nodes
    sg.add_node("book_car_rental", Assistant(runnable))
    sg.add_node("book_car_rental_safe_tools", create_tool_node_with_fallback(book_car_rental_safe_tools))
    sg.add_node("book_car_sensitive_tools", create_tool_node_with_fallback(book_car_rental_sensitive_tools))
    sg.add_node("leave_skill", pop_dialog_state) # internal — answers CompleteOrEscalate

    # Edges
    sg.add_edge(START, "book_car_rental")
    sg.add_edge("book_car_rental_safe_tools", "book_car_rental")
    sg.add_edge("book_car_rental_sensitive_tools", "book_car_rental")
    sg.add_edge("leave_skill", END) # leave sub-graph after updating the states stack

    sg.add_conditional_edges(
        "book_car_rental",
        router,
        [
            "book_car_rental_safe_tools", 
            "book_car_rental_sensitive_tools", 
            "leave_skill", 
            END
        ]
    )

    return sg.compile(interrupt_before = ["book_car_rental_sensitive_tools"])