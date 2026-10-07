from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from tools.excursions import (
    book_excursion,
    cancel_excursion,
    search_trip_recommendations,
    update_excursion,
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

class ToBookExcursion(BaseModel):
    """Transfers work to a specialized assistant to handle trip recommendations
    and other excursion bookings."""

    location: str = Field(
        description="The location where the user wants to book a recommended trip."
    )
    request: str = Field(
        description="Any additional information or requests from the user regarding the trip recommendation."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location": "Lucerne",
                "request": "The user is interested in outdoor activities and scenic views.",
            }
        }


# ---------------------------------------------------------------------------
# Prompt & runnable
# ---------------------------------------------------------------------------

book_excursion_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling trip recommendations. "
            "The primary assistant delegates work to you whenever the user needs help booking a recommended trip. "
            "Search for available trip recommendations based on the user's preferences and confirm the booking details with the customer. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant. "
            "When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "Remember that a booking isn't completed until after the relevant tool has successfully been used."
            "\nCurrent time: {time}."
            '\n\nIf the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant. '
            "Do not waste the user's time. Do not make up invalid tools or functions."
            "\n\nSome examples for which you should CompleteOrEscalate:\n"
            " - 'nevermind i think I'll book separately'\n"
            " - 'i need to figure out transportation while i'm there'\n"
            " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
            " - 'Excursion booking confirmed!'",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

book_excursion_safe_tools = [search_trip_recommendations]
book_excursion_sensitive_tools = [book_excursion, update_excursion, cancel_excursion]
book_excursion_tools = book_excursion_safe_tools + book_excursion_sensitive_tools


def build_graph(llm): #-> CompiledStateGraph
    runnable = book_excursion_prompt | llm.bind_tools(book_excursion_tools + [CompleteOrEscalate])
    router = make_skill_router(book_excursion_sensitive_tools, "book_excursion")
    sg = StateGraph(State)

    #  Nodes
    sg.add_node("book_excursion", Assistant(runnable))
    sg.add_node("book_excursion_safe_tools", create_tool_node_with_fallback(book_excursion_safe_tools))
    sg.add_node("book_excursion_sensitive_tools", create_tool_node_with_fallback(book_excursion_sensitive_tools))
    sg.add_node("leave_skill", pop_dialog_state) # internal — answers CompleteOrEscalate

    # Edges
    sg.add_edge(START, "book_excursion")
    sg.add_edge("book_excursion_safe_tools", "book_excursion")
    sg.add_edge("book_excursion_sensitive_tools", "book_excursion")
    sg.add_edge("leave_skill", END) # leave sub-graph after updating the states stack

    sg.add_conditional_edges(
        "book_excursion",
        router,
        [
            "book_excursion_safe_tools", 
            "book_excursion_sensitive_tools", 
            "leave_skill", 
            END
        ]
    )

    return sg.compile(interrupt_before = ["book_excursion_sensitive_tools"])