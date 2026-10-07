from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from tools.flights import search_flights
from tools.lookup_company_policies import lookup_policy
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from langchain.messages import ToolMessage, AIMessage
from budget_caping import budget_middleware, BudgetPolicy

from .common import (
    format_prompt_middleware,
    TravelState
)

primary_assistant_tools = [
    TavilySearchResults(max_results=1),
    search_flights,
    lookup_policy,
]

# Handoff tools, used for transfer from prime assistant to specialized assistant
@tool
def transfer_to_flight_agent(
    request: str,
    runtime: ToolRuntime[None, TravelState],
) -> Command:
    """Transfer to the flight assistant for flight updates and cancellations.
    Args:
        request: Any necessary follow-up questions the flight assistant
                 should clarify before proceeding.
    """
    last_ai_message = next(
            (
                msg for msg in reversed(runtime.state["messages"]) 
                if isinstance(msg, AIMessage)
                and any(
                    tc["id"] == runtime.tool_call_id
                    for tc in msg.tool_calls
                )
            ),
            None
        )

    transfer_message = ToolMessage(
        content = "Transferred to flight agent.",
        tool_call_id=runtime.tool_call_id,
    )

    return Command(
    goto="flight_agent",
    update={
        "active_agent": "flight_agent",
        "handoff_data": {
            "request": request,
        },
        "messages": [last_ai_message, transfer_message],
    },
    graph=Command.PARENT,
)

@tool
def transfer_to_car_rental_agent(
    location: str,
    start_date: str,
    end_date: str,
    request: str,
    runtime: ToolRuntime[None, TravelState],
) -> Command:
    """Transfer to the car rental assistant for car rental bookings.
    Args:
        location: The location where the user wants to rent a car.
        start_date: The start date of the car rental.
        end_date: The end date of the car rental.
        request: Any additional information or requests from the user.
    """
    last_ai_message = next(
        (
            msg for msg in reversed(runtime.state["messages"]) 
            if isinstance(msg, AIMessage)
            and any(
                tc["id"] == runtime.tool_call_id
                for tc in msg.tool_calls
            )
        ),
        None
    )

    transfer_message = ToolMessage(
        content = "Transferred to car rental agent.",
        tool_call_id=runtime.tool_call_id,
    )

    return Command(
        goto="car_rental_agent",
        update={
            "active_agent": "car_rental_agent",
            "handoff_data": {
                "location": location,
                "start_date": start_date,
                "end_date": end_date,
                "request": request,
            },
            "messages": [last_ai_message, transfer_message],
        },
        graph=Command.PARENT,
    )


@tool
def transfer_to_hotel_agent(
    location: str,
    checkin_date: str,
    checkout_date: str,
    request: str,
    runtime: ToolRuntime[None, TravelState],
) -> Command:
    """Transfer to the hotel agent for hotel bookings.

    Args:
        location: The location where the user wants to book a hotel.
        checkin_date: The check-in date for the hotel.
        checkout_date: The check-out date for the hotel.
        request: Any additional information or requests from the user.
    """
    last_ai_message = next(
        (
            msg for msg in reversed(runtime.state["messages"]) 
            if isinstance(msg, AIMessage)
            and any(
                tc["id"] == runtime.tool_call_id
                for tc in msg.tool_calls
            )
        ),
        None
    )

    transfer_message = ToolMessage(
        content = "Transferred to hotel agent.",
        tool_call_id=runtime.tool_call_id,
    )

    return Command(
        goto="hotel_agent",
        update={
            "active_agent": "hotel_agent",
            "handoff_data": {
                "location": location,
                "checkin_date": checkin_date,
                "checkout_date": checkout_date,
                "request": request,
            },
            "messages": [last_ai_message, transfer_message],
        },
        graph=Command.PARENT,
    )


@tool
def transfer_to_excursion_agent(
    location: str,
    request: str,
    runtime: ToolRuntime[None, TravelState],
) -> Command:
    """Transfer to the excursion agent for trip recommendations and bookings.

    Args:
        location: The location where the user wants to book a recommended trip.
        request: Any additional information or requests from the user.
    """
    last_ai_message = next(
        (
            msg for msg in reversed(runtime.state["messages"]) 
            if isinstance(msg, AIMessage)
            and any(
                tc["id"] == runtime.tool_call_id
                for tc in msg.tool_calls
            )
        ),
        None
    )

    transfer_message = ToolMessage(
        content = "Transferred to excursion agent.",
        tool_call_id=runtime.tool_call_id,
    )

    return Command(
        goto="excursion_agent",
        update={
            "active_agent": "excursion_agent",
            "handoff_data": {
                "location": location,
                "request": request,
            },
            "messages": [last_ai_message, transfer_message],
        },
        graph=Command.PARENT,
    )
    

handoff_tools = [
    transfer_to_flight_agent,
    transfer_to_car_rental_agent,
    transfer_to_hotel_agent,
    transfer_to_excursion_agent
]



# The primary assistant performs general Q&A and routes specialized tasks to
# the appropriate sub-agent via handoff tools
PRIMARY_PROMPT = """
    You are a helpful customer support assistant for Swiss Airlines.
    Your primary role is to search for flight information and company policies to answer customer queries.
    If a customer requests to update or cancel a flight, book a car rental, book a hotel, or get trip recommendations, 
        delegate the task to the appropriate specialized assistant by invoking the corresponding tool.
    You are not able to make these types of changes yourself.
    Only the specialized assistants are given permission to do this for the user.
    The user is not aware of the different specialized assistants, so do not mention them, just quietly delegate through function calls.
    Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable.
    When searching, be persistent. Expand your query bounds if the first search returns no results.
    If a search comes up empty, expand your search before giving up.

    Current user flight information:
    <Flights>
        {user_info}
    </Flights>

    Current time: {time}
"""

PRIMARY_BUDGET_POLICY = BudgetPolicy(
    budget_fraction=0.20,
    primary_model="openai:gpt-4o",
    primary_max_tokens=1000, #maximal output tokens for primary model
    fallback_model="openai:gpt-4o-mini",
    fallback_max_tokens=600, #maximal output tokens for fallback model
    trend_cntr=0 #if number of node calls > trend_cntr, early swith to fallback model to preserve the node's budget
)

model = init_chat_model(model=PRIMARY_BUDGET_POLICY.primary_model, temperature=0)

primary_agent = create_agent(
    model = model.bind(parallel_tool_calls=False),
    tools = primary_assistant_tools + handoff_tools,
    middleware = [
        format_prompt_middleware(PRIMARY_PROMPT),
        budget_middleware(PRIMARY_BUDGET_POLICY, "primary_agent"),
        ]
)