from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from tools.flights import search_flights
from tools.car_rental import search_car_rentals
from tools.lookup_company_policies import lookup_policy
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from langchain.messages import ToolMessage
from .budget_caping import budget_middleware, BudgetPolicy
from tools.excursions import search_trip_recommendations
from tools.hotels import search_hotels

from .common import (
    format_prompt_middleware,
    prune_read_only_tool_traffic,
    summarization_middleware,
    find_calling_ai_message,
    TravelState,
)

primary_assistant_tools = [
    TavilySearchResults(max_results=1),
    search_flights,
    lookup_policy,
]


# ---------------------------------------------------------------------------
# Handoff tools (primary -> specialized assistants)
# ---------------------------------------------------------------------------

def _handoff(
    goto: str,
    handoff_data: dict,
    runtime: ToolRuntime[None, TravelState],
) -> Command:
    """Shared body of every transfer tool."""
    messages = [
        ToolMessage(
            content=f"Transferred to {goto.replace('_', ' ')}.",
            tool_call_id=runtime.tool_call_id,
        )
    ]
    # The AI message that issued the call must precede its ToolMessage in the
    # sub-agent's history; guard against the lookup failing.
    last_ai_message = find_calling_ai_message(runtime)
    if last_ai_message is not None:
        messages.insert(0, last_ai_message)

    return Command(
        goto=goto,
        update={ #update the state 
            "active_agent": goto,
            "handoff_data": handoff_data,
            "messages": messages,
        },
        graph=Command.PARENT,
    )


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
    return _handoff("flight_agent", {"request": request}, runtime)


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
    return _handoff(
        "car_rental_agent",
        {
            "location": location,
            "start_date": start_date,
            "end_date": end_date,
            "request": request,
        },
        runtime,
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
    return _handoff(
        "hotel_agent",
        {
            "location": location,
            "checkin_date": checkin_date,
            "checkout_date": checkout_date,
            "request": request,
        },
        runtime,
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
    return _handoff(
        "excursion_agent",
        {"location": location, "request": request},
        runtime,
    )


handoff_tools = [
    transfer_to_flight_agent,
    transfer_to_car_rental_agent,
    transfer_to_hotel_agent,
    transfer_to_excursion_agent,
]


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------

# Tools whose old results are safe to forget (they go stale, and the assistant
# restates what matters in its replies). Messages are shared across agents, so
# this also cleans up the sub-agents' search traffic.
# NEVER add booking/update/cancel tools or handoff tools here.
READ_ONLY_TOOLS = {t.name for t in primary_assistant_tools + 
                   [
                       search_car_rentals, 
                       search_trip_recommendations,
                       search_hotels
                    ]} 

# The primary assistant performs general Q&A and routes specialized tasks to
# the appropriate sub-agent via handoff tools
PRIMARY_PROMPT = """
    You are a helpful customer support assistant for Swiss Airlines.
    Your primary role is to search for flight information and company policies to answer customer queries.
    If a customer requests to update or cancel a flight, book a car rental, book a hotel, or get trip recommendations, 
        delegate the task to the appropriate specialized assistant by invoking the corresponding tool.
    Delegate to one specialized assistant at a time. If the request needs several, start with the first
        and handle the others once control returns to you.
    You are not able to make these types of changes yourself.
    Only the specialized assistants are given permission to do this for the user.
    The user is not aware of the different specialized assistants, so do not mention them, just quietly delegate through function calls.
    Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable.
    When searching, be persistent. Expand your query bounds if the first search returns no results.
    If a search comes up empty, expand your search before giving up.
    After any search, restate the key facts (prices, times, IDs) in your reply so they are not lost from the history.

    Current user flight information:
    <Flights>
        {user_info}
    </Flights>

    Current time: {time}
"""

PRIMARY_BUDGET_POLICY = BudgetPolicy(
    budget_fraction=0.40,
    primary_model="openai:gpt-4o",
    primary_max_tokens=300, #maximal output tokens for primary model
    fallback_model="openai:gpt-4o-mini",
    fallback_max_tokens=150, #maximal output tokens for fallback model
    trend_cntr=1 #if number of node calls > trend_cntr, early swith to fallback model to preserve the node's budget
)

model = init_chat_model(model=PRIMARY_BUDGET_POLICY.primary_model, temperature=0)

excursion_agent = build_specialist(
    "excursion_agent", excursion_tools, sensitive_tools_names, EXCURSION_PROMPT, EXCURSION_BUDGET_POLICY
)

primary_agent = create_agent(
    # One tool call at a time: parallel handoffs would emit two Commands with
    # different `goto` values and both write active_agent / handoff_data.
    model=model.bind(parallel_tool_calls=False),
    tools=primary_assistant_tools + handoff_tools,
    state_schema=TravelState,
    middleware=[
        # prune first, then summarize the rest.
        prune_read_only_tool_traffic(READ_ONLY_TOOLS, keep_last=6),
        summarization_middleware(trigger=3000, keep_last=6),
        format_prompt_middleware(PRIMARY_PROMPT),
        budget_middleware(PRIMARY_BUDGET_POLICY, "primary_agent", parallel_tool_calls = False),
    ],
)