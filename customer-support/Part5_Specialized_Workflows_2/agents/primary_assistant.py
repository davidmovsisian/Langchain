from datetime import datetime
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

from tools.flights import search_flights
from tools.lookup_company_policies import lookup_policy

from .flight_assistant import ToFlightBookingAssistant
from .car_rental_assistant import ToBookCarRental
from .hotel_assistant import ToHotelBookingAssistant
from .excursion_assistant import ToBookExcursion


# ---------------------------------------------------------------------------
# Prompt & runnable
# ---------------------------------------------------------------------------

# The primary assistant performs general Q&A and routes specialized tasks to
# the appropriate sub-agent via handoff tools (simple semantic routing /
# intent detection).
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            "Your primary role is to search for flight information and company policies to answer customer queries. "
            "If a customer requests to update or cancel a flight, book a car rental, book a hotel, or get trip recommendations, "
            "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. "
            "You are not able to make these types of changes yourself. "
            "Only the specialized assistants are given permission to do this for the user. "
            "The user is not aware of the different specialized assistants, so do not mention them; "
            "just quietly delegate through function calls. "
            "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
            "When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

primary_assistant_tools = [
    TavilySearchResults(max_results=1),
    search_flights,
    lookup_policy,
]

# All handoff tools the primary assistant can invoke to delegate to sub-agents.
handoff_tools = [
    ToFlightBookingAssistant,
    ToBookCarRental,
    ToHotelBookingAssistant,
    ToBookExcursion,
]


def build_runnable(llm):
    """Return the primary-assistant runnable bound to *llm*."""
    return primary_assistant_prompt | llm.bind_tools(
        primary_assistant_tools + handoff_tools
    )