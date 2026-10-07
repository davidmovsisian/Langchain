from .budget_caping import BudgetPolicy
from .specialist import build_specialist
 
from tools.flights import (
    cancel_ticket,
    search_flights,
    update_ticket_to_new_flight,
)

#  Tools
flight_tools = [search_flights, update_ticket_to_new_flight, cancel_ticket]
sensitive_tools_names = ["update_ticket_to_new_flight", "cancel_ticket"]

FLIGHT_PROMPT = """
    You are a specialized assistant for handling flight updates and cancellations.
    The primary assistant delegates work to you whenever the user needs help updating their flights. 
    Confirm updated flight details with the customer and inform them of any additional fees.
    When searching, be persistent. Expand your query bounds if the first search returns no results.

    Remember: a booking isn't completed until after the relevant tool has successfully been used.

    If the user needs help, and none of your tools are appropriate for it, then call "complete_or_escalate"
    to hand control back to the primary assistant with a short reason.
    Do not waste the user's time. Do not make up invalid tools or functions.

    Current user flight information:
        <Flights>
            {user_info}
        </Flights>

    Details passed from the primary assistant (use these to start; confirm with the user if unclear):
        <Handoff>
            {handoff}
        </Handoff>

    Current time: {time}.
"""

FLIGHT_BUDGET_POLICY = BudgetPolicy(
    budget_fraction=0.15,
    primary_model="openai:gpt-4o",
    primary_max_tokens=300, #maximal output tokens for primary model
    fallback_model="openai:gpt-4o-mini",
    fallback_max_tokens=150, #maximal output tokens for fallback model
    trend_cntr=2 #if number of node calls > trend_cntr, early swith to fallback model to preserve the node's budget
)

flight_agent = build_specialist(
    "flight_agent", flight_tools, sensitive_tools_names, FLIGHT_PROMPT, FLIGHT_BUDGET_POLICY
)


