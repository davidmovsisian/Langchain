from .budget_caping import BudgetPolicy
from .specialist import build_specialist

from tools.excursions import (
    book_excursion,
    cancel_excursion,
    search_trip_recommendations,
    update_excursion,
)

excursion_tools = [book_excursion, cancel_excursion, search_trip_recommendations, update_excursion]
sensitive_tools_names = ["book_excursion", "cancel_excursion", "update_excursion"]

EXCURSION_PROMPT = """
    You are a specialized assistant for handling trip recommendations.
    The primary assistant delegates work to you whenever the user needs help booking a recommended trip.
    Search for available trip recommendations based on the user's preferences and confirm the booking details with the customer.

    Remember: a booking isn't completed until after the relevant tool has successfully been used.

    If the user needs help, and none of your tools are appropriate for it, then call "complete_or_escalate" 
        to hand control back to the primary assistant with a short reason.
        Do not waste the user's time. Do not make up invalid tools or functions.

    
    Some examples for which you should complete_or_escalate:
    - 'nevermind i think I'll book separately.'
    - 'i need to figure out transportation while i'm there.'
    - 'Oh wait i haven't booked my flight yet i'll do that first.'
    - 'Excursion booking confirmed!'

    Current user flight information:
        <Flights>
            {user_info}
        </Flights>
    
    Details passed from the primary assistant (use these to start; confirm with the user if unclear):
        <Handoff>
            {handoff}
        </Handoff>

    Current time: {time}
"""

EXCURSION_BUDGET_POLICY = BudgetPolicy(
    budget_fraction=0.15,
    primary_model="openai:gpt-4o",
    primary_max_tokens=300, #maximal output tokens for primary model
    fallback_model="openai:gpt-4o-mini",
    fallback_max_tokens=150, #maximal output tokens for fallback model,
    trend_cntr=2 #if number of node calls > trend_cntr, early swith to fallback model to preserve the node's budget
)

excursion_agent = build_specialist(
    "excursion_agent", excursion_tools, sensitive_tools_names, EXCURSION_PROMPT, EXCURSION_BUDGET_POLICY
)