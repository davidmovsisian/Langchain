from .budget_caping import BudgetPolicy
from .specialist import build_specialist

from tools.car_rental import (
    book_car_rental,
    cancel_car_rental,
    search_car_rentals,
    update_car_rental,
)

car_rental_tools = [book_car_rental, cancel_car_rental, search_car_rentals, update_car_rental]
sensitive_tools_names = ["book_car_rental", "cancel_car_rental", "update_car_rental"]

CAR_RENTAL_PROMPT = """
    You are a specialized assistant for handling car rental bookings.
    The primary assistant delegates work to you whenever the user needs help booking a car rental.
    Search for available car rentals based on the user's preferences and confirm the booking details with the customer.
    When searching, be persistent. Expand your query bounds if the first search returns no results.

    Remember: a booking isn't completed until after the relevant tool has successfully been used.

    If the user needs help, and none of your tools are appropriate for it, then call "complete_or_escalate"
    to hand control back to the primary assistant with a short reason.
    Do not waste the user's time. Do not make up invalid tools or functions.

    Some examples for which you should complete_or_escalate:
    - 'what's the weather like this time of year?'
    - 'What flights are available?'
    - 'nevermind i think I'll book separately'
    - 'Oh wait i haven't booked my flight yet i'll do that first'
    - 'Car rental booking confirmed'

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

CAR_RENTAL_BUDGET_POLICY = BudgetPolicy(
    budget_fraction=0.15,
    primary_model="openai:gpt-4o",
    primary_max_tokens=300, #maximal output tokens for primary model
    fallback_model="openai:gpt-4o-mini",
    fallback_max_tokens=150, #maximal output tokens for fallback model,
    trend_cntr=2 #if number of node calls > trend_cntr, early swith to fallback model to preserve the node's budget
)

car_rental_agent = build_specialist(
    "car_rental_agent", car_rental_tools, sensitive_tools_names, CAR_RENTAL_PROMPT, CAR_RENTAL_BUDGET_POLICY
)
