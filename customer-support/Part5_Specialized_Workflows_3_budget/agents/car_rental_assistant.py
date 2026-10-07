from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from budget_caping import budget_middleware, BudgetPolicy

from tools.car_rental import (
    book_car_rental,
    cancel_car_rental,
    search_car_rentals,
    update_car_rental,
)
from .common import (
    sensitive_tools_middleware,
    complete_or_escalate,
    format_prompt_middleware,
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

    Current time: {time}
"""

CAR_RENTAL_BUDGET_POLICY = BudgetPolicy(
    budget_fraction=0.20,
    primary_model="openai:gpt-4o",
    primary_max_tokens=1000, #maximal output tokens for primary model
    fallback_model="openai:gpt-4o-mini",
    fallback_max_tokens=600, #maximal output tokens for fallback model,
    trend_cntr=2 #if number of node calls > trend_cntr, early swith to fallback model to preserve the node's budget
)

model = init_chat_model(model=CAR_RENTAL_BUDGET_POLICY.primary_model, temperature=0)

car_rental_agent = create_agent(
    model=model.bind(parallel_tool_calls=False),
    tools=car_rental_tools + [complete_or_escalate],
    middleware=[
        sensitive_tools_middleware(sensitive_tools_names),
        format_prompt_middleware(CAR_RENTAL_PROMPT),
        budget_middleware(CAR_RENTAL_BUDGET_POLICY, "car_rental_agent"),
    ],
)
