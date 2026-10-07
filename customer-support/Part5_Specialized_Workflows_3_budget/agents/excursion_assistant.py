from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from budget_caping import budget_middleware, BudgetPolicy

from tools.excursions import (
    book_excursion,
    cancel_excursion,
    search_trip_recommendations,
    update_excursion,
)
from .common import (
    sensitive_tools_middleware, 
    complete_or_escalate,
    format_prompt_middleware,
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

    
    Some examples for which you should CompleteOrEscalate:
    - 'nevermind i think I'll book separately.'
    - 'i need to figure out transportation while i'm there.'
    - 'Oh wait i haven't booked my flight yet i'll do that first.'
    - 'Excursion booking confirmed!'

    Current user flight information:
        <Flights>
            {user_info}
        </Flights>
        
    Current time: {time}
"""

EXCURSION_BUDGET_POLICY = BudgetPolicy(
    budget_fraction=0.20,
    primary_model="openai:gpt-4o",
    primary_max_tokens=1000, #maximal output tokens for primary model
    fallback_model="openai:gpt-4o-mini",
    fallback_max_tokens=600, #maximal output tokens for fallback model,
    trend_cntr=2 #if number of node calls > trend_cntr, early swith to fallback model to preserve the node's budget
)

model = init_chat_model(model=EXCURSION_BUDGET_POLICY.primary_model, temperature=0)

excursion_agent = create_agent(
    model = model.bind(parallel_tool_calls=False),
    tools = excursion_tools + [complete_or_escalate],
    middleware = [
        sensitive_tools_middleware(sensitive_tools_names),
        format_prompt_middleware(EXCURSION_PROMPT),
        budget_middleware(EXCURSION_BUDGET_POLICY, "excursion_agent"),
        ]
)