from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
 
from tools.flights import (
    cancel_ticket,
    search_flights,
    update_ticket_to_new_flight,
)
from .common import (
    sensitive_tools_middleware, 
    complete_or_escalate,
    format_prompt_middleware
)

model = init_chat_model(model="openai:gpt-4o", temperature=0)

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

    Current time: {time}.
"""

flight_agent = create_agent(
    model=model.bind(parallel_tool_calls=False),
    tools=flight_tools +[complete_or_escalate],
    middleware = [
        sensitive_tools_middleware(sensitive_tools_names),
        format_prompt_middleware(FLIGHT_PROMPT)
        ]
)


