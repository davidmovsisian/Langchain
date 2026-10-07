from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from tools.hotels import (
    book_hotel,
    cancel_hotel,
    search_hotels,
    update_hotel,
)
from .common import (
    sensitive_tools_middleware, 
    complete_or_escalate,
    format_prompt_middleware
)

model = init_chat_model(model="openai:gpt-4o", temperature=0)

hotel_tools = [search_hotels, book_hotel, update_hotel, cancel_hotel]
sensitive_tools_names = ["book_hotel", "update_hotel", "cancel_hotel"]

HOTEL_PROMPT = """
    You are a specialized assistant for handling hotel bookings.
    The primary assistant delegates work to you whenever the user needs help booking a hotel.
    Search for available hotels based on the user's preferences and confirm the booking details with the customer.
    When searching, be persistent. Expand your query bounds if the first search returns no results.

    Remember: a booking isn't completed until after the relevant tool has successfully been used.

    If the user needs help, and none of your tools are appropriate for it, then call "complete_or_escalate" 
    to hand control back to the primary assistant with a short reason.
    Do not waste the user's time. Do not make up invalid tools or functions.
  
    Some examples for which you should complete_or_escalate:
    - "what's the weather like this time of year?"
    - "nevermind i think I'll book separately."
    - "'i need to figure out transportation while i'm there."
    - "Oh wait i haven't booked my flight yet i'll do that first."
    - "Hotel booking confirmed."

    Current user flight information:
        <Flights>
            {user_info}
        </Flights>
    
    Current time: {time}
"""

hotel_agent = create_agent(
    model = model.bind(parallel_tool_calls=False),
    tools = hotel_tools + [complete_or_escalate],
    middleware = [
        sensitive_tools_middleware(sensitive_tools_names),
        format_prompt_middleware(HOTEL_PROMPT)
        ]
)