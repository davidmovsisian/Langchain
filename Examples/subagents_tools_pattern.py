import os
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from datetime import date
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# implements tools pattern where each subagent is wrapped in a tool so that the supervisor agent can orchestrate the sub-agents using the tools.

model = ChatOpenAI(model="gpt-5.5", temperature=0.0)

@tool
def create_calendar_event(
    title: str,
    start_time: str,       # ISO format: "2024-01-15T14:00:00"
    end_time: str,         # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # email addresses
    location: str = ""
) -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    # Stub: In practice, this would call Google Calendar API, Outlook API, etc.
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"

@tool
def send_email(
    to: list[str],  # email addresses
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """Send an email via email API. Requires properly formatted addresses."""
    # Stub: In practice, this would call SendGrid, Gmail API, etc.
    return f"Email sent to {', '.join(to)} - Subject: {subject}\n Body: {body}"


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO format: "2024-01-15"
    duration_minutes: int
) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    # Stub: In practice, this would query calendar APIs
    return ["09:00", "14:00", "16:00"]


CALENDAR_AGENT_PROMPT = (
    f"Today's date is {date.today().isoformat()}. "
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
    "into proper ISO datetime formats. "
    "Use get_available_time_slots tool to check availability before creating calendar event."
    "If there is no suitable time slot, stop and confirm unavailability in your response. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response."
)

calendar_agent = create_agent(
    model=model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
    middleware=[HumanInTheLoopMiddleware(
        interrupt_on={"create_calendar_event": True},
        description_prefix="Calendar event pending approval"
    )]
)

# test calenda_agent
# query = "Schedule a team meeting next Tuesday at 2pm for 1 hour"
# stream = calendar_agent.stream_events(
#     {"messages": [{"role": "user", "content": query}]},
#     version="v3",
# )

# for kind, item in stream.interleave("messages", "tool_calls"):
#     if kind == "messages":
#         for token in item.text:
#             print(token, end="", flush=True)
#     elif kind == "tool_calls":
#         print(f"\nTool call: {item.tool_name}({item.input})")
#         print(f"Tool result: {item.output}")


EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response."
)

email_agent = create_agent(
    model=model,
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True},
            description_prefix="Outbound email pending approval",
        ),
    ]
)

# test email_agent
# query = "Send the design team a reminder about reviewing the new mockups"

# stream = email_agent.stream_events(
#     {"messages": [{"role": "user", "content": query}]},
#     version="v3",
# )
# for kind, item in stream.interleave("messages", "tool_calls"):
#     if kind == "messages":
#         for token in item.text:
#             print(token, end="", flush=True)
#     elif kind == "tool_calls":
#         print(f"\nTool call: {item.tool_name}({item.input})")
#         print(f"Tool result: {item.output}")

# warap each agent in the tool so assistant agent can handle handle subagents using the tools.
@tool
def schedule_event(request: str, runtime: ToolRuntime ) -> str: #use runtime to get access to shared state and context if needed
    """Schedule calendar events using natural language.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, availability checking, and event creation.

    Input: Natural language scheduling request (e.g., 'meeting with design team
    next Tuesday at 2pm')
    """

    # Customize context received by sub-agent
    original_user_message = next(
        message for message in runtime.state["messages"]
        if message.type == "human"
    )
    prompt = (
        "You are assisting with the following user inquiry:\n\n"
        f"{original_user_message.text}\n\n"
        "You are tasked with the following sub-request:\n\n"
        f"{request}"
    )

    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": prompt}]
    })
    return result["messages"][-1].text

@tool
def manage_email(request: str) -> str:
    """Send emails using natural language.

    Use this when the user wants to send notifications, reminders, or any email
    communication. Handles recipient extraction, subject generation, and email
    composition.

    Input: Natural language email request (e.g., 'send them a reminder about
    the meeting')
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

# supervisor agent that orchestrates the sub-agents

SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events and send emails. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple actions, use multiple tools in sequence or in parallel as appropriate."
)

supervisor_agent = create_agent(
    model=model,
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
    checkpointer=InMemorySaver(),
)

graph = supervisor_agent #for LangGraph dev server to load the graph from this file



if __name__ == "__main__":
    # query = "Schedule a team standup for tomorrow at 9am for Alice and Bob, and send them a reminder email about it to email@example.com."
    
    query = (
        "Schedule a meeting with the design team next Tuesday at 2pm for 1 hour, "
        "and send them an email to email@example.com about reviewing the new mockups."
    )

    # human in the loop interrupts
    config = {"configurable": {"thread_id": "6"}}
    interrupts = []

    stream = supervisor_agent.stream_events(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
        version="v3"
    )
    for kind, item in stream.interleave("messages", "tool_calls"):
        if kind == "messages":
            for token in item.text:
                print(token, end="", flush=True)
        elif kind == "tool_calls":
            print(f"\nTool call: {item.tool_name}({item.input})")
            print(f"Tool result: {item.output}")

    if stream.interrupted:
        for interrupt_ in stream.interrupts:
            interrupts.append(interrupt_)
            print(f"\nINTERRUPTED: {interrupt_.id}")

    resume = {}
    for interrupt_ in interrupts:
        if interrupt_.value["action_requests"][0]["name"] == "send_email":
            # Edit email
            edited_action = interrupt_.value["action_requests"][0].copy()
            edited_action["args"]["subject"] = "Mockups reminder"
            resume[interrupt_.id] = {
                "decisions": [{"type": "edit", "edited_action": edited_action}]
            }
        else:
            resume[interrupt_.id] = {"decisions": [{"type": "approve"}]}

    interrupts = []
    stream = supervisor_agent.stream_events(
        Command(resume=resume),
        config,
        version="v3",
    )
    for kind, item in stream.interleave("messages", "tool_calls"):
        if kind == "messages":
            for token in item.text:
                print(token, end="", flush=True)
        elif kind == "tool_calls":
            print(f"\nTool call: {item.tool_name}({item.input})")
    if stream.interrupted:
        for interrupt_ in stream.interrupts:
            interrupts.append(interrupt_)
            print(f"\nINTERRUPTED: {interrupt_.id}")

