"""
Personal assistant using the single dispatch tool pattern.

A single `task` tool dispatches to named subagents from a registry, rather than
wrapping each subagent in its own dedicated tool. This lets teams develop and
register agents independently without touching the supervisor or its tool list.

Pattern: Agent registry with task dispatcher
  - SUBAGENTS dict maps agent names → compiled agents
  - One `task(agent_name, description)` tool looks up and invokes the right agent
  - Supervisor's system prompt enumerates available agents (static registry < 10 agents)
  - Human-in-the-loop middleware on each subagent; interrupts bubble up through the task tool
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


# ---------------------------------------------------------------------------
# Shared model
# ---------------------------------------------------------------------------

model = ChatOpenAI(model="gpt-5.5", temperature=0.0)


# ---------------------------------------------------------------------------
# Primitive tools (leaf-level API calls)
# ---------------------------------------------------------------------------

@tool
def create_calendar_event(
    title: str,
    start_time: str,        # ISO-8601: "2024-01-15T14:00:00"
    end_time: str,          # ISO-8601: "2024-01-15T15:00:00"
    attendees: list[str],   # e-mail addresses
    location: str = "",
) -> str:
    """Create a calendar event. Requires exact ISO datetime strings."""
    # Stub: replace with Google Calendar / Outlook API call.
    return (
        f"Event created: '{title}' from {start_time} to {end_time} "
        f"with {len(attendees)} attendee(s)"
    )


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,              # ISO-8601: "2024-01-15"
    duration_minutes: int,
) -> list[str]:
    """Check calendar availability for attendees on a given date."""
    # Stub: replace with calendar API query.
    return ["09:00", "14:00", "16:00"]


@tool
def send_email(
    to: list[str],          # e-mail addresses
    subject: str,
    body: str,
    cc: list[str] = [],
) -> str:
    """Send an e-mail. Requires properly formatted addresses."""
    # Stub: replace with SendGrid / Gmail API call.
    recipients = ", ".join(to)
    return f"E-mail sent to {recipients} — Subject: {subject}\nBody: {body}"


# ---------------------------------------------------------------------------
# Subagents
# ---------------------------------------------------------------------------

_CALENDAR_SYSTEM_PROMPT = (
    f"Today's date is {date.today().isoformat()}. "
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g. 'next Tuesday at 2 pm') "
    "into proper ISO-8601 datetime strings. "
    "Use get_available_time_slots to check availability before creating an event. "
    "If no suitable slot exists, stop and report unavailability. "
    "Use create_calendar_event to schedule the event. "
    "Always summarise what was scheduled in your final response."
)

calendar_agent = create_agent(
    model=model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=_CALENDAR_SYSTEM_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"create_calendar_event": True},
            description_prefix="Calendar event pending approval",
        )
    ],
)

_EMAIL_SYSTEM_PROMPT = (
    "You are an e-mail assistant. "
    "Compose professional e-mails from natural language requests. "
    "Extract recipient information and craft an appropriate subject line and body. "
    "Use send_email to deliver the message. "
    "Always confirm what was sent in your final response."
)

email_agent = create_agent(
    model=model,
    tools=[send_email],
    system_prompt=_EMAIL_SYSTEM_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True},
            description_prefix="Outbound e-mail pending approval",
        )
    ],
)


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

class AgentName(str, Enum):
    """Enum constraint: the supervisor can only dispatch to registered agents."""
    CALENDAR = "calendar"
    EMAIL = "email"


# Maps AgentName → compiled agent.  Add new agents here without touching the
# supervisor or its tool schema (apart from the AgentName enum and the system
# prompt listing if the registry is small).
SUBAGENTS: dict[AgentName, object] = {
    AgentName.CALENDAR: calendar_agent,
    AgentName.EMAIL: email_agent,
}


# ---------------------------------------------------------------------------
# Single dispatch tool
# ---------------------------------------------------------------------------

@tool
def task(agent_name: AgentName, description: str) -> str:
    """Launch an ephemeral subagent for an independent subtask.

    Agents available:
    - calendar: Parse natural language scheduling requests, check availability,
                and create calendar events.
    - email:    Compose and send professional e-mails.

    Pass a self-contained description of the work to be done.  The subagent
    runs in an isolated context window and returns a concise summary.
    """
    agent = SUBAGENTS[agent_name]
    result = agent.invoke(
        {"messages": [{"role": "user", "content": description}]}
    )
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# Supervisor agent
# ---------------------------------------------------------------------------

_SUPERVISOR_SYSTEM_PROMPT = (
    "You are a helpful personal assistant. "
    "Delegate work to specialised subagents using the task tool.\n\n"
    "Available agents:\n"
    "- calendar: scheduling, availability checks, and calendar events\n"
    "- email:    composing and sending e-mails\n\n"
    "Break compound requests into independent subtasks and invoke the task "
    "tool once per subtask. When tasks are independent, invoke them in "
    "parallel. Combine the results into a coherent final reply."
)

supervisor_agent = create_agent(
    model=model,
    tools=[task],
    system_prompt=_SUPERVISOR_SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),   # persistent memory across turns
)


# ---------------------------------------------------------------------------
# Example run with human-in-the-loop interrupt handling
# ---------------------------------------------------------------------------

def run_with_hitl(query: str, thread_id: str = "1") -> None:
    """Run the supervisor, handle any HITL interrupts, then resume."""
    config = {"configurable": {"thread_id": thread_id}}
    interrupts: list = []

    # --- First pass ---------------------------------------------------
    stream = supervisor_agent.stream_events(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
        version="v3",
    )

    for kind, item in stream.interleave("messages", "tool_calls"):
        if kind == "messages":
            for token in item.text:
                print(token, end="", flush=True)
        elif kind == "tool_calls":
            print(f"\n[Tool] {item.tool_name}({item.input})")
            print(f"[Result] {item.output}")

    if stream.interrupted:
        for interrupt_ in stream.interrupts:
            interrupts.append(interrupt_)
            print(f"\n[INTERRUPTED] {interrupt_.id}")

    if not interrupts:
        return

    # --- Resolve interrupts -------------------------------------------
    resume: dict = {}
    for interrupt_ in interrupts:
        action_name = interrupt_.value["action_requests"][0]["name"]

        if action_name == "send_email":
            # Example: edit the e-mail subject before approving.
            edited_action = interrupt_.value["action_requests"][0].copy()
            edited_action["args"]["subject"] = "Mockups reminder"
            resume[interrupt_.id] = {
                "decisions": [{"type": "edit", "edited_action": edited_action}]
            }
        else:
            # Approve everything else as-is.
            resume[interrupt_.id] = {"decisions": [{"type": "approve"}]}

    interrupts = []

    # --- Second pass (resume) -----------------------------------------
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
            print(f"\n[Tool] {item.tool_name}({item.input})")

    if stream.interrupted:
        for interrupt_ in stream.interrupts:
            interrupts.append(interrupt_)
            print(f"\n[INTERRUPTED] {interrupt_.id}")


if __name__ == "__main__":
    query = (
        "Schedule a meeting with the design team next Tuesday at 2 pm for 1 hour, "
        "and send them an email to email@example.com about reviewing the new mockups."
    )
    run_with_hitl(query, thread_id="1")