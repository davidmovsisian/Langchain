import uuid
from .agents import part_6_graph
from .agents.budget_caping import (
    GraphBudget, 
    set_graph_budget, 
    reset_graph_budget, 
    ALL_POLICIES,
    BudgetExceededError
)
from sqlite_db import db, update_dates
from utils.utils import _print_event
from langgraph.types import Command

tutorial_questions = [
    "Hi there, what time is my flight?",
    "Am i allowed to update my flight to something sooner? I want to leave later today.",
    "Update my flight to sometime next week then",
    "The next available option is great",
    "what about lodging and transportation?",
    "Yeah i think i'd like an affordable hotel for my week-long stay (7 days). And I'll want to rent a car.",
    "OK could you place a reservation for your recommended hotel? It sounds nice.",
    "yes go ahead and book anything that's moderate expense and has availability.",
    "Now for a car, what are my options?",
    "Awesome let's just get the cheapest option. Go ahead and book for 7 days",
    "Cool so now what recommendations do you have on excursions?",
    "Are they available while I'm there?",
    "interesting - i like the museums, what options are there? ",
    "OK great pick one and book it for my second day there.",
]

AGENT_NAMES = [
    "primary_agent",
    "flight_agent",
    "hotel_agent",
    "car_rental_agent",
    "excursion_agent",
]

def print_budget(graph_budget: GraphBudget, header: str):
    print(f"\n{header}")
    print(f" global remaining: {graph_budget.remaining:.6f}")
    for name in AGENT_NAMES:
        print(f"  {name}: ${graph_budget.node_remaining(name):.6f}")

def pending_interrupts(config: dict) -> list:
    """All interrupts the graph is currently paused on.
 
    Several can be pending at once (the agents use parallel_tool_calls=True),
    and they can come from inside a sub-agent, so read them off the tasks.
    """

    snapshot = part_6_graph.get_state(config)
    return [i for task in snapshot.tasks for i in task.interrupts]

def ask_approval(payload: dict) -> dict:
    """Ask the human and return the resume value the middleware expects."""
    print("\n--- Approval required ---")
    print(f"Action: {payload.get('action')}")
    print(f"Args:   {payload.get('args')}")

    try:
        answer = input(
            "Type 'y' to approve; otherwise explain the change you want:\n> "
        ).strip()
    except EOFError:  # non-interactive run: auto-approve
        answer = "y"

    if answer.lower() in {'y', 'yes'}:
        return {"approved": True}
    return {"approved": False, "reason": answer}

def run_graph(graph_input, config: dict, printed: set):
    for event in part_6_graph.stream(graph_input, config, stream_mode="values"):
        _print_event(event, printed)

# Update with the backup file so we can restart from the original place in each section
db = update_dates(db)

def main():
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "passenger_id": "3442 587242",
            "thread_id": thread_id,
        }
    }

    graph_budget = GraphBudget(total_budget=0.10, overflow_fraction=0.1)
    for node_name, policy in ALL_POLICIES.items():
        graph_budget.register_policy(node_name, policy)
    budget_token = set_graph_budget(graph_budget)

    try:
        print_budget(graph_budget, "Initial budget")
        printed: set = set()

        for question in tutorial_questions:
            run_graph({"messages": ("user", question)}, config, printed)

            # The graph pauses whenever a sensitive tool needs approval.
            # Resume with Command(resume=...)

            while interrupts := pending_interrupts(config):
                decisions = {i.id: ask_approval(i.value) for i in interrupts}
                run_graph(Command(resume=decisions), config, printed)
    except BudgetExceededError as e:
        return {
            "response": "I've used the available budget for this conversation. "
                        "Please start a new conversation to continue.",
            "thread_id": thread_id,
            "primary_expected": e.primary_expected,
            "fallback_expected": e.fallback_expected,
            "budget_remaining": e.remaining,
        }
    finally:
        reset_graph_budget(budget_token)
        graph_budget.report()
        graph_budget.estimation_accuracy_report()

    graph_budget.report()
    graph_budget.estimation_accuracy_report()

if __name__ == "__main__":
    main()