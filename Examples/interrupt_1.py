"""
Human-in-the-Loop (HITL) Demo — LangGraph
==========================================
A minimal graph with NO LLM calls.
Focus: understand exactly what happens with interrupt().

Graph flow:
  START → step_1 → human_review → step_2 → END
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


# ─────────────────────────────────────────
# 1. STATE
# ─────────────────────────────────────────
class MyState(TypedDict):
    message: str          # set at start
    draft: str            # set by step_1
    final: str            # set by step_2 after human approves


# ─────────────────────────────────────────
# 2. NODES
# ─────────────────────────────────────────

def step_1(state: MyState) -> dict:
    print("\n[step_1] ▶ ENTERED")
    print(f"[step_1]   state so far: {dict(state)}")

    draft = f"Auto-draft based on: '{state['message']}'"
    print(f"[step_1]   produced draft: '{draft}'")
    print("[step_1] ◀ RETURNING update → draft")
    return {"draft": draft}


def human_review(state: MyState) -> Command[Literal["step_2", END]]:
    print("\n[human_review] ▶ ENTERED")
    print(f"[human_review]   state so far: {dict(state)}")

    # ── THE PAUSE POINT ──────────────────────────────────────────────────────
    # Everything ABOVE this line re-runs when the graph resumes.
    # Everything BELOW this line runs ONLY after the human responds.
    # ─────────────────────────────────────────────────────────────────────────
    print("[human_review]   calling interrupt() → graph will FREEZE here")

    human_decision = interrupt({
        "question":       "Do you approve this draft?",
        "draft_to_review": state["draft"],
    })

    # ── RESUMES HERE after Command(resume=...) is sent ───────────────────────
    print(f"\n[human_review] ▶ RESUMED — human sent: {human_decision}")

    if human_decision.get("approved"):
        edited = human_decision.get("edited_text", state["draft"])
        print(f"[human_review]   Approved! final text: '{edited}'")
        print("[human_review] ◀ ROUTING → step_2")
        return Command(update={"draft": edited}, goto="step_2")
    else:
        print("[human_review]   Rejected. Ending graph.")
        print("[human_review] ◀ ROUTING → END")
        return Command(update={}, goto=END)


def step_2(state: MyState) -> dict:
    print("\n[step_2] ▶ ENTERED")
    print(f"[step_2]   state so far: {dict(state)}")

    final = f"[SENT] {state['draft']}"
    print(f"[step_2]   produced final: '{final}'")
    print("[step_2] ◀ RETURNING update → final")
    return {"final": final}


# ─────────────────────────────────────────
# 3. BUILD GRAPH
# ─────────────────────────────────────────
memory = MemorySaver()   # <-- Required: persists state across the pause

workflow = (
    StateGraph(MyState)
    .add_node("step_1",      step_1)
    .add_node("human_review", human_review)
    .add_node("step_2",      step_2)
    .add_edge(START,          "step_1")
    .add_edge("step_1",       "human_review")
    # step_2 → END is implicit when it returns a plain dict
    .add_edge("step_2",       END)
    .compile(checkpointer=memory)
)

# ─────────────────────────────────────────
# 4. RUN
# ─────────────────────────────────────────

# thread_id ties all runs of this "conversation" together.
# The checkpointer uses it to find the saved state on resume.
config = {"configurable": {"thread_id": "demo-thread-1"}}

print("=" * 60)
print("PHASE 1: First invocation — graph runs until interrupt()")
print("=" * 60)

initial_input = {
    "message": "Please handle my billing issue",
    "draft": "",
    "final": "",
}

# stream() drives execution and yields events.
# We just exhaust it here; what matters is what the nodes print.
for event in workflow.stream(initial_input, config):
    pass   # events are internal graph updates; we use our own prints

# After stream() returns, check what interrupted
state_snapshot = workflow.get_state(config)
print("\n──────────────────────────────────────")
print("GRAPH IS NOW PAUSED.")
print(f"  Pending interrupt payload: {state_snapshot.tasks[0].interrupts[0].value}")
print(f"  State after phase 1: {dict(state_snapshot.values)}")
print("──────────────────────────────────────")

# ── SCENARIO A: Human APPROVES (possibly with an edit) ────────────────────

print("\n" + "=" * 60)
print("PHASE 2: Human responds → APPROVE with an edit")
print("=" * 60)

resume_payload = Command(resume={
    "approved":    True,
    "edited_text": "Approved draft: We will refund you within 3 business days.",
})

for event in workflow.stream(resume_payload, config):
    pass

final_snapshot = workflow.get_state(config)
print("\n──────────────────────────────────────")
print("GRAPH FINISHED.")
print(f"  Final state: {dict(final_snapshot.values)}")
print("──────────────────────────────────────")

# ── SCENARIO B: run again on a NEW thread and REJECT ──────────────────────

print("\n" + "=" * 60)
print("BONUS — SCENARIO B: Human REJECTS on a fresh thread")
print("=" * 60)

config_b = {"configurable": {"thread_id": "demo-thread-2"}}

for event in workflow.stream(initial_input, config_b):
    pass

print("\n[main] Sending rejection...")
for event in workflow.stream(Command(resume={"approved": False}), config_b):
    pass

snap_b = workflow.get_state(config_b)
print(f"\n  Final state (rejected run): {dict(snap_b.values)}")
print("  (no 'final' key because step_2 was never reached)")