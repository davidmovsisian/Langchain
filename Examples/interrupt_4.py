from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt

class FormState(TypedDict):
    age: int | None
    pending_question: str | None

def get_age_node(state: FormState) :
    question = state.get("pending_question") or "What is your age?"
    answer = interrupt({"question": question})
    print(f"I got {answer}")
    if isinstance(answer, int) and answer >= 0:
        return {"age": answer, "pending_question": None}
    return {"age": None, "pending_question": f"'{answer}' is not a valid age. Please enter a positive number."}

def should_continue(state: FormState):
    if state.get("age") is None:
        return "get_age"
    return END

workflow = (
    StateGraph(FormState)
    .add_node("get_age", get_age_node)
    .add_conditional_edges("get_age", should_continue, ["get_age", END])
    .add_edge(START, "get_age")
    .compile(checkpointer=InMemorySaver()))

config = {"configurable": {"thread_id": "age-form"}}
first = workflow.stream_events({"age": None, "pending_question": None}, config, version="v3")
_ = first.output  # drive the stream to completion
print("First Interrupts:", first.interrupts)

#resume with invalid input
retry = workflow.stream_events(Command(resume="thirty"), config, version="v3")
_ = retry.output  # drive the stream to completion
print("Retry Interrupts:", retry.interrupts)

#resume with valid input
valid_retry = workflow.stream_events(Command(resume=25), config, version="v3")
_ = valid_retry.output  # drive the stream to completion
print("Valid Retry Interrupts:", valid_retry.interrupts)
print(valid_retry.output)  