from langchain.messages import RemoveMessage
from langchain_ollama import ChatOllama
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model, SummarizationMiddleware
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Middleware to trim messages before sending to the model."""
    messages = state.get("messages", [])
    if len(messages) <= 3:
        return None  # No trimming needed
    
    first_message = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_message] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

@after_model
def delete_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None

model = ChatOllama(model="llama3.1", temperature=0.7, max_tokens=300, keep_alive="30m")

# agent = create_agent(
#     model=model,
#     middleware=[delete_messages],
#     checkpointer=InMemorySaver()
# )

config=RunnableConfig(configurable={"thread_id": "1"})

# agent.invoke({"messages": "hi, my name is bob"}, config)
# agent.invoke({"messages": "write a short sentence about cats"}, config)
# agent.invoke({"messages": "now do the same but for dogs"}, config)
# agent.invoke({"messages": "What's the capital of France?"}, config)
# final_response = agent.invoke({"messages": "what's my name?"}, config)

# final_response["messages"][-1].pretty_print()

# example of messages deletion after the model call
# agent = create_agent(
#     model=model,
#     system_prompt="Please be concise and to the point.",
#     middleware=[delete_messages],
#     checkpointer=InMemorySaver()
# )

# stream = agent.stream_events(
#     {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
#     config,
#     version="v3",
# )
# for snapshot in stream.values:
#     print([(message.type, message.content) for message in snapshot["messages"]])

# stream = agent.stream_events(
#     {"messages": [{"role": "user", "content": "write a short poem about cats"}]},
#     config,
#     version="v3",
# )
# for snapshot in stream.values:
#     print([(message.type, message.content) for message in snapshot["messages"]])

# stream = agent.stream_events(
#     {"messages": [{"role": "user", "content": "what's my name?"}]},
#     config,
#     version="v3",
# )
# for snapshot in stream.values:
#     print([(message.type, message.content) for message in snapshot["messages"]])


# Summarization middleware example

agent = create_agent(
    model=model,
    middleware=[SummarizationMiddleware(
        model=model,
        trigger=("messages", 3),
        keep=("messages", 2)
    )],
    checkpointer=InMemorySaver()
)

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

for msg in final_response["messages"]:
    msg.pretty_print()