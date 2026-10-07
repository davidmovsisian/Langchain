from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class UserState(AgentState):
    user_name: str

class UserContext(BaseModel):
    user_id: str

@tool
def update_user_info(runtime: ToolRuntime[UserContext, UserState]) -> Command:
    """Update the user's name in the agent's state.
        Args: None
    """
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(
        update={
            "user_name": name,
            "messages": [ToolMessage(content=f"User name updated to {name}.", tool_call_id=runtime.tool_call_id)]
        }
    )

@tool
def greet_user(runtime: ToolRuntime[UserState]) -> Command | str:
    """Greet the user based on their name in the agent's state.
        Args: None
    """
    user_name = runtime.state.get("user_name", None)
    if user_name is None:
        return Command(update={
            "messages": [
                ToolMessage(
                    content="SYSTEM: User name is missing. You MUST call the 'update_user_info' tool first to retrieve the name.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })
    return f"Hello, {user_name}! How can I assist you today?"

# model = ChatOllama(model="llama3.1", temperature=0.7, max_tokens=300, keep_alive="30m")
# model_with_tools = model.bind_tools([update_user_info, greet_user])

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o", parallel_tool_calls=False),
    tools=[update_user_info, greet_user],
    system_prompt="You are a helpful assistant. "
    "FIRST ensure their name is updated using 'update_user_info' "
    "Then greet the user.",
    state_schema=UserState,
    context_schema=UserContext
)

graph = agent #for LangGraph dev server to load the graph from this file

if __name__ == "__main__":
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "greet the user"}]},
        context=UserContext(user_id="user_123"),
        version="v3"
    )
    for msg in response["messages"]:
        msg.pretty_print()