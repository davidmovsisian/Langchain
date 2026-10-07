from dataclasses import dataclass
from langchain.agents import create_agent
from pydantic import BaseModel
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables.config import RunnableConfig
from langchain_core.utils.uuid import uuid7
from langchain_ollama import ChatOllama


USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com",
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com",
    },
}


@dataclass
class UserContext:
    user_id: str

class AccountInfo(BaseModel):
    account_holder: str
    type: str
    balance: float

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return (
            f"Account holder: {user['name']}\n"
            f"Type: {user['account_type']}\n"
            f"Balance: ${user['balance']}"
        )
    return "User not found"


model = ChatOllama(model="llama3.1", temperature=0.0, max_tokens=300, keep_alive="30m")

agent = create_agent(
    model,
    tools=[get_account_info],
    context_schema=UserContext,
    system_prompt="You are a financial assistant.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's Alice Johnson's current balance?"}]},
    config={"configurable": {"thread_id": str(uuid7())}},
    context=UserContext(user_id="user123"),
)

content = result["messages"][-1].content

structured_model = model.with_structured_output(AccountInfo)
structured_result = structured_model.invoke(f"Extract the account info from this text. if there is no strict match leave empty: {content}")

print(f"Structured result: {structured_result}")