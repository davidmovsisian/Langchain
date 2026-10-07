from langchain_ollama import ChatOllama
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, ToolMessage, SystemMessage

@tool
def get_last_user_message(history: list) -> str:
    """
    Get the last user message from the conversation.
    """
    for msg in reversed(history):
        # Handle both dict messages and LangChain Message objects
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg["content"]
        elif isinstance(msg, HumanMessage):
            return msg.content
        
    return "No user message found."

tools_map = {"get_last_user_message": get_last_user_message}

model = ChatOllama(model="llama3.1", temperature=0.0, max_tokens=300, keep_alive="30m")
model_with_tools = model.bind_tools([get_last_user_message])

messages = [
    SystemMessage(content="You are a helpful assistant. You have access to tools to help answer questions. "
                        ), #"If a tool is needed, call it directly without explaining your reasoning."
                        HumanMessage(content="Translate to French: I love building applications.")
]

response = model.invoke(messages)
print(f"First Translation: {response.content}\n")
messages.append(response)

question = HumanMessage(content="What's the last user message?")
current_turn_messages = messages + [question]
response = model_with_tools.invoke(current_turn_messages)

if response.tool_calls:
    messages.append(response)
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = {"history": current_turn_messages}
        print(f"\n🤖 [Agent Decision]: The model decided to use the tool '{tool_name}'")

        selected_tool = tools_map.get(tool_name)
        tool_output = selected_tool.invoke(tool_args)
        # Feed the retrieved context back to the model
        messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))

    response = model_with_tools.invoke(messages)
    print(f"\n🤖 [Final Response]: {response.content}")
else:
    print(f"\n❌ Tool was not called. Model output text instead:")
    print(response.content)

