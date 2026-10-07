import os
from langchain.agents import create_agent
from deepagents import create_deep_agent
from deepagents.middleware import FilesystemMiddleware
from deepagents.backends import LocalShellBackend
from pathlib import Path


# Expose the real workspace folder to the agent instead of a virtual "/" root.
sandbox_backend = LocalShellBackend(root_dir=Path.cwd(), virtual_mode=False)

# Build the agent stack with the middleware
agent = create_agent(
    model="openai:gpt-4.1-mini",
    middleware=[FilesystemMiddleware(backend=sandbox_backend)],
)

# 4. Invoke the agent with a prompt that requires running a system `ls`
response = agent.invoke({
    "messages": [
        ("human", "Print the path of the current folder.")
    ]}
)

print(response["messages"][-1].content)