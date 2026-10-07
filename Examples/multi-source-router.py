"""
Multi-Source Knowledge Router Example

This example demonstrates the router pattern for multi-agent systems.
A router classifies queries, routes them to specialized agents in parallel,
and synthesizes results into a combined response.
"""

import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel, Field

# State definitions
class AgentInput(TypedDict):
    """Simple input state for each subagent."""
    query: str


class AgentOutput(TypedDict):
    """Output from each subagent."""
    source: str
    result: str


class Classification(TypedDict):
    """A single routing decision: which agent to call with what query."""
    source: Literal["github", "notion", "slack"]
    query: str


class RouterState(TypedDict):
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]
    final_answer: str


# Structured output schema for classifier
class ClassificationResult(BaseModel):
    """Result of classifying a user query into agent-specific sub-questions."""
    classifications: list[Classification] = Field(
        description="List of agents to invoke with their targeted sub-questions"
    )

# Tools
@tool
def search_code(query: str) -> str:
    """Search GitHub code repositories for relevant code snippets."""
    # Placeholder implementation
    return f"Searching GitHub code for: {query}"

@tool
def search_issues(query: str) -> str:
    """Search GitHub issues and pull requests."""
    return f"Found 3 issues matching '{query}': #142 (API auth docs), #89 (OAuth flow), #203 (token refresh)"


@tool
def search_prs(query: str) -> str:
    """Search pull requests for implementation details."""
    return f"PR #156 added JWT authentication, PR #178 updated OAuth scopes"

@tool
def search_notion(query: str) -> str:
    """Search Notion workspace for documentation."""
    return f"Found documentation: 'API Authentication Guide' - covers OAuth2 flow, API keys, and JWT tokens"


@tool
def get_page(page_id: str) -> str:
    """Get a specific Notion page by ID."""
    return f"Page content: Step-by-step authentication setup instructions"


@tool
def search_slack(query: str) -> str:
    """Search Slack messages and threads."""
    return f"Found discussion in #engineering: 'Use Bearer tokens for API auth, see docs for refresh flow'"


@tool
def get_thread(thread_id: str) -> str:
    """Get a specific Slack thread."""
    return f"Thread discusses best practices for API key rotation"

# Models and agents
model = init_chat_model("openai:gpt-5.5")
router_llm = init_chat_model("openai:gpt-5.4-mini")

github_agent = create_agent(
    model=model,
    tools=[search_code, search_issues, search_prs],
    system_prompt=(
        "You are a GitHub expert. Answer questions about code, "
        "API references, and implementation details by searching "
        "repositories, issues, and pull requests."
        "IMPORTANT: Never ask for clarification or additional input — use your tools "
        "to find the best available answer with what you have."
    )
)

notion_agent = create_agent(
    model=model,
    tools=[search_notion, get_page],
    system_prompt=(
        "You are a Notion expert. Answer questions about internal "
        "processes, policies, and team documentation by searching "
        "the organization's Notion workspace."
        "IMPORTANT: Never ask for clarification or additional input — use your tools "
        "to find the best available answer with what you have."
    )
)

slack_agent = create_agent(
    model=model,
    tools=[search_slack, get_thread],
    system_prompt=(
        "You are a Slack expert. Answer questions by searching "
        "relevant threads and discussions where team members have "
        "shared knowledge and solutions."
        "IMPORTANT: Never ask for clarification or additional input — use your tools "
        "to find the best available answer with what you have."
    )
)

# Workflow nodes
def classify_query(state: RouterState) -> dict:
    """Classify query and determine which agents to invoke."""
    structured_llm = router_llm.with_structured_output(ClassificationResult)

    result = structured_llm.invoke(
        [
            {
                "role": "system",
                "content": """Analyze this query and determine which knowledge bases to consult.
            For each source, generate a SHORT, keyword-focused sub-query suitable for 
            direct use as a search term (e.g. "API authentication headers", not an 
            open-ended instruction). Do NOT use imperative language or ask the agent 
            to "identify" or "please provide" anything.

            Available sources:
            - github: Code, API references, implementation details, issues, pull requests
            - notion: Internal documentation, processes, policies, team wikis
            - slack: Team discussions, informal knowledge sharing, recent conversations

            Return ONLY the sources that are relevant to the query."""
            },
            {
                "role": "user", "content": f"Classify this query: {state['query']}"
            }
        ]
    )

    # update the state with the classification results
    return {"classifications": result.classifications}

def route_to_agent(state: RouterState) -> list[Send]:
    """Fan out to agents based on classifications."""
    return [Send(c["source"], {"query": c["query"]}) #{"query": c["query"]} -> AgentInput
            for c in state["classifications"]
            ]

#invoke the agent, agent will invole the tools, the tools will return the results to the agent, and agent will update the RouterState.reslts
def query_github(state: AgentInput) -> dict: #partial update of the RouterState
    """Query the GitHub agent."""
    result = github_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    return {"results": [{"source": "github", "result": result["messages"][-1].content}]} # RouterState.reslts : Annotated[list[AgentOutput], operator.add]

def query_notion(state: AgentInput) -> dict:
    """Query the Notion agent."""
    result = notion_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    return {"results": [{"source": "notion", "result": result["messages"][-1].content}]} 

def query_slack(state: AgentInput) -> dict:
    """Query the Slack agent."""
    result = slack_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    return {"results": [{"source": "slack", "result": result["messages"][-1].content}]}

def synthesize_results(state: RouterState) -> dict:
    """Combine results from all agents into a coherent answer."""
    if not state["results"]:
        return {"final_answer": "No results found from any knowledge source."}

    formatted = [
        f"**From {r['source'].title()}:**\n{r['result']}"
        for r in state["results"]
    ]

    synthesis_response = router_llm.invoke([
        {
            "role": "system",
            "content": f"""Synthesize these search results to answer the original question: "{state['query']}"

- Combine information from multiple sources without redundancy
- Highlight the most relevant and actionable information
- Note any discrepancies between sources
- Keep the response concise and well-organized"""
        },
        {"role": "user", "content": "\n\n".join(formatted)}
    ])

    return {"final_answer": synthesis_response.content}

# Build workflow
builder = StateGraph(RouterState)
builder.add_node("classify", classify_query)
builder.add_node("github", query_github)
builder.add_node("notion", query_notion)
builder.add_node("slack", query_slack)
builder.add_node("synthesize", synthesize_results)
builder.add_conditional_edges("classify", route_to_agent, ["github", "notion", "slack"])
builder.add_edge("github", "synthesize")
builder.add_edge("notion", "synthesize")
builder.add_edge("slack", "synthesize")
builder.add_edge("synthesize", END)
builder.add_edge(START, "classify")

graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({
        "query": "How do I authenticate API requests?"
    })

    print("Original query:", result["query"])
    print("\nClassifications:")
    for c in result["classifications"]:
        print(f"  {c['source']}: {c['query']}")
    print("\n" + "=" * 60 + "\n")
    print("Final Answer:")
    print(result["final_answer"])