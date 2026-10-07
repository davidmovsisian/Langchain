# agents/specialist.py
from langchain.agents import create_agent
from .budget_caping import budget_middleware, BudgetPolicy
from .common import (
    TravelState,
    sensitive_tools_middleware,
    complete_or_escalate,
    format_prompt_middleware,
    clear_old_search_results_middleware,
)


def build_specialist(
    name: str,
    tools: list,
    sensitive: list[str],
    prompt: str,
    policy: BudgetPolicy,
    *,
    parallel_tool_calls: bool = True,
):
    """Build a specialist sub-agent with the shared middleware stack.

    name       graph node / budget key, e.g. "hotel_agent"
    tools      domain tools (complete_or_escalate is added automatically)
    sensitive  names of tools that need human approval
    prompt     system prompt with {user_info}, {handoff} and {time} placeholders
    policy     budget policy for this node
    """
    return create_agent(
        # Required argument of create_agent. budget_middleware swaps in the
        # real primary/fallback model on every call, so this is only a placeholder.
        model=policy.primary_model,
        tools=[*tools, complete_or_escalate],
        state_schema=TravelState,
        name=name,
        middleware=[
            # Order matters: outermost first. Clear old results, then the prompt
            # is formatted, and the budget sees the final, cleared request.
            clear_old_search_results_middleware(sensitive),
            sensitive_tools_middleware(sensitive),  # interrupt on sensitive tools
            format_prompt_middleware(prompt),
            budget_middleware(policy, name, parallel_tool_calls=parallel_tool_calls),
        ],
    )