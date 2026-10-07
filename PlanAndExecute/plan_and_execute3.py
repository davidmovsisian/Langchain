from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, Base
from langchain_core.prompts import ChatPromptTemplate
from typing import Union, Annotated, List, Tuple
from langgraph.graph import END, START, StateGraph
import operator
from typing_extensions import TypedDict, Any
import asyncio
from dataclasses import dataclass, field
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool

# add tool regestry. At initial step after initial plan creation the toos for each step are retrived from the registry.
# tool name and description are stored in the RAG. description used for search, name is retrived. 
# if tool is not found use defualt tool

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Fallback tool used when no relevant tool is retrieved from the registry.
DEFAULT_TOOL = TavilySearchResults(max_results=3, search_depth="basic")

#custom tool
@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> float:
    """ Calculate the tip amount for a given restaurant bill.
        Args:
         bill_amount - bill amount for payment
         tip_percentage - recomended tip percentage
    """
    return round(bill_amount * (tip_percentage / 100), 2)

# --------------
# Tool registry
# --------------

@dataclass
class ToolMetaData:
    name: str,          # Unique tool identifier
    description: str,   # description used for embedding and retrieval
    tool: BaseTool      # The actual LangChain tool object

#In-memory tool registry
TOOL_REGISTRY : List[ToolMetaData] = [
    ToolMetaData(
        name="tavily_search",
        description="Search the web for current information, recent events, news, and facts using Tavily.",
        tool=TavilySearchResults(max_results=3, search_depth="basic"),
    ),
    ToolMetadata(
        name="wikipedia",
        description="Look up factual, encyclopedic information about people, places, history, science, and concepts using Wikipedia.",
        tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
    ),
    ToolMetadata(
        name="open_weather_map",
        description="Get current weather conditions and forecasts for any location using OpenWeatherMap.",
        tool=OpenWeatherMapQueryRun(api_wrapper=OpenWeatherMapAPIWrapper()),
    ),
    ToolMetadata(
        name="calculate_tip",
        description="Calculate the tip amount for a given restaurant bill.",
        tool=calculate_tip,
    ),
]

# Build the vector store from tool descriptions at startup.
# Each document's metadata carries the tool name for lookup after retrieval.

_tool_documents = [
    Document(page_content = t.description, metadata = {"name": t.name})
             for t in TOOL_REGISTRY
]

tool_vector_store = InMemoryVectorStore.from_documents(_tool_documents, embeddings)
_tool_by_name = {t.name: t for t in TOOL_REGISTRY}


# --------------
# Step and Plan models
# --------------
class Step(TypedDict):
    idx: int = Field(description="Step number (1-based, auto-incremented by the planner).")
    task: str = Field(description="The task to perform in this step.")
    dependencies: List[int] = Field(description="Indices of steps that must be resolved before this one.")
    retries: int = Field(description="Number of execution attempts made so far (starts at 0)")

class Plan(BaseModel):
    pending_steps: List[Step] = Field(description=(
        "All steps to complete, in sorted order. Each step has an idx (1-based), "
        "a task description, and a dependencies list of idx values that must be "
        "resolved before this step can run. Steps with no prerequisites have an "
        "empty dependencies list."
    ))
    resolved_steps: List[Step] = Field(
        default_factory=list,
        description="Steps that have already been completed. Always empty on initial plan creation.",
    )
    failed_steps: List[Step] = Field(
        default_factory=list,
        description="Steps that failed during execution. Always empty on initial plan creation."
    )

# --------------
# Planner
# --------------
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """For a given objective, come up witha simple step-by-step plan.

Each step must be represented as a STEP with four fields:
- idx: a unique 1-based integer assigned in the order the step appears.
- task: a clear, self-contained description of what to do in this step. Include all information needed — do not assume the executor can infer context from other steps.
- dependencies: a list of idx values of steps that MUST be completed before this step can run. If a step has no prerequisites, use an empty list [].
- retries: always set to 0 on creation.
 
Rules:
- Do not add superfluous steps.
- If step i depends on the result of step j, then j must appear in i's dependencies list.
- The result of the final step should be the final answer.
- On creation, place ALL steps in pending_steps and leave resolved_steps and failed_steps empty.
"""
        ),
        ("placeholder", "{messages}")
    ]
)
llm_with_plan = llm.with_structured_output(schema=Plan)
planner = planner_prompt | llm_with_plan

# plan =planner.invoke(
#     {
#         "messages": ["user", "Does temperature id SF higher then in LA?"]
#     }
# )
# print(plan.pending_steps)

class Response(BaseModel):
    response: str = Field(description="Final response to the user.")

class Act(BaseModel):
    action: Union[Response, Plan] = Field(
        description=
        "Action to take. Use Response when no more steps are needed and you can "
        "answer the user directly. Use Plan when further plan use is required."
    )

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, revise the plan based on what has already been done.
    
    Your objective was:
    {input}
    
    Current plan:
      Pending steps (still to be executed):
      {pending_steps}
    
      Resolved steps (already completed successfully):
      {resolved_steps}
    
      Failed steps (execution failed — error messages are in past_steps):
      {failed_steps}
    
    Steps already completed or attempted (each entry is a (task, result) pair; errors start with ERROR:):
    {past_steps}
    
    Instructions:
    - If no more steps are needed, respond with a Response containing the final answer.
    - Otherwise, respond with a revised Plan.
      - pending_steps must contain ONLY steps that still need to be done.
      - Do NOT repeat already-completed steps — they are listed in resolved_steps above.
      - For failed steps, decide whether to retry them (possibly with a different approach), rewrite them, or drop them along with any steps that depended on them.
    """
)
replanner = replanner_prompt | llm.with_structured_output(Act)

# --------------
# Shared graph state
# --------------
class PlanExecute(TypedDict):
    input: str,                                                 # The user's original question
    plan: Plan,                                                 # Current plan
    past_steps: Annotated[List[Tuple[str, str]], operator.add]  # (task, result) pairs accumulated across nodes
    response: str                                               # final response
    plan_executor: Any

async def initial_plan(state: PlanExecute):
    """
    Create the first plan and place all steps in pending_steps.
 
    After planning, retrieve relevant tools from the vector store for each step,
    union the results across all steps, and bind them to a single agent_executor
    that will be shared for the entire run.
 
    Falls back to DEFAULT_TOOL if no tools are retrieved for any step.
    """
    plan = await planner.ainvoke({
        "messages":[
            ("user", state["input"])
        ]
    })

    # Retrieve tools for each step and union by name to deduplicate.

    top_k = config.get("configurable", {}).get("tools_per_step", 2)
    retrieved_tool_names = set[str] = set()

    for step in plan.pending_steps:
        # serach for matching tool name in vector store
        docs = await tool_vector_store.asimilarity_search(step["task"], top_k)
        if docs:
            for doc in docs:
                retrieved_tool_names.add(doc.metadata["name"])
        else:
            retrieved_tool_names.add("tavily_search")

    # resolve tool names to tools
    retrieved_tools = [_tool_by_name[name].tool for name in retrieved_tool_names if name in _tool_by_name]

    executor = create_react_agent(
        model=llm,
        tools=retrieved_tools,
        prompt="You are a helpful assistant.")

    return {"plan": plan, "plan_executor": executor}

async def execute_step(state: PlanExecute, config: dict):
    """
    Execute all currently runnable steps (i.e., steps whose dependencies are
    all present in resolved_steps). Runs sequentially — no multithreading.

    Moves each completed step from pending_steps to resolved_steps and
    appends its (task, result) pair to past_steps.
    """

    plan: Plan = state["plan"]
    resolved_ids = {s["idx"] for s in plan.resolved_steps} #set of ids of resolved steps
    failed_ids = {s["idx"] for s in plan.failed_steps}
    newly_resolved: List[Step] = []
    newly_failed: List[Step] = []
    new_past_steps: List[Tuple[str, str]] = []

    # Keep iterating until no more steps become runnable in this pass.
    # This handles chains like 1 → 2 → 3 where resolving 1 unlocks 2, etc.
    # Steps whose dependencies include a failed idx are not runnable and stay
    # in pending_steps for the replanner to handle.
    made_progress = True
    pending = list(plan.pending_steps)

    while made_progress:
        made_progress = False
        still_pending: List[Step] = []
        for step in pending:
            if all(dep in resolved_ids for dep in step["dependencies"]):
                # All dependencies satisfied — execute this step.
                max_retries = config.get("configurable", {}).get("max_retries",3)

                # Build a lookup from task string → result/error for all past attempts.
                past_steps_map = {task: result for task, result in state["past_steps"] + new_past_steps}

                # Inject results from dependency steps so the agent can use them
                dep_context = ""
                if step["dependencies"]:
                    dep_lines = []
                    for resolved in plan.resolved_steps + newly_resolved:
                        if resolved["idx"] in step["dependencies"]:
                            dep_result = past_steps_map.get(resolved["task"], "No result available")
                            dep_lines.append(f"Step {resolved['idx']} ({resolved['task']}):\n{dep_result}")
                    if dep_lines:
                        dep_context = "\n\nResults from dependency steps:\n" + "\n\n".join(dep_lines)

                # Inject previous error messages for this step so the agent
                # can try a different approach on each retry.
                all_past = state["past_steps"] + new_past_steps
                prior_errors = 
                [
                    result for task, result in all_past
                    if task == step["task"] and result.startswith("ERROR:")
                ]

                error_context = ""
                if prior_errors:
                    error_lines = "\n".join(
                        f"- Attempt {i + 1}: {err}" for i, err in enumerate(prior_errors)
                    )
                    error_context = f"\n\nPrevious attempts for this step failed:\n{error_lines}\nPlease try a different approach."

                task_prompt = (
                    f"You are tasked with executing the following step: {step['task']}"
                    f"{dep_context}"
                    f"{error_context}"
                )

                try:
                    agent_response = await state["plan_executor"].ainvoke(
                        {
                            "messages": [("user", task_prompt)]
                        }
                    )
                    result = agent_response["messages"][-1].content
                    resolved_ids.add(step["idx"])
                    newly_resolved.append({**step}) #create shallow copy of step to avoid accident sharing
                    new_past_steps.append((step["task"], result))
                except  Exception as e:
                    updated_step = {**step, "retries": step["retries"] + 1}
                    error_msg = f"ERROR: {type(e).__name__}: {e}"
                    if updated_step["retries"] >= max_retries:
                        # Exhausted all retries — move to failed_steps.
                        failed_ids.add(step["idx"])
                        newly_failed.append(updated_step)
                        new_past_steps.append((
                            step["task"],
                            f"FAILED after {max_retries} retries. Last error: {error_msg}",
                        ))
                    else:
                        # Still has retries remaining — keep in pending with incremented count.
                        still_pending.append(updated_step)

                made_progress = True
            else:
                still_pending.append(step)

        pending = still_pending

    updated_plan = new Plan(
        pending_steps=pending,
        resolved_steps=plan.resolved_steps + newly_resolved,
        failed_steps=plan.failed_steps + newly_failed
    )

    return {
        "plan": updated_plan,
        "past_steps": new_past_steps,
    }

async def replan(state: PlanExecute):
    """
    Inspect progress and either finish (return a Response) or emit a
    revised Plan with only the remaining pending_steps.
    """
    act = await replanner.ainvoke(
        {
            "input": state["input"],
            "pending_steps": state["plan"].pending_steps,
            "resolved_steps": state["plan"].resolved_steps,
            "failed_steps": state["plan"].failed_steps,
            "past_steps": state["past_steps"],
        }
    )

    if isinstance(act.action, Response):
            return {"response": act.action.response}

    # Replanner returned a new Plan — keep resolved_steps and failed_steps
        # from current state so history is not lost, and replace pending with
        # the revised list.
        new_plan = Plan(
            pending_steps=act.action.pending_steps,
            resolved_steps=state["plan"].resolved_steps,
            failed_steps=state["plan"].failed_steps,
        )
        return {"plan": new_plan, "response": None}

def should_end(state: PlanExecute) -> str:
    """Route to END when a final response exists, otherwise loop back to agent."""
    if state.get("response"):
        return END
    return "agent"


# --------------
# Build the graph
# --------------
workflow = StateGraph(PlanExecute)

workflow.add_node("initial_plan", initial_plan)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan)

workflow.add_edge(START, "initial_plan")
workflow.add_edge("initial_plan", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges("replan", should_end, ["agent", END])

graph = workflow.compile()


# --------------
# Entry point
# --------------
config = {"recursion_limit": 50, "configurable": {"max_retries": 3, "tools_per_step": 2}}
initial_input = {"input": "What is the hometown of the men's 2024 Australian Open winner?"}


async def main():
    async for event in graph.astream(initial_input, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)


if __name__ == "__main__":
    asyncio.run(main())
