from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import Union, Literal, Annotated, List, Tuple
from langgraph.graph import END, START, StateGraph
import operator
from typing_extensions import TypedDict
import asyncio

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --------------    
# execute the step from the plan
# --------------    
tools = [TavilySearchResults(max_results=3,search_depth="basic")]
agent_executor = create_react_agent(
    model=model,
    tools=tools,
    prompt="You are a helpful assistant."
)

# --------------    
# Create the initila plan steps
# --------------    
class Plan(BaseModel):
    """Plan to follow in future"""
    steps: list[str] = Field(description="A list of steps to follow, should be in sorted order")

planner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """For the given objective, come up with a simple step by step plan.
            This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps.
            Make sure that each step has all the information needed - do not skip steps.
            if step {{i}} is dependent on step {{j}}, make sure to include step {{j}} before step {{i}} in the plan.
            The result of the final step should be the final answer. """
        ),
        ("placeholder", "{messages}"),
    ]
)
model_with_structured_output = model.with_structured_output(Plan)
planner = planner_prompt | model_with_structured_output

# plan =planner.invoke(
#     {
#         "messages": ["user", "what is the hometown of the current Australia open winner??"]
#     }
# )
# print(plan.steps)

# Re-Plan th plan. receiving the state from the agent_executor and decides if need to replan or not.
class Response(BaseModel):
    response: str = Field(description="Response to user.")

class Act(BaseModel):
    action: Union[Response, Plan] = Field(description="Action to take, If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan.")

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. 
Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Act)


# shared state between nodes
class PlanExecute(TypedDict):
    input: str # The input to the plan
    plan: List[str] # The plan to execute
    past_steps: Annotated[List[Tuple], operator.add] # The steps that have been executed (step, response) pairs)
    response: str # The response to the user

#----------------------
#  define graph nodes
#----------------------
async def execute_step(state: PlanExecute):
    """Execute the next step in the plan. Store the (step, result) pair in past_steps."""
    plan = state["plan"]
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    step = plan[0]
    task = f"""For the following plan:
    {plan_str}
    
    You are tasked with executing step {1}, {step}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task)]}
    )
    return {
        "past_steps": [(step, agent_response["messages"][-1].content)],
    }

async def initial_plan(state: PlanExecute):
    plan = await planner.ainvoke(
        {
            "messages": [("user", state["input"])]
        }
    )
    return {
        "plan": plan.steps,
    }

async def replan(state: PlanExecute):
    act = await replanner.ainvoke(
        {
            "input": state["input"],
            "plan": state["plan"],
            "past_steps": state["past_steps"],
        }
    )
    if isinstance(act.action, Response):
        return {
            "response": act.action.response,
        }
    elif isinstance(act.action, Plan):
        return {
            "plan": act.action.steps,
            "response": None,
        }

def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    return "agent"

workflow = StateGraph(PlanExecute)
workflow.add_node("initial_plan", initial_plan)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan)

workflow.add_edge(START, "initial_plan")
workflow.add_edge("initial_plan", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges("replan", should_end, ["agent", END])

graph = workflow.compile()


config = {"recursion_limit": 50}
input = {"input": "what is the hometown of the mens 2024 Australia open winner?"}

async def main():
    async for event in graph.astream(input, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)


asyncio.run(main())