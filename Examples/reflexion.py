import json
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import ValidationError
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from typing import Literal
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from datetime import datetime

llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)

#tools
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)

# Responder. Used by initial response and revisiones

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing")
    superfluous: str = Field(description="Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    """Answer the question. Provide an answer, reflection, and then follow up with search queries to improve the answer."""
    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    queries: list[str] = Field(description=
                               "1-3 search queries for researching improvements to address the critique of your current answer.")

class ResponderWithReties:
    def __init__(self, runnable:Runnable, validator:PydanticToolsParser):
        self.runnable = runnable
        self.validator = validator


    def respond(self, state: dict):
        """invoke runnable to get the response. If response validation failed, add ToolMessage to the state and retry. Else return response """
        response = []
        for attempt in range(3):
            response = self.runnable.invoke(
                {"messages": state["messages"]}, {"tags": [f"attemp{attempt}"]}
            )
            try:
                self.validator.invoke(response)
                return {"messages": response}
            except ValidationError as e:
                # update state with ToolMessage
                state = state + [
                    response, #response is AIMessage. ToolMessage should have matching AIMessage, else get error
                    ToolMessage(
                        content=f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                        + self.validator.schema_json()
                        + " Respond by fixing all validation errors.",
                        tool_call_id = response.tool_calls[0]["id"]
                    )
                ]
        return {"messages": response}
    
# prompt template 

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """You are expert researcher.
Current time: {time}

1. {instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer."""),
MessagesPlaceholder(variable_name="messages"),
(
    "user", 
    "\n\n<reminder>Reflect on the user's original question and the"
    " actions taken thus far. Respond using the {function_name} .</reminder>",
)
    ]
).partial(time = lambda: datetime.now().isoformat())

initial_instructions = "Provide a detailed ~250 word answer."

# binding AnswerQuestion as a tool ensures that the output will be stored in AIMessage -> tool_calls["args"] as a Json string(schema AnswerQuestion)
# then validator wiil try to parse the output to validate the fields of the schema
initial_answer_chain = actor_prompt_template.partial(
    instruction = initial_instructions,
    function_name = AnswerQuestion.__name__
) | llm.bind_tools(tools= [AnswerQuestion])

validator = PydanticToolsParser(tools=[AnswerQuestion])

initial_responder = ResponderWithReties(runnable=initial_answer_chain, validator=validator)

# ----------------------------------------------------------------

# The second part of the actor is a revision step.

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

class RevisionQuestion(AnswerQuestion):
    """Revise your original answer to your question. Provide an answer, reflection,

    cite your reflection with references, and finally
    add search queries to improve the answer."""

    references: list[str] = Field(description="Citations motivating your updated answer.")

revision_chain = actor_prompt_template.partial(
    instruction = revise_instructions,
    function_name = RevisionQuestion.__name__
) | llm.bind_tools(tools=[RevisionQuestion])

revision_validator = PydanticToolsParser(tools=[RevisionQuestion])
revisor = ResponderWithReties(runnable=revision_chain, validator=revision_validator)

# -----------------------------------------------------------------
# Tool Node

def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])

# when ToolNode is invoked, it decides which tool to execute based on the tool_calls[0]["name"]
tool_node = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__), 
        StructuredTool.from_function(run_queries, name=RevisionQuestion.__name__),
    ]
)

class State(TypedDict) : 
    messages: Annotated[list, add_messages]

MAX_ITERATIONS = 5
builder = StateGraph(State)

builder.add_node("draft", initial_responder.respond)
builder.add_node("revise", revisor.respond)
builder.add_node("tools", tool_node)

def _get_num_iterations(state: list):
    i = 0
    for m in state[::-1]:
        if m.type not in {"tool", "ai"}:
            break
        i += 1
    return i


def event_loop(state: State):
    # in our case, we'll just stop after N plans
    num_iterations = _get_num_iterations(state["messages"])
    if num_iterations > MAX_ITERATIONS:
        return END
    return "tools"

builder.add_edge(START, "draft")
builder.add_edge("tools", "revisor")

builder.add_conditional_edges("draft", event_loop ,["tools", END])
builder.add_conditional_edges("revisor", event_loop, ["tools", END])

graph = builder.compile()

events = graph.stream(
    {"messages": [("user", "How should we handle the climate crisis?")]},
    stream_mode="values",
)

for i, step in enumerate(events):
    print(f"Step {i}")
    step["messages"][-1].pretty_print()


# example of messages in the state after initial_responder. Used as input to revisor
# example_question = "Why is reflection useful in AI?" 
# initial = initial_responder.respond(
#     {"messages": [HumanMessage(content=example_question)]}
# )

# revised = revisor.respond(
#     {
#         "messages": [
#             HumanMessage(content=example_question), # HummanMessage
#             initial["messages"], # AIMessage from initial_answer_chain with tool call
#             ToolMessage( # ToolMessage from tool_node
#                 tool_call_id=initial["messages"].tool_calls[0]["id"],
#                 content=json.dumps(
#                     tavily_tool.invoke(
#                         {
#                             "query": initial["messages"].tool_calls[0]["args"][
#                                 "search_queries"
#                             ][0]
#                         }
#                     )
#                 ),
#             ),
#         ]
#     }
# )
# revised["messages"] # AIMessage from revisor appended to the messages
