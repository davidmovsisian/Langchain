import json
from pathlib import Path

from langchain_core.utils.uuid import uuid7
from typing import TypedDict, NotRequired
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing import Callable
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import AgentState
from langgraph.types import Command  
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage  

class Skill(TypedDict):
    name: str
    description: str
    content: str

class SkillState(AgentState):
    """State to track the currently loaded skill in the agent's context."""
    skills_loaded: NotRequired[list[str]]  # List of skill names that have been loaded into the context

@tool
def write_sql_query(
    query: str,
    vertical: str,
    runtime: ToolRuntime,
) -> str:
    """Write and validate a SQL query for a specific business vertical.

    This tool helps format and validate SQL queries. You must load the
    appropriate skill first to understand the database schema.

    Args:
        query: The SQL query to write
        vertical: The business vertical (sales_analytics or inventory_management)
    """
    # Check if the required skill has been loaded
    skills_loaded = runtime.state.get("skills_loaded", [])

    if vertical not in skills_loaded:
        return (
            f"Error: You must load the '{vertical}' skill first "
            f"to understand the database schema before writing queries. "
            f"Use load_skill('{vertical}') to load the schema."
        )

    # Validate and format the query
    return (
        f"SQL Query for {vertical}:\n\n"
        f"```sql\n{query}\n```\n\n"
        f"✓ Query validated against {vertical} schema\n"
        f"Ready to execute against the database."
    )

def make_skill_tool(skill: Skill):
    """
    Build a dedicated load_skill_<name> tool for a single skill.
    The skill content is captured in the closure — no file I/O at call time.
    """
    skill_name = skill["name"]
    skill_content = skill["content"]

    # The docstring is what the LLM sees — make it specific to this skill
    @tool
    def skill_tool(runtime: ToolRuntime) -> Command:
        """Placeholder — replaced below."""
        content = f"Loaded skill: {skill_name}\n\n{skill_content}"
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
                "skills_loaded": [skill_name],
            }
        )

    # Set dynamic docstring and name after decoration
    skill_tool.__doc__ = (
        f"Load the full schema and business logic for the {skill_name} skill.\n\n"
        f"Call this before writing any queries or analysis related to {skill_name}.\n"
        f"It provides the complete table definitions, business rules, and example queries."
    )
    # Give the tool a unique name so the LLM can distinguish them
    skill_tool.name = f"{skill_name}"
    return skill_tool

class SkillMiddleware(AgentMiddleware):
    """Middleware to load skill content into the agent's context based on the skill name."""

    state_schema = SkillState
    
    # Register the load_skill tool as a class variable
    # tools =[]

    def __init__(self, skills_dir: Path):
        self.skilld_dir = skills_dir
        self.skills : list[Skill] = self._discover_skills()

        # Build skills prompt from the SKILLS list
        self.skills_prompt = "\n".join([f"- {skill['name']}: {skill['description']}" for skill in self.skills])
        self.tools = [write_sql_query, *[make_skill_tool(s) for s in self.skills]]

    # def get_tools(self):
    #     """Called by the framework when binding tools to the agent."""
    #     return self.tools
    
    def _discover_skills(self) -> list[Skill]:
        """Discover skills in the skills directory."""
        discovered = []

        for skill_dir in sorted(self.skilld_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            manifest_path = skill_dir / "skill.json"
            if not manifest_path.exists():
                continue

            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                content_path = skill_dir / manifest["content_file"]  # e.g. "SKILL.md"
                content = content_path.read_text(encoding="utf-8")
                discovered.append(Skill(
                    name=manifest["name"],
                    description=manifest["description"],
                    content=content,
                ))
                print(f"[SkillMiddleware] Loaded skill: {manifest['name']}")
            except Exception as e:
                print(f"[SkillMiddleware] Skipping {skill_dir.name}: {e}")
            
        return discovered
    
    #called before the model call
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Sync: Inject skill descriptions into system prompt."""
        skills_addendum = (
            f"\n\n## Available Skills\n\n{self.skills_prompt}\n\n"
            "Use the load_skill tool when you need detailed information "
            "about handling a specific type of request."
        )

        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]

        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)

        return handler(modified_request)

agent = create_agent(
    model=ChatOpenAI(model="gpt-5.5"),
    system_prompt=(
    "You are a SQL query assistant that helps users "
    "write queries against business databases."
    ),
    middleware=[SkillMiddleware(skills_dir=Path(__file__).parent / "SQL Skills")],
    checkpointer=InMemorySaver(),
)

if __name__ == "__main__":
    # Configuration for this conversation thread
    thread_id = str(uuid7())
    config = {"configurable": {"thread_id": thread_id}}

    # Ask for a SQL query
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a SQL query to find all customers "
                        "who made orders over $1000 in the last month"
                    ),
                }
            ]
        },
        config
    )

    # Print the conversation
    for message in result["messages"]:
        if hasattr(message, 'pretty_print'):
            message.pretty_print()
        else:
            print(f"{message.type}: {message.content}")

    # result = agent.invoke(
    #         {
    #             "messages": [
    #                 {
    #                     "role": "user",
    #                     "content": (
    #                         "Find all products that moved in from the warehouse to the store in the last week."
    #                         "Return the product name, warehouse name, quantity, date of transfer."
    #                     ),
    #                 }
    #             ]
    #         },
    #         config
    #     )

    #     # Print the conversation
        
    # for message in result["messages"]:
    #     if hasattr(message, 'pretty_print'):
    #         message.pretty_print()
    #     else:
    #         print(f"{message.type}: {message.content}")