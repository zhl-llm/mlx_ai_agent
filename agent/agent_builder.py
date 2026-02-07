import datetime
import json
import re

from pydantic import BaseModel, Field
from typing import Dict, Any, TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from my_llm import MyChatLLM
from tool_registry import get_all_tools

# ---------------------------
#  ReAct Prompt Template
# ---------------------------
PLANNER_PROMPT = """You are a planning agent.

You have access to these tools:
{tools}

The current date is {current_date}.

Decide the next step.

{format_instructions}

Rules:
1. Decide whether a tool is required.
   - If yes, set "action" to the tool name and provide "args".
   - If no, set "action" to "Final Answer" and provide "final_answer".

2. Web Search & URL Handling:
   - If web search returns one or more URLs, you MUST:
     a. Fetch the content of each URL.
     b. Analyze the fetched content before answering.
   - Each URL must be fetched and analyzed at most once.
   - Do NOT duplicate analysis for the same URL.

3. When answering:
   - Base conclusions only on fetched and analyzed content when URLs are involved.
   - Reuse previous analysis instead of re-fetching URLs.

User question:
{question}

Observation (if any):
{observation}

"""

# ---------------------------
#  Correct LangGraph State Schema
# ---------------------------
class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    plan: Optional[dict]
    observation: Optional[str]

class Plan(BaseModel):
    action: str = Field(
        description="Tool name to call, or 'Final Answer'"
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool"
    )
    final_answer: str | None = Field(
        default=None,
        description="Final answer to the user, if action is Final Answer"
    )

def extract_first_json(text: str) -> dict:
    """
    Safely find the first JSON object inside text
    and load it. Raises if no JSON found.
    """
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON object found in LLM output:\n{text}")

    return json.loads(match.group())

def build_agent():
    llm = MyChatLLM()
    tools = get_all_tools()
    tool_map = {t.name: t for t in tools}
    tools_text = "\n".join(f"{t.name}: {t.description}" for t in tools)

    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)

    # -------------------------
    # Planner Node
    # -------------------------
    def planner(state: AgentState):
        question = next(
            m.content for m in state["messages"]
            if isinstance(m, HumanMessage)
        )

        current_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Create the structured output parser
        parser = JsonOutputParser(pydantic_object=Plan)

        msgs = prompt.format_messages(
            tools=tools_text,
            question=question,
            current_date=current_date,
            observation=state.get("observation"),
            format_instructions=parser.get_format_instructions(),
        )

        response = llm.invoke(msgs)
        raw = response.content

        plan: dict

        try:
            # First try structured parser (strict)
            plan = parser.parse(raw)
        except Exception:
            # Fallback: extract first JSON object manually
            print("⚠️ Fallback JSON extraction triggered")
            plan = extract_first_json(raw)

        # Ensure plan is a dict
        if not isinstance(plan, dict):
            plan = plan.model_dump()  # For parsers returning Pydantic model

        return {
            "messages": state["messages"] + [response],
            "plan": plan,
            "observation": None,
        }

    # -------------------------
    # Tool Executor Node
    # -------------------------
    def tool_executor(state: AgentState):
        plan = state["plan"]
        action = plan["action"]
        args = plan.get("args", {})

        print(f"=== [DEBUG] Executing tool: {action} args={args}")

        tool = tool_map[action]
        result = tool.invoke(args)

        obs = f"Tool {action} result:\n{result}"

        print(f"=== [DEBUG] Tool observation:\n{obs}")

        return {
            "messages": state["messages"] + [AIMessage(content=obs)],
            "plan": None,
            "observation": obs,
        }

    # -------------------------
    # Router
    # -------------------------
    def router(state: AgentState):
        action = state["plan"]["action"]

        if action == "Final Answer":
            return END

        if action in tool_map:
            return "tool"

        raise ValueError(f"Unknown action: {action}")

    # -------------------------
    # Graph
    # -------------------------
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("tool", tool_executor)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", router)
    graph.add_edge("tool", "planner")

    return graph.compile()
