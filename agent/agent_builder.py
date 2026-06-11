import datetime
import json
import logging
import re

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, TypedDict

from my_llm import MyChatLLM
from tool_registry import get_all_tools

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """You are a ReAct-style planning agent.

Your job is to decide the NEXT action only.
You must either:
- call exactly ONE tool, or
- produce the FINAL answer.

You have access to the following tools:
{tools}

Current date: {current_date}

You must respond ONLY with a valid JSON object that follows the schema below.
Do not include explanations, markdown, or extra text.

{format_instructions}

Decision rules:
1. Tool usage
   - Use a tool ONLY if it is necessary to answer the question.
   - If a tool is required, set:
     - "action" = the exact tool name
     - "args" = the arguments for that tool
     - Do NOT include "final_answer"

2. Final answer
   - If no tool is needed, set:
     - "action" = "Final Answer"
     - "final_answer" = the complete answer to the user
     - Do NOT include "args"

3. Web search & URLs
   - If any tool returns URLs:
     a. Each URL MUST be fetched exactly once.
     b. You MUST analyze fetched content before answering.
     c. Reuse prior observations; never re-fetch the same URL.

4. Observations
   - The "Observation" section contains results from previous tool calls.
   - Base decisions ONLY on the user question and available observations.

User question:
{question}

Observation:
{observation}
"""


class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    plan: Optional[dict]
    observation: Optional[str]


class Plan(BaseModel):
    action: str = Field(
        description="Tool name to call, or 'Final Answer'",
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool",
    )
    final_answer: str | None = Field(
        default=None,
        description="Final answer to the user, if action is Final Answer",
    )


def extract_first_json(text: str) -> dict:
    """Load the first JSON object found in model output."""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON object found in LLM output:\n{text}")

    return json.loads(match.group())


def first_user_question(messages: List[HumanMessage | AIMessage]) -> str:
    """Return the first user message in the conversation."""
    return next(message.content for message in messages if isinstance(message, HumanMessage))


def parse_plan(raw_output: str, parser: JsonOutputParser) -> dict:
    """Parse planner output, falling back to JSON extraction for noisy model responses."""
    try:
        return parser.parse(raw_output)
    except Exception:
        logger.warning("Falling back to first JSON object extraction for planner output.")
        return extract_first_json(raw_output)


def build_agent():
    llm = MyChatLLM()
    tools = get_all_tools()
    tool_map = {t.name: t for t in tools}
    tools_text = "\n".join(f"{t.name}: {t.description}" for t in tools)
    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
    parser = JsonOutputParser(pydantic_object=Plan)

    def planner(state: AgentState):
        msgs = prompt.format_messages(
            tools=tools_text,
            question=first_user_question(state["messages"]),
            current_date=datetime.datetime.now().strftime("%Y-%m-%d"),
            observation=state.get("observation"),
            format_instructions=parser.get_format_instructions(),
        )
        response = llm.invoke(msgs)
        plan = parse_plan(response.content, parser)

        return {
            "messages": state["messages"] + [response],
            "plan": plan,
            "observation": None,
        }

    def tool_executor(state: AgentState):
        plan = state["plan"]
        action = plan["action"]
        args = plan.get("args", {})

        logger.debug("Executing tool %s with args=%s", action, args)

        tool = tool_map[action]
        result = tool.invoke(args)
        obs = f"Tool {action} result:\n{result}"

        return {
            "messages": state["messages"] + [AIMessage(content=obs)],
            "plan": None,
            "observation": obs,
        }

    def router(state: AgentState):
        action = state["plan"]["action"]

        if action == "Final Answer":
            return END

        if action in tool_map:
            return "tool"

        raise ValueError(f"Unknown action: {action}")

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("tool", tool_executor)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", router)
    graph.add_edge("tool", "planner")

    return graph.compile()
