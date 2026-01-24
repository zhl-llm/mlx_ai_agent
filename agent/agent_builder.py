import datetime
import json
import re

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

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

Respond ONLY in valid JSON using this schema:

{{
  "action": "<tool_name | Final Answer>",
  "args": {{ ... }},
  "final_answer": "<string | null>"
}}

Rules:
- If you need a tool, set "action" to the tool name and fill "args"
- If you are done, set "action" to "Final Answer" and provide "final_answer"
- Do NOT include explanations outside JSON

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

def extract_json(text: str) -> dict:
    """
    Extract the first JSON object from text.
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

        msgs = prompt.format_messages(
            tools=tools_text,
            question=question,
            current_date=current_date,
            observation=state.get("observation"),
        )

        response = llm.invoke(msgs)
        content = response.content

        try:
            plan = extract_json(content)
        except Exception as e:
            raise ValueError(
                f"Planner output could not be parsed.\n\nRaw output:\n{content}"
            ) from e

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

