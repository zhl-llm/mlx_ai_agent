import datetime
from typing import TypedDict, List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from my_llm import MyChatLLM
from tool_registry import get_all_tools

# ---------------------------
#  ReAct Prompt Template
# ---------------------------
REACT_PROMPT = """You are a helpful assistant that can use tools to answer questions.
The current date is {current_date}. When a query involves a date, you must compare it to the current date to determine if it is a past, present, or future date.
You have access to the following tools:
{tools}

To answer the question, you must use the following format:

Thought: Your reasoning and plan to solve the problem.
Action: The name of the tool to use (from the list above) or "Final Answer" if you have the answer.
Action Input: The input to the tool.

You will then receive an "Observation" with the result of the tool call. You should use this observation to continue your reasoning.

Repeat the Thought/Action/Action Input/Observation cycle until you have the final answer.

When you have the final answer, use the "Final Answer" action.

Begin!

Question: {question}

{scratchpad}"""
# ---------------------------
#  Correct LangGraph State Schema
# ---------------------------
class AgentState(TypedDict):
    messages: List[AIMessage | HumanMessage]
    scratchpad: str
    action: Optional[str]
    action_input: Optional[str]

# ---------------------------
#  Parser for Thought/Action
# ---------------------------
def parse_react(text: str):
    """
    Extract Action and Action Input or Final Answer from LLM output.
    """
    if "Final Answer:" in text:
        return "Final Answer", text.split("Final Answer:")[1].strip()

    action = None
    action_input = None

    import re
    action_match = re.search(r"Action:\s*(.*)", text)
    if not action_match:
        action_match = re.search(r"Action\n(.*)", text)

    if action_match:
        action = action_match.group(1).strip()

    action_input_match = re.search(r"Action Input:\s*(.*)", text, re.DOTALL)
    if action_input_match:
        action_input = action_input_match.group(1).strip()
    return action, action_input

# ---------------------------
#  Agent builder
# ---------------------------
def build_agent():
    llm = MyChatLLM()
    tools = get_all_tools()
    tool_map = {t.name: t for t in tools}
    tools_text = "\n".join(f"{t.name}: {t.description}" for t in tools)
    prompt = ChatPromptTemplate.from_template(REACT_PROMPT)

    # ---------------------------
    #  Node: Planner (LLM)
    # ---------------------------
    def planner(state: AgentState):
        user_question = next(
            (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
        )
        scratch = state["scratchpad"]
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        prompt_msgs = prompt.format_messages(
            tools=tools_text,
            question=user_question,
            scratchpad=scratch,
            current_date=current_date,
        )
        # print("=== [DEBUG] Planner prompt ===")
        # print(prompt_msgs)
        # print("================================")
        llm_msg = llm.invoke(prompt_msgs)
        raw_content = llm_msg.content
        if isinstance(raw_content, dict) and "parts" in raw_content:
            content_text = "".join(p.get("text", "") for p in raw_content["parts"])
        else:
            content_text = str(raw_content)
        # print("=== [DEBUG] Planner content ===")
        # print(content_text)
        # print("================================")
        new_scratchpad = scratch + "\n" + content_text + "\n"
        action, action_input = parse_react(content_text)
        return {
            "messages": state["messages"] + [llm_msg],
            "scratchpad": new_scratchpad,
            "action": action,
            "action_input": action_input,
        }

    # ---------------------------
    #  Node: Tool Execution
    # ---------------------------
    def tool_executor(state: AgentState):
        action = state["action"]
        tool = tool_map.get(action)
        inp = state["action_input"]
        print(f"=== [DEBUG] Tool executor called: {action=} {inp=}")
        if not tool:
            obs_msg = AIMessage(content=f"Observation: Unknown tool {action}")
        else:
            result = tool.invoke(inp)
            obs_msg = AIMessage(content=f"Observation: {result}")
        return {
            "messages": state["messages"] + [obs_msg],
            "scratchpad": state["scratchpad"] + f"\nObservation: {result}\n",
            "action": None,
            "action_input": None,
        }

    # ---------------------------
    #  LangGraph Build
    # ---------------------------
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("tool", tool_executor)
    graph.set_entry_point("planner")

    def router(state: AgentState):
        print(f"=== [DEBUG] Router called: action={state['action']}")
        if state["action"] == "Final Answer":
            return END
        if state["action"] in tool_map:
            return "tool"
        return END
    
    graph.add_conditional_edges(
        "planner",
        router,
        {
            "tool": "tool",
            END: END,
            "planner": "planner",
        }
    )
    graph.add_edge("tool", "planner")
    return graph.compile()

