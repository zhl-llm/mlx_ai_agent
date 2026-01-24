import sys

from tool_registry import get_all_tools
from agent_builder import build_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def run_agent(user_query):
    agent = build_agent()
    tools = get_all_tools()
    tools_info = "\n".join(f"- {t.name}: {t.description}" for t in tools)

    system_prompt = (
        "You are a helpful assistant and ReAct agent. "
        "You can use the following tools when needed:\n"
        f"{tools_info}\n"
        "Answer questions carefully. If a tool is appropriate, indicate which one to use."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]

    state = {
        "messages": messages,
        "scratchpad": "",
        "action": None,
        "action_input": None,
    }
    result = agent.invoke(state)

    # Return final AI answer
    last_ai = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
    if last_ai:
        return last_ai.content
    return "Sorry, I couldn't find an answer."

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "Find the latest price of Tesla stock."

    result = run_agent(user_query)
    print(result)

