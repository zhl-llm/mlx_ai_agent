import sys

from tool_registry import get_all_tools
from agent_builder import build_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def run_agent(user_query):
    agent = build_agent()

    state = {
        "messages": [
            HumanMessage(content=user_query)
        ],
        "plan": None,
        "observation": None,
    }

    result = agent.invoke(state)

    last_ai = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
        None
    )

    return last_ai.content if last_ai else "Sorry, I couldn't find an answer."

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "Find the latest price of Tesla stock."

    result = run_agent(user_query)
    print(result)

