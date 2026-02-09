from langchain_core.messages import HumanMessage, AIMessage
from agent_builder import build_agent

agent = build_agent()  # reuse across requests

def run_agent_with_history(chat_history):
    """
    chat_history: list of tuples [(user, assistant), ...]
    """
    messages = []

    for user, assistant in chat_history:
        messages.append(HumanMessage(content=user))
        messages.append(AIMessage(content=assistant))

    state = {
        "messages": messages,
        "plan": None,
        "observation": None,
    }

    result = agent.invoke(state)

    last_ai = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
        None
    )

    return last_ai.content if last_ai else "Sorry, I couldn't find an answer."
