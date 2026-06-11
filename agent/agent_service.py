"""Helpers for running a fresh LangGraph agent per request."""

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

DEFAULT_ERROR_MESSAGE = "Sorry, I couldn't find an answer."


def build_agent() -> Any:
    """Build a new compiled agent for one request."""
    from agent_builder import build_agent as _build_agent

    return _build_agent()


def history_to_messages(chat_history: Sequence[Sequence[str]]) -> list[BaseMessage]:
    """Convert ``[[user, assistant], ...]`` history into LangChain messages."""
    messages: list[BaseMessage] = []
    for turn in chat_history:
        if len(turn) != 2:
            raise ValueError("Each chat history item must contain user and assistant text.")

        user, assistant = turn
        messages.append(HumanMessage(content=user))
        messages.append(AIMessage(content=assistant))

    return messages


def extract_last_ai_message(messages: Sequence[BaseMessage]) -> str:
    """Return the most recent AI response from an agent result."""
    last_ai = next(
        (message for message in reversed(messages) if isinstance(message, AIMessage)),
        None,
    )
    return last_ai.content if last_ai else DEFAULT_ERROR_MESSAGE


def run_agent_messages(messages: Sequence[BaseMessage]) -> str:
    """Run the ReAct agent with pre-built LangChain messages."""
    try:
        agent = build_agent()
        state = {
            "messages": list(messages),
            "plan": None,
            "observation": None,
        }
        result = agent.invoke(state)
        return extract_last_ai_message(result["messages"])

    except Exception as e:
        logger.error(f"Agent invocation failed: {e}", exc_info=True)
        return f"Error: Failed to process request. Details: {str(e)}"


def run_agent_with_history(chat_history: Sequence[Sequence[str]]) -> str:
    """Run the ReAct agent with conversation history."""
    return run_agent_messages(history_to_messages(chat_history))


def run_agent_single_turn(user_query: str) -> str:
    """Run the agent with a single user query."""
    return run_agent_messages([HumanMessage(content=user_query)])


if __name__ == "__main__":
    # Test the agent service
    test_query = "What is the capital of France?"

    print("\n=== Testing single-turn query ===")
    print(f"Query: {test_query}")
    response = run_agent_single_turn(test_query)
    print(f"Response:\n{response}\n")

    test_history = [
        ("Hello, who are you?", "I'm a helpful AI assistant."),
        ("What is 2+2?", "2+2 equals 4."),
    ]

    print("=== Testing multi-turn conversation ===")
    response = run_agent_with_history(test_history)
    print(f"Response:\n{response}")
