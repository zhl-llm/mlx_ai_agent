"""Command-line entry point for the MLX AI agent."""

import logging
import sys
from collections.abc import Sequence

from agent_service import run_agent_single_turn, run_agent_with_history as run_history

logger = logging.getLogger(__name__)


def run_agent(user_query: str) -> str:
    """Run the agent with a single user query."""
    try:
        return run_agent_single_turn(user_query)
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}", exc_info=True)
        return f"Error: Failed to process request. Details: {str(e)}"


def run_chat_history(chat_history: Sequence[Sequence[str]]) -> str:
    """Run the agent with conversation history."""
    try:
        return run_history(chat_history)
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}", exc_info=True)
        return f"Error: Failed to process request. Details: {str(e)}"


def main() -> None:
    """Main CLI entry point."""
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "Find the latest price of Tesla stock."

    print("\n=== Running Agent ===")
    print(f"Query: {user_query}")
    print(f"\nResponse:\n{run_agent(user_query)}\n")


if __name__ == "__main__":
    main()
