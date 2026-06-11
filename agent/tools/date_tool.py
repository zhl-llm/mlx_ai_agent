from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_date(query: str) -> str:
    """
    Get the current date.

    Args:
        query (str): A query string, which is ignored by this tool.

    Returns:
        str: The current date in a string format.
    """
    return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."
