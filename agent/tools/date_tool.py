from langchain_core.tools import tool
from datetime import datetime

@tool
def get_current_date(query: str) -> str:
    """
    Get the current date.

    Args:
        query (str): A query string, which is ignored by this tool.

    Returns:
        str: The current date in a string format.
    """
    date_str = f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."
    # print(f"DEBUG: from get_current_date tool: {date_str}")
    return date_str

