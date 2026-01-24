import requests
import settings
from my_llm import MyChatLLM

from langchain_core.tools import tool

def _tavily_search_raw(query: str, max_results: int = 5) -> str:
    """Get the raw search results from Tavily API."""
    api_key = settings.TAVILY_API_KEY
    if not api_key:
        return "ERROR: TAVILY_API_KEY not found in settings."

    data = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
    }

    try:
        resp = requests.post("https://api.tavily.com/search", json=data, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"ERROR: Failed to fetch results from Tavily. {e}"

    # Extract and format results
    items = []
    for r in data.get("results", [])[:max_results]:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        items.append(f"{title}\n{snippet}\n{link}")

    if not items:
        return "No search results found."

    return "\n\n---\n\n".join(items)

@tool
def tavily_search(query: str, max_results: int = 3) -> str:
    """
    Search the web using Tavily API and return a summarized answer.

    Args:
        query (str): The search query.
        max_results (int, optional): The number of search results to consider for the summary. Defaults to 3.

    Returns:
        str: A summarized answer based on the search results.
    """
    print(f"DEBUG: Summarizing search results for query: {query}")

    # 1. Get raw search results
    raw_results = _tavily_search_raw(query, max_results=max_results)

    if "ERROR:" in raw_results or "No search results found." in raw_results:
        return raw_results

    # 2. Initialize the LLM
    llm = MyChatLLM()

    # 3. Create a prompt for summarization
    prompt = (
        f"Please summarize the following search results for the query '{query}'. "
        "Provide a concise, relevant summary and include the source links for the information."
        f"Search Results:\n{raw_results}"
    )

    # 4. Get the summary from the LLM
    summary_result = llm.invoke(prompt)
    return summary_result.content

