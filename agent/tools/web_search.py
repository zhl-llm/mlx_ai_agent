import requests
import settings
from tavily import TavilyClient
from my_llm import MyChatLLM

from langchain_core.tools import tool

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
    api_key = settings.TAVILY_API_KEY
    client = TavilyClient(api_key)
    response = client.search(
        query=query,
        search_depth="advanced", # advanced|basic|fast|ultra-fast
        max_results=3
    )
    # print("=== [DEBUG] tavily_search response ===")
    # print(response)
    # print("======================================")

    if "ERROR:" in response or "No search results found." in response:
        return response

    # 2. Initialize the LLM
    llm = MyChatLLM()

    # 3. Create a prompt for summarization
    prompt = (
        f"Please summarize the following search results for the query '{query}'. "
        "Provide a concise, relevant summary and include the source links for the information."
        f"Search Results:\n{response}"
    )

    # 4. Get the summary from the LLM
    summary_result = llm.invoke(prompt)
    return summary_result.content

