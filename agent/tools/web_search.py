import requests
import settings
from tavily import TavilyClient
from serpapi.baidu_search import BaiduSearch
from my_llm import MyChatLLM
from langchain_core.tools import tool

# https://www.tavily.com/
@tool
def tavily_search(query: str, max_results: int = 1) -> str:
    """
    Search the web using Tavily API and return a summarized answer.

    Args:
        query (str): The search query.
        max_results (int, optional): The number of search results to consider for the summary. Defaults to 1.

    Returns:
        str: A summarized answer based on the search results.
    """
    print(f"DEBUG: Summarizing tavily search results for query: {query}")

    # 1. Get raw search results
    api_key = settings.TAVILY_API_KEY
    client = TavilyClient(api_key)
    response = client.search(
        query=query,
        search_depth="advanced", # advanced|basic|fast|ultra-fast
        max_results=max_results
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


# https://serpapi.com/
@tool
def baidu_search(query: str, top_k: int = 3) -> str:
    """
    Search the web using Serp API and return a summarized answer.

    Args:
        query (str): The search query.
        top_k (int, optional): The number of search results to consider for the summary. Defaults to 3.

    Returns:
        str: A summarized answer based on the search results.
    """
    print(f"DEBUG: Summarizing baidu search results for query: {query}")

    # 1. Prepare the paramters for SerpAPI
    api_key = settings.SERP_API_KEY
    params = {
        "engine": "baidu",
        "q": {query},
        "api_key": {api_key},
        "rn": {top_k},
        "oq": True
    }

    search = BaiduSearch(params)
    results = search.get_dict()
    # Handle SerpAPI-level errors
    if "error" in results:
        return {
            "query": query,
            "results": [],
            "error": results["error"],
        }
    organic_results = results.get("organic_results", [])

    # Optimize the search results
    normalized = normalize_baidu_results(organic_results, top_k)
    filtered = [r for r in normalized if is_high_quality(r["url"])]

    # 2. Initialize the LLM
    llm = MyChatLLM()

    # 3. Create a prompt for summarization
    prompt = (
        f"Please summarize the following search results for the query '{query}'. "
        "Provide a concise, relevant summary and include the source links for the information."
        f"Search Results:\n{filtered}"
    )

    # 4. Get the summary from the LLM
    summary_result = llm.invoke(prompt)
    return summary_result.content

def normalize_baidu_results(raw_results, top_k=5):
    normalized = []
    for r in raw_results[:top_k]:
        link = r.get("link")
        title = r.get("title")

        if not link or not title:
            continue

        normalized.append({
            "title": title,
            "snippet": r.get("snippet", ""),
            "url": link,
            "source": "baidu",
        })

    return normalized

def is_high_quality(url: str) -> bool:
    BLOCK_DOMAINS = (
        "baike.baidu.com",
        "zhidao.baidu.com",
        "tieba.baidu.com",
    )
    return not any(d in url for d in BLOCK_DOMAINS)
