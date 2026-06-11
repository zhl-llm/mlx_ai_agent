import logging

import settings
from langchain_core.tools import tool
from my_llm import MyChatLLM
from serpapi.baidu_search import BaiduSearch
from tavily import TavilyClient

logger = logging.getLogger(__name__)

BLOCKED_BAIDU_DOMAINS = (
    "baike.baidu.com",
    "zhidao.baidu.com",
    "tieba.baidu.com",
)


@tool
def tavily_search(query: str, max_results: int = 1) -> str:
    """Search the web using Tavily and return an LLM summary."""
    logger.debug("Summarizing Tavily search results for query: %s", query)

    client = TavilyClient(settings.TAVILY_API_KEY)
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
    )

    if "ERROR:" in response or "No search results found." in response:
        return response

    return summarize_search_results(query, response)


@tool
def baidu_search(query: str, top_k: int = 3) -> str:
    """Search the web using SerpAPI's Baidu engine and return an LLM summary."""
    logger.debug("Summarizing Baidu search results for query: %s", query)

    search = BaiduSearch({
        "engine": "baidu",
        "q": query,
        "api_key": settings.SERP_API_KEY,
        "rn": top_k,
        "oq": True,
    })
    results = search.get_dict()
    if "error" in results:
        return {
            "query": query,
            "results": [],
            "error": results["error"],
        }

    normalized = normalize_baidu_results(results.get("organic_results", []), top_k)
    filtered = [result for result in normalized if is_high_quality(result["url"])]
    return summarize_search_results(query, filtered)


def summarize_search_results(query: str, results) -> str:
    llm = MyChatLLM()
    prompt = (
        f"Please summarize the following search results for the query '{query}'. "
        "Provide a concise, relevant summary and include the source links for the information."
        f"Search Results:\n{results}"
    )
    return llm.invoke(prompt).content


def normalize_baidu_results(raw_results, top_k: int = 5):
    normalized = []
    for result in raw_results[:top_k]:
        link = result.get("link")
        title = result.get("title")
        if not link or not title:
            continue

        normalized.append({
            "title": title,
            "snippet": result.get("snippet", ""),
            "url": link,
            "source": "baidu",
        })

    return normalized


def is_high_quality(url: str) -> bool:
    return not any(domain in url for domain in BLOCKED_BAIDU_DOMAINS)
