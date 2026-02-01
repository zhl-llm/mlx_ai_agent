from langchain_core.tools import tool
from tools.local_rag import local_rag
from tools.web_search import tavily_search, baidu_search
from tools.date_tool import get_current_date

_all_tools = [
    get_current_date,
    tavily_search,
    # baidu_search,
]

@tool
def build_rag_index(docs_dir: str) -> str:
    """Builds a local RAG index from a directory of text files."""
    return local_rag.build_index(docs_dir)

@tool
def query_rag_index(query: str) -> str:
    """Queries the local RAG index."""
    return local_rag.query_index(query)

@tool
def query_url_index(url: str, query: str) -> str:
    """
    Queries the content from a given URL based on given query input
    Args:
        url (str): The URL to fetch content from.
        query (str): The given query input.

    Returns:
        str: A summarized answer based on the search results.
    """
    return local_rag.query_url(url, query)

def get_all_tools():
    """Returns a list of all available tools."""
    return _all_tools + [build_rag_index, query_rag_index, query_url_index]

