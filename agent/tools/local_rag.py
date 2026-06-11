import logging
import os

import settings
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from my_embedding import MyCustomEmbeddings
from my_llm import MyChatLLM
from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = (".txt", ".md")
HTML_CONTENT_TAGS = ["h1", "h2", "h3", "p", "li"]
HTML_TAGS_TO_REMOVE = ["script", "style", "noscript", "header", "footer", "nav"]


class LocalRAG:
    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or settings.VECTORSTORE_PERSIST_DIR
        self.vectorstore = None
        self.llm = MyChatLLM()
        self.embeddings = MyCustomEmbeddings()

    def text_splitter(self, chunk_size: int = 300, chunk_overlap: int = 50):
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n",
                "\n",
                " ",
                "\uff0e",
                "\u3000",
                "\u3002",
                "",
            ],
        )

    def load_documents(self, docs_dir: str) -> list[Document]:
        documents = []
        for root, _, files in os.walk(docs_dir):
            for filename in files:
                if filename.lower().endswith(TEXT_EXTENSIONS):
                    path = os.path.join(root, filename)
                    documents.extend(TextLoader(path, encoding="utf-8").load())
        return documents

    def build_index(self, docs_dir: str):
        documents = self.load_documents(docs_dir)
        if not documents:
            raise ValueError(f"No text files found under {docs_dir}")

        chunks = self.text_splitter().split_documents(documents)

        if settings.VECTORSTORE_TYPE != "chroma":
            raise NotImplementedError("Other vectorstores not implemented yet")

        self.vectorstore = Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory=self.persist_dir,
        )
        self.vectorstore.persist()
        return f"Indexed {len(chunks)} chunks into {self.persist_dir}"

    def fetch_url_html(self, url: str) -> str:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            )
            page = browser.new_page()
            page.goto(url, timeout=20000, wait_until="networkidle")
            html = page.content()
            browser.close()
            return html

    def documents_from_html(self, html: str, source: str) -> list[Document]:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(HTML_TAGS_TO_REMOVE):
            tag.decompose()

        texts = [
            tag.get_text(strip=True)
            for tag in soup.find_all(HTML_CONTENT_TAGS)
            if len(tag.get_text(strip=True)) > 50
        ]
        if not texts:
            return []

        return [Document(page_content="\n".join(texts), metadata={"source": source})]

    def query_url(self, url: str, query: str, k: int = 4):
        logger.debug("Querying URL %s for %s", url, query)

        try:
            html = self.fetch_url_html(url)
        except Exception as e:
            logger.warning("Failed to load URL %s: %s", url, e)
            return f"[ERROR] Failed to load URL: {url}. Reason: {e}"

        if not html:
            return f"[ERROR] Failed to load URL: {url}. Reason: not html"

        docs = self.documents_from_html(html, url)
        if not docs:
            return f"[ERROR] Failed to load URL: {url}. Reason: not texts"

        chunks = self.text_splitter().split_documents(docs)
        if not chunks:
            return f"[ERROR] Failed to load URL: {url}. Reason: not chunks"

        vectorstore = Chroma.from_documents(chunks[:settings.MAX_CHUNKS], self.embeddings)
        results = vectorstore.max_marginal_relevance_search(query, k=k)
        return "\n\n---\n\n".join(result.page_content[:2000] for result in results)

    def query_index(self, query: str, k: int = 4, expand_query: bool = True):
        logger.debug("Querying local RAG index for %s", query)

        if not self.vectorstore:
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )

        queries = self.expand_query(query) if expand_query else [query]
        candidate_docs = []
        for candidate_query in queries:
            candidate_docs.extend(
                self.vectorstore.max_marginal_relevance_search(
                    candidate_query,
                    k=max(k, 6),
                    fetch_k=20,
                ),
            )

        final_docs = self.select_unique_documents(candidate_docs, k)
        return "\n\n---\n\n".join(doc.page_content[:2000] for doc in final_docs)

    def expand_query(self, query: str) -> list[str]:
        queries = [query]
        try:
            expanded = self.llm.invoke(
                f"""Expand the search query with 3 short alternative phrasings.
Return one per line, no explanations.

Query: {query}
""",
            )
            queries.extend(
                line.strip("-* ").strip()
                for line in expanded.content.splitlines()
                if len(line.strip()) > 5
            )
        except Exception as e:
            logger.warning("Query expansion failed: %s", e)

        return queries

    def select_unique_documents(self, documents: list[Document], k: int) -> list[Document]:
        seen = set()
        unique_docs = []
        for document in documents:
            key = document.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique_docs.append(document)

        return [doc for doc in unique_docs if len(doc.page_content) > 200][:k]


local_rag = LocalRAG()
