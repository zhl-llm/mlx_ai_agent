import os
import settings
from my_llm import MyChatLLM
from my_embedding import MyCustomEmbeddings

from bs4 import BeautifulSoup
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from playwright.sync_api import sync_playwright
from langchain_core.documents import Document
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

class LocalRAG:
    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or settings.VECTORSTORE_PERSIST_DIR
        self.vectorstore = None
        self.llm = MyChatLLM()
        self.embeddings = MyCustomEmbeddings()

    ##
    ## Read source documents and extract return the list contains documents
    ##
    def text_splitter(self, chunk_size:int=300, chunk_overlap:int=50, max_doc_count:int=-1):
        my_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n",
                "\n",
                " ",
                "\uff0e",  # Fullwidth full stop
                "\u3000",
                "\u3002",  # Ideographic full stop
                "",
            ],
        )
        return my_splitter

    def build_index(self, docs_dir: str):
        from os import walk
        loaders = []
        for root, _, files in walk(docs_dir):
            for fname in files:
                if fname.lower().endswith((".txt", ".md")):
                    loaders.append(TextLoader(os.path.join(root, fname), encoding="utf-8"))
        if not loaders:
            raise ValueError(f"No text files found under {docs_dir}")
        documents = []
        for loader in loaders:
            documents.extend(loader.load())

        splitter = self.text_splitter()
        chunks = splitter.split_documents(documents)

        # Vectorstore selection
        if settings.VECTORSTORE_TYPE == "chroma":
            self.vectorstore = Chroma.from_documents(chunks, self.embeddings, persist_directory=self.persist_dir)
            self.vectorstore.persist()
        else:
            raise NotImplementedError("Other vectorstores not implemented yet")
        return f"Indexed {len(chunks)} chunks into {self.persist_dir}"


    def query_url(self, url: str, query: str, k: int = 4):
        print(f"DEBUG: Querying URL '{url}' for '{query}'")
        html = ""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                )
                page = browser.new_page()
                page.goto(url, timeout=20000, wait_until="networkidle")
                html = page.content()
                browser.close()
        except Exception as e:
            print(f"WARN: Playwright failed to load {url}: {e}")
            return ""
        if not html:
            return ""
        # ---- HTML CLEANING ----
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        texts = [
            tag.get_text(strip=True)
            for tag in soup.find_all(["h1", "h2", "h3", "p", "li"])
            if len(tag.get_text(strip=True)) > 50
        ]
        if not texts:
            return ""
        docs = [Document(page_content="\n".join(texts), metadata={"source": url})]
        # ---- SPLITTING ----
        splitter = self.text_splitter()
        chunks = splitter.split_documents(docs)
        if not chunks:
            return ""
        # ---- VECTOR SEARCH ----
        vectorstore = Chroma.from_documents(chunks[:settings.MAX_CHUNKS], self.embeddings)
        results = vectorstore.max_marginal_relevance_search(query, k=k)

        return "\n\n---\n\n".join(r.page_content[:2000] for r in results)

    def query_index(self, query: str, k: int = 4, expand_query: bool = True):
        print(f"DEBUG: Query Index with LocalRAG for query: {query}")

        if not self.vectorstore:
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )

        queries = [query]
        if expand_query:
            try:
                expanded = self.llm.invoke(
                    f"""Expand the search query with 3 short alternative phrasings.
    Return one per line, no explanations.

    Query: {query}
    """
                )
                queries.extend(
                    q.strip("-• ").strip()
                    for q in expanded.content.splitlines()
                    if len(q.strip()) > 5
                )
            except Exception as e:
                print(f"WARN: Query expansion failed: {e}")

        # ---- SEARCH WITH MMR ----
        candidate_docs = []
        for q in queries:
            docs = self.vectorstore.max_marginal_relevance_search(
                q,
                k=max(k, 6),
                fetch_k=20,
            )
            candidate_docs.extend(docs)

        # ---- DEDUPLICATE ----
        seen = set()
        unique_docs = []
        for d in candidate_docs:
            key = d.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)
        # ---- FILTER LOW-SIGNAL CHUNKS ----
        filtered = [
            d for d in unique_docs
            if len(d.page_content) > 200
        ]
        # ---- FINAL SELECTION ----
        final_docs = filtered[:k]
        ctx = "\n\n---\n\n".join(
            d.page_content[:2000] for d in final_docs
        )
        return ctx

# Singleton
local_rag = LocalRAG()

