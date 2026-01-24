import os
import settings
from my_llm import MyChatLLM
from my_embedding import MyCustomEmbeddings

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

class LocalRAG:
    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or settings.VECTORSTORE_PERSIST_DIR
        self.vectorstore = None
        self.llm = MyChatLLM()
        self.embeddings = MyCustomEmbeddings()

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
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        # Vectorstore selection
        if settings.VECTORSTORE_TYPE == "chroma":
            self.vectorstore = Chroma.from_documents(chunks, self.embeddings, persist_directory=self.persist_dir)
            self.vectorstore.persist()
        else:
            raise NotImplementedError("Other vectorstores not implemented yet")
        return f"Indexed {len(chunks)} chunks into {self.persist_dir}"

    def query_url(self, url: str, query: str, k: int = 4):
        """
        Fetches content from a URL, creates a temporary vectorstore, and performs a similarity search.
        """
        print(f"DEBUG: Querying URL '{url}' for '{query}'")
        loader = WebBaseLoader(web_paths=[url])
        documents = loader.load()

        if not documents:
            print(f"WARN: No documents loaded from {url}")
            return ""

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        if not chunks:
            print(f"WARN: No chunks created from {url}")
            return ""

        # Create an in-memory vectorstore for this operation
        vectorstore = Chroma.from_documents(chunks, self.embeddings)
        docs = vectorstore.similarity_search(query, k=k)

        return "\n\n---\n\n".join([d.page_content[:2000] for d in docs])

    def query_index(self, query: str, k: int = 4):
        print(f"DEBUG: Query Index with LocalRAG for query: {query}")
        if not self.vectorstore:
            self.vectorstore = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
        docs = self.vectorstore.similarity_search(query, k=k)
        ctx = "\n\n---\n\n".join([d.page_content[:2000] for d in docs])
        return ctx

# Singleton
local_rag = LocalRAG()

