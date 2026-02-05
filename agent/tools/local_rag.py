import os
import settings
from my_llm import MyChatLLM
from my_embedding import MyCustomEmbeddings

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, WebBaseLoader
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
        """
        Fetches content from a URL, creates a temporary vectorstore, and performs a similarity search.
        """
        print(f"DEBUG: Querying URL '{url}' for '{query}'")
        loader = WebBaseLoader(web_paths=[url])
        documents = loader.load()

        if not documents:
            print(f"WARN: No documents loaded from {url}")
            return ""

        splitter = self.text_splitter()
        chunks = splitter.split_documents(documents)

        clean_chunks = []
        for c in chunks:
            if isinstance(c, tuple):
                clean_chunks.append(c[0])
            else:
                clean_chunks.append(c)

        # Optional: hard limit for safety
        clean_chunks = clean_chunks[:settings.MAX_CHUNKS]
        # Create an in-memory vectorstore for this operation
        vectorstore = Chroma.from_documents(clean_chunks, self.embeddings)
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

