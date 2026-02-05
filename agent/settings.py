import os
from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()

# -----------------------
# LLM Configuration
# -----------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-pro")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
# LLM_API_KEY = os.getenv("LLM_API_KEY", "")
OAUTH_ID_TOKEN = os.getenv("OAUTH_ID_TOKEN", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -----------------------
# Embeddings Configuration
# -----------------------
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openai")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")

# -----------------------
# Vectorstore Configuration
# -----------------------
RAG_DATA_SOURCE = os.getenv("RAG_DATA_SOURCE", "source_data")
VECTORSTORE_TYPE = os.getenv("VECTORSTORE_TYPE", "chroma")
VECTORSTORE_PERSIST_DIR = os.getenv("VECTORSTORE_PERSIST_DIR", "./chroma_store")

# -----------------------
# API Keys
# -----------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

MAX_CHUNKS = os.getenv("MAX_CHUNKS", 30)