import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import time

# =========================
# Configuration
# =========================
MODEL_PATH = "/Users/zhlsunshine/Projects/inference/models/chatmodels/qwen3-embedding-4b"
MAX_LENGTH = 512
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# =========================
# Load model (once)
# =========================
print(f"[INFO] Loading model on device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    trust_remote_code=True
).to(DEVICE)

model.eval()

print("[INFO] Model loaded successfully")

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Qwen3 Embedding Service")

class EmbeddingRequest(BaseModel):
    input: List[str]

# =========================
# Embedding logic
# =========================
@torch.no_grad()
def compute_embeddings(texts: List[str]):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = model(**inputs)

    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)

    # Normalize (important for cosine similarity)
    embeddings = F.normalize(embeddings, dim=1)

    return embeddings.cpu().tolist()

# =========================
# API endpoint
# =========================
@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest):
    start = time.time()

    vectors = compute_embeddings(req.input)

    return {
        "object": "list",
        "model": "qwen3-embedding-4b",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": vectors[i]
            }
            for i in range(len(vectors))
        ],
        "usage": {
            "prompt_tokens": None,
            "total_tokens": None
        },
        "latency_ms": int((time.time() - start) * 1000)
    }

# =========================
# Health check
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE
    }
