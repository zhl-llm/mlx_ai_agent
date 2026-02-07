import requests
from typing import List
from langchain.embeddings.base import Embeddings


class MyCustomEmbeddings(Embeddings):
    def __init__(
        self,
        endpoint: str = "http://localhost:6000/v1/embeddings",
        timeout: int = 180,
    ):
        self.endpoint = endpoint
        self.timeout = timeout

    def _embed(self, texts: List[str]) -> List[List[float]]:
        payload = {"input": texts}

        response = requests.post(
            self.endpoint,
            headers={
                "USER_AGENT": "mlx-ai-agent/0.1 (local)"
            },
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()

        return [item["embedding"] for item in result["data"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]
