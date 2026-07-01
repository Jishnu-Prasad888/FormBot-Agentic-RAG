import numpy as np
from openai import OpenAI
import httpx

from app.config import settings


class EmbeddingClient:
    def __init__(self):
        self.provider = settings.embedding_provider
        self.model = settings.embedding_model
        self._openai = None
        self._ollama_base = settings.ollama_base_url

    def _get_openai(self):
        if self._openai is None:
            self._openai = OpenAI(api_key=settings.openai_api_key)
        return self._openai

    def embed(self, text: str) -> list[float]:
        if self.provider == "ollama":
            return self._embed_ollama(text)
        return self._embed_openai(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "ollama":
            return [self._embed_ollama(t) for t in texts]
        return self._embed_openai_batch(texts)

    def _embed_openai(self, text: str) -> list[float]:
        client = self._get_openai()
        resp = client.embeddings.create(input=text, model=self.model)
        return resp.data[0].embedding

    def _embed_openai_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._get_openai()
        resp = client.embeddings.create(input=texts, model=self.model)
        sorted_data = sorted(resp.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]

    def _embed_ollama(self, text: str) -> list[float]:
        resp = httpx.post(
            f"{self._ollama_base}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        A = np.array(a, dtype=np.float32)
        B = np.array(b, dtype=np.float32)
        norm = np.linalg.norm(A) * np.linalg.norm(B)
        if norm == 0:
            return 0.0
        return float(np.dot(A, B) / norm)


embedder = EmbeddingClient()
