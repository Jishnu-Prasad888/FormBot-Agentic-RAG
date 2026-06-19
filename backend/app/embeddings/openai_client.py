import asyncio
from typing import AsyncGenerator, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.config import settings
from app.core.exceptions import OpenAIConnectionError


OPENAI_API_BASE = "https://api.openai.com/v1"


class OpenAIClient:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.llm_model = settings.OPENAI_LLM_MODEL
        self.embed_model = settings.OPENAI_EMBED_MODEL
        self.timeout = settings.OPENAI_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=OPENAI_API_BASE,
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    )
    async def generate(self, prompt: str, model: Optional[str] = None, system: Optional[str] = None) -> str:
        """Single-turn generation via chat completions."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self._chat_completions(messages, model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    )
    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        system: Optional[str] = None,
    ) -> str:
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        return await self._chat_completions(chat_messages, model)

    async def _chat_completions(self, messages: list[dict], model: Optional[str] = None) -> str:
        client = await self._get_client()
        payload = {
            "model": model or self.llm_model,
            "messages": messages,
        }
        try:
            response = await client.post(
                "/chat/completions",
                json=payload,
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            raise OpenAIConnectionError(str(e))
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise OpenAIConnectionError(str(e))

    async def generate_stream(
        self, prompt: str, model: Optional[str] = None, system: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        async for token in self._chat_stream(messages, model):
            yield token

    async def chat_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        system: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        async for token in self._chat_stream(chat_messages, model):
            yield token

    async def _chat_stream(
        self, messages: list[dict], model: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        client = await self._get_client()
        payload = {
            "model": model or self.llm_model,
            "messages": messages,
            "stream": True,
        }
        try:
            async with client.stream(
                "POST",
                "/chat/completions",
                json=payload,
                headers=self._get_headers(),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        import json
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield token
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise OpenAIConnectionError(str(e))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    )
    async def embeddings(self, text: str, model: Optional[str] = None) -> list[float]:
        """Embed text with a fallback model if the primary fails."""
        primary_model = model or self.embed_model
        fallback_model = settings.OPENAI_EMBED_FALLBACK_MODEL
        client = await self._get_client()

        async def _embed(m: str) -> list[float]:
            payload = {"model": m, "input": text}
            response = await client.post(
                "/embeddings",
                json=payload,
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]

        try:
            return await _embed(primary_model)
        except Exception:
            if fallback_model and fallback_model != primary_model:
                try:
                    return await _embed(fallback_model)
                except Exception as e:
                    raise OpenAIConnectionError(str(e))
            raise

    async def batch_embeddings(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
        """
        Use OpenAI's native batch input for efficiency (up to 2048 inputs per request).
        Falls back to individual calls for very large batches.
        """
        BATCH_SIZE = 100  # safe limit well within OpenAI's 2048 max
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i: i + BATCH_SIZE]
            all_embeddings.extend(await self._batch_embeddings_chunk(batch, model))
        return all_embeddings

    async def _batch_embeddings_chunk(
        self, texts: list[str], model: Optional[str] = None
    ) -> list[list[float]]:
        client = await self._get_client()
        payload = {
            "model": model or self.embed_model,
            "input": texts,
        }
        try:
            response = await client.post(
                "/embeddings",
                json=payload,
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()
            # OpenAI returns data sorted by index
            items = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in items]
        except httpx.HTTPStatusError as e:
            raise OpenAIConnectionError(str(e))
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise OpenAIConnectionError(str(e))

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get("/models", headers=self._get_headers())
            return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[dict]:
        try:
            client = await self._get_client()
            response = await client.get("/models", headers=self._get_headers())
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as e:
            return []


# Module-level singleton — same name pattern as before so imports stay clean
openai_client = OpenAIClient()

# Alias: every existing import of `ollama_client` still works without any
# other file change, because we also export the name `ollama_client` here.
ollama_client = openai_client
