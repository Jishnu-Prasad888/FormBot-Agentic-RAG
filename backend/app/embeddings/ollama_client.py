import asyncio
import json
from typing import AsyncGenerator, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.config import settings
from app.core.exceptions import OllamaConnectionError



class OllamaClient:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.llm_model = settings.OLLAMA_LLM_MODEL
        self.embed_model = settings.OLLAMA_EMBED_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
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
        client = await self._get_client()
        payload = {
            "model": model or self.llm_model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        try:
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except httpx.HTTPStatusError as e:
            raise OllamaConnectionError(str(e))
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise OllamaConnectionError(str(e))

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
        client = await self._get_client()
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        payload = {
            "model": model or self.llm_model,
            "messages": chat_messages,
            "stream": False,
        }
        try:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except httpx.HTTPStatusError as e:
            raise OllamaConnectionError(str(e))
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise OllamaConnectionError(str(e))

    async def generate_stream(
        self, prompt: str, model: Optional[str] = None, system: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        client = await self._get_client()
        payload = {
            "model": model or self.llm_model,
            "prompt": prompt,
            "stream": True,
        }
        if system:
            payload["system"] = system
        try:
            async with client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise OllamaConnectionError(str(e))

    async def chat_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        system: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        client = await self._get_client()
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        payload = {
            "model": model or self.llm_model,
            "messages": chat_messages,
            "stream": True,
        }
        try:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                yield token
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise OllamaConnectionError(str(e))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    )
    async def embeddings(self, text: str, model: Optional[str] = None) -> list[float]:
        client = await self._get_client()
        payload = {
            "model": model or self.embed_model,
            "prompt": text,
        }
        try:
            response = await client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])
        except httpx.HTTPStatusError as e:
            raise OllamaConnectionError(str(e))
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise OllamaConnectionError(str(e))

    async def batch_embeddings(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
        tasks = [self.embeddings(text, model) for text in texts]
        return await asyncio.gather(*tasks)

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[dict]:
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            return []


ollama_client = OllamaClient()
