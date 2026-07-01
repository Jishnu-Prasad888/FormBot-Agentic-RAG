import json
from typing import AsyncIterator
from openai import OpenAI
import httpx

from app.config import settings


class LLMClient:
    def __init__(self):
        self.provider = settings.llm_provider
        self.openai_model = settings.openai_model
        self.ollama_model = settings.ollama_model
        self.ollama_base = settings.ollama_base_url
        self._openai = None

    def _get_openai(self):
        if self._openai is None:
            self._openai = OpenAI(api_key=settings.openai_api_key)
        return self._openai

    def generate(self, system: str, prompt: str) -> str:
        if self.provider == "ollama":
            return self._generate_ollama(system, prompt)
        return self._generate_openai(system, prompt)

    def generate_stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        if self.provider == "ollama":
            return self._stream_ollama(system, prompt)
        return self._stream_openai(system, prompt)

    def _generate_openai(self, system: str, prompt: str) -> str:
        client = self._get_openai()
        resp = client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""

    async def _stream_openai(self, system: str, prompt: str) -> AsyncIterator[str]:
        client = self._get_openai()
        stream = client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _generate_ollama(self, system: str, prompt: str) -> str:
        resp = httpx.post(
            f"{self.ollama_base}/api/chat",
            json={
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.3},
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    async def _stream_ollama(self, system: str, prompt: str) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self.ollama_base}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": True,
                    "options": {"temperature": 0.3},
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]


llm = LLMClient()
