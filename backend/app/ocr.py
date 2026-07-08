import base64
import json
import mimetypes
import re
from typing import Iterable, List, Tuple

from google import genai
import httpx
from openai import OpenAI

from app.config import settings


def _parse_qas_from_text(text: str) -> list[dict]:
    """Parse model output into a list of {question, expected_answer} dicts."""
    if not text:
        return []

    candidates: List[str] = [text.strip()]
    for block in re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
        if block.strip():
            candidates.append(block.strip())

    def _coerce(items: Iterable) -> list[dict]:
        qas: list[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or item.get("q") or "").strip()
            answer = str(item.get("expected_answer") or item.get("answer") or item.get("a") or "").strip()
            if question:
                qas.append({"question": question, "expected_answer": answer})
        return qas

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue

        if isinstance(data, list):
            parsed = _coerce(data)
            if parsed:
                return parsed
        elif isinstance(data, dict):
            if "questions" in data and isinstance(data["questions"], list):
                parsed = _coerce(data["questions"])
                if parsed:
                    return parsed
            parsed = _coerce(data.values())
            if parsed:
                return parsed

    # Fallback: try to read lines like "Q: ..." / "A: ..."
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    pairs: list[dict] = []
    question, answer = "", ""
    for line in lines:
        if line.lower().startswith("q"):
            if question:
                pairs.append({"question": question, "expected_answer": answer})
                answer = ""
            question = line.split(":", 1)[-1].strip() or line
        elif line.lower().startswith("a"):
            answer = line.split(":", 1)[-1].strip() or line
    if question:
        pairs.append({"question": question, "expected_answer": answer})
    return pairs


PROMPT = (
    "You are an OCR parser. Read every question and its expected answer from the image. "
    "Return strict JSON: an array of objects with keys 'question' and 'expected_answer'. "
    "Preserve wording; if an answer is missing use an empty string."
)


class OCRClient:
    def __init__(self):
        self._gemini_client = None
        self._openai_client = None

    def _ensure_gemini(self):
        if not settings.gemini_api_key:
            raise RuntimeError("Gemini API key is not configured (GEMINI_API_KEY)")
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=settings.gemini_api_key)
        return self._gemini_client

    def _ensure_openai(self):
        if not settings.openai_api_key:
            raise RuntimeError("OpenAI API key is not configured (OPENAI_API_KEY)")
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        return self._openai_client

    def _run_gemini(self, image_bytes: bytes, mime_type: str) -> str:
        client = self._ensure_gemini()
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=[PROMPT, {"mime_type": mime_type, "data": image_bytes}],
            config={"temperature": 0.0},
        )
        return response.text or ""

    def _run_openai(self, image_bytes: bytes, mime_type: str) -> str:
        client = self._ensure_openai()
        b64 = base64.b64encode(image_bytes).decode()
        image_url = f"data:{mime_type};base64,{b64}"
        resp = client.chat.completions.create(
            model=settings.ocr_openai_model,
            messages=[
                {"role": "system", "content": PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract questions and answers"},
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                    ],
                },
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""

    def _run_ollama(self, image_bytes: bytes, mime_type: str) -> str:
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": settings.ocr_ollama_model,
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": "Extract questions and answers", "images": [b64]},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        resp = httpx.post(f"{settings.ollama_base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        # Fallback for generate endpoint
        if "response" in data:
            return data["response"]
        return ""

    def _run_provider(self, image_bytes: bytes, mime_type: str) -> str:
        provider = (settings.ocr_provider or "gemini").lower()
        if provider == "gemini":
            return self._run_gemini(image_bytes, mime_type)
        if provider == "openai":
            return self._run_openai(image_bytes, mime_type)
        if provider == "ollama":
            return self._run_ollama(image_bytes, mime_type)
        raise RuntimeError(f"Unsupported OCR provider: {provider}")

    def extract_questions(self, image_bytes: bytes, mime_type: str, filename: str = "image") -> list[dict]:
        text = self._run_provider(image_bytes, mime_type)
        parsed = _parse_qas_from_text(text)
        return parsed

    def extract_from_images(self, images: List[Tuple[str, bytes, str]]):
        """Extract Q&A pairs from a list of (filename, bytes, mime) tuples."""
        questions: list[dict] = []
        errors: list[dict] = []
        for fname, data, mime in images:
            try:
                mime_type = mime or mimetypes.guess_type(fname)[0] or "application/octet-stream"
                qas = self.extract_questions(data, mime_type, fname)
                questions.extend(qas)
            except Exception as exc:  # noqa: BLE001
                errors.append({"file": fname, "error": str(exc)})
        return questions, errors


ocr = OCRClient()
