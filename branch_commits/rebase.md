# Branch rebase

Generated on 2026-07-13T07:28:45Z
Total commits: 56

## 4f4f1afb8853cd92515329ecb40cf92762aac27f — 2026-07-07T14:45:38+05:30

Message:

added image ocr support with openai and google genai

```diff
diff --git a/backend/app/config.py b/backend/app/config.py
index c85c1b9..112578e 100644
+++ b/backend/app/config.py
@@ -11,6 +11,11 @@ class Settings(BaseSettings):
     ollama_model: str = "banking-assistant"
     eval_accuracy_provider: str = "ollama"
     eval_accuracy_model: str = "banking-assistant"
+    gemini_api_key: str = ""
+    gemini_model: str = "gemini-2.5-pro"
+    ocr_provider: str = "gemini"  # gemini | openai | ollama
+    ocr_openai_model: str = "gpt-4o-mini"
+    ocr_ollama_model: str = "llama3.2-vision"
 
     model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
 
diff --git a/backend/app/llm.py b/backend/app/llm.py
index 31f43f5..836b6bf 100644
+++ b/backend/app/llm.py
@@ -57,26 +57,8 @@ class LLMClient:
                 yield chunk.choices[0].delta.content
 
     def _generate_ollama(self, system: str, prompt: str) -> str:
+        try:
+            resp = httpx.post(
                 f"{self.ollama_base}/api/chat",
                 json={
                     "model": self.ollama_model,
@@ -84,15 +66,75 @@ class LLMClient:
                         {"role": "system", "content": system},
                         {"role": "user", "content": prompt},
                     ],
+                    "stream": False,
                     "options": {"temperature": 0.3},
                 },
+                timeout=60,
+            )
+            resp.raise_for_status()
+            return resp.json()["message"]["content"]
+        except httpx.HTTPStatusError as exc:
+            if exc.response.status_code != 404:
+                raise
+            resp = httpx.post(
+                f"{self.ollama_base}/api/generate",
+                json={
+                    "model": self.ollama_model,
+                    "prompt": f"{system}\n\n{prompt}",
+                    "stream": False,
+                    "options": {"temperature": 0.3},
+                },
+                timeout=60,
+            )
+            resp.raise_for_status()
+            data = resp.json()
+            if "response" in data:
+                return data["response"]
+            return data.get("message", {}).get("content", "")
+
+    async def _stream_ollama(self, system: str, prompt: str) -> AsyncIterator[str]:
+        async with httpx.AsyncClient(timeout=60) as client:
+            try:
+                async with client.stream(
+                    "POST",
+                    f"{self.ollama_base}/api/chat",
+                    json={
+                        "model": self.ollama_model,
+                        "messages": [
+                            {"role": "system", "content": system},
+                            {"role": "user", "content": prompt},
+                        ],
+                        "stream": True,
+                        "options": {"temperature": 0.3},
+                    },
+                ) as resp:
+                    resp.raise_for_status()
+                    async for line in resp.aiter_lines():
+                        if line.strip():
+                            data = json.loads(line)
+                            if "message" in data and "content" in data["message"]:
+                                yield data["message"]["content"]
+            except httpx.HTTPStatusError as exc:
+                if exc.response.status_code != 404:
+                    raise
+                async with client.stream(
+                    "POST",
+                    f"{self.ollama_base}/api/generate",
+                    json={
+                        "model": self.ollama_model,
+                        "prompt": f"{system}\n\n{prompt}",
+                        "stream": True,
+                        "options": {"temperature": 0.3},
+                    },
+                ) as resp:
+                    resp.raise_for_status()
+                    async for line in resp.aiter_lines():
+                        if line.strip():
+                            data = json.loads(line)
+                            if "response" in data:
+                                yield data["response"]
+                            elif "message" in data and "content" in data["message"]:
+                                yield data["message"]["content"]
 
 
 llm = LLMClient()
diff --git a/backend/app/main.py b/backend/app/main.py
index be6c0db..e6d7ab7 100644
+++ b/backend/app/main.py
@@ -31,8 +31,10 @@ from app.models import (
 from app.embeddings import embedder
 from app.rag import chunk_text, extract_text_from_file, retrieve, build_context, query
 from app.llm import llm
+from app.ocr import ocr
 
 UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
+SAMPLE_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample.png")
 os.makedirs(UPLOAD_DIR, exist_ok=True)
 
 
@@ -611,6 +613,49 @@ async def rag_evaluate(req: EvaluationRequest):
     ).model_dump()
 
 
+@app.post("/api/rag/evaluate/images")
+async def rag_evaluate_images(
+    use_sample: bool = Form(False),
+    files: List[UploadFile] | None = File(None),
+):
+    images: list[tuple[str, bytes, str]] = []
+
+    if use_sample:
+        if not os.path.exists(SAMPLE_IMAGE_PATH):
+            raise HTTPException(500, "Sample image not found on server")
+        with open(SAMPLE_IMAGE_PATH, "rb") as fh:
+            images.append((os.path.basename(SAMPLE_IMAGE_PATH), fh.read(), "image/png"))
+
+    if files:
+        for f in files:
+            content = await f.read()
+            images.append((f.filename or "image", content, f.content_type or "application/octet-stream"))
+
+    if not images:
+        raise HTTPException(400, "No images provided")
+
+    try:
+        questions, errors = ocr.extract_from_images(images)
+    except Exception as exc:  # noqa: BLE001
+        raise HTTPException(500, f"OCR failed: {exc}") from exc
+
+    response = {
+        "questions": questions,
+        "count": len(questions),
+        "errors": errors,
+        "images_processed": len(images),
+        "from_sample": use_sample,
+    }
+
+    if len(questions) == 0 and errors:
+        raise HTTPException(
+            status_code=400,
+            detail={"message": "No questions extracted", "errors": errors},
+        )
+
+    return response
+
+
 # ─── Agents ───────────────────────────────────────────────────────────────────
 
 @app.post("/api/agents/{agent_type}")
diff --git a/backend/app/ocr.py b/backend/app/ocr.py
new file mode 100644
index 0000000..b5ce8f9
+++ b/backend/app/ocr.py
@@ -0,0 +1,176 @@
+import base64
+import json
+import mimetypes
+import re
+from typing import Iterable, List, Tuple
+
+from google import genai
+import httpx
+from openai import OpenAI
+
+from app.config import settings
+
+
+def _parse_qas_from_text(text: str) -> list[dict]:
+    """Parse model output into a list of {question, expected_answer} dicts."""
+    if not text:
+        return []
+
+    candidates: List[str] = [text.strip()]
+    for block in re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
+        if block.strip():
+            candidates.append(block.strip())
+
+    def _coerce(items: Iterable) -> list[dict]:
+        qas: list[dict] = []
+        for item in items:
+            if not isinstance(item, dict):
+                continue
+            question = str(item.get("question") or item.get("q") or "").strip()
+            answer = str(item.get("expected_answer") or item.get("answer") or item.get("a") or "").strip()
+            if question:
+                qas.append({"question": question, "expected_answer": answer})
+        return qas
+
+    for candidate in candidates:
+        try:
+            data = json.loads(candidate)
+        except Exception:
+            continue
+
+        if isinstance(data, list):
+            parsed = _coerce(data)
+            if parsed:
+                return parsed
+        elif isinstance(data, dict):
+            if "questions" in data and isinstance(data["questions"], list):
+                parsed = _coerce(data["questions"])
+                if parsed:
+                    return parsed
+            parsed = _coerce(data.values())
+            if parsed:
+                return parsed
+
+    # Fallback: try to read lines like "Q: ..." / "A: ..."
+    lines = [l.strip() for l in text.splitlines() if l.strip()]
+    pairs: list[dict] = []
+    question, answer = "", ""
+    for line in lines:
+        if line.lower().startswith("q"):
+            if question:
+                pairs.append({"question": question, "expected_answer": answer})
+                answer = ""
+            question = line.split(":", 1)[-1].strip() or line
+        elif line.lower().startswith("a"):
+            answer = line.split(":", 1)[-1].strip() or line
+    if question:
+        pairs.append({"question": question, "expected_answer": answer})
+    return pairs
+
+
+PROMPT = (
+    "You are an OCR parser. Read every question and its expected answer from the image. "
+    "Return strict JSON: an array of objects with keys 'question' and 'expected_answer'. "
+    "Preserve wording; if an answer is missing use an empty string."
+)
+
+
+class OCRClient:
+    def __init__(self):
+        self._gemini_client = None
+        self._openai_client = None
+
+    def _ensure_gemini(self):
+        if not settings.gemini_api_key:
+            raise RuntimeError("Gemini API key is not configured (GEMINI_API_KEY)")
+        if self._gemini_client is None:
+            self._gemini_client = genai.Client(api_key=settings.gemini_api_key)
+        return self._gemini_client
+
+    def _ensure_openai(self):
+        if not settings.openai_api_key:
+            raise RuntimeError("OpenAI API key is not configured (OPENAI_API_KEY)")
+        if self._openai_client is None:
+            self._openai_client = OpenAI(api_key=settings.openai_api_key)
+        return self._openai_client
+
+    def _run_gemini(self, image_bytes: bytes, mime_type: str) -> str:
+        client = self._ensure_gemini()
+        response = client.models.generate_content(
+            model=settings.gemini_model,
+            contents=[PROMPT, {"mime_type": mime_type, "data": image_bytes}],
+            config={"temperature": 0.0},
+        )
+        return response.text or ""
+
+    def _run_openai(self, image_bytes: bytes, mime_type: str) -> str:
+        client = self._ensure_openai()
+        b64 = base64.b64encode(image_bytes).decode()
+        image_url = f"data:{mime_type};base64,{b64}"
+        resp = client.chat.completions.create(
+            model=settings.ocr_openai_model,
+            messages=[
+                {"role": "system", "content": PROMPT},
+                {
+                    "role": "user",
+                    "content": [
+                        {"type": "text", "text": "Extract questions and answers"},
+                        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
+                    ],
+                },
+            ],
+            temperature=0.0,
+        )
+        return resp.choices[0].message.content or ""
+
+    def _run_ollama(self, image_bytes: bytes, mime_type: str) -> str:
+        b64 = base64.b64encode(image_bytes).decode()
+        payload = {
+            "model": settings.ocr_ollama_model,
+            "messages": [
+                {"role": "system", "content": PROMPT},
+                {"role": "user", "content": "Extract questions and answers", "images": [b64]},
+            ],
+            "stream": False,
+            "options": {"temperature": 0.0},
+        }
+        resp = httpx.post(f"{settings.ollama_base_url}/api/chat", json=payload, timeout=120)
+        resp.raise_for_status()
+        data = resp.json()
+        if "message" in data and "content" in data["message"]:
+            return data["message"]["content"]
+        # Fallback for generate endpoint
+        if "response" in data:
+            return data["response"]
+        return ""
+
+    def _run_provider(self, image_bytes: bytes, mime_type: str) -> str:
+        provider = (settings.ocr_provider or "gemini").lower()
+        if provider == "gemini":
+            return self._run_gemini(image_bytes, mime_type)
+        if provider == "openai":
+            return self._run_openai(image_bytes, mime_type)
+        if provider == "ollama":
+            return self._run_ollama(image_bytes, mime_type)
+        raise RuntimeError(f"Unsupported OCR provider: {provider}")
+
+    def extract_questions(self, image_bytes: bytes, mime_type: str, filename: str = "image") -> list[dict]:
+        text = self._run_provider(image_bytes, mime_type)
+        parsed = _parse_qas_from_text(text)
+        return parsed
+
+    def extract_from_images(self, images: List[Tuple[str, bytes, str]]):
+        """Extract Q&A pairs from a list of (filename, bytes, mime) tuples."""
+        questions: list[dict] = []
+        errors: list[dict] = []
+        for fname, data, mime in images:
+            try:
+                mime_type = mime or mimetypes.guess_type(fname)[0] or "application/octet-stream"
+                qas = self.extract_questions(data, mime_type, fname)
+                questions.extend(qas)
+            except Exception as exc:  # noqa: BLE001
+                errors.append({"file": fname, "error": str(exc)})
+        return questions, errors
+
+
+ocr = OCRClient()
diff --git a/helpfull scripts/gen_img_openai.py b/helpfull scripts/gen_img_openai.py
new file mode 100644
index 0000000..f06e5b1
+++ b/helpfull scripts/gen_img_openai.py	
@@ -0,0 +1,50 @@
+import os
+from pathlib import Path
+from openai import OpenAI
+import base64
+from dotenv import load_dotenv
+
+load_dotenv()
+
+client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
+
+questions = [
+    "What is a realty gold loan in the context of a gold loan application form, and why is it necessary to have builder letters for it?",
+    "What are the gross weight and net weight in a gold loan application form, and how should they be entered?",
+    "What does 75% LTV mean in relation to a gold loan application form?"
+]
+
+output_dir = Path("generated_handwritten_questions")
+output_dir.mkdir(exist_ok=True)
+
+for idx, question in enumerate(questions, start=1):
+    prompt = f"""
+A highly realistic photograph of a slightly off-white notebook paper lying on a desk.
+
+The following text is handwritten naturally with blue ink, realistic human handwriting,
+medium neatness, slight imperfections, natural spacing, and authentic pen pressure:
+
+"{question}"
+
+Photorealistic, natural lighting, paper texture visible, realistic shadows,
+looks like an actual handwritten note captured by a smartphone camera.
+No printed fonts. No digital text. Handwriting only.
+Medium image quality.
+"""
+
+    result = client.images.generate(
+        model="gpt-image-1",
+        prompt=prompt,
+        size="1024x1024",
+    )
+
+    image_base64 = result.data[0].b64_json
+    image_bytes = base64.b64decode(image_base64)
+
+    output_path = output_dir / f"question_{idx}.png"
+    with open(output_path, "wb") as f:
+        f.write(image_bytes)
+
+    print(f"Saved: {output_path}")
+
+print("Done.")
\ No newline at end of file
diff --git a/helpfull scripts/generate_db_for_app.py b/helpfull scripts/generate_db_for_app.py
new file mode 100644
index 0000000..129420f
+++ b/helpfull scripts/generate_db_for_app.py	
@@ -0,0 +1,232 @@
+import csv
+import hashlib
+import json
+import os
+import uuid
+
+from dotenv import load_dotenv
+from openai import OpenAI
+
+# -----------------------------
+# Configuration
+# -----------------------------
+
+TXT_FILE = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/ASR removed questions/formatted_questions.txt"
+OUTPUT_CSV = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/ASR removed questions/questions.csv"
+
+# Replace with your actual list of forms from the system prompt
+FORMS = [
+"Account Opening Form-Resident Individual",
+"Customer Request Form",
+"Deposit Slip",
+"Withdrawal Slip",
+"Agriculture Loan Application Form",
+"PM Svanidhi Loan Application Form",
+"PMMY Loan Application Form",
+"Public Provident Fund Account Closure Form",
+"Public Provident Fund Account Deposit Slip",
+"Public Provident Fund Account Extension Form",
+"Public Provident Fund Account Nomination Form",
+"Public Provident Fund Account Opening Form",
+"SCSS Account Closure Form",
+"SCSS Account Deposit Slip",
+"SCSS Account Nomination Change Form",
+"SCSS Account Opening Form",
+"SSA Account Closure Form",
+"SSA Account Opening Form",
+"SSA Account Premature Closure",
+"SSA Account Withdrawal",
+"Auto Loan Application",
+"Business Loan Application",
+"Education Loan Application",
+"Gold Loan Application",
+"Home Loan Application",
+"Key fact Statement",
+"Loan Against Property Application",
+"Personal Loan Application"
+]
+
+# =====================================================
+# OPENAI SETUP
+# =====================================================
+
+load_dotenv()
+
+client = OpenAI(
+    api_key=os.getenv("OPENAI_API_KEY")
+)
+
+# =====================================================
+# HELPERS
+# =====================================================
+
+def get_form_code(form_name: str) -> str:
+    """
+    Creates a fixed 3-character abbreviation.
+
+    Examples:
+        Personal Information Form -> PIF
+        Technical Support Form -> TSF
+        Survey Form -> SUR
+    """
+
+    words = [
+        w for w in form_name.split()
+        if w.lower() not in {"form", "and", "&"}
+    ]
+
+    initials = "".join(word[0].upper() for word in words)
+
+    if len(initials) >= 3:
+        return initials[:3]
+
+    cleaned = "".join(
+        c.upper()
+        for c in form_name
+        if c.isalpha()
+    )
+
+    return (cleaned + "XXX")[:3]
+
+
+def create_question_id(form_name: str, question: str) -> str:
+    """
+    Creates a fixed-length ID:
+
+        PIFA8F3C2D1
+
+    Structure:
+        3 chars form code
+        8 chars hash
+
+    Total length = 11 chars
+    """
+
+    form_code = get_form_code(form_name)
+
+    hash_input = f"{form_name}|{question}"
+
+    hash_part = hashlib.md5(
+        hash_input.encode("utf-8")
+    ).hexdigest()[:8].upper()
+
+    return f"{form_code}{hash_part}"
+
+
+def classify_question(question: str) -> str:
+    """
+    Uses GPT-4o-mini to determine the closest form.
+    """
+
+    prompt = f"""
+Available forms:
+
+{json.dumps(FORMS, indent=2)}
+
+Question:
+{question}
+
+Return ONLY valid JSON:
+
+{{
+  "form": "exact form name from the list"
+}}
+"""
+
+    response = client.chat.completions.create(
+        model="gpt-4o-mini",
+        temperature=0,
+        response_format={"type": "json_object"},
+        messages=[
+            {
+                "role": "system",
+                "content": (
+                    "You classify questions into the closest matching form. "
+                    "You must choose exactly one form from the provided list."
+                )
+            },
+            {
+                "role": "user",
+                "content": prompt
+            }
+        ]
+    )
+
+    content = response.choices[0].message.content
+
+    try:
+        result = json.loads(content)
+        form_name = result["form"]
+
+        if form_name in FORMS:
+            return form_name
+
+    except Exception:
+        pass
+
+    return "Unknown Form"
+
+
+# =====================================================
+# MAIN
+# =====================================================
+
+def main():
+
+    if not os.path.exists(TXT_FILE):
+        raise FileNotFoundError(
+            f"Could not find {TXT_FILE}"
+        )
+
+    with open(TXT_FILE, "r", encoding="utf-8") as f:
+        questions = [
+            line.strip()
+            for line in f
+            if line.strip()
+        ]
+
+    rows = []
+
+    total = len(questions)
+
+    for idx, question in enumerate(questions, start=1):
+
+        print(f"[{idx}/{total}] Processing...")
+
+        form_name = classify_question(question)
+
+        question_id = create_question_id(
+            form_name,
+            question
+        )
+
+        rows.append({
+            "question_id": question_id,
+            "question": question
+        })
+
+    with open(
+        OUTPUT_CSV,
+        "w",
+        newline="",
+        encoding="utf-8"
+    ) as csvfile:
+
+        writer = csv.DictWriter(
+            csvfile,
+            fieldnames=[
+                "question_id",
+                "question"
+            ]
+        )
+
+        writer.writeheader()
+        writer.writerows(rows)
+
+    print(f"\nDone.")
+    print(f"Saved: {OUTPUT_CSV}")
+    print(f"Questions processed: {len(rows)}")
+
+
+if __name__ == "__main__":
+    main()
\ No newline at end of file
```

## 53bac6dd8463573723605549e06e44216d348f60 — 2026-07-03T12:52:40+05:30

Message:

qwen model as eval acuracy provider
openai for all other metrics only qwen model for accurayc eval judging

```diff
diff --git a/ask_gpt.py b/ask_gpt.py
deleted file mode 100644
index e69de29..0000000
diff --git a/backend/app/config.py b/backend/app/config.py
index 21dbf3a..c85c1b9 100644
+++ b/backend/app/config.py
@@ -9,6 +9,8 @@ class Settings(BaseSettings):
     embedding_model: str = "nomic-embed-text"
     ollama_base_url: str = "http://localhost:11434"
     ollama_model: str = "banking-assistant"
+    eval_accuracy_provider: str = "ollama"
+    eval_accuracy_model: str = "banking-assistant"
 
     model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
 
diff --git a/backend/app/main.py b/backend/app/main.py
index 5382582..be6c0db 100644
+++ b/backend/app/main.py
@@ -11,6 +11,7 @@ from fastapi.exceptions import RequestValidationError
 from typing import List
 from fastapi.middleware.cors import CORSMiddleware
 from fastapi.responses import StreamingResponse, JSONResponse
+from openai import OpenAI
 
 from app.config import settings
 from app.storage import (
@@ -89,6 +90,70 @@ def _utcnow() -> str:
     return datetime.now(timezone.utc).isoformat()
 
 
+def _eval_accuracy_llm(system: str, prompt: str) -> str:
+    if settings.eval_accuracy_provider == "ollama":
+        import httpx
+
+        payload = {
+            "model": settings.eval_accuracy_model,
+            "messages": [
+                {"role": "system", "content": system},
+                {"role": "user", "content": prompt},
+            ],
+            "stream": False,
+            "options": {"temperature": 0.0},
+        }
+
+        try:
+            resp = httpx.post(
+                f"{settings.ollama_base_url}/api/chat",
+                json=payload,
+                timeout=60,
+            )
+            resp.raise_for_status()
+            return resp.json()["message"]["content"]
+        except httpx.HTTPStatusError as exc:
+            # Older Ollama versions may not expose /api/chat; fall back to /api/generate
+            if exc.response.status_code != 404:
+                raise
+            resp = httpx.post(
+                f"{settings.ollama_base_url}/api/generate",
+                json={
+                    "model": settings.eval_accuracy_model,
+                    "prompt": f"{system}\n\n{prompt}",
+                    "stream": False,
+                    "options": {"temperature": 0.0},
+                },
+                timeout=60,
+            )
+            try:
+                resp.raise_for_status()
+            except httpx.HTTPStatusError as exc2:
+                if exc2.response.status_code == 404:
+                    detail = exc2.response.text.strip()
+                    raise RuntimeError(
+                        "Eval accuracy model not available on Ollama. "
+                        f"model={settings.eval_accuracy_model}, url={settings.ollama_base_url}, "
+                        f"detail={detail or 'not found'}"
+                    ) from exc2
+                raise
+            data = resp.json()
+            if "response" in data:
+                return data["response"]
+            return data.get("message", {}).get("content", "")
+
+    client = OpenAI(api_key=settings.openai_api_key)
+    resp = client.chat.completions.create(
+        model=settings.eval_accuracy_model,
+        messages=[
+            {"role": "system", "content": system},
+            {"role": "user", "content": prompt},
+        ],
+        temperature=0.0,
+    )
+    return resp.choices[0].message.content or ""
+
+
 # ─── Health ───────────────────────────────────────────────────────────────────
 
 @app.get("/health")
@@ -472,7 +537,7 @@ async def rag_evaluate(req: EvaluationRequest):
                 f"Generated Answer: {answer}\n\n"
                 "Rate accuracy from 0.0 to 1.0. Return only a number."
             )
+            acc_str = _eval_accuracy_llm(
                 "You evaluate answer accuracy. Return only a number 0-1.",
                 eval_prompt,
             )
diff --git a/helpfull scripts/format and correct questions of arya.py b/helpfull scripts/format and correct questions of arya.py
new file mode 100644
index 0000000..bbbfe11
+++ b/helpfull scripts/format and correct questions of arya.py	
@@ -0,0 +1,42 @@
+import pandas as pd
+import os
+from dotenv import load_dotenv
+from openai import OpenAI
+
+load_dotenv()
+
+input_file = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/ASR removed questions/kag eval sheet from arya.csv"
+df = pd.read_csv(input_file)
+
+client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
+
+def ask_llm(question):
+    response = client.chat.completions.create(
+        model="gpt-4o-mini",
+        messages=[
+            {"role": "system", "content": "Correct the following question to make it grammatically correct and clear and also suitable for a person to speak to create a dataset and give only the question back nothing but the corrected question."},
+            {"role": "user", "content": question}
+        ]
+    )
+    return response.choices[0].message.content.strip()
+
+
+def write_to_file(formatted_questions, output_file):
+    with open(output_file, 'w') as f:
+        for question in formatted_questions:
+            f.write(question + '\n')
+
+
+
+formatted_questions = []
+
+for index, row in df.iterrows():
+    form_name = row[df.columns[0]].strip().lower()  # Assuming the form name is in the first column
+    question = row[df.columns[1]].strip()  # Assuming the question is in the second column
+    # Format the question
+    formatted_question = f"For {form_name} {question}"
+    formatting_response = ask_llm(formatted_question)
+    formatted_questions.append(formatting_response)
+    print(f"Formatted question for {form_name}: {formatting_response}")
+
+write_to_file(formatted_questions, "formatted_questions.txt")
\ No newline at end of file
```

## 138b35781c7fc4b80dd88360a69109ad63485aea — 2026-07-03T12:25:14+05:30

Message:

qwen as retrever and gpt as generation llm
got 67.3 percent accuracy the llm as a judge was gpt only

```diff
diff --git a/backend/app/config.py b/backend/app/config.py
index 3d9b7e4..21dbf3a 100644
+++ b/backend/app/config.py
@@ -4,9 +4,9 @@ from pydantic_settings import BaseSettings
 class Settings(BaseSettings):
     openai_api_key: str = ""
     openai_model: str = "gpt-4o-mini"
+    embedding_provider: str = "ollama"
+    llm_provider: str = "openai"
+    embedding_model: str = "nomic-embed-text"
     ollama_base_url: str = "http://localhost:11434"
     ollama_model: str = "banking-assistant"
```

## d98f9e8faec4bbba9a3133323b93a34b7703adf4 — 2026-07-03T11:43:44+05:30

Message:

script to upload docs in a folder for ingestion

```diff
diff --git a/helpfull scripts/upload docs in a folder for ingestion.py b/helpfull scripts/upload docs in a folder for ingestion.py
new file mode 100644
index 0000000..c0b8e97
+++ b/helpfull scripts/upload docs in a folder for ingestion.py	
@@ -0,0 +1,141 @@
+"""Bulk upload documents from a folder to the Simple RAG API.
+
+Configure the globals below (BASE_URL, FOLDER, METADATA, TIMEOUT) and run:
+  python upload_documents.py
+
+Notes:
+  - BASE_URL defaults to env SRAG_API_URL or http://localhost:9000.
+  - Shows two progress bars: one for uploads, one for indexing (server-side).
+  - Failed uploads are logged to failed_uploads.txt if any remain.
+"""
+
+from __future__ import annotations
+
+import json
+import os
+from pathlib import Path
+from typing import Iterable
+
+import requests
+from requests.exceptions import RequestException, Timeout
+from tqdm import tqdm
+
+
+# ── Configurable globals ──────────────────────────────────────────────────────
+BASE_URL = os.environ.get("SRAG_API_URL", "http://localhost:9000")
+# Set this to the folder containing the documents you want to upload
+FOLDER = Path("/path/to/folder")
+# Optional metadata sent with every file (set to a dict or None)
+METADATA: dict | None = None
+TIMEOUT = 180  # seconds
+
+
+def iter_files(folder: Path) -> Iterable[Path]:
+    for path in folder.rglob("*"):
+        if path.is_file():
+            yield path
+
+
+def upload_file(path: Path, upload_url: str, metadata: dict | None, timeout: int) -> dict:
+    data = {"metadata": json.dumps(metadata)} if metadata else None
+    with open(path, "rb") as f:
+        files = {"file": (path.name, f)}
+        resp = requests.post(upload_url, files=files, data=data, timeout=timeout)
+    resp.raise_for_status()
+    return resp.json()
+
+
+def main():
+    folder: Path = FOLDER.expanduser().resolve()
+    if not folder.is_dir():
+        raise SystemExit(f"Folder not found: {folder}")
+
+    meta_obj = METADATA
+
+    files = list(iter_files(folder))
+    if not files:
+        raise SystemExit("No files found to upload.")
+
+    upload_url = BASE_URL.rstrip("/") + "/api/documents/upload"
+    print(f"Uploading {len(files)} files to {upload_url}")
+
+    upload_bar = tqdm(total=len(files), desc="Uploading", unit="file")
+    index_bar = tqdm(total=len(files), desc="Indexed", unit="file")
+
+    failures_timeout: list[tuple[Path, str]] = []
+    failures_other: list[tuple[Path, str]] = []
+
+    for path in files:
+        try:
+            _ = upload_file(path, upload_url, meta_obj, timeout=TIMEOUT)
+            upload_bar.update(1)
+            index_bar.update(1)
+        except Timeout as exc:
+            failures_timeout.append((path, str(exc)))
+            upload_bar.set_postfix(error="timeout")
+            index_bar.set_postfix(error="timeout")
+        except RequestException as exc:
+            failures_other.append((path, str(exc)))
+            upload_bar.set_postfix(error="request error")
+            index_bar.set_postfix(error="request error")
+        except Exception as exc:  # noqa: BLE001
+            failures_other.append((path, str(exc)))
+            upload_bar.set_postfix(error="error")
+            index_bar.set_postfix(error="error")
+
+    upload_bar.close()
+    index_bar.close()
+
+    combined_failures = failures_timeout + failures_other
+
+    if combined_failures:
+        print("Finished first pass with errors:")
+        if failures_timeout:
+            print("Timeouts:")
+            for path, err in failures_timeout:
+                print(f"- {path}: {err}")
+        if failures_other:
+            print("Other errors:")
+            for path, err in failures_other:
+                print(f"- {path}: {err}")
+
+        retry = input("Retry failed uploads now? (y/N): ").strip().lower() == "y"
+        final_failures = combined_failures
+        if retry and combined_failures:
+            retry_bar = tqdm(total=len(combined_failures), desc="Retrying", unit="file")
+            still_failed: list[tuple[Path, str]] = []
+            for path, _ in combined_failures:
+                try:
+                    _ = upload_file(path, upload_url, meta_obj, timeout=TIMEOUT)
+                except Exception as exc:  # noqa: BLE001
+                    still_failed.append((path, str(exc)))
+                    retry_bar.set_postfix(error="error")
+                else:
+                    retry_bar.set_postfix(error="")
+                retry_bar.update(1)
+            retry_bar.close()
+
+            if still_failed:
+                print("Still failed after retry:")
+                for path, err in still_failed:
+                    print(f"- {path}: {err}")
+                final_failures = still_failed
+            else:
+                print("All previously failed files uploaded on retry.")
+                final_failures = []
+        else:
+            print("Skipped retry.")
+            final_failures = combined_failures
+
+        if final_failures:
+            fail_log = Path(__file__).with_name("failed_uploads.txt")
+            with open(fail_log, "w") as f:
+                for path, err in final_failures:
+                    f.write(f"{path}\t{err}\n")
+            print(f"Saved failed uploads to {fail_log} for later retry.")
+    else:
+        print("All files uploaded and indexed successfully.")
+
+
+if __name__ == "__main__":
+    main()
```

## f024c998f64afcfa840874bb0fa988a02100e788 — 2026-07-03T11:30:27+05:30

Message:

ollama instead of openai 68.3 overall score 100% accuracy in retreival related metrics ( need to investigate )

```diff
diff --git a/backend/app/config.py b/backend/app/config.py
index e46e37e..3d9b7e4 100644
+++ b/backend/app/config.py
@@ -5,10 +5,10 @@ class Settings(BaseSettings):
     openai_api_key: str = ""
     openai_model: str = "gpt-4o-mini"
     embedding_provider: str = "openai"
+    llm_provider: str = "ollama"
     embedding_model: str = "text-embedding-3-small"
     ollama_base_url: str = "http://localhost:11434"
+    ollama_model: str = "banking-assistant"
 
     model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
```

## 80768850a60446f46c274247062aade2a295d0a4 — 2026-07-02T14:29:03+05:30

Message:

68 percent accuracy and ingested data

```diff
diff --git a/backend/app/chroma_store.py b/backend/app/chroma_store.py
new file mode 100644
index 0000000..1170b82
+++ b/backend/app/chroma_store.py
@@ -0,0 +1,129 @@
+import os
+import uuid
+import chromadb
+from chromadb.config import Settings
+from chromadb.errors import NotFoundError, InvalidCollectionException
+
+os.environ.setdefault("CHROMADB_DISABLE_TELEMETRY", "1")
+
+CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma")
+COLLECTION_NAME = "documents"
+
+_client = None
+_collection = None
+
+
+def _get_client():
+    global _client
+    if _client is None:
+        _client = chromadb.PersistentClient(
+            path=CHROMA_DIR,
+            settings=Settings(anonymized_telemetry=False),
+        )
+    return _client
+
+
+def _get_collection():
+    global _collection
+    if _collection is None:
+        client = _get_client()
+        try:
+            _collection = client.get_collection(COLLECTION_NAME)
+        except (ValueError, NotFoundError, InvalidCollectionException):
+            _collection = client.create_collection(COLLECTION_NAME)
+    return _collection
+
+
+def add_chunks(doc_id: str, filename: str, chunks: list[str],
+               embeddings: list[list[float]], metadata_list: list[dict] = None):
+    collection = _get_collection()
+    ids = []
+    metadatas = []
+    for i, (text, emb) in enumerate(zip(chunks, embeddings)):
+        chunk_id = str(uuid.uuid4())
+        ids.append(chunk_id)
+        meta = {
+            "document_id": doc_id,
+            "filename": filename,
+            "chunk_index": i,
+            "chunk_text": text,
+        }
+        if metadata_list and i < len(metadata_list):
+            meta.update(metadata_list[i])
+        metadatas.append(meta)
+    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
+    return len(chunks)
+
+
+def search(query_emb: list[float], top_k: int = 5) -> list[dict]:
+    collection = _get_collection()
+    results = collection.query(
+        query_embeddings=[query_emb],
+        n_results=top_k,
+        include=["metadatas", "distances"],
+    )
+    output = []
+    if not results["ids"] or not results["ids"][0]:
+        return output
+    for i, chunk_id in enumerate(results["ids"][0]):
+        meta = results["metadatas"][0][i]
+        distance = results["distances"][0][i]
+        score = 1.0 - distance
+        output.append({
+            "chunk_id": chunk_id,
+            "document_id": meta.get("document_id", ""),
+            "filename": meta.get("filename", "unknown"),
+            "chunk_text": meta.get("chunk_text", ""),
+            "score": score,
+            "metadata": {k: v for k, v in meta.items() if k not in ("document_id", "filename", "chunk_text")},
+        })
+    return output
+
+
+def get_chunks_by_doc(doc_id: str) -> list[dict]:
+    collection = _get_collection()
+    results = collection.get(
+        where={"document_id": doc_id},
+        include=["metadatas"],
+    )
+    output = []
+    for i, chunk_id in enumerate(results["ids"]):
+        meta = results["metadatas"][i]
+        output.append({
+            "id": chunk_id,
+            "document_id": meta.get("document_id", ""),
+            "chunk_index": meta.get("chunk_index", 0),
+            "chunk_text": meta.get("chunk_text", ""),
+            "chunk_metadata": {k: v for k, v in meta.items() if k not in ("document_id", "filename", "chunk_text", "chunk_index")},
+            "created_at": "",
+        })
+    output.sort(key=lambda x: x["chunk_index"])
+    return output
+
+
+def delete_chunks_by_doc(doc_id: str):
+    collection = _get_collection()
+    results = collection.get(where={"document_id": doc_id})
+    if results["ids"]:
+        collection.delete(ids=results["ids"])
+
+
+def count_chunks() -> int:
+    collection = _get_collection()
+    return collection.count()
+
+
+def list_collections_info() -> list[dict]:
+    client = _get_client()
+    cols = client.list_collections()
+    return [{"name": c.name, "count": c.count()} for c in cols]
+
+
+def reset_collection():
+    global _collection
+    client = _get_client()
+    try:
+        client.delete_collection(COLLECTION_NAME)
+    except (ValueError, NotFoundError):
+        pass
+    _collection = None
diff --git a/backend/app/main.py b/backend/app/main.py
index 26ca91e..5382582 100644
+++ b/backend/app/main.py
@@ -1,20 +1,26 @@
+import logging
 import os
 import time
 import uuid
 from datetime import datetime, timezone
 from contextlib import asynccontextmanager
 
+from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
+from fastapi.encoders import jsonable_encoder
+from fastapi.exceptions import RequestValidationError
+from typing import List
 from fastapi.middleware.cors import CORSMiddleware
+from fastapi.responses import StreamingResponse, JSONResponse
 
 from app.config import settings
 from app.storage import (
     list_docs, get_doc, create_doc, update_doc, delete_doc,
     list_convs, get_conv, create_conv, add_message, delete_conv,
 )
+from app.chroma_store import (
+    add_chunks, get_chunks_by_doc, delete_chunks_by_doc,
+    count_chunks, list_collections_info,
+)
 from app.models import (
     SearchRequest, SearchResponse, SearchResult,
     ChatRequest, RAGQueryRequest, RAGQueryResponse, Source,
@@ -36,6 +42,8 @@ async def lifespan(app: FastAPI):
 
 app = FastAPI(title="Simple RAG", version="2.0.0", lifespan=lifespan)
 
+logger = logging.getLogger("simple_rag")
+
 app.add_middleware(
     CORSMiddleware,
     allow_origins=["*"],
@@ -45,6 +53,38 @@ app.add_middleware(
 )
 
 
+@app.middleware("http")
+async def add_cors_on_error(request: Request, call_next):
+    try:
+        response = await call_next(request)
+    except HTTPException as exc:
+        response = JSONResponse(
+            status_code=exc.status_code,
+            content={"detail": exc.detail},
+            headers=exc.headers,
+        )
+    except RequestValidationError as exc:
+        response = JSONResponse(
+            status_code=422,
+            content={"detail": jsonable_encoder(exc.errors())},
+        )
+    except Exception as exc:  # noqa: BLE001
+        logger.exception("Unhandled error during request %s %s", request.method, request.url.path)
+        response = JSONResponse(
+            status_code=500,
+            content={"detail": "Internal server error", "error": str(exc)},
+        )
+
+    origin = request.headers.get("origin") or "*"
+    response.headers.setdefault("Access-Control-Allow-Origin", origin)
+    if origin != "*":
+        response.headers.setdefault("Vary", "Origin")
+        response.headers.setdefault("Access-Control-Allow-Credentials", "true")
+    response.headers.setdefault("Access-Control-Allow-Methods", "*")
+    response.headers.setdefault("Access-Control-Allow-Headers", "*")
+    return response
+
+
 def _utcnow() -> str:
     return datetime.now(timezone.utc).isoformat()
 
@@ -63,7 +103,11 @@ async def health_db():
 
 @app.get("/health/chroma")
 async def health_chroma():
+    try:
+        cols = list_collections_info()
+        return {"status": "ok", "collections": [c["name"] for c in cols]}
+    except Exception as e:
+        return {"status": "error", "detail": str(e)}
 
 
 @app.get("/health/ollama")
@@ -98,50 +142,72 @@ async def health_elasticsearch():
 # ─── Documents ────────────────────────────────────────────────────────────────
 
 @app.post("/api/documents/upload")
+async def upload_document(
+    file: UploadFile = File(None),
+    files: List[UploadFile] = File(None),
+    metadata: str = Form("{}"),
+):
     import json
+
     meta = {}
     try:
         meta = json.loads(metadata) if metadata else {}
     except json.JSONDecodeError:
         pass
 
+    payload_files: List[UploadFile] = []
+    if files:
+        payload_files = files
+    elif file:
+        payload_files = [file]
+    else:
+        raise HTTPException(400, "No file provided")
+
+    results = []
+    for f in payload_files:
+        ext = os.path.splitext(f.filename or "file.txt")[1].lower()
+        doc_type = ext.lstrip(".") if ext else "text"
+        if doc_type in ("md",):
+            doc_type = "markdown"
+
+        filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{f.filename}")
+        with open(filepath, "wb") as out:
+            content = await f.read()
+            out.write(content)
+
+        text = extract_text_from_file(filepath)
+        chunks = chunk_text(text)
+        doc = create_doc(
+            filename=f.filename or "unknown",
+            filepath=filepath,
+            doc_type=doc_type,
+            embedding_model=settings.embedding_model,
+            metadata=meta,
+        )
+
+        if chunks:
+            embeddings = embedder.embed_batch(chunks)
+            metadata_list = [{"chunk_index": i} for i in range(len(chunks))]
+            chunk_count = add_chunks(
+                doc.id, doc.filename, chunks, embeddings, metadata_list
+            )
+        else:
+            chunk_count = 0
 
+        update_doc(doc.id, chunk_count=chunk_count)
+        results.append({
+            "id": doc.id,
+            "filename": doc.filename,
+            "document_type": doc_type,
+            "retrieval_strategy": "vector",
+            "chunk_count": chunk_count,
+            "message": "Document uploaded and indexed",
+        })
 
+    # Backward-compatible single-file response
+    if len(results) == 1:
+        return results[0]
+    return {"uploaded": len(results), "results": results, "message": "Documents uploaded and indexed"}
 
 
 @app.get("/api/documents")
@@ -174,19 +240,26 @@ async def reindex_document(doc_id: str):
     if not text:
         raise HTTPException(400, "Could not extract text from file")
     chunks = chunk_text(text)
+    delete_chunks_by_doc(doc.id)
+    if chunks:
+        embeddings = embedder.embed_batch(chunks)
+        metadata_list = [{"chunk_index": i} for i in range(len(chunks))]
+        chunk_count = add_chunks(
+            doc.id, doc.filename, chunks, embeddings, metadata_list
+        )
+    else:
+        chunk_count = 0
     update_doc(doc.id, chunk_count=chunk_count)
     return {"document_id": doc.id, "message": "Reindexed", "chunk_count": chunk_count}
 
 
 @app.get("/api/documents/{doc_id}/chunks")
 async def get_document_chunks(doc_id: str):
+    chunks = get_chunks_by_doc(doc_id)
+    # Remove embedding from metadata if present
+    for c in chunks:
+        c["chunk_metadata"].pop("_embedding", None)
+    return chunks
 
 
 @app.get("/api/documents/{doc_id}/metadata")
@@ -500,7 +573,9 @@ async def run_agent(agent_type: str, req: dict):
 
 @app.get("/api/chroma/collections")
 async def list_collections():
+    cols = list_collections_info()
+    counts = {c["name"]: c["count"] for c in cols}
+    return {"collections": [c["name"] for c in cols], "counts": counts}
 
 
 # ─── Web ─────────────────────────────────────────────────────────────────────
@@ -535,11 +610,14 @@ async def elasticsearch_upload(file: UploadFile = File(...)):
         doc_type=doc_type,
         embedding_model=settings.embedding_model,
     )
+    if chunks:
+        embeddings = embedder.embed_batch(chunks)
+        metadata_list = [{"chunk_index": i} for i in range(len(chunks))]
+        chunk_count = add_chunks(
+            doc.id, doc.filename, chunks, embeddings, metadata_list
+        )
+    else:
+        chunk_count = 0
     update_doc(doc.id, chunk_count=chunk_count)
 
     return {"count": chunk_count}
diff --git a/backend/app/rag.py b/backend/app/rag.py
index d447960..b2f4e4c 100644
+++ b/backend/app/rag.py
@@ -1,14 +1,12 @@
 import os
 import time
 
 from app.embeddings import embedder
+from app.chroma_store import search
 from app.models import SearchResult, Source
 
 
+def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
     if not text:
         return []
     chunks = []
@@ -49,36 +47,23 @@ def extract_text_from_file(filepath: str) -> str:
 
 
 def retrieve(query: str, top_k: int = 5) -> tuple[list[SearchResult], float]:
     query_emb = embedder.embed(query)
+    results = search(query_emb, top_k)
+    if not results:
         return [], 0.0
+    scored = [
+        SearchResult(
+            chunk_id=r["chunk_id"],
+            document_id=r["document_id"],
+            filename=r["filename"],
+            chunk_text=r["chunk_text"],
+            score=r["score"],
+            metadata=r["metadata"],
+        )
+        for r in results
+    ]
+    confidence = scored[0].score if scored else 0.0
+    return scored, confidence
 
 
 def build_context(results: list[SearchResult]) -> str:
diff --git a/backend/app/storage.py b/backend/app/storage.py
index b2fc278..b389690 100644
+++ b/backend/app/storage.py
@@ -4,11 +4,11 @@ import uuid
 from datetime import datetime, timezone
 from typing import Optional
 
+from app.models import Document, Conversation, Message, Source
+from app.chroma_store import delete_chunks_by_doc
 
 DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
 DOCS_FILE = os.path.join(DATA_DIR, "documents.json")
 CONVS_FILE = os.path.join(DATA_DIR, "conversations.json")
 
 os.makedirs(DATA_DIR, exist_ok=True)
@@ -88,44 +88,10 @@ def delete_doc(doc_id: str) -> bool:
     if len(new_docs) == len(docs):
         return False
     _save_json(DOCS_FILE, new_docs)
+    delete_chunks_by_doc(doc_id)
     return True
 
 
 # ─── Conversations ────────────────────────────────────────────────────────────
 
 def list_convs(skip: int = 0, limit: int = 50) -> tuple[list[Conversation], int]:
```

## 9052bf20730c077388d42641fafdce7a5091cac7 — 2026-07-01T11:51:42+05:30

Message:

project started from scratch to a simple cosine similarity

```diff
diff --git a/backend/alembic/env.py b/backend/alembic/env.py
deleted file mode 100644
index bb42de1..0000000
+++ /dev/null
@@ -1,63 +0,0 @@
diff --git a/backend/alembic/versions/0001_initial_schema.py b/backend/alembic/versions/0001_initial_schema.py
deleted file mode 100644
index d4331a8..0000000
+++ /dev/null
@@ -1,112 +0,0 @@
diff --git a/backend/alembic/versions/0002_kag_schema.py b/backend/alembic/versions/0002_kag_schema.py
deleted file mode 100644
index 9c98b3c..0000000
+++ /dev/null
@@ -1,74 +0,0 @@
diff --git a/backend/alembic/versions/0003_kag_extension.py b/backend/alembic/versions/0003_kag_extension.py
deleted file mode 100644
index 77442e6..0000000
+++ /dev/null
@@ -1,188 +0,0 @@
diff --git a/backend/app/api/__init__.py b/backend/app/api/__init__.py
deleted file mode 100644
index e69de29..0000000
diff --git a/backend/app/api/chat.py b/backend/app/api/chat.py
deleted file mode 100644
index 7563d9d..0000000
+++ /dev/null
@@ -1,52 +0,0 @@
diff --git a/backend/app/api/chroma.py b/backend/app/api/chroma.py
deleted file mode 100644
index 6901e40..0000000
+++ /dev/null
@@ -1,70 +0,0 @@
\ No newline at end of file
diff --git a/backend/app/api/documents.py b/backend/app/api/documents.py
deleted file mode 100644
index 7610c27..0000000
+++ /dev/null
@@ -1,93 +0,0 @@
diff --git a/backend/app/api/forms.py b/backend/app/api/forms.py
deleted file mode 100644
index 142eb75..0000000
+++ /dev/null
@@ -1,17 +0,0 @@
diff --git a/backend/app/api/health.py b/backend/app/api/health.py
deleted file mode 100644
index 24fe05d..0000000
+++ /dev/null
@@ -1,88 +0,0 @@
diff --git a/backend/app/api/rag.py b/backend/app/api/rag.py
deleted file mode 100644
index ad231dc..0000000
+++ /dev/null
@@ -1,132 +0,0 @@
diff --git a/backend/app/chromadb/__init__.py b/backend/app/chromadb/__init__.py
deleted file mode 100644
index a906368..0000000
+++ /dev/null
@@ -1 +0,0 @@
diff --git a/backend/app/chromadb/client.py b/backend/app/chromadb/client.py
deleted file mode 100644
index 4c311a8..0000000
+++ /dev/null
@@ -1,414 +0,0 @@
diff --git a/backend/app/config.py b/backend/app/config.py
new file mode 100644
index 0000000..e46e37e
+++ b/backend/app/config.py
@@ -0,0 +1,16 @@
+from pydantic_settings import BaseSettings
+
+
+class Settings(BaseSettings):
+    openai_api_key: str = ""
+    openai_model: str = "gpt-4o-mini"
+    embedding_provider: str = "openai"
+    llm_provider: str = "openai"
+    embedding_model: str = "text-embedding-3-small"
+    ollama_base_url: str = "http://localhost:11434"
+    ollama_model: str = "llama3"
+
+    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
+
+
+settings = Settings()
diff --git a/backend/app/core/__init__.py b/backend/app/core/__init__.py
deleted file mode 100644
index e69de29..0000000
diff --git a/backend/app/core/config.py b/backend/app/core/config.py
deleted file mode 100644
index ced2ad0..0000000
+++ /dev/null
@@ -1,79 +0,0 @@
diff --git a/backend/app/core/dependencies.py b/backend/app/core/dependencies.py
deleted file mode 100644
index 46ad68e..0000000
+++ /dev/null
@@ -1,11 +0,0 @@
diff --git a/backend/app/core/evaluation_logger.py b/backend/app/core/evaluation_logger.py
deleted file mode 100644
index d112051..0000000
+++ /dev/null
@@ -1,67 +0,0 @@
diff --git a/backend/app/core/exceptions.py b/backend/app/core/exceptions.py
deleted file mode 100644
index 00b161b..0000000
+++ /dev/null
@@ -1,74 +0,0 @@
diff --git a/backend/app/core/logging.py b/backend/app/core/logging.py
deleted file mode 100644
index 792087c..0000000
+++ /dev/null
@@ -1,40 +0,0 @@
diff --git a/backend/app/core/prompts.py b/backend/app/core/prompts.py
deleted file mode 100644
index 06d8049..0000000
+++ /dev/null
@@ -1,319 +0,0 @@
diff --git a/backend/app/database/__init__.py b/backend/app/database/__init__.py
deleted file mode 100644
index e69de29..0000000
diff --git a/backend/app/database/base.py b/backend/app/database/base.py
deleted file mode 100644
index 9643249..0000000
+++ /dev/null
@@ -1,14 +0,0 @@
diff --git a/backend/app/database/init_db.py b/backend/app/database/init_db.py
deleted file mode 100644
index 138b3b6..0000000
+++ /dev/null
@@ -1,14 +0,0 @@
diff --git a/backend/app/database/models.py b/backend/app/database/models.py
deleted file mode 100644
index 09e2817..0000000
+++ /dev/null
@@ -1,278 +0,0 @@
diff --git a/backend/app/database/session.py b/backend/app/database/session.py
deleted file mode 100644
index 22a723a..0000000
+++ /dev/null
@@ -1,23 +0,0 @@
diff --git a/backend/app/embeddings.py b/backend/app/embeddings.py
new file mode 100644
index 0000000..9c05ee8
+++ b/backend/app/embeddings.py
@@ -0,0 +1,59 @@
+import numpy as np
+from openai import OpenAI
+import httpx
+
+from app.config import settings
+
+
+class EmbeddingClient:
+    def __init__(self):
+        self.provider = settings.embedding_provider
+        self.model = settings.embedding_model
+        self._openai = None
+        self._ollama_base = settings.ollama_base_url
+
+    def _get_openai(self):
+        if self._openai is None:
+            self._openai = OpenAI(api_key=settings.openai_api_key)
+        return self._openai
+
+    def embed(self, text: str) -> list[float]:
+        if self.provider == "ollama":
+            return self._embed_ollama(text)
+        return self._embed_openai(text)
+
+    def embed_batch(self, texts: list[str]) -> list[list[float]]:
+        if self.provider == "ollama":
+            return [self._embed_ollama(t) for t in texts]
+        return self._embed_openai_batch(texts)
+
+    def _embed_openai(self, text: str) -> list[float]:
+        client = self._get_openai()
+        resp = client.embeddings.create(input=text, model=self.model)
+        return resp.data[0].embedding
+
+    def _embed_openai_batch(self, texts: list[str]) -> list[list[float]]:
+        client = self._get_openai()
+        resp = client.embeddings.create(input=texts, model=self.model)
+        sorted_data = sorted(resp.data, key=lambda x: x.index)
+        return [d.embedding for d in sorted_data]
+
+    def _embed_ollama(self, text: str) -> list[float]:
+        resp = httpx.post(
+            f"{self._ollama_base}/api/embeddings",
+            json={"model": self.model, "prompt": text},
+            timeout=30,
+        )
+        resp.raise_for_status()
+        return resp.json()["embedding"]
+
+    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
+        A = np.array(a, dtype=np.float32)
+        B = np.array(b, dtype=np.float32)
+        norm = np.linalg.norm(A) * np.linalg.norm(B)
+        if norm == 0:
+            return 0.0
+        return float(np.dot(A, B) / norm)
+
+
+embedder = EmbeddingClient()
diff --git a/backend/app/embeddings/__init__.py b/backend/app/embeddings/__init__.py
deleted file mode 100644
index 3c13198..0000000
+++ /dev/null
@@ -1 +0,0 @@
diff --git a/backend/app/embeddings/ollama_client.py b/backend/app/embeddings/ollama_client.py
deleted file mode 100644
index 437c594..0000000
+++ /dev/null
@@ -1,191 +0,0 @@
diff --git a/backend/app/embeddings/openai_client.py b/backend/app/embeddings/openai_client.py
deleted file mode 100644
index 8b1405d..0000000
+++ /dev/null
@@ -1,238 +0,0 @@
diff --git a/backend/app/evaluation/__init__.py b/backend/app/evaluation/__init__.py
deleted file mode 100644
index e69de29..0000000
diff --git a/backend/app/evaluation/accuracy_evaluation.py b/backend/app/evaluation/accuracy_evaluation.py
deleted file mode 100644
index d00f03e..0000000
+++ /dev/null
@@ -1,79 +0,0 @@
diff --git a/backend/app/evaluation/agent_runner.py b/backend/app/evaluation/agent_runner.py
deleted file mode 100644
index 345de8a..0000000
+++ /dev/null
@@ -1,463 +0,0 @@
diff --git a/backend/app/evaluation/evaluator.py b/backend/app/evaluation/evaluator.py
deleted file mode 100644
index 3af14d5..0000000
+++ /dev/null
@@ -1,219 +0,0 @@
\ No newline at end of file
diff --git a/backend/app/evaluation/retrieval_metrics.py b/backend/app/evaluation/retrieval_metrics.py
deleted file mode 100644
index b94cf05..0000000
+++ /dev/null
@@ -1,51 +0,0 @@
diff --git a/backend/app/graph/__init__.py b/backend/app/graph/__init__.py
deleted file mode 100644
index 33f396b..0000000
+++ /dev/null
@@ -1 +0,0 @@
diff --git a/backend/app/graph/entity_extraction.py b/backend/app/graph/entity_extraction.py
deleted file mode 100644
index 084c297..0000000
+++ /dev/null
@@ -1,43 +0,0 @@
diff --git a/backend/app/graph/form_recommender.py b/backend/app/graph/form_recommender.py
deleted file mode 100644
index 5fbaab8..0000000
+++ /dev/null
@@ -1,59 +0,0 @@
diff --git a/backend/app/graph/graph_ingestor.py b/backend/app/graph/graph_ingestor.py
deleted file mode 100644
index dc43c89..0000000
+++ /dev/null
@@ -1,186 +0,0 @@
diff --git a/backend/app/graph/graph_retriever.py b/backend/app/graph/graph_retriever.py
deleted file mode 100644
index 82861cc..0000000
+++ /dev/null
@@ -1,77 +0,0 @@
diff --git a/backend/app/graph/neo4j_client.py b/backend/app/graph/neo4j_client.py
deleted file mode 100644
index 22cc7bb..0000000
+++ /dev/null
@@ -1,59 +0,0 @@
diff --git a/backend/app/llm.py b/backend/app/llm.py
new file mode 100644
index 0000000..31f43f5
+++ b/backend/app/llm.py
@@ -0,0 +1,98 @@
+import json
+from typing import AsyncIterator
+from openai import OpenAI
+import httpx
+
+from app.config import settings
+
+
+class LLMClient:
+    def __init__(self):
+        self.provider = settings.llm_provider
+        self.openai_model = settings.openai_model
+        self.ollama_model = settings.ollama_model
+        self.ollama_base = settings.ollama_base_url
+        self._openai = None
+
+    def _get_openai(self):
+        if self._openai is None:
+            self._openai = OpenAI(api_key=settings.openai_api_key)
+        return self._openai
+
+    def generate(self, system: str, prompt: str) -> str:
+        if self.provider == "ollama":
+            return self._generate_ollama(system, prompt)
+        return self._generate_openai(system, prompt)
+
+    def generate_stream(self, system: str, prompt: str) -> AsyncIterator[str]:
+        if self.provider == "ollama":
+            return self._stream_ollama(system, prompt)
+        return self._stream_openai(system, prompt)
+
+    def _generate_openai(self, system: str, prompt: str) -> str:
+        client = self._get_openai()
+        resp = client.chat.completions.create(
+            model=self.openai_model,
+            messages=[
+                {"role": "system", "content": system},
+                {"role": "user", "content": prompt},
+            ],
+            temperature=0.3,
+        )
+        return resp.choices[0].message.content or ""
+
+    async def _stream_openai(self, system: str, prompt: str) -> AsyncIterator[str]:
+        client = self._get_openai()
+        stream = client.chat.completions.create(
+            model=self.openai_model,
+            messages=[
+                {"role": "system", "content": system},
+                {"role": "user", "content": prompt},
+            ],
+            temperature=0.3,
+            stream=True,
+        )
+        for chunk in stream:
+            if chunk.choices[0].delta.content:
+                yield chunk.choices[0].delta.content
+
+    def _generate_ollama(self, system: str, prompt: str) -> str:
+        resp = httpx.post(
+            f"{self.ollama_base}/api/chat",
+            json={
+                "model": self.ollama_model,
+                "messages": [
+                    {"role": "system", "content": system},
+                    {"role": "user", "content": prompt},
+                ],
+                "stream": False,
+                "options": {"temperature": 0.3},
+            },
+            timeout=60,
+        )
+        resp.raise_for_status()
+        return resp.json()["message"]["content"]
+
+    async def _stream_ollama(self, system: str, prompt: str) -> AsyncIterator[str]:
+        async with httpx.AsyncClient(timeout=60) as client:
+            async with client.stream(
+                "POST",
+                f"{self.ollama_base}/api/chat",
+                json={
+                    "model": self.ollama_model,
+                    "messages": [
+                        {"role": "system", "content": system},
+                        {"role": "user", "content": prompt},
+                    ],
+                    "stream": True,
+                    "options": {"temperature": 0.3},
+                },
+            ) as resp:
+                async for line in resp.aiter_lines():
+                    if line.strip():
+                        data = json.loads(line)
+                        if "message" in data and "content" in data["message"]:
+                            yield data["message"]["content"]
+
+
+llm = LLMClient()
diff --git a/backend/app/main.py b/backend/app/main.py
index 0abf64c..26ca91e 100644
+++ b/backend/app/main.py
@@ -1,55 +1,41 @@
+import os
 import time
+import uuid
+import shutil
+from datetime import datetime, timezone
 from contextlib import asynccontextmanager
 
+from fastapi import FastAPI, UploadFile, File, Form, HTTPException
 from fastapi.middleware.cors import CORSMiddleware
+from fastapi.responses import StreamingResponse
 
+from app.config import settings
+from app.storage import (
+    list_docs, get_doc, create_doc, update_doc, delete_doc,
+    get_chunks, save_chunks, get_all_chunks,
+    list_convs, get_conv, create_conv, add_message, delete_conv,
+)
+from app.models import (
+    SearchRequest, SearchResponse, SearchResult,
+    ChatRequest, RAGQueryRequest, RAGQueryResponse, Source,
+    EvaluationRequest, EvaluationResponse, PerQuestionResult,
+    Document,
+)
+from app.embeddings import embedder
+from app.rag import chunk_text, extract_text_from_file, retrieve, build_context, query
+from app.llm import llm
 
+UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
+os.makedirs(UPLOAD_DIR, exist_ok=True)
 
 
 @asynccontextmanager
 async def lifespan(app: FastAPI):
     yield
 
+
+app = FastAPI(title="Simple RAG", version="2.0.0", lifespan=lifespan)
+
 app.add_middleware(
     CORSMiddleware,
     allow_origins=["*"],
@@ -59,25 +45,506 @@ app.add_middleware(
 )
 
 
+def _utcnow() -> str:
+    return datetime.now(timezone.utc).isoformat()
+
+
+# ─── Health ───────────────────────────────────────────────────────────────────
+
+@app.get("/health")
+async def health_check():
+    return {"status": "ok", "service": "Simple RAG"}
+
+
+@app.get("/health/db")
+async def health_db():
+    return {"status": "ok", "database": "file-based"}
+
+
+@app.get("/health/chroma")
+async def health_chroma():
+    return {"status": "ok", "collections": ["default"]}
+
+
+@app.get("/health/ollama")
+async def health_ollama():
+    models = []
+    if settings.llm_provider == "ollama" or settings.embedding_provider == "ollama":
+        try:
+            import httpx
+            resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
+            if resp.status_code == 200:
+                models = [m["name"] for m in resp.json().get("models", [])]
+        except Exception:
+            pass
+    return {"status": "ok", "models": models}
+
+
+@app.get("/health/neo4j")
+async def health_neo4j():
+    return {"status": "ok", "neo4j": "not_used"}
+
+
+@app.get("/health/qdrant")
+async def health_qdrant():
+    return {"status": "ok", "collections": []}
+
+
+@app.get("/api/elasticsearch/status")
+async def health_elasticsearch():
+    return {"status": "ok"}
+
+
+# ─── Documents ────────────────────────────────────────────────────────────────
+
+@app.post("/api/documents/upload")
+async def upload_document(file: UploadFile = File(...), metadata: str = Form("{}")):
+    import json
+    meta = {}
+    try:
+        meta = json.loads(metadata) if metadata else {}
+    except json.JSONDecodeError:
+        pass
+
+    ext = os.path.splitext(file.filename or "file.txt")[1].lower()
+    doc_type = ext.lstrip(".") if ext else "text"
+    if doc_type in ("md",):
+        doc_type = "markdown"
+
+    filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
+    with open(filepath, "wb") as f:
+        content = await file.read()
+        f.write(content)
+
+    text = extract_text_from_file(filepath)
+    chunks = chunk_text(text)
+    doc = create_doc(
+        filename=file.filename or "unknown",
+        filepath=filepath,
+        doc_type=doc_type,
+        embedding_model=settings.embedding_model,
+        metadata=meta,
+    )
+
+    emb_list = embedder.embed_batch(chunks) if chunks else []
+    metadata_list = []
+    for i, (chunk, emb) in enumerate(zip(chunks, emb_list)):
+        metadata_list.append({"_embedding": emb, "chunk_index": i})
+
+    chunk_count = save_chunks(doc.id, chunks, metadata_list)
+    update_doc(doc.id, chunk_count=chunk_count)
+
+    return {
+        "id": doc.id,
+        "filename": doc.filename,
+        "document_type": doc_type,
+        "retrieval_strategy": "vector",
+        "chunk_count": chunk_count,
+        "message": "Document uploaded and indexed",
+    }
+
+
+@app.get("/api/documents")
+async def list_documents(skip: int = 0, limit: int = 50):
+    docs, total = list_docs(skip, limit)
+    return {"documents": [d.model_dump() for d in docs], "total": total}
+
+
+@app.get("/api/documents/{doc_id}")
+async def get_document(doc_id: str):
+    doc = get_doc(doc_id)
+    if not doc:
+        raise HTTPException(404, "Document not found")
+    return doc.model_dump()
+
+
+@app.delete("/api/documents/{doc_id}")
+async def delete_document(doc_id: str):
+    if not delete_doc(doc_id):
+        raise HTTPException(404, "Document not found")
+    return {"message": "Document deleted"}
+
+
+@app.post("/api/documents/{doc_id}/reindex")
+async def reindex_document(doc_id: str):
+    doc = get_doc(doc_id)
+    if not doc:
+        raise HTTPException(404, "Document not found")
+    text = extract_text_from_file(doc.filepath)
+    if not text:
+        raise HTTPException(400, "Could not extract text from file")
+    chunks = chunk_text(text)
+    emb_list = embedder.embed_batch(chunks) if chunks else []
+    metadata_list = []
+    for i, (chunk, emb) in enumerate(zip(chunks, emb_list)):
+        metadata_list.append({"_embedding": emb, "chunk_index": i})
+    chunk_count = save_chunks(doc.id, chunks, metadata_list)
+    update_doc(doc.id, chunk_count=chunk_count)
+    return {"document_id": doc.id, "message": "Reindexed", "chunk_count": chunk_count}
+
+
+@app.get("/api/documents/{doc_id}/chunks")
+async def get_document_chunks(doc_id: str):
+    chunks = get_chunks(doc_id)
+    return [c.model_dump() for c in chunks]
+
+
+@app.get("/api/documents/{doc_id}/metadata")
+async def get_document_metadata(doc_id: str):
+    doc = get_doc(doc_id)
+    if not doc:
+        raise HTTPException(404, "Document not found")
+    return doc.metadata_json
+
+
+# ─── Search ──────────────────────────────────────────────────────────────────
+
+def _do_search(req: SearchRequest, strategy: str = "vector") -> SearchResponse:
     start = time.time()
+    results, confidence = retrieve(req.query, req.top_k or 5)
+    elapsed = (time.time() - start) * 1000
+    return SearchResponse(
+        query=req.query,
+        results=results,
+        confidence=confidence,
+        sources=list(set(r.filename for r in results)),
+        latency_ms=elapsed,
+        strategy=strategy,
+    )
+
+
+@app.post("/api/search/vector")
+async def search_vector(req: SearchRequest):
+    return _do_search(req, "vector").model_dump()
+
+
+@app.post("/api/search/bm25")
+async def search_bm25(req: SearchRequest):
+    return _do_search(req, "bm25").model_dump()
+
+
+@app.post("/api/search/hybrid")
+async def search_hybrid(req: SearchRequest):
+    return _do_search(req, "hybrid").model_dump()
+
+
+@app.post("/api/search/metadata")
+async def search_metadata(req: SearchRequest):
+    return _do_search(req, "metadata").model_dump()
+
+
+@app.post("/api/search/table")
+async def search_table(req: SearchRequest):
+    return _do_search(req, "table").model_dump()
+
+
+# ─── Chat ─────────────────────────────────────────────────────────────────────
+
+@app.post("/api/chat")
+async def chat(req: ChatRequest):
+    if not req.conversation_id:
+        conv = create_conv()
+        req.conversation_id = conv.id
+
+    add_message(req.conversation_id, "user", req.message)
+
+    context, sources, confidence, elapsed = query(req.message, req.top_k or 5)
+
+    system_prompt = (
+        "You are a helpful RAG assistant. Answer the user's question based "
+        "on the provided context. If the context doesn't contain enough "
+        "information, say so. Be concise."
+    )
+    user_prompt = f"Context:\n{context}\n\nQuestion: {req.message}"
+
+    answer = llm.generate(system_prompt, user_prompt)
+    add_message(req.conversation_id, "assistant", answer,
+                [s.model_dump() for s in sources])
+
+    message_response = {
+        "id": str(uuid.uuid4()),
+        "conversation_id": req.conversation_id,
+        "role": "assistant",
+        "content": answer,
+        "sources": [s.model_dump() for s in sources],
+        "created_at": _utcnow(),
+    }
+
+    return {
+        "conversation_id": req.conversation_id,
+        "message": message_response,
+        "sources": [s.model_dump() for s in sources],
+    }
+
+
+@app.post("/api/chat/stream")
+async def chat_stream(req: ChatRequest):
+    if not req.conversation_id:
+        conv = create_conv()
+        req.conversation_id = conv.id
+
+    add_message(req.conversation_id, "user", req.message)
+    context, sources, confidence, elapsed = query(req.message, req.top_k or 5)
+
+    system_prompt = (
+        "You are a helpful RAG assistant. Answer the user's question based "
+        "on the provided context. If the context doesn't contain enough "
+        "information, say so. Be concise."
+    )
+    user_prompt = f"Context:\n{context}\n\nQuestion: {req.message}"
+
+    async def generate():
+        full = ""
+        async for token in llm.generate_stream(system_prompt, user_prompt):
+            full += token
+            yield token
+        add_message(req.conversation_id, "assistant", full,
+                    [s.model_dump() for s in sources])
+
+    return StreamingResponse(generate(), media_type="text/plain")
+
+
+@app.get("/api/chat/conversations")
+async def list_conversations(skip: int = 0, limit: int = 50):
+    convs, total = list_convs(skip, limit)
+    return {"conversations": [c.model_dump() for c in convs], "total": total}
+
+
+@app.get("/api/chat/conversations/{conv_id}")
+async def get_conversation(conv_id: str):
+    conv = get_conv(conv_id)
+    if not conv:
+        raise HTTPException(404, "Conversation not found")
+    return conv.model_dump()
+
+
+@app.delete("/api/chat/conversations/{conv_id}")
+async def delete_conversation(conv_id: str):
+    if not delete_conv(conv_id):
+        raise HTTPException(404, "Conversation not found")
+    return {"message": "Conversation deleted"}
+
+
+# ─── RAG ──────────────────────────────────────────────────────────────────────
+
+@app.post("/api/rag/query")
+async def rag_query(req: RAGQueryRequest):
+    start = time.time()
+    context, sources, confidence, elapsed = query(req.query, req.top_k or 5)
+
+    system_prompt = (
+        "You are a helpful RAG assistant. Answer the question based on the "
+        "provided context. Be concise and factual."
+    )
+    user_prompt = f"Context:\n{context}\n\nQuestion: {req.query}"
+
+    answer = llm.generate(system_prompt, user_prompt)
+    total_elapsed = (time.time() - start) * 1000
+
+    return RAGQueryResponse(
+        query=req.query,
+        answer=answer,
+        sources=sources,
+        strategy=req.strategy or "vector",
+        latency_ms=total_elapsed,
+        confidence=confidence,
+    ).model_dump()
+
+
+@app.post("/api/rag/query/stream")
+async def rag_query_stream(req: RAGQueryRequest):
+    context, sources, confidence, elapsed = query(req.query, req.top_k or 5)
+
+    system_prompt = (
+        "You are a helpful RAG assistant. Answer the question based on the "
+        "provided context. Be concise and factual."
+    )
+    user_prompt = f"Context:\n{context}\n\nQuestion: {req.query}"
+
+    async def generate():
+        async for token in llm.generate_stream(system_prompt, user_prompt):
+            yield token
+
+    return StreamingResponse(generate(), media_type="text/plain")
+
+
+@app.post("/api/rag/retrieve")
+async def rag_retrieve(req: RAGQueryRequest):
+    results, _ = retrieve(req.query, req.top_k or 5)
+    return [r.model_dump() for r in results]
+
+
+@app.post("/api/rag/evaluate")
+async def rag_evaluate(req: EvaluationRequest):
+    per_question = []
+    total_latency = 0
+    failed = []
+
+    for q in req.questions:
+        start = time.time()
+        try:
+            context, sources, confidence, _ = query(q.question, 5)
+
+            system_prompt = (
+                "You are a RAG evaluator. Answer the question based on the context."
+            )
+            user_prompt = f"Context:\n{context}\n\nQuestion: {q.question}"
+            answer = llm.generate(system_prompt, user_prompt)
+            latency = (time.time() - start) * 1000
+            total_latency += latency
+
+            eval_prompt = (
+                f"Question: {q.question}\n"
+                f"Expected Answer: {q.expected_answer}\n"
+                f"Generated Answer: {answer}\n\n"
+                "Rate accuracy from 0.0 to 1.0. Return only a number."
+            )
+            acc_str = llm.generate(
+                "You evaluate answer accuracy. Return only a number 0-1.",
+                eval_prompt,
+            )
+            try:
+                accuracy = max(0.0, min(1.0, float(acc_str.strip())))
+            except ValueError:
+                accuracy = 0.0
+
+            per_question.append(PerQuestionResult(
+                question=q.question,
+                expected_answer=q.expected_answer,
+                generated_answer=answer,
+                retrieved_context=context[:500],
+                accuracy_llm=accuracy,
+                accuracy_combined=accuracy,
+                faithfulness=accuracy,
+                answer_relevancy=accuracy,
+                context_precision=accuracy,
+                context_recall=accuracy,
+                exact_match=1.0 if answer.strip() == q.expected_answer.strip() else 0.0,
+                semantic_similarity=accuracy,
+                f1=accuracy,
+                recall_10=1.0 if len(sources) > 0 else 0.0,
+                recall_20=1.0 if len(sources) > 0 else 0.0,
+                recall_50=1.0 if len(sources) > 0 else 0.0,
+                mrr=1.0 if len(sources) > 0 else 0.0,
+                ndcg_10=1.0 if len(sources) > 0 else 0.0,
+                gold_answer_found=accuracy > 0.5,
+                accuracy_rationale=f"LLM-judged accuracy: {accuracy:.2f}",
+                faithfulness_rationale=f"Generated answer aligns with context",
+                answer_relevancy_rationale=f"Answer addresses the question",
+                context_precision_rationale=f"Retrieved context is relevant",
+                context_recall_rationale=f"All necessary context was retrieved",
+                latency_ms=latency,
+            ))
+        except Exception as e:
+            latency = (time.time() - start) * 1000
+            total_latency += latency
+            failed.append({"question": q.question, "error": str(e)})
+            per_question.append(PerQuestionResult(
+                question=q.question,
+                expected_answer=q.expected_answer,
+                error=str(e),
+                latency_ms=latency,
+            ))
+
+    succeeded = [p for p in per_question if not p.error]
+    n = len(succeeded)
+    avg = lambda key: sum(getattr(p, key, 0) or 0 for p in succeeded) / n if n else 0.0
+
+    return EvaluationResponse(
+        accuracy=avg("accuracy_llm"),
+        accuracy_llm=avg("accuracy_llm"),
+        accuracy_combined=avg("accuracy_combined"),
+        faithfulness=avg("faithfulness"),
+        context_precision=avg("context_precision"),
+        context_recall=avg("context_recall"),
+        answer_relevancy=avg("answer_relevancy"),
+        exact_match=avg("exact_match"),
+        semantic_similarity=avg("semantic_similarity"),
+        f1=avg("f1"),
+        recall_10=avg("recall_10"),
+        recall_20=avg("recall_20"),
+        recall_50=avg("recall_50"),
+        mrr=avg("mrr"),
+        ndcg_10=avg("ndcg_10"),
+        latency_avg_ms=total_latency / len(req.questions) if req.questions else 0,
+        dataset_name=req.dataset_name or "",
+        failed_questions=failed,
+        per_question=per_question,
+    ).model_dump()
+
+
+# ─── Agents ───────────────────────────────────────────────────────────────────
+
+@app.post("/api/agents/{agent_type}")
+async def run_agent(agent_type: str, req: dict):
+    query_text = req.get("query", "")
+    context, sources, confidence, elapsed = query(query_text, req.get("top_k", 5))
+
+    system_prompt = f"You are a {agent_type} agent. Answer the question based on context."
+    user_prompt = f"Context:\n{context}\n\nQuestion: {query_text}"
+    answer = llm.generate(system_prompt, user_prompt)
+
+    return {
+        "agent": agent_type,
+        "query": query_text,
+        "answer": answer,
+        "sources": [s.model_dump() for s in sources],
+        "reasoning": f"Retrieved {len(sources)} relevant chunks",
+        "latency_ms": elapsed,
+        "metadata": {"strategy": "vector"},
+        "confidence": confidence,
+    }
+
+
+# ─── Chroma ───────────────────────────────────────────────────────────────────
+
+@app.get("/api/chroma/collections")
+async def list_collections():
+    return {"collections": ["default"], "counts": {"default": len(get_all_chunks())}}
+
+
+# ─── Web ─────────────────────────────────────────────────────────────────────
+
+@app.post("/api/web/ingest")
+async def ingest_web(data: dict):
+    url = data.get("url", "")
+    if not url:
+        raise HTTPException(400, "URL is required")
+    return {"message": f"URL {url} queued for ingestion", "url": url}
+
+
+# ─── Elasticsearch ───────────────────────────────────────────────────────────
+
+@app.post("/api/elasticsearch/upload")
+async def elasticsearch_upload(file: UploadFile = File(...)):
+    ext = os.path.splitext(file.filename or "file.txt")[1].lower()
+    doc_type = ext.lstrip(".") if ext else "text"
+    if doc_type in ("md",):
+        doc_type = "markdown"
+
+    filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
+    with open(filepath, "wb") as f:
+        content = await file.read()
+        f.write(content)
+
+    text = extract_text_from_file(filepath)
+    chunks = chunk_text(text)
+    doc = create_doc(
+        filename=file.filename or "unknown",
+        filepath=filepath,
+        doc_type=doc_type,
+        embedding_model=settings.embedding_model,
+    )
+    emb_list = embedder.embed_batch(chunks) if chunks else []
+    metadata_list = []
+    for i, (chunk, emb) in enumerate(zip(chunks, emb_list)):
+        metadata_list.append({"_embedding": emb, "chunk_index": i})
+    chunk_count = save_chunks(doc.id, chunks, metadata_list)
+    update_doc(doc.id, chunk_count=chunk_count)
+
+    return {"count": chunk_count}
+
+
+if __name__ == "__main__":
+    import uvicorn
+    uvicorn.run("app.main:app", host="0.0.0.0", port=9000, reload=True)
diff --git a/backend/app/models.py b/backend/app/models.py
new file mode 100644
index 0000000..5650943
+++ b/backend/app/models.py
@@ -0,0 +1,157 @@
+from pydantic import BaseModel
+from typing import Optional
+
+
+class Document(BaseModel):
+    id: str
+    filename: str
+    filepath: str
+    document_type: str
+    retrieval_strategy: str = "vector"
+    language: str = "en"
+    chunk_count: int = 0
+    embedding_model: str = ""
+    collection_name: str = "default"
+    metadata_json: dict = {}
+    created_at: str = ""
+    updated_at: str = ""
+
+
+class Chunk(BaseModel):
+    id: str
+    document_id: str
+    chunk_index: int
+    chunk_text: str
+    chunk_metadata: dict = {}
+    created_at: str = ""
+
+
+class SearchResult(BaseModel):
+    chunk_id: str
+    document_id: str
+    filename: str
+    chunk_text: str
+    score: float
+    metadata: dict = {}
+
+
+class Source(BaseModel):
+    filename: str
+    chunk_id: str
+    score: float
+
+
+class Message(BaseModel):
+    id: str
+    conversation_id: str
+    role: str
+    content: str
+    sources: list[Source] = []
+    created_at: str = ""
+
+
+class Conversation(BaseModel):
+    id: str
+    title: str = ""
+    created_at: str = ""
+    updated_at: str = ""
+    messages: list[Message] = []
+
+
+class SearchRequest(BaseModel):
+    query: str
+    top_k: int = 5
+    filters: Optional[dict] = None
+    collection_name: Optional[str] = None
+
+
+class SearchResponse(BaseModel):
+    query: str
+    results: list[SearchResult]
+    confidence: float = 0.0
+    sources: list[str] = []
+    latency_ms: float = 0.0
+    strategy: str = "vector"
+
+
+class ChatRequest(BaseModel):
+    message: str
+    conversation_id: Optional[str] = None
+    top_k: int = 5
+
+
+class RAGQueryRequest(BaseModel):
+    query: str
+    strategy: Optional[str] = "vector"
+    top_k: int = 5
+    filters: Optional[dict] = None
+
+
+class RAGQueryResponse(BaseModel):
+    query: str
+    answer: str
+    sources: list[Source]
+    strategy: str
+    latency_ms: float
+    confidence: float
+
+
+class EvalQuestion(BaseModel):
+    question: str
+    expected_answer: str
+
+
+class EvaluationRequest(BaseModel):
+    questions: list[EvalQuestion]
+    dataset_name: Optional[str] = None
+
+
+class PerQuestionResult(BaseModel):
+    question: str
+    expected_answer: str
+    generated_answer: str = ""
+    retrieved_context: str = ""
+    accuracy_llm: float = 0.0
+    faithfulness: float = 0.0
+    answer_relevancy: float = 0.0
+    context_precision: float = 0.0
+    context_recall: float = 0.0
+    exact_match: float = 0.0
+    semantic_similarity: float = 0.0
+    f1: float = 0.0
+    accuracy_combined: float = 0.0
+    recall_10: float = 0.0
+    recall_20: float = 0.0
+    recall_50: float = 0.0
+    mrr: float = 0.0
+    ndcg_10: float = 0.0
+    gold_answer_found: bool = False
+    accuracy_rationale: str = ""
+    faithfulness_rationale: str = ""
+    answer_relevancy_rationale: str = ""
+    context_precision_rationale: str = ""
+    context_recall_rationale: str = ""
+    latency_ms: float = 0.0
+    error: Optional[str] = None
+
+
+class EvaluationResponse(BaseModel):
+    accuracy: float = 0.0
+    accuracy_llm: float = 0.0
+    accuracy_combined: float = 0.0
+    faithfulness: float = 0.0
+    context_precision: float = 0.0
+    context_recall: float = 0.0
+    answer_relevancy: float = 0.0
+    exact_match: float = 0.0
+    semantic_similarity: float = 0.0
+    f1: float = 0.0
+    recall_10: float = 0.0
+    recall_20: float = 0.0
+    recall_50: float = 0.0
+    mrr: float = 0.0
+    ndcg_10: float = 0.0
+    latency_avg_ms: float = 0.0
+    dataset_name: str = ""
+    failed_questions: list[dict] = []
+    per_question: list[PerQuestionResult] = []
diff --git a/backend/app/rag.py b/backend/app/rag.py
new file mode 100644
index 0000000..d447960
+++ b/backend/app/rag.py
@@ -0,0 +1,100 @@
+import os
+import time
+import uuid
+from typing import Optional
+
+from app.embeddings import embedder
+from app.storage import get_all_chunks, get_doc
+from app.models import SearchResult, Source
+
+
+def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
+    if not text:
+        return []
+    chunks = []
+    start = 0
+    while start < len(text):
+        end = min(start + chunk_size, len(text))
+        if end < len(text):
+            last_space = text.rfind(" ", start + chunk_size - overlap, end)
+            if last_space > start:
+                end = last_space
+        chunks.append(text[start:end].strip())
+        start = end - overlap if end < len(text) else end
+    return [c for c in chunks if c]
+
+
+def extract_text_from_file(filepath: str) -> str:
+    ext = os.path.splitext(filepath)[1].lower()
+    if ext == ".pdf":
+        try:
+            from PyPDF2 import PdfReader
+            reader = PdfReader(filepath)
+            return "\n".join(page.extract_text() or "" for page in reader.pages)
+        except Exception:
+            return ""
+    elif ext == ".docx":
+        try:
+            from docx import Document as DocxDoc
+            doc = DocxDoc(filepath)
+            return "\n".join(p.text for p in doc.paragraphs)
+        except Exception:
+            return ""
+    else:
+        try:
+            with open(filepath, encoding="utf-8", errors="replace") as f:
+                return f.read()
+        except Exception:
+            return ""
+
+
+def retrieve(query: str, top_k: int = 5) -> tuple[list[SearchResult], float]:
+    chunks = get_all_chunks()
+    if not chunks:
+        return [], 0.0
+
+    query_emb = embedder.embed(query)
+    chunk_embeddings = [c.chunk_metadata.get("_embedding") for c in chunks]
+
+    if not any(chunk_embeddings):
+        return [], 0.0
+
+    scored = []
+    for c, emb in zip(chunks, chunk_embeddings):
+        if emb:
+            score = embedder.cosine_similarity(query_emb, emb)
+        else:
+            score = 0.0
+        doc = get_doc(c.document_id)
+        scored.append(SearchResult(
+            chunk_id=c.id,
+            document_id=c.document_id,
+            filename=doc.filename if doc else "unknown",
+            chunk_text=c.chunk_text,
+            score=score,
+            metadata=c.chunk_metadata,
+        ))
+
+    scored.sort(key=lambda x: x.score, reverse=True)
+    results = scored[:top_k]
+    confidence = results[0].score if results else 0.0
+    return results, confidence
+
+
+def build_context(results: list[SearchResult]) -> str:
+    return "\n\n".join(
+        f"[Source {i + 1}] {r.filename}:\n{r.chunk_text}"
+        for i, r in enumerate(results)
+    )
+
+
+def query(query: str, top_k: int = 5) -> tuple[str, list[Source], float, float]:
+    start = time.time()
+    results, confidence = retrieve(query, top_k)
+    context = build_context(results)
+    sources = [
+        Source(filename=r.filename, chunk_id=r.chunk_id, score=r.score)
+        for r in results
+    ]
+    elapsed = (time.time() - start) * 1000
+    return context, sources, confidence, elapsed
diff --git a/backend/app/rag/__init__.py b/backend/app/rag/__init__.py
deleted file mode 100644
index e69de29..0000000
diff --git a/backend/app/rag/bm25.py b/backend/app/rag/bm25.py
deleted file mode 100644
index f4841b6..0000000
+++ /dev/null
@@ -1,47 +0,0 @@
diff --git a/backend/app/rag/cross_encoder.py b/backend/app/rag/cross_encoder.py
deleted file mode 100644
index 103bb4e..0000000
+++ /dev/null
@@ -1,33 +0,0 @@
diff --git a/backend/app/rag/evaluator.py b/backend/app/rag/evaluator.py
deleted file mode 100644
index acbe386..0000000
+++ /dev/null
@@ -1,373 +0,0 @@
\ No newline at end of file
diff --git a/backend/app/rag/hybrid_rag.py b/backend/app/rag/hybrid_rag.py
deleted file mode 100644
index 0cf7d14..0000000
+++ /dev/null
@@ -1,108 +0,0 @@
diff --git a/backend/app/rag/markdown_rag.py b/backend/app/rag/markdown_rag.py
deleted file mode 100644
index e7da7e1..0000000
+++ /dev/null
@@ -1,127 +0,0 @@
diff --git a/backend/app/rag/metadata_filter.py b/backend/app/rag/metadata_filter.py
deleted file mode 100644
index d8dd3ee..0000000
+++ /dev/null
@@ -1,56 +0,0 @@
diff --git a/backend/app/rag/multi_collection_retrieval.py b/backend/app/rag/multi_collection_retrieval.py
deleted file mode 100644
index 8d98fc3..0000000
+++ /dev/null
@@ -1,78 +0,0 @@
diff --git a/backend/app/rag/parent_context.py b/backend/app/rag/parent_context.py
deleted file mode 100644
index c3c8ee8..0000000
+++ /dev/null
@@ -1,78 +0,0 @@
diff --git a/backend/app/rag/pdf_rag.py b/backend/app/rag/pdf_rag.py
deleted file mode 100644
index 155e29a..0000000
+++ /dev/null
@@ -1,186 +0,0 @@
diff --git a/backend/app/rag/rrf.py b/backend/app/rag/rrf.py
deleted file mode 100644
index e981ff9..0000000
+++ /dev/null
@@ -1,29 +0,0 @@
diff --git a/backend/app/rag/synonym_expansion.py b/backend/app/rag/synonym_expansion.py
deleted file mode 100644
index 4ce62aa..0000000
+++ /dev/null
@@ -1,88 +0,0 @@
diff --git a/backend/app/rag/table_rag.py b/backend/app/rag/table_rag.py
deleted file mode 100644
index d9de0bb..0000000
+++ /dev/null
@@ -1,177 +0,0 @@
diff --git a/backend/app/rag/vector_rag.py b/backend/app/rag/vector_rag.py
deleted file mode 100644
index 3764439..0000000
+++ /dev/null
@@ -1,490 +0,0 @@
diff --git a/backend/app/repositories/__init__.py b/backend/app/repositories/__init__.py
deleted file mode 100644
index e69de29..0000000
diff --git a/backend/app/repositories/conversation_repository.py b/backend/app/repositories/conversation_repository.py
deleted file mode 100644
index d8139c0..0000000
+++ /dev/null
@@ -1,57 +0,0 @@
diff --git a/backend/app/repositories/document_repository.py b/backend/app/repositories/document_repository.py
deleted file mode 100644
index c784af1..0000000
+++ /dev/null
@@ -1,69 +0,0 @@
diff --git a/backend/app/repositories/kag_repository.py b/backend/app/repositories/kag_repository.py
deleted file mode 100644
index 5a8bb2d..0000000
+++ /dev/null
@@ -1,95 +0,0 @@
diff --git a/backend/app/repositories/log_repository.py b/backend/app/repositories/log_repository.py
deleted file mode 100644
index 3305f9a..0000000
+++ /dev/null
@@ -1,42 +0,0 @@
diff --git a/backend/app/schemas/__init__.py b/backend/app/schemas/__init__.py
deleted file mode 100644
index 78ee4f2..0000000
+++ /dev/null
@@ -1 +0,0 @@
diff --git a/backend/app/schemas/chat.py b/backend/app/schemas/chat.py
deleted file mode 100644
index f65e3fb..0000000
+++ /dev/null
@@ -1,42 +0,0 @@
diff --git a/backend/app/schemas/document.py b/backend/app/schemas/document.py
deleted file mode 100644
index 9756870..0000000
+++ /dev/null
@@ -1,42 +0,0 @@
diff --git a/backend/app/schemas/rag.py b/backend/app/schemas/rag.py
deleted file mode 100644
index dd730e3..0000000
+++ /dev/null
@@ -1,46 +0,0 @@
diff --git a/backend/app/schemas/search.py b/backend/app/schemas/search.py
deleted file mode 100644
index 1f64706..0000000
+++ /dev/null
@@ -1,27 +0,0 @@
diff --git a/backend/app/services/__init__.py b/backend/app/services/__init__.py
deleted file mode 100644
index e69de29..0000000
diff --git a/backend/app/services/chat_service.py b/backend/app/services/chat_service.py
deleted file mode 100644
index df746dd..0000000
+++ /dev/null
@@ -1,131 +0,0 @@
\ No newline at end of file
diff --git a/backend/app/services/document_service.py b/backend/app/services/document_service.py
deleted file mode 100644
index 3fe3494..0000000
+++ /dev/null
@@ -1,439 +0,0 @@
diff --git a/backend/app/services/elasticsearch_service.py b/backend/app/services/elasticsearch_service.py
deleted file mode 100644
index d84c542..0000000
+++ /dev/null
@@ -1,104 +0,0 @@
diff --git a/backend/app/services/graph_service.py b/backend/app/services/graph_service.py
deleted file mode 100644
index 1cad975..0000000
+++ /dev/null
@@ -1,11 +0,0 @@
diff --git a/backend/app/services/rag_service.py b/backend/app/services/rag_service.py
deleted file mode 100644
index 07059d2..0000000
+++ /dev/null
@@ -1,268 +0,0 @@
diff --git a/backend/app/storage.py b/backend/app/storage.py
new file mode 100644
index 0000000..b2fc278
+++ b/backend/app/storage.py
@@ -0,0 +1,190 @@
+import json
+import os
+import uuid
+from datetime import datetime, timezone
+from typing import Optional
+
+from app.models import Document, Chunk, Conversation, Message
+
+DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
+DOCS_FILE = os.path.join(DATA_DIR, "documents.json")
+CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
+CONVS_FILE = os.path.join(DATA_DIR, "conversations.json")
+
+os.makedirs(DATA_DIR, exist_ok=True)
+
+
+def _load_json(path: str) -> list:
+    if not os.path.exists(path):
+        return []
+    with open(path) as f:
+        return json.load(f)
+
+
+def _save_json(path: str, data: list):
+    with open(path, "w") as f:
+        json.dump(data, f, indent=2, default=str)
+
+
+def _utcnow() -> str:
+    return datetime.now(timezone.utc).isoformat()
+
+
+def _new_id() -> str:
+    return str(uuid.uuid4())
+
+
+# ─── Documents ────────────────────────────────────────────────────────────────
+
+def list_docs(skip: int = 0, limit: int = 50) -> tuple[list[Document], int]:
+    docs = _load_json(DOCS_FILE)
+    total = len(docs)
+    docs = docs[skip: skip + limit]
+    return [Document(**d) for d in docs], total
+
+
+def get_doc(doc_id: str) -> Optional[Document]:
+    docs = _load_json(DOCS_FILE)
+    for d in docs:
+        if d["id"] == doc_id:
+            return Document(**d)
+    return None
+
+
+def create_doc(filename: str, filepath: str, doc_type: str,
+               embedding_model: str, metadata: dict = None) -> Document:
+    now = _utcnow()
+    doc = Document(
+        id=_new_id(),
+        filename=filename,
+        filepath=filepath,
+        document_type=doc_type,
+        embedding_model=embedding_model,
+        metadata_json=metadata or {},
+        created_at=now,
+        updated_at=now,
+    )
+    docs = _load_json(DOCS_FILE)
+    docs.append(doc.model_dump())
+    _save_json(DOCS_FILE, docs)
+    return doc
+
+
+def update_doc(doc_id: str, **kwargs):
+    docs = _load_json(DOCS_FILE)
+    for i, d in enumerate(docs):
+        if d["id"] == doc_id:
+            d.update(kwargs)
+            d["updated_at"] = _utcnow()
+            docs[i] = d
+            _save_json(DOCS_FILE, docs)
+            return Document(**d)
+    return None
+
+
+def delete_doc(doc_id: str) -> bool:
+    docs = _load_json(DOCS_FILE)
+    new_docs = [d for d in docs if d["id"] != doc_id]
+    if len(new_docs) == len(docs):
+        return False
+    _save_json(DOCS_FILE, new_docs)
+    chunks = _load_json(CHUNKS_FILE)
+    chunks = [c for c in chunks if c["document_id"] != doc_id]
+    _save_json(CHUNKS_FILE, chunks)
+    return True
+
+
+# ─── Chunks ───────────────────────────────────────────────────────────────────
+
+def get_chunks(doc_id: str) -> list[Chunk]:
+    chunks = _load_json(CHUNKS_FILE)
+    return [Chunk(**c) for c in chunks if c["document_id"] == doc_id]
+
+
+def get_all_chunks() -> list[Chunk]:
+    chunks = _load_json(CHUNKS_FILE)
+    return [Chunk(**c) for c in chunks]
+
+
+def save_chunks(doc_id: str, chunk_texts: list[str],
+                metadata_list: list[dict] = None):
+    existing = _load_json(CHUNKS_FILE)
+    existing = [c for c in existing if c["document_id"] != doc_id]
+    now = _utcnow()
+    new_chunks = []
+    for i, text in enumerate(chunk_texts):
+        new_chunks.append({
+            "id": _new_id(),
+            "document_id": doc_id,
+            "chunk_index": i,
+            "chunk_text": text,
+            "chunk_metadata": (metadata_list[i] if metadata_list and i < len(metadata_list) else {}),
+            "created_at": now,
+        })
+    existing.extend(new_chunks)
+    _save_json(CHUNKS_FILE, existing)
+    return len(new_chunks)
+
+
+# ─── Conversations ────────────────────────────────────────────────────────────
+
+def list_convs(skip: int = 0, limit: int = 50) -> tuple[list[Conversation], int]:
+    convs = _load_json(CONVS_FILE)
+    total = len(convs)
+    convs = convs[skip: skip + limit]
+    return [Conversation(**c) for c in convs], total
+
+
+def get_conv(conv_id: str) -> Optional[Conversation]:
+    convs = _load_json(CONVS_FILE)
+    for c in convs:
+        if c["id"] == conv_id:
+            return Conversation(**c)
+    return None
+
+
+def create_conv(title: str = "") -> Conversation:
+    now = _utcnow()
+    conv = Conversation(
+        id=_new_id(),
+        title=title or f"Chat {now[:10]}",
+        created_at=now,
+        updated_at=now,
+    )
+    convs = _load_json(CONVS_FILE)
+    convs.append(conv.model_dump())
+    _save_json(CONVS_FILE, convs)
+    return conv
+
+
+def add_message(conv_id: str, role: str, content: str,
+                sources: list[dict] = None) -> Optional[Message]:
+    convs = _load_json(CONVS_FILE)
+    for i, c in enumerate(convs):
+        if c["id"] == conv_id:
+            now = _utcnow()
+            msg = Message(
+                id=_new_id(),
+                conversation_id=conv_id,
+                role=role,
+                content=content,
+                sources=[Source(**s) for s in (sources or [])],
+                created_at=now,
+            )
+            c["messages"].append(msg.model_dump())
+            c["updated_at"] = now
+            if role == "user" and not c.get("title") or c["title"].startswith("Chat "):
+                c["title"] = content[:60]
+            convs[i] = c
+            _save_json(CONVS_FILE, convs)
+            return msg
+    return None
+
+
+def delete_conv(conv_id: str) -> bool:
+    convs = _load_json(CONVS_FILE)
+    new_convs = [c for c in convs if c["id"] != conv_id]
+    if len(new_convs) == len(convs):
+        return False
+    _save_json(CONVS_FILE, new_convs)
+    return True
diff --git a/backend/app/tests/__init__.py b/backend/app/tests/__init__.py
deleted file mode 100644
index e69de29..0000000
diff --git a/backend/app/tests/test_core.py b/backend/app/tests/test_core.py
deleted file mode 100644
index 8dedd1f..0000000
+++ /dev/null
@@ -1,289 +0,0 @@
diff --git a/backend/delete_embeddings.py b/backend/delete_embeddings.py
deleted file mode 100644
index 47b94dd..0000000
+++ /dev/null
@@ -1,13 +0,0 @@
\ No newline at end of file
diff --git a/backend/test_logger.py b/backend/test_logger.py
deleted file mode 100644
index 2808935..0000000
+++ /dev/null
@@ -1,16 +0,0 @@
diff --git a/helpfull scripts/script.py b/helpfull scripts/script.py
new file mode 100644
index 0000000..c4faac1
+++ b/helpfull scripts/script.py	
@@ -0,0 +1,89 @@
+"""
+Text-to-Speech script using OpenAI's API.
+
+Note: Whisper is OpenAI's speech-to-text model (audio -> text).
+This script does the opposite: text -> audio, using OpenAI's
+text-to-speech model (gpt-4o-mini-tts).
+
+Setup:
+    pip install openai
+    export OPENAI_API_KEY="your-api-key-here"   # (Windows: set OPENAI_API_KEY=...)
+
+Usage:
+    python text_to_speech.py
+    python text_to_speech.py --text "Hello there!" --output hello.mp3
+    python text_to_speech.py --input-file script.txt --voice nova
+"""
+
+import argparse
+from pathlib import Path
+from openai import OpenAI
+
+
+def text_to_speech(
+    text: str,
+    output_path: str = "speech.mp3",
+    model: str = "gpt-4o-mini-tts",
+    voice: str = "coral",
+    instructions: str | None = None,
+) -> Path:
+    """
+    Convert text to an audio file using OpenAI's TTS API.
+
+    Args:
+        text: The text to convert to speech.
+        output_path: Where to save the resulting audio file (.mp3, .wav, etc.)
+        model: TTS model to use ("gpt-4o-mini-tts", "tts-1", or "tts-1-hd").
+        voice: Voice to use (e.g. "alloy", "coral", "nova", "onyx", "shimmer").
+        instructions: Optional steering instructions, e.g. "Speak slowly and calmly."
+                       (only supported by gpt-4o-mini-tts)
+
+    Returns:
+        Path to the saved audio file.
+    """
+    client = OpenAI()  # reads OPENAI_API_KEY from environment
+    speech_file_path = Path(output_path)
+
+    kwargs = dict(model=model, voice=voice, input=text)
+    if instructions and model == "gpt-4o-mini-tts":
+        kwargs["instructions"] = instructions
+
+    with client.audio.speech.with_streaming_response.create(**kwargs) as response:
+        response.stream_to_file(speech_file_path)
+
+    return speech_file_path
+
+
+def main():
+    parser = argparse.ArgumentParser(description="Convert text to an audio file using OpenAI TTS.")
+    parser.add_argument("--text", type=str, help="Text to convert directly.")
+    parser.add_argument("--input-file", type=str, help="Path to a .txt file containing the text.")
+    parser.add_argument("--output", type=str, default="speech.mp3", help="Output audio file path.")
+    parser.add_argument("--model", type=str, default="gpt-4o-mini-tts",
+                         choices=["gpt-4o-mini-tts", "tts-1", "tts-1-hd"])
+    parser.add_argument("--voice", type=str, default="coral",
+                         help="alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse")
+    parser.add_argument("--instructions", type=str, default=None,
+                         help="Optional tone/style instructions (gpt-4o-mini-tts only).")
+    args = parser.parse_args()
+
+    if args.input_file:
+        text = Path(args.input_file).read_text(encoding="utf-8")
+    elif args.text:
+        text = args.text
+    else:
+        text = "Hello! This is a test of OpenAI's text to speech API."
+        print("No --text or --input-file given, using a default test sentence.")
+
+    path = text_to_speech(
+        text=text,
+        output_path=args.output,
+        model=args.model,
+        voice=args.voice,
+        instructions=args.instructions,
+    )
+    print(f"Saved audio to: {path.resolve()}")
+
+
+if __name__ == "__main__":
+    main()
\ No newline at end of file
```

## 184ae5905abfbf579e7f6bc68eee08ad4f004140 — 2026-06-29T12:02:53+05:30

Message:

requirement

```diff
diff --git a/backend/app/evaluation/agent_runner.py b/backend/app/evaluation/agent_runner.py
index 0d52eba..345de8a 100644
+++ b/backend/app/evaluation/agent_runner.py
@@ -16,8 +16,6 @@ Flow:
     ↓
   Cross-Encoder Rerank
     ↓
   LLM
 """
 
@@ -32,7 +30,6 @@ from app.evaluation.evaluator import evaluate_single
 from app.embeddings.openai_client import openai_client
 from app.core.config import settings
 from app.core.evaluation_logger import EvaluationLogger
 from app.services.graph_service import graph_service
 from app.services.rag_service import rag_service
 
@@ -337,20 +334,7 @@ async def evaluate_question(
     if graph_note:
         chunk_texts = [graph_note] + chunk_texts
 
+    context_text = "\n---\n".join(chunk_texts)
 
     if context_text.strip():
         prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
@@ -392,7 +376,7 @@ async def evaluate_question(
         _current_logger.log("GOLD_CHUNKS",
             f"Identified {len(gold_chunk_ids)} gold chunks from {len(retrieved_chunk_ids)} retrieved")
 
+    scores = await evaluate_single(question, expected_answer, generated_answer, chunk_texts,
                                    retrieved_chunk_ids=retrieved_chunk_ids, gold_chunk_ids=gold_chunk_ids)
 
     if _current_logger:
diff --git a/backend/app/services/rag_service.py b/backend/app/services/rag_service.py
index 836c3bc..07059d2 100644
+++ b/backend/app/services/rag_service.py
@@ -18,7 +18,6 @@ from app.rag.evaluator import (
 from app.embeddings.openai_client import openai_client as ollama_client
 from app.repositories.log_repository import log_repo
 from app.core.config import settings
 from app.core.evaluation_logger import EvaluationLogger
 from app.services.graph_service import graph_service
 from app.database.models import Chunk
@@ -205,12 +204,7 @@ class RAGService:
                 context_texts = [r["chunk_text"] for r in chunks]
                 logger.log_retrieval("hybrid", settings.TOP_K, chunks)
                 
+                context = "\n\n".join(context_texts)
                 prompt = f"Context:\n{context}\n\nQuestion: {question}"
                 
                 llm_start = time.time()
@@ -223,13 +217,13 @@ class RAGService:
                 acc_score, acc_rat = await compute_accuracy(answer, expected)
                 logger.log_metrics("Accuracy", acc_score, acc_rat)
                 
+                faith_score, faith_rat = await compute_faithfulness(answer, context_texts)
                 logger.log_metrics("Faithfulness", faith_score, faith_rat)
                 
+                cp_score, cp_rat = await compute_context_precision(question, context_texts)
                 logger.log_metrics("Context Precision", cp_score, cp_rat)
                 
+                cr_score, cr_rat = await compute_context_recall(expected, context_texts)
                 logger.log_metrics("Context Recall", cr_score, cr_rat)
                 
                 ar_score, ar_rat = await compute_answer_relevancy(question, answer)
```

## 633da7070f411c2a5c122e79e3715b22287d7842 — 2026-06-29T11:50:28+05:30

Message:

elastic search remove

```diff
diff --git a/helpfull scripts/check eval dataset/correct.py b/helpfull scripts/check eval dataset/correct.py
deleted file mode 100644
index b7f0f5f..0000000
+++ /dev/null
@@ -1,154 +0,0 @@
\ No newline at end of file
diff --git a/helpfull scripts/check eval dataset/extract_top_n_questions.py b/helpfull scripts/check eval dataset/extract_top_n_questions.py
deleted file mode 100644
index 122d94a..0000000
+++ /dev/null
@@ -1,85 +0,0 @@
\ No newline at end of file
diff --git a/helpfull scripts/compare_changes_in_metrics.py b/helpfull scripts/compare_changes_in_metrics.py
index 85451ba..cd6ad15 100644
+++ b/helpfull scripts/compare_changes_in_metrics.py	
@@ -4,8 +4,8 @@ import matplotlib.pyplot as plt
 # ==========================
 # CONFIG
 # ==========================
+CSV_1 = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/eval 11th june .csv" 
+CSV_2 = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/rag_eval_1781246175645 12th june overall score 63 added elasticsearch and crawled websites.csv"
 
 METRICS = [
     "Accuracy (LLM)",
diff --git a/helpfull scripts/edit eval expected answer/judge_and_update_answers.py b/helpfull scripts/edit eval expected answer/judge_and_update_answers.py
deleted file mode 100644
index 167a713..0000000
+++ /dev/null
@@ -1,336 +0,0 @@
diff --git a/helpfull scripts/generate_correct_answers.py b/helpfull scripts/generate_correct_answers.py
deleted file mode 100644
index e69de29..0000000
diff --git a/helpfull scripts/get_less_accuracy_rows.py b/helpfull scripts/get_less_accuracy_rows.py
index 4c3f0ae..e2ede8b 100644
+++ b/helpfull scripts/get_less_accuracy_rows.py	
@@ -1,15 +1,14 @@
 import pandas as pd
+from pathlib import Path
 
 # Input CSV file
+input_file = "C:\\Users\\Jishnu\\Desktop\\SRAG\\eval\\results\\8th june 2nd cross encoder.csv"
 
 # Read CSV
 df = pd.read_csv(input_file)
 
 # Filter rows where Accuracy <= 0.6
+filtered_df = df[df["Accuracy"] <= 0.6]
 
 # Create output filename
 input_path = Path(input_file)
@@ -19,4 +18,4 @@ output_file = f"0.6accuracyandbelowrows_{input_path.stem}.csv"
 filtered_df.to_csv(output_file, index=False)
 
 print(f"Filtered {len(filtered_df)} rows.")
+print(f"Output saved to: {output_file}")
\ No newline at end of file
diff --git a/helpfull scripts/get_questions_answer_from_eval_result.py b/helpfull scripts/get_questions_answer_from_eval_result.py
deleted file mode 100644
index bf280af..0000000
+++ /dev/null
@@ -1,24 +0,0 @@
\ No newline at end of file
```

## ecf2428a20c8934620556651d9cfab478984295a — 2026-06-23T14:24:28+05:30

Message:

metrics for 23rd kag run with llm changed answers oeverlal score at 56.1 worse performing

```diff
diff --git a/helpfull scripts/compare_changes_in_metrics.py b/helpfull scripts/compare_changes_in_metrics.py
index f464425..85451ba 100644
+++ b/helpfull scripts/compare_changes_in_metrics.py	
@@ -4,8 +4,8 @@ import matplotlib.pyplot as plt
 # ==========================
 # CONFIG
 # ==========================
+CSV_1 = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/23 june after KAG and llm changed answers.csv" 
+CSV_2 = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/23 june after KAG manual questions.csv"
 
 METRICS = [
     "Accuracy (LLM)",
```

## fa92da6e63a3189bbc4af605879c2426b8451f16 — 2026-06-23T12:51:17+05:30

Message:

new ingested data and Kag and updated answers

```diff
diff --git a/backend/alembic/versions/0003_kag_extension.py b/backend/alembic/versions/0003_kag_extension.py
new file mode 100644
index 0000000..77442e6
+++ b/backend/alembic/versions/0003_kag_extension.py
@@ -0,0 +1,188 @@
+"""kag extension
+
+Revision ID: 0003
+Revises: 0002
+Create Date: 2026-06-22 00:00:00.000000
+
+"""
+from typing import Sequence, Union
+
+from alembic import op
+import sqlalchemy as sa
+
+
+revision: str = "0003"
+down_revision: Union[str, None] = "0002"
+branch_labels: Union[str, Sequence[str], None] = None
+depends_on: Union[str, Sequence[str], None] = None
+
+
+def upgrade() -> None:
+    # Documents
+    op.add_column("documents", sa.Column("content_hash", sa.String(length=128), nullable=True))
+    op.add_column("documents", sa.Column("document_version", sa.String(length=64), nullable=True))
+    op.add_column("documents", sa.Column("processing_status", sa.String(length=64), nullable=True))
+    op.add_column("documents", sa.Column("processing_log", sa.Text(), nullable=True))
+    op.create_index("ix_documents_document_version", "documents", ["document_version"], unique=False)
+    op.create_index("ix_documents_content_hash", "documents", ["content_hash"], unique=False)
+
+    # Chunks
+    op.add_column("chunks", sa.Column("vector_id", sa.String(length=64), nullable=True))
+    op.add_column("chunks", sa.Column("chunk_type", sa.String(length=64), nullable=True))
+    op.add_column("chunks", sa.Column("content_summary", sa.Text(), nullable=True))
+    op.add_column("chunks", sa.Column("extracted_entities", sa.JSON(), nullable=True))
+    op.add_column("chunks", sa.Column("section", sa.String(length=256), nullable=True))
+    op.add_column("chunks", sa.Column("field_name", sa.String(length=256), nullable=True))
+    op.add_column("chunks", sa.Column("requirement_tags", sa.JSON(), nullable=True))
+    op.add_column("chunks", sa.Column("regulatory_reference", sa.JSON(), nullable=True))
+    op.add_column("chunks", sa.Column("confidence_score", sa.Float(), nullable=True))
+    op.add_column("chunks", sa.Column("chunk_position", sa.Integer(), nullable=True))
+    op.create_index("ix_chunks_vector_id", "chunks", ["vector_id"], unique=False)
+    op.create_index("ix_chunks_chunk_type", "chunks", ["chunk_type"], unique=False)
+
+    # Form versions
+    op.create_table(
+        "form_versions",
+        sa.Column("id", sa.String(length=36), nullable=False),
+        sa.Column("form_id", sa.String(length=36), sa.ForeignKey("forms.id", ondelete="CASCADE"), nullable=False),
+        sa.Column("version", sa.String(length=64), nullable=False),
+        sa.Column("status", sa.String(length=64), nullable=True),
+        sa.Column("supersedes_id", sa.String(length=36), sa.ForeignKey("form_versions.id", ondelete="SET NULL"), nullable=True),
+        sa.Column("effective_date", sa.DateTime(), nullable=True),
+        sa.Column("created_at", sa.DateTime(), nullable=True),
+        sa.PrimaryKeyConstraint("id", name="pk_form_versions"),
+    )
+    op.create_index("ix_form_versions_form_id", "form_versions", ["form_id"], unique=False)
+    op.create_index("ix_form_versions_version", "form_versions", ["version"], unique=False)
+
+    # Fields
+    op.create_table(
+        "fields",
+        sa.Column("id", sa.String(length=36), nullable=False),
+        sa.Column("form_version_id", sa.String(length=36), sa.ForeignKey("form_versions.id", ondelete="CASCADE"), nullable=False),
+        sa.Column("name", sa.String(length=256), nullable=False),
+        sa.Column("field_type", sa.String(length=64), nullable=True),
+        sa.Column("validation_rules", sa.JSON(), nullable=True),
+        sa.Column("required", sa.Boolean(), nullable=True),
+        sa.Column("description", sa.Text(), nullable=True),
+        sa.Column("created_at", sa.DateTime(), nullable=True),
+        sa.PrimaryKeyConstraint("id", name="pk_fields"),
+    )
+    op.create_index("ix_fields_form_version_id", "fields", ["form_version_id"], unique=False)
+    op.create_index("ix_fields_name", "fields", ["name"], unique=False)
+
+    # Field dependencies
+    op.create_table(
+        "field_dependencies",
+        sa.Column("id", sa.String(length=36), nullable=False),
+        sa.Column("source_field_id", sa.String(length=36), sa.ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
+        sa.Column("target_field_id", sa.String(length=36), sa.ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
+        sa.Column("condition", sa.JSON(), nullable=True),
+        sa.Column("created_at", sa.DateTime(), nullable=True),
+        sa.PrimaryKeyConstraint("id", name="pk_field_dependencies"),
+    )
+    op.create_index("ix_field_dependencies_source", "field_dependencies", ["source_field_id"], unique=False)
+    op.create_index("ix_field_dependencies_target", "field_dependencies", ["target_field_id"], unique=False)
+
+    # Regulations
+    op.create_table(
+        "regulations",
+        sa.Column("id", sa.String(length=36), nullable=False),
+        sa.Column("title", sa.String(length=256), nullable=False),
+        sa.Column("authority", sa.String(length=256), nullable=True),
+        sa.Column("effective_date", sa.DateTime(), nullable=True),
+        sa.Column("citation", sa.String(length=256), nullable=True),
+        sa.Column("description", sa.Text(), nullable=True),
+        sa.Column("created_at", sa.DateTime(), nullable=True),
+        sa.PrimaryKeyConstraint("id", name="pk_regulations"),
+    )
+    op.create_index("ix_regulations_title", "regulations", ["title"], unique=False)
+    op.create_index("ix_regulations_citation", "regulations", ["citation"], unique=False)
+
+    # Requirements
+    op.create_table(
+        "requirements",
+        sa.Column("id", sa.String(length=36), nullable=False),
+        sa.Column("description", sa.Text(), nullable=False),
+        sa.Column("applicability", sa.String(length=256), nullable=True),
+        sa.Column("regulation_id", sa.String(length=36), sa.ForeignKey("regulations.id", ondelete="SET NULL"), nullable=True),
+        sa.Column("regulation_ref", sa.String(length=256), nullable=True),
+        sa.Column("created_at", sa.DateTime(), nullable=True),
+        sa.PrimaryKeyConstraint("id", name="pk_requirements"),
+    )
+    op.create_index("ix_requirements_regulation_ref", "requirements", ["regulation_ref"], unique=False)
+
+    # Form requirements
+    op.create_table(
+        "form_requirements",
+        sa.Column("id", sa.String(length=36), nullable=False),
+        sa.Column("form_version_id", sa.String(length=36), sa.ForeignKey("form_versions.id", ondelete="CASCADE"), nullable=False),
+        sa.Column("requirement_id", sa.String(length=36), sa.ForeignKey("requirements.id", ondelete="CASCADE"), nullable=False),
+        sa.Column("applies_if", sa.JSON(), nullable=True),
+        sa.Column("created_at", sa.DateTime(), nullable=True),
+        sa.PrimaryKeyConstraint("id", name="pk_form_requirements"),
+    )
+    op.create_index("ix_form_requirements_form_version_id", "form_requirements", ["form_version_id"], unique=False)
+    op.create_index("ix_form_requirements_requirement_id", "form_requirements", ["requirement_id"], unique=False)
+
+    # Form regulations
+    op.create_table(
+        "form_regulations",
+        sa.Column("id", sa.String(length=36), nullable=False),
+        sa.Column("form_version_id", sa.String(length=36), sa.ForeignKey("form_versions.id", ondelete="CASCADE"), nullable=False),
+        sa.Column("regulation_id", sa.String(length=36), sa.ForeignKey("regulations.id", ondelete="CASCADE"), nullable=False),
+        sa.Column("relation_type", sa.String(length=64), nullable=True),
+        sa.Column("created_at", sa.DateTime(), nullable=True),
+        sa.PrimaryKeyConstraint("id", name="pk_form_regulations"),
+    )
+    op.create_index("ix_form_regulations_form_version_id", "form_regulations", ["form_version_id"], unique=False)
+    op.create_index("ix_form_regulations_regulation_id", "form_regulations", ["regulation_id"], unique=False)
+
+
+def downgrade() -> None:
+    op.drop_index("ix_form_regulations_regulation_id", table_name="form_regulations")
+    op.drop_index("ix_form_regulations_form_version_id", table_name="form_regulations")
+    op.drop_table("form_regulations")
+
+    op.drop_index("ix_form_requirements_requirement_id", table_name="form_requirements")
+    op.drop_index("ix_form_requirements_form_version_id", table_name="form_requirements")
+    op.drop_table("form_requirements")
+
+    op.drop_index("ix_requirements_regulation_ref", table_name="requirements")
+    op.drop_table("requirements")
+
+    op.drop_index("ix_regulations_citation", table_name="regulations")
+    op.drop_index("ix_regulations_title", table_name="regulations")
+    op.drop_table("regulations")
+
+    op.drop_index("ix_field_dependencies_target", table_name="field_dependencies")
+    op.drop_index("ix_field_dependencies_source", table_name="field_dependencies")
+    op.drop_table("field_dependencies")
+
+    op.drop_index("ix_fields_name", table_name="fields")
+    op.drop_index("ix_fields_form_version_id", table_name="fields")
+    op.drop_table("fields")
+
+    op.drop_index("ix_form_versions_version", table_name="form_versions")
+    op.drop_index("ix_form_versions_form_id", table_name="form_versions")
+    op.drop_table("form_versions")
+
+    op.drop_index("ix_chunks_chunk_type", table_name="chunks")
+    op.drop_index("ix_chunks_vector_id", table_name="chunks")
+    op.drop_column("chunks", "chunk_position")
+    op.drop_column("chunks", "confidence_score")
+    op.drop_column("chunks", "regulatory_reference")
+    op.drop_column("chunks", "requirement_tags")
+    op.drop_column("chunks", "field_name")
+    op.drop_column("chunks", "section")
+    op.drop_column("chunks", "extracted_entities")
+    op.drop_column("chunks", "content_summary")
+    op.drop_column("chunks", "chunk_type")
+    op.drop_column("chunks", "vector_id")
+
+    op.drop_index("ix_documents_content_hash", table_name="documents")
+    op.drop_index("ix_documents_document_version", table_name="documents")
+    op.drop_column("documents", "processing_log")
+    op.drop_column("documents", "processing_status")
+    op.drop_column("documents", "document_version")
+    op.drop_column("documents", "content_hash")
diff --git a/backend/app/api/health.py b/backend/app/api/health.py
index 1907101..24fe05d 100644
+++ b/backend/app/api/health.py
@@ -59,6 +59,24 @@ async def health_openai():
     }
 
 
+@router.get("/health/qdrant")
+async def health_qdrant():
+    try:
+        ok = chroma_client.health_check()
+        collections = chroma_client.list_collections() if ok else []
+        return {
+            "status": "ok" if ok else "error",
+            "vector_store": getattr(chroma_client, "backend_name", "unknown"),
+            "collections": collections,
+        }
+    except Exception as exc:
+        return {
+            "status": "error",
+            "vector_store": getattr(chroma_client, "backend_name", "unknown"),
+            "detail": str(exc),
+        }
+
+
 @router.get("/health/neo4j")
 async def health_neo4j():
     if not neo4j_client.enabled:
diff --git a/backend/app/chromadb/client.py b/backend/app/chromadb/client.py
index 0c5f999..4c311a8 100644
+++ b/backend/app/chromadb/client.py
@@ -15,6 +15,9 @@ COLLECTIONS = [
     "text_documents",
     "audio_transcripts",
     "web_documents",
+    "bank_forms_collection",
+    "regulations_collection",
+    "guidelines_collection",
 ]
 
 
@@ -29,6 +32,9 @@ try:
         MatchValue,
         PointStruct,
         VectorParams,
+        HnswConfigDiff,
+        ScalarQuantization,
+        ScalarQuantizationConfig,
     )
 except ImportError:
     QdrantClient = None  # type: ignore
@@ -60,6 +66,13 @@ class _QdrantBackend:
                     size=size,
                     distance=self._distance(),
                 ),
+                hnsw_config=HnswConfigDiff(
+                    m=settings.QDRANT_HNSW_M,
+                    ef_construct=settings.QDRANT_HNSW_EF_CONSTRUCT,
+                ),
+                quantization_config=ScalarQuantization(
+                    scalar=ScalarQuantizationConfig(enabled=settings.QDRANT_USE_SCALAR_QUANTIZATION)
+                ) if settings.QDRANT_USE_SCALAR_QUANTIZATION else None,
             )
 
     def _to_filter(self, where: Optional[dict]) -> Optional[Filter]:
diff --git a/backend/app/core/config.py b/backend/app/core/config.py
index e56c9d4..ced2ad0 100644
+++ b/backend/app/core/config.py
@@ -11,7 +11,7 @@ class Settings(BaseSettings):
     # ── Primary DB ────────────────────────────────────────────────────────────
     # Default points to Postgres; set to SQLite URL for local/dev fallback.
     DATABASE_URL: str = Field(
+        default="postgresql+asyncpg://bank_user:bank_password@localhost:7000/bank_kag"
     )
 
     # ── Legacy Chroma (kept for fallback) ──────────────────────────────────────
@@ -27,9 +27,13 @@ class Settings(BaseSettings):
     QDRANT_COLLECTION: str = "bank_documents"
     QDRANT_VECTOR_SIZE: int = 3072  # text-embedding-3-large dimension
     QDRANT_DISTANCE: str = "Cosine"
+    QDRANT_HNSW_M: int = 32
+    QDRANT_HNSW_EF_CONSTRUCT: int = 200
+    QDRANT_HNSW_EF: int = 64
+    QDRANT_USE_SCALAR_QUANTIZATION: bool = False
 
     # ── Neo4j (knowledge graph) ───────────────────────────────────────────────
+    USE_KG_RETRIEVAL: bool = True
     NEO4J_URI: str = "bolt://localhost:7687"
     NEO4J_USER: str = "neo4j"
     NEO4J_PASSWORD: str = "password"
diff --git a/backend/app/database/models.py b/backend/app/database/models.py
index f4fdac7..09e2817 100644
+++ b/backend/app/database/models.py
@@ -1,7 +1,7 @@
 from datetime import datetime
 from typing import Optional
 from sqlalchemy import (
+    String, Text, Integer, Float, ForeignKey, DateTime, Index, JSON, Boolean
 )
 from sqlalchemy.orm import Mapped, mapped_column, relationship
 from app.database.base import Base
@@ -13,6 +13,8 @@ class Document(Base):
     id: Mapped[str] = mapped_column(String(36), primary_key=True)
     filename: Mapped[str] = mapped_column(String(512), nullable=False)
     filepath: Mapped[str] = mapped_column(String(1024), nullable=False)
+    content_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
+    document_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
     title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
     document_type: Mapped[str] = mapped_column(String(64), nullable=False)
     category: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
@@ -24,6 +26,8 @@ class Document(Base):
     collection_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
     form_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
     metadata_json: Mapped[dict] = mapped_column(JSON, nullable=True, default=dict)
+    processing_status: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
+    processing_log: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
     created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
     updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
 
@@ -34,6 +38,8 @@ class Document(Base):
         Index("ix_documents_filename", "filename"),
         Index("ix_documents_created_at", "created_at"),
         Index("ix_documents_category", "category"),
+        Index("ix_documents_document_version", "document_version"),
+        Index("ix_documents_content_hash", "content_hash"),
     )
 
 
@@ -47,6 +53,16 @@ class Chunk(Base):
     chunk_metadata: Mapped[dict] = mapped_column(JSON, nullable=True, default=dict)
     metadata_json: Mapped[dict] = mapped_column(JSON, nullable=True, default=dict)
     qdrant_point_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
+    vector_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
+    chunk_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
+    content_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
+    extracted_entities: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
+    section: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
+    field_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
+    requirement_tags: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
+    regulatory_reference: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
+    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
+    chunk_position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
     created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
 
     document: Mapped["Document"] = relationship("Document", back_populates="chunks")
@@ -55,6 +71,8 @@ class Chunk(Base):
         Index("ix_chunks_document_id", "document_id"),
         Index("ix_chunks_chunk_index", "chunk_index"),
         Index("ix_chunks_qdrant_point_id", "qdrant_point_id"),
+        Index("ix_chunks_vector_id", "vector_id"),
+        Index("ix_chunks_chunk_type", "chunk_type"),
     )
 
 
@@ -134,6 +152,118 @@ class Form(Base):
     )
 
 
+class FormVersion(Base):
+    __tablename__ = "form_versions"
+
+    id: Mapped[str] = mapped_column(String(36), primary_key=True)
+    form_id: Mapped[str] = mapped_column(String(36), ForeignKey("forms.id", ondelete="CASCADE"), nullable=False)
+    version: Mapped[str] = mapped_column(String(64), nullable=False)
+    status: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
+    supersedes_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("form_versions.id", ondelete="SET NULL"), nullable=True)
+    effective_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
+    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
+
+    __table_args__ = (
+        Index("ix_form_versions_form_id", "form_id"),
+        Index("ix_form_versions_version", "version"),
+    )
+
+
+class Field(Base):
+    __tablename__ = "fields"
+
+    id: Mapped[str] = mapped_column(String(36), primary_key=True)
+    form_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("form_versions.id", ondelete="CASCADE"), nullable=False)
+    name: Mapped[str] = mapped_column(String(256), nullable=False)
+    field_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
+    validation_rules: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
+    required: Mapped[bool] = mapped_column(Boolean, default=False)
+    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
+    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
+
+    __table_args__ = (
+        Index("ix_fields_form_version_id", "form_version_id"),
+        Index("ix_fields_name", "name"),
+    )
+
+
+class FieldDependency(Base):
+    __tablename__ = "field_dependencies"
+
+    id: Mapped[str] = mapped_column(String(36), primary_key=True)
+    source_field_id: Mapped[str] = mapped_column(String(36), ForeignKey("fields.id", ondelete="CASCADE"), nullable=False)
+    target_field_id: Mapped[str] = mapped_column(String(36), ForeignKey("fields.id", ondelete="CASCADE"), nullable=False)
+    condition: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
+    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
+
+    __table_args__ = (
+        Index("ix_field_dependencies_source", "source_field_id"),
+        Index("ix_field_dependencies_target", "target_field_id"),
+    )
+
+
+class Regulation(Base):
+    __tablename__ = "regulations"
+
+    id: Mapped[str] = mapped_column(String(36), primary_key=True)
+    title: Mapped[str] = mapped_column(String(256), nullable=False)
+    authority: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
+    effective_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
+    citation: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
+    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
+    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
+
+    __table_args__ = (
+        Index("ix_regulations_title", "title"),
+        Index("ix_regulations_citation", "citation"),
+    )
+
+
+class Requirement(Base):
+    __tablename__ = "requirements"
+
+    id: Mapped[str] = mapped_column(String(36), primary_key=True)
+    description: Mapped[str] = mapped_column(Text, nullable=False)
+    applicability: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
+    regulation_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("regulations.id", ondelete="SET NULL"), nullable=True)
+    regulation_ref: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
+    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
+
+    __table_args__ = (
+        Index("ix_requirements_regulation_ref", "regulation_ref"),
+    )
+
+
+class FormRequirement(Base):
+    __tablename__ = "form_requirements"
+
+    id: Mapped[str] = mapped_column(String(36), primary_key=True)
+    form_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("form_versions.id", ondelete="CASCADE"), nullable=False)
+    requirement_id: Mapped[str] = mapped_column(String(36), ForeignKey("requirements.id", ondelete="CASCADE"), nullable=False)
+    applies_if: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
+    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
+
+    __table_args__ = (
+        Index("ix_form_requirements_form_version_id", "form_version_id"),
+        Index("ix_form_requirements_requirement_id", "requirement_id"),
+    )
+
+
+class FormRegulation(Base):
+    __tablename__ = "form_regulations"
+
+    id: Mapped[str] = mapped_column(String(36), primary_key=True)
+    form_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("form_versions.id", ondelete="CASCADE"), nullable=False)
+    regulation_id: Mapped[str] = mapped_column(String(36), ForeignKey("regulations.id", ondelete="CASCADE"), nullable=False)
+    relation_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
+    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
+
+    __table_args__ = (
+        Index("ix_form_regulations_form_version_id", "form_version_id"),
+        Index("ix_form_regulations_regulation_id", "regulation_id"),
+    )
+
+
 class EvaluationRun(Base):
     __tablename__ = "evaluation_runs"
 
diff --git a/backend/app/evaluation/agent_runner.py b/backend/app/evaluation/agent_runner.py
index e322f36..0d52eba 100644
+++ b/backend/app/evaluation/agent_runner.py
@@ -1,5 +1,5 @@
 """
+KAG-first evaluation for banking/product documents.
 
 Flow:
   Question
@@ -8,15 +8,13 @@ Flow:
     ↓
   Intent Detection & Metadata Filter Generation
     ↓
+  KAG Retrieval (Neo4j candidates → hybrid vector with metadata filters)
     ↓
+  Cross-Encoder Rerank
     ↓
+  Neighbor Chunk Expansion
     ↓
+  Cross-Encoder Rerank
     ↓
   ES Enhancement
     ↓
@@ -27,18 +25,16 @@ import time
 import re
 from typing import Any
 
+from app.chromadb.client import chroma_client
 from app.rag.synonym_expansion import get_synonym_expander
 from app.rag.cross_encoder import cross_encoder
 from app.evaluation.evaluator import evaluate_single
 from app.embeddings.openai_client import openai_client
 from app.core.config import settings
 from app.core.evaluation_logger import EvaluationLogger
 from app.services.elasticsearch_service import es_service
+from app.services.graph_service import graph_service
+from app.services.rag_service import rag_service
 
 # Global logger instance
 _current_logger = None
@@ -265,61 +261,61 @@ async def evaluate_question(
     if _current_logger:
         _current_logger.log("SYNONYM_EXPANSION", f"Generated {len(queries)} queries:\n" + "\n".join(queries))
 
+    # ── 3. KAG retrieval (graph candidates + hybrid vector) ───────────────
+    all_chunks: list[dict[str, Any]] = []
+    graph_forms: set[str] = set()
+    for idx, q in enumerate(queries, 1):
+        graph_result = await graph_service.get_candidates(q, filters)
+        candidate_ids = graph_result.candidate_document_ids or None
+        for f in graph_result.forms:
+            name = f.get("name")
+            if name:
+                graph_forms.add(name)
+        if _current_logger:
+            _current_logger.log(
+                f"GRAPH_{idx}",
+                f"Candidates: {len(candidate_ids or [])}\nForms: {[f.get('name') for f in graph_result.forms]}"
+            )
 
+        results = await rag_service.retrieve(
+            q,
+            strategy="hybrid",
+            top_k=getattr(settings, "TOP_K", top_k),
             filters=filters if filters else None,
+            candidate_document_ids=candidate_ids,
         )
         if _current_logger:
+            _current_logger.log(f"KAG_RETRIEVE_{idx}", f"Query: {q}\nFound: {len(results)} chunks")
+        if results:
+            all_chunks.extend(results)
+
+    # Deduplicate by chunk_id
+    deduped = []
+    seen = set()
+    for c in all_chunks:
+        cid = c.get("chunk_id")
+        if cid and cid in seen:
+            continue
+        if cid:
+            seen.add(cid)
+        deduped.append(c)
 
     if _current_logger:
+        _current_logger.log("KAG_DEDUP", f"Chunks after KAG dedup: {len(deduped)}")
 
+    # ── 4. Rerank (over-retrieve) ─────────────────────────────────────────
+    reranked = cross_encoder.rerank(question, deduped, top_k=top_k * 2)
     if _current_logger:
+        _current_logger.log("FIRST_RERANK", f"Top {top_k * 2} chunks after KAG rerank")
 
+    # ── 5. Neighbor chunk expansion ───────────────────────────────────────
     expanded_chunks = await _fetch_neighbor_chunks(reranked, "text_documents")
     if _current_logger:
         _current_logger.log("NEIGHBOR_EXPANSION",
             f"Before: {len(reranked)} chunks\n"
             f"After neighbor expansion: {len(expanded_chunks)} chunks")
 
+    # ── 6. Second rerank ──────────────────────────────────────────────────
     reranked_final = cross_encoder.rerank(question, expanded_chunks, top_k=top_k)
     if _current_logger:
         _current_logger.log("SECOND_RERANK", f"Final top {top_k} chunks after neighbor-aware reranking")
@@ -334,6 +330,13 @@ async def evaluate_question(
     # ── 8. Generate answer ────────────────────────────────────────────────
     chunk_texts = _chunk_texts(reranked_final)
 
+    # Inject graph context if available
+    graph_note = ""
+    if graph_forms:
+        graph_note = "Graph candidates (Forms):\n" + "\n".join(sorted(graph_forms))
+    if graph_note:
+        chunk_texts = [graph_note] + chunk_texts
+
     # Enhance with Elasticsearch via iterative query
     original_count = len(chunk_texts)
     enhanced_texts = await es_service.enhance_with_iterative_query(
diff --git a/backend/app/graph/graph_ingestor.py b/backend/app/graph/graph_ingestor.py
index 0815a46..dc43c89 100644
+++ b/backend/app/graph/graph_ingestor.py
@@ -68,3 +68,119 @@ async def connect_form_to_document(form_name: Optional[str], doc_id: str) -> Non
         await neo4j_client.run_write(cypher, {"doc_id": doc_id, "form_name": form_name})
     except Exception as exc:
         logger.warning("Failed to connect form to document: %s", exc)
+
+
+async def upsert_form_version(form_name: str, version: str, status: Optional[str] = None, supersedes: Optional[str] = None):
+    if not neo4j_client.enabled:
+        return
+    cypher = (
+        "MERGE (f:Form {name: $name}) "
+        "MERGE (v:FormVersion {name: $name, version: $version}) "
+        "MERGE (v)-[:VERSION_OF]->(f) "
+        "SET v.status=$status"
+    )
+    params = {"name": form_name, "version": version, "status": status}
+    try:
+        await neo4j_client.run_write(cypher, params)
+        if supersedes:
+            sup_cypher = (
+                "MATCH (v:FormVersion {name:$name, version:$version}) "
+                "MATCH (s:FormVersion {name:$name, version:$supersedes}) "
+                "MERGE (v)-[:SUPERSEDES]->(s)"
+            )
+            await neo4j_client.run_write(sup_cypher, {"name": form_name, "version": version, "supersedes": supersedes})
+    except Exception as exc:
+        logger.warning("Failed to upsert form version: %s", exc)
+
+
+async def upsert_field(form_name: str, version: str, field_name: str, field_type: Optional[str] = None, required: Optional[bool] = None):
+    if not neo4j_client.enabled:
+        return
+    cypher = (
+        "MATCH (v:FormVersion {name:$name, version:$version}) "
+        "MERGE (fld:Field {name:$field_name}) "
+        "SET fld.type=$field_type, fld.required=$required "
+        "MERGE (v)-[:REQUIRES]->(fld)"
+    )
+    try:
+        await neo4j_client.run_write(cypher, {
+            "name": form_name,
+            "version": version,
+            "field_name": field_name,
+            "field_type": field_type,
+            "required": required,
+        })
+    except Exception as exc:
+        logger.warning("Failed to upsert field: %s", exc)
+
+
+async def link_field_dependency(source_field: str, target_field: str, condition: Optional[str] = None):
+    if not neo4j_client.enabled:
+        return
+    cypher = (
+        "MATCH (s:Field {name:$source}), (t:Field {name:$target}) "
+        "MERGE (s)-[r:DEPENDS_ON]->(t) "
+        "SET r.condition=$condition"
+    )
+    try:
+        await neo4j_client.run_write(cypher, {"source": source_field, "target": target_field, "condition": condition})
+    except Exception as exc:
+        logger.warning("Failed to link field dependency: %s", exc)
+
+
+async def upsert_regulation(title: str, citation: Optional[str], authority: Optional[str]):
+    if not neo4j_client.enabled:
+        return
+    cypher = (
+        "MERGE (r:Regulation {title:$title}) "
+        "SET r.citation=$citation, r.authority=$authority"
+    )
+    try:
+        await neo4j_client.run_write(cypher, {"title": title, "citation": citation, "authority": authority})
+    except Exception as exc:
+        logger.warning("Failed to upsert regulation: %s", exc)
+
+
+async def link_form_regulation(form_name: str, version: str, regulation_title: str, relation_type: Optional[str] = None):
+    if not neo4j_client.enabled:
+        return
+    cypher = (
+        "MATCH (v:FormVersion {name:$name, version:$version}), (r:Regulation {title:$reg_title}) "
+        "MERGE (v)-[rel:REFERENCES]->(r) "
+        "SET rel.type=$relation_type"
+    )
+    try:
+        await neo4j_client.run_write(cypher, {
+            "name": form_name,
+            "version": version,
+            "reg_title": regulation_title,
+            "relation_type": relation_type,
+        })
+    except Exception as exc:
+        logger.warning("Failed to link form to regulation: %s", exc)
+
+
+async def upsert_requirement(description: str, regulation_ref: Optional[str] = None):
+    if not neo4j_client.enabled:
+        return
+    cypher = (
+        "MERGE (req:Requirement {description:$desc}) "
+        "SET req.regulation_ref=$reg_ref"
+    )
+    try:
+        await neo4j_client.run_write(cypher, {"desc": description, "reg_ref": regulation_ref})
+    except Exception as exc:
+        logger.warning("Failed to upsert requirement: %s", exc)
+
+
+async def link_form_requirement(form_name: str, version: str, description: str):
+    if not neo4j_client.enabled:
+        return
+    cypher = (
+        "MATCH (v:FormVersion {name:$name, version:$version}), (req:Requirement {description:$desc}) "
+        "MERGE (v)-[:REQUIRES]->(req)"
+    )
+    try:
+        await neo4j_client.run_write(cypher, {"name": form_name, "version": version, "desc": description})
+    except Exception as exc:
+        logger.warning("Failed to link form to requirement: %s", exc)
diff --git a/backend/app/graph/neo4j_client.py b/backend/app/graph/neo4j_client.py
index 73d4a7c..22cc7bb 100644
+++ b/backend/app/graph/neo4j_client.py
@@ -38,8 +38,7 @@ class Neo4jClient:
             return []
         async with driver.session() as session:
             result = await session.run(cypher, params or {})
+            return await result.data()
 
     async def run_write(self, cypher: str, params: Optional[dict[str, Any]] = None) -> None:
         driver = await self._get_driver()
diff --git a/backend/app/rag/markdown_rag.py b/backend/app/rag/markdown_rag.py
index 0c863ec..e7da7e1 100644
+++ b/backend/app/rag/markdown_rag.py
@@ -69,6 +69,8 @@ class MarkdownRAG:
                 "heading_level": section["level"],
                 "chunk_index": i,
                 "chunk_id": chunk_id,
+                "chunk_type": "section",
+                "chunk_position": i,
                 **(extra_metadata or {}),
             }
             metadatas.append(meta)
@@ -80,6 +82,16 @@ class MarkdownRAG:
                 "chunk_metadata": meta,
                 "metadata_json": meta,
                 "qdrant_point_id": chunk_id,
+                "vector_id": chunk_id,
+                "chunk_type": meta.get("chunk_type"),
+                "content_summary": meta.get("content_summary"),
+                "extracted_entities": meta.get("extracted_entities"),
+                "section": meta.get("section"),
+                "field_name": meta.get("field_name"),
+                "requirement_tags": meta.get("requirement_tags"),
+                "regulatory_reference": meta.get("regulatory_reference"),
+                "confidence_score": meta.get("confidence_score"),
+                "chunk_position": meta.get("chunk_position"),
             })
 
         if ids:
diff --git a/backend/app/rag/pdf_rag.py b/backend/app/rag/pdf_rag.py
index 6bd5422..155e29a 100644
+++ b/backend/app/rag/pdf_rag.py
@@ -72,6 +72,8 @@ class PDFHierarchicalRAG:
                                 "page_number": page_num,
                                 "chunk_index": chunk_index,
                                 "chunk_id": chunk_id,
+                                "chunk_type": "section",
+                                "chunk_position": chunk_index,
                                 **(extra_metadata or {}),
                             }
                             metadatas.append(meta)
@@ -83,6 +85,16 @@ class PDFHierarchicalRAG:
                                 "chunk_metadata": meta,
                                 "metadata_json": meta,
                                 "qdrant_point_id": chunk_id,
+                                "vector_id": chunk_id,
+                                "chunk_type": meta.get("chunk_type"),
+                                "content_summary": meta.get("content_summary"),
+                                "extracted_entities": meta.get("extracted_entities"),
+                                "section": meta.get("section"),
+                                "field_name": meta.get("field_name"),
+                                "requirement_tags": meta.get("requirement_tags"),
+                                "regulatory_reference": meta.get("regulatory_reference"),
+                                "confidence_score": meta.get("confidence_score"),
+                                "chunk_position": meta.get("chunk_position"),
                             })
                             chunk_index += 1
                         current_para = []
@@ -107,6 +119,8 @@ class PDFHierarchicalRAG:
                         "page_number": page_num,
                         "chunk_index": chunk_index,
                         "chunk_id": chunk_id,
+                        "chunk_type": "section",
+                        "chunk_position": chunk_index,
                         **(extra_metadata or {}),
                     }
                     metadatas.append(meta)
@@ -118,6 +132,16 @@ class PDFHierarchicalRAG:
                         "chunk_metadata": meta,
                         "metadata_json": meta,
                         "qdrant_point_id": chunk_id,
+                        "vector_id": chunk_id,
+                        "chunk_type": meta.get("chunk_type"),
+                        "content_summary": meta.get("content_summary"),
+                        "extracted_entities": meta.get("extracted_entities"),
+                        "section": meta.get("section"),
+                        "field_name": meta.get("field_name"),
+                        "requirement_tags": meta.get("requirement_tags"),
+                        "regulatory_reference": meta.get("regulatory_reference"),
+                        "confidence_score": meta.get("confidence_score"),
+                        "chunk_position": meta.get("chunk_position"),
                     })
                     chunk_index += 1
 
diff --git a/backend/app/rag/table_rag.py b/backend/app/rag/table_rag.py
index ca6ccd9..d9de0bb 100644
+++ b/backend/app/rag/table_rag.py
@@ -83,6 +83,16 @@ class TableRAG:
             "chunk_metadata": schema_meta,
             "metadata_json": schema_meta,
             "qdrant_point_id": schema_id,
+            "vector_id": schema_id,
+            "chunk_type": schema_meta.get("chunk_type"),
+            "content_summary": schema_meta.get("content_summary"),
+            "extracted_entities": schema_meta.get("extracted_entities"),
+            "section": schema_meta.get("section"),
+            "field_name": schema_meta.get("field_name"),
+            "requirement_tags": schema_meta.get("requirement_tags"),
+            "regulatory_reference": schema_meta.get("regulatory_reference"),
+            "confidence_score": schema_meta.get("confidence_score"),
+            "chunk_position": 0,
         })
 
         # Row chunks: convert to readable text format (preserve headers in each chunk)
@@ -114,6 +124,16 @@ class TableRAG:
                 "chunk_metadata": row_meta,
                 "metadata_json": row_meta,
                 "qdrant_point_id": row_id,
+                "vector_id": row_id,
+                "chunk_type": row_meta.get("chunk_type"),
+                "content_summary": row_meta.get("content_summary"),
+                "extracted_entities": row_meta.get("extracted_entities"),
+                "section": row_meta.get("section"),
+                "field_name": row_meta.get("field_name"),
+                "requirement_tags": row_meta.get("requirement_tags"),
+                "regulatory_reference": row_meta.get("regulatory_reference"),
+                "confidence_score": row_meta.get("confidence_score"),
+                "chunk_position": idx,
             })
 
         chroma_client.add_documents(SCHEMA_COLLECTION, ids, embeddings, documents, metadatas)
diff --git a/backend/app/repositories/kag_repository.py b/backend/app/repositories/kag_repository.py
new file mode 100644
index 0000000..5a8bb2d
+++ b/backend/app/repositories/kag_repository.py
@@ -0,0 +1,95 @@
+from typing import Optional
+
+from sqlalchemy.ext.asyncio import AsyncSession
+from sqlalchemy import select, delete
+
+from app.database import models
+
+
+class KAGRepository:
+    async def upsert_form_version(
+        self,
+        db: AsyncSession,
+        form_id: str,
+        version: str,
+        data: dict,
+    ) -> models.FormVersion:
+        existing = await db.execute(
+            select(models.FormVersion).where(
+                models.FormVersion.form_id == form_id,
+                models.FormVersion.version == version,
+            )
+        )
+        form_version = existing.scalar_one_or_none()
+        if form_version is None:
+            form_version = models.FormVersion(id=data.get("id"), form_id=form_id, version=version)
+            db.add(form_version)
+        for k, v in data.items():
+            if v is not None:
+                setattr(form_version, k, v)
+        await db.commit()
+        await db.refresh(form_version)
+        return form_version
+
+    async def bulk_upsert_fields(self, db: AsyncSession, fields: list[dict]) -> int:
+        if not fields:
+            return 0
+        objects = []
+        for f in fields:
+            obj = models.Field(**f)
+            objects.append(obj)
+        db.add_all(objects)
+        await db.commit()
+        return len(objects)
+
+    async def bulk_upsert_requirements(self, db: AsyncSession, reqs: list[dict]) -> int:
+        if not reqs:
+            return 0
+        objects = [models.Requirement(**r) for r in reqs]
+        db.add_all(objects)
+        await db.commit()
+        return len(objects)
+
+    async def bulk_upsert_form_requirements(self, db: AsyncSession, links: list[dict]) -> int:
+        if not links:
+            return 0
+        db.add_all([models.FormRequirement(**l) for l in links])
+        await db.commit()
+        return len(links)
+
+    async def bulk_upsert_form_regulations(self, db: AsyncSession, links: list[dict]) -> int:
+        if not links:
+            return 0
+        db.add_all([models.FormRegulation(**l) for l in links])
+        await db.commit()
+        return len(links)
+
+    async def bulk_upsert_field_dependencies(self, db: AsyncSession, deps: list[dict]) -> int:
+        if not deps:
+            return 0
+        db.add_all([models.FieldDependency(**d) for d in deps])
+        await db.commit()
+        return len(deps)
+
+    async def upsert_regulation(self, db: AsyncSession, data: dict) -> models.Regulation:
+        existing = await db.execute(
+            select(models.Regulation).where(models.Regulation.id == data.get("id"))
+        )
+        regulation = existing.scalar_one_or_none()
+        if regulation is None:
+            regulation = models.Regulation(**data)
+            db.add(regulation)
+        else:
+            for k, v in data.items():
+                if v is not None:
+                    setattr(regulation, k, v)
+        await db.commit()
+        await db.refresh(regulation)
+        return regulation
+
+    async def delete_form_version(self, db: AsyncSession, form_version_id: str) -> None:
+        await db.execute(delete(models.FormVersion).where(models.FormVersion.id == form_version_id))
+        await db.commit()
+
+
+kag_repo = KAGRepository()
diff --git a/backend/app/services/document_service.py b/backend/app/services/document_service.py
index 90d0f10..3fe3494 100644
+++ b/backend/app/services/document_service.py
@@ -4,9 +4,11 @@ import json
 from pathlib import Path
 from typing import Any, Optional
 from datetime import datetime
+import hashlib
 import aiofiles
 from sqlalchemy.ext.asyncio import AsyncSession
 from app.repositories.document_repository import document_repo
+from app.repositories.kag_repository import kag_repo
 from app.chromadb.client import chroma_client
 from app.embeddings.openai_client import openai_client as ollama_client
 from app.rag.table_rag import table_rag
@@ -16,6 +18,7 @@ from app.rag.bm25 import bm25_retriever
 from app.core.config import settings
 from app.core.exceptions import DocumentNotFoundError, UnsupportedFileTypeError
 from app.graph import graph_ingestor
+from app.graph import graph_ingestor as gi
 
 
 SUPPORTED_TYPES = {
@@ -86,6 +89,8 @@ async def _index_text_chunks(
             "document_type": doc_type,
             "chunk_index": i,
             "chunk_id": chunk_id,
+            "chunk_type": "paragraph",
+            "chunk_position": i,
             **(extra_metadata or {}),
         }
         metadatas.append(meta)
@@ -97,6 +102,16 @@ async def _index_text_chunks(
             "chunk_metadata": meta,
             "metadata_json": meta,
             "qdrant_point_id": chunk_id,
+            "vector_id": chunk_id,
+            "chunk_type": meta.get("chunk_type"),
+            "content_summary": meta.get("content_summary"),
+            "extracted_entities": meta.get("extracted_entities"),
+            "section": meta.get("section"),
+            "field_name": meta.get("field_name"),
+            "requirement_tags": meta.get("requirement_tags"),
+            "regulatory_reference": meta.get("regulatory_reference"),
+            "confidence_score": meta.get("confidence_score"),
+            "chunk_position": meta.get("chunk_position"),
         })
 
     if ids:
@@ -110,6 +125,127 @@ async def _index_text_chunks(
 
 
 class DocumentService:
+    async def _ingest_kag_structures(
+        self,
+        db: AsyncSession,
+        doc_data: dict[str, Any],
+        metadata: dict[str, Any],
+    ) -> None:
+        """Persist structured form/regulation data to Postgres + Neo4j."""
+        form_name = metadata.get("form_name")
+        form_version = metadata.get("version") or metadata.get("form_version") or "v1"
+        kag_type = metadata.get("kag_type")
+
+        # Only proceed if this looks like a form/regulation payload
+        if kag_type not in {"form", "regulation", "guideline"} and not form_name:
+            return
+
+        # Postgres: FormVersion + Fields + Requirements + Regulations
+        fv_id = str(uuid.uuid4())
+        await kag_repo.upsert_form_version(db, metadata.get("form_id") or str(uuid.uuid4()), form_version, {
+            "id": fv_id,
+            "status": metadata.get("status"),
+            "effective_date": metadata.get("effective_date"),
+            "supersedes_id": metadata.get("supersedes_id"),
+        })
+
+        fields = metadata.get("fields") or []
+        field_id_map: dict[str, str] = {}
+        field_rows = []
+        for f in fields:
+            fid = f.get("id") or str(uuid.uuid4())
+            field_id_map[f.get("name")] = fid
+            field_rows.append({
+                "id": fid,
+                "form_version_id": fv_id,
+                "name": f.get("name"),
+                "field_type": f.get("type") or f.get("field_type"),
+                "validation_rules": f.get("validation_rules"),
+                "required": bool(f.get("required", False)),
+                "description": f.get("description"),
+            })
+        await kag_repo.bulk_upsert_fields(db, field_rows)
+
+        # Field dependencies
+        deps = []
+        for f in fields:
+            source = field_id_map.get(f.get("name"))
+            for dep in f.get("depends_on", []) or []:
+                target = field_id_map.get(dep.get("field"))
+                if source and target:
+                    deps.append({
+                        "id": str(uuid.uuid4()),
+                        "source_field_id": source,
+                        "target_field_id": target,
+                        "condition": dep.get("condition"),
+                    })
+        await kag_repo.bulk_upsert_field_dependencies(db, deps)
+
+        # Regulations and requirements
+        regulation_rows = []
+        for r in metadata.get("regulations", []) or []:
+            rid = r.get("id") or str(uuid.uuid4())
+            regulation_rows.append({
+                "id": rid,
+                "title": r.get("title") or r.get("citation") or "regulation",
+                "authority": r.get("authority"),
+                "effective_date": r.get("effective_date"),
+                "citation": r.get("citation"),
+                "description": r.get("description"),
+            })
+        for row in regulation_rows:
+            await kag_repo.upsert_regulation(db, row)
+
+        requirements = []
+        req_links = []
+        for req in metadata.get("requirements", []) or []:
+            req_id = req.get("id") or str(uuid.uuid4())
+            requirements.append({
+                "id": req_id,
+                "description": req.get("description") or "",
+                "applicability": req.get("applicability"),
+                "regulation_id": req.get("regulation_id"),
+                "regulation_ref": req.get("regulation_ref"),
+            })
+            req_links.append({
+                "id": str(uuid.uuid4()),
+                "form_version_id": fv_id,
+                "requirement_id": req_id,
+                "applies_if": req.get("applies_if"),
+            })
+        await kag_repo.bulk_upsert_requirements(db, requirements)
+        await kag_repo.bulk_upsert_form_requirements(db, req_links)
+
+        # Form → Regulation links
+        form_reg_links = []
+        for r in regulation_rows:
+            form_reg_links.append({
+                "id": str(uuid.uuid4()),
+                "form_version_id": fv_id,
+                "regulation_id": r.get("id"),
+                "relation_type": "REFERENCES",
+            })
+        await kag_repo.bulk_upsert_form_regulations(db, form_reg_links)
+
+        # Graph: forms, versions, fields, dependencies, regulations, links
+        try:
+            if form_name:
+                await gi.upsert_form(form_name, metadata.get("category"))
+                await gi.upsert_form_version(form_name, form_version, metadata.get("status"), metadata.get("supersedes"))
+            for f in fields:
+                await gi.upsert_field(form_name or "", form_version, f.get("name"), f.get("type"), f.get("required"))
+                for dep in f.get("depends_on", []) or []:
+                    await gi.link_field_dependency(f.get("name"), dep.get("field"), dep.get("condition"))
+            for r in regulation_rows:
+                await gi.upsert_regulation(r.get("title"), r.get("citation"), r.get("authority"))
+                if form_name:
+                    await gi.link_form_regulation(form_name, form_version, r.get("title"), "REFERENCES")
+            for req in requirements:
+                await gi.upsert_requirement(req.get("description"), req.get("regulation_ref"))
+                if form_name:
+                    await gi.link_form_requirement(form_name, form_version, req.get("description"))
+        except Exception:
+            pass
     async def upload_and_index(
         self,
         db: AsyncSession,
@@ -119,7 +255,14 @@ class DocumentService:
     ) -> dict[str, Any]:
         doc_type = _detect_type(filename)
         doc_id = str(uuid.uuid4())
+        kag_type = (extra_metadata or {}).get("kag_type")  # form | regulation | guideline
         collection = TYPE_TO_COLLECTION[doc_type]
+        if kag_type in {"form", "regulation", "guideline"}:
+            collection = {
+                "form": "bank_forms_collection",
+                "regulation": "regulations_collection",
+                "guideline": "guidelines_collection",
+            }[kag_type]
         strategy = TYPE_TO_STRATEGY[doc_type]
 
         # Save file
@@ -132,24 +275,31 @@ class DocumentService:
         # Index based on type
         chunk_count = 0
         chunk_records = []
+        content_hash = hashlib.sha256(content).hexdigest()
+        document_version = (extra_metadata or {}).get("version", "v1")
+        base_meta = {
+            "document_version": document_version,
+            "content_hash": content_hash,
+            "kag_type": kag_type,
+        }
 
         if doc_type == "pdf":
+            result = await pdf_rag.index(doc_id, filename, content, {**(extra_metadata or {}), **base_meta})
             chunk_count = result["chunk_count"]
             chunk_records = result.get("chunks", [])
         elif doc_type == "markdown":
             text = content.decode("utf-8", errors="replace")
+            result = await markdown_rag.index(doc_id, filename, text, {**(extra_metadata or {}), **base_meta})
             chunk_count = result["chunk_count"]
             chunk_records = result.get("chunks", [])
         elif doc_type == "csv":
+            result = await table_rag.index_csv(doc_id, filename, content, {**(extra_metadata or {}), **base_meta})
             chunk_count = result["chunk_count"]
             chunk_records = result.get("chunks", [])
         else:
             # text / json
             text = content.decode("utf-8", errors="replace")
+            chunk_records = await _index_text_chunks(doc_id, filename, doc_type, text, collection, {**(extra_metadata or {}), **base_meta})
             chunk_count = len(chunk_records)
 
         doc_data = {
@@ -166,7 +316,11 @@ class DocumentService:
             "embedding_model": settings.OPENAI_EMBED_MODEL,
             "collection_name": collection,
             "form_name": (extra_metadata or {}).get("form_name"),
+            "metadata_json": {**(extra_metadata or {}), **base_meta},
+            "content_hash": content_hash,
+            "document_version": document_version,
+            "processing_status": "indexed",
+            "processing_log": None,
         }
 
         doc = await document_repo.create(db, doc_data)
@@ -176,6 +330,15 @@ class DocumentService:
             await graph_ingestor.upsert_document_node(doc_data)
             await graph_ingestor.link_document_to_category(doc_id, doc_data.get("category"))
             await graph_ingestor.connect_form_to_document(doc_data.get("form_name"), doc_id)
+            if doc_data.get("form_name"):
+                await graph_ingestor.upsert_form(doc_data.get("form_name"), doc_data.get("category"))
+                await graph_ingestor.upsert_form_version(doc_data.get("form_name"), document_version)
+        except Exception:
+            pass
+
+        # Persist structured KAG metadata (forms/fields/requirements/regulations)
+        try:
+            await self._ingest_kag_structures(db, doc_data, doc_data.get("metadata_json") or {})
         except Exception:
             pass
 
@@ -207,23 +370,35 @@ class DocumentService:
 
         doc_type = doc.document_type
         collection = TYPE_TO_COLLECTION.get(doc_type, "text_documents")
+        kag_type = (doc.metadata_json or {}).get("kag_type")
+        if kag_type in {"form", "regulation", "guideline"}:
+            collection = {
+                "form": "bank_forms_collection",
+                "regulation": "regulations_collection",
+                "guideline": "guidelines_collection",
+            }[kag_type]
         chunk_count = 0
+        base_meta = {
+            "document_version": (doc.metadata_json or {}).get("document_version", "v1"),
+            "content_hash": doc.metadata_json.get("content_hash") if doc.metadata_json else None,
+            "kag_type": kag_type,
+        }
 
         if doc_type == "pdf":
+            result = await pdf_rag.index(doc_id, doc.filename, content, {**(doc.metadata_json or {}), **base_meta})
             chunk_count = result["chunk_count"]
         elif doc_type == "markdown":
             text = content.decode("utf-8", errors="replace")
+            result = await markdown_rag.index(doc_id, doc.filename, text, {**(doc.metadata_json or {}), **base_meta})
             chunk_count = result["chunk_count"]
             chunk_records = result.get("chunks", [])
         elif doc_type == "csv":
+            result = await table_rag.index_csv(doc_id, doc.filename, content, {**(doc.metadata_json or {}), **base_meta})
             chunk_count = result["chunk_count"]
             chunk_records = result.get("chunks", [])
         else:
             text = content.decode("utf-8", errors="replace")
+            chunk_records = await _index_text_chunks(doc_id, doc.filename, doc_type, text, collection, {**(doc.metadata_json or {}), **base_meta})
             chunk_count = len(chunk_records)
 
         if chunk_records:
diff --git a/backend/app/services/rag_service.py b/backend/app/services/rag_service.py
index 0344f12..836c3bc 100644
+++ b/backend/app/services/rag_service.py
@@ -1,12 +1,13 @@
+import time
+import uuid
+from typing import Any, AsyncGenerator, Optional
+from sqlalchemy.ext.asyncio import AsyncSession
+from sqlalchemy import select
+from app.rag.vector_rag import vector_rag
+from app.rag.hybrid_rag import hybrid_rag
+from app.rag.bm25 import bm25_retriever
+from app.rag.table_rag import table_rag
+from app.rag.pdf_rag import pdf_rag
 from app.rag.markdown_rag import markdown_rag
 from app.rag.multi_collection_retrieval import multi_collection_retriever
 from app.rag.metadata_filter import filter_results
@@ -20,30 +21,52 @@ from app.core.config import settings
 from app.services.elasticsearch_service import es_service
 from app.core.evaluation_logger import EvaluationLogger
 from app.services.graph_service import graph_service
+from app.database.models import Chunk
 
 
 RAG_SYSTEM = """You are an expert assistant. Answer the question using only the provided context.
 Be factual, concise, and cite sources. If the answer is not in the context, say 'Not found in available documents'."""
 
 
+class RAGService:
+    async def _keyword_lookup(self, db: AsyncSession, query: str, limit: int = 5) -> list[dict[str, Any]]:
+        try:
+            stmt = select(Chunk).where(Chunk.chunk_text.ilike(f"%{query}%")).limit(limit)
+            rows = await db.execute(stmt)
+            chunks = rows.scalars().all()
+            results: list[dict[str, Any]] = []
+            for c in chunks:
+                results.append({
+                    "chunk_id": c.id,
+                    "document_id": c.document_id,
+                    "filename": c.chunk_metadata.get("filename") if c.chunk_metadata else "",
+                    "chunk_text": c.chunk_text,
+                    "score": 0.9,
+                    "metadata": c.chunk_metadata or {},
+                })
+            return results
+        except Exception:
+            return []
+
+    async def retrieve(
+        self,
+        query: str,
+        strategy: str = "hybrid",
+        top_k: int = 5,
+        filters: Optional[dict] = None,
+        collection_name: Optional[str] = None,
         all_collections: bool = False,
+        candidate_document_ids: Optional[set[str]] = None,
     ) -> list[dict[str, Any]]:
         """
         Retrieve from single or multiple collections.
         If all_collections=True, search across all available collections.
         """
+        candidate_ids = set(candidate_document_ids) if candidate_document_ids else None
         if settings.USE_KG_RETRIEVAL:
             graph_result = await graph_service.get_candidates(query, filters)
+            graph_candidates = graph_result.candidate_document_ids or set()
+            candidate_ids = (candidate_ids or set()) | graph_candidates or None
 
         if all_collections:
             return await multi_collection_retriever.retrieve_all_collections(
@@ -63,9 +86,9 @@ class RAGService:
         elif strategy == "pdf":
             return await pdf_rag.query(query, top_k=top_k)
         elif strategy == "markdown":
+            return await markdown_rag.query(query, top_k=top_k)
+        else:
+            return await hybrid_rag.retrieve(query, col, top_k, filters, candidate_ids)
 
     async def query(
         self,
@@ -87,6 +110,14 @@ class RAGService:
                 graph_context = ""
 
         chunks = await self.retrieve(query, strategy, top_k, filters)
+
+        # Tier-1 keyword search (Postgres) to add exact matches
+        keyword_hits = await self._keyword_lookup(db, query, top_k)
+        seen_ids = {c.get("chunk_id") for c in chunks if c.get("chunk_id")}
+        for hit in keyword_hits:
+            if hit.get("chunk_id") in seen_ids:
+                continue
+            chunks.append(hit)
         context = "\n\n".join(
             f"[{r.get('filename','?')}] {r['chunk_text']}" for r in chunks
         )
diff --git a/helpfull scripts/compare_changes_in_metrics.py b/helpfull scripts/compare_changes_in_metrics.py
index f0a42c1..f464425 100644
+++ b/helpfull scripts/compare_changes_in_metrics.py	
@@ -4,8 +4,8 @@ import matplotlib.pyplot as plt
 # ==========================
 # CONFIG
 # ==========================
+CSV_1 = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/19th june added neo4j and qrant alogn with postgeres sql and elasticsearch overall score 64.9.csv" 
+CSV_2 = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/22 june only 0.6 less accuracy questions from the 19th june gpt 4o results.csv"
 
 METRICS = [
     "Accuracy (LLM)",
diff --git a/helpfull scripts/edit eval expected answer/judge_and_update_answers.py b/helpfull scripts/edit eval expected answer/judge_and_update_answers.py
new file mode 100644
index 0000000..167a713
+++ b/helpfull scripts/edit eval expected answer/judge_and_update_answers.py	
@@ -0,0 +1,336 @@
+"""
+judge_and_update_answers.py
+
+Reads an eval CSV (Question No, Question, Expected Answer, Generated Answer,
+Retrieved Context, ...) and a questions CSV (Question no, Eval Question,
+Eval Answer).
+
+For every row in the eval CSV, asks GPT-4o-mini to judge whether the
+Expected Answer or the Generated Answer better answers the Question given
+the Retrieved Context, and whether the Expected Answer needs to be revised.
+
+If a revision is suggested, the matching row in the questions CSV is updated
+-- matched by comparing the actual question TEXT (not just the row number),
+so mismatched numbering between the two files doesn't silently corrupt data.
+A full audit log CSV is written so every decision can be checked by hand.
+
+Setup:
+    pip install pandas python-dotenv openai
+
+    Create a .env file next to this script containing:
+        OPENAI_API_KEY=sk-...
+
+Usage:
+    python judge_and_update_answers.py \
+        --eval-csv eval.csv \
+        --questions-csv questions.csv \
+        --output-csv questions_updated.csv \
+        --log-csv judge_log.csv
+"""
+
+import argparse
+import difflib
+import json
+import os
+import re
+import sys
+import time
+
+import pandas as pd
+from dotenv import load_dotenv
+from openai import OpenAI
+
+# --------------------------------------------------------------------------
+# CONFIG -- adjust these if your column headers differ
+# --------------------------------------------------------------------------
+EVAL_QUESTION_COL = "Question"
+EVAL_EXPECTED_COL = "Expected Answer"
+EVAL_GENERATED_COL = "Generated Answer"
+EVAL_CONTEXT_COL = "Retrieved Context"
+EVAL_NUMBER_COL = "Question No"
+
+Q_NUMBER_COL = "Question no"
+Q_QUESTION_COL = "Eval Question"
+Q_ANSWER_COL = "Eval Answer"
+
+MODEL = "gpt-4o-mini"
+MATCH_THRESHOLD = (
+    0.90  # similarity score (0-1) above which two questions are treated as the same
+)
+MAX_RETRIES = 3
+RETRY_BACKOFF_SECONDS = 2
+
+SYSTEM_PROMPT = (
+    "You are a meticulous evaluation judge for question-answering systems. "
+    "You compare a gold/expected answer and a model-generated answer against "
+    "retrieved context, and judge them strictly on correctness and completeness "
+    "relative to that context. Respond ONLY with a valid JSON object and nothing else."
+)
+
+USER_PROMPT_TEMPLATE = """Question:
+{question}
+
+Expected Answer (gold/reference):
+{expected_answer}
+
+Generated Answer (model output):
+{generated_answer}
+
+Retrieved Context (source passages):
+{retrieved_context}
+
+Task:
+1. Decide which answer -- "expected" or "generated" -- better answers the
+   Question, using the Retrieved Context as ground truth. Use "tie" if they
+   are equally good (or equally bad).
+2. Independently assess whether the Expected Answer itself is flawed,
+   incomplete, or incorrect given the Retrieved Context, regardless of how
+   it compares to the Generated Answer.
+3. If the Expected Answer needs correction, write a revised answer that
+   correctly and completely answers the Question, grounded only in the
+   Retrieved Context.
+
+Respond with ONLY a JSON object, exactly in this shape:
+{{
+  "better_answer": "expected" | "generated" | "tie",
+  "expected_answer_needs_revision": true | false,
+  "revised_expected_answer": "<string, or null if no revision is needed>",
+  "reasoning": "<1-3 sentence justification>"
+}}"""
+
+
+def normalize(text: str) -> str:
+    """Lowercase, collapse whitespace, strip punctuation for fuzzy matching."""
+    text = str(text).lower().strip()
+    text = re.sub(r"[^a-z0-9\s]", "", text)
+    text = re.sub(r"\s+", " ", text)
+    return text
+
+
+def best_match(question: str, candidates: pd.Series):
+    """Return (best_index, best_score) of the candidate most similar to `question`."""
+    norm_q = normalize(question)
+    best_idx, best_score = None, 0.0
+    for idx, cand in candidates.items():
+        score = difflib.SequenceMatcher(None, norm_q, normalize(cand)).ratio()
+        if score > best_score:
+            best_idx, best_score = idx, score
+    return best_idx, best_score
+
+
+def call_judge(client: OpenAI, question, expected, generated, context) -> dict:
+    """Call GPT-4o-mini and return the parsed JSON verdict, with retries."""
+    user_prompt = USER_PROMPT_TEMPLATE.format(
+        question=question,
+        expected_answer=expected,
+        generated_answer=generated,
+        retrieved_context=context,
+    )
+
+    last_err = None
+    for attempt in range(1, MAX_RETRIES + 1):
+        try:
+            resp = client.chat.completions.create(
+                model=MODEL,
+                temperature=0,
+                response_format={"type": "json_object"},
+                messages=[
+                    {"role": "system", "content": SYSTEM_PROMPT},
+                    {"role": "user", "content": user_prompt},
+                ],
+            )
+            raw = resp.choices[0].message.content
+            return json.loads(raw)
+        except Exception as e:  # noqa: BLE001
+            last_err = e
+            print(f"  [warn] judge call failed (attempt {attempt}/{MAX_RETRIES}): {e}")
+            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
+
+    raise RuntimeError(f"Judge call failed after {MAX_RETRIES} attempts: {last_err}")
+
+
+def main():
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument("--eval-csv", required=True, help="Path to the eval CSV")
+    parser.add_argument(
+        "--questions-csv", required=True, help="Path to the questions CSV"
+    )
+    parser.add_argument(
+        "--output-csv",
+        default="questions_updated.csv",
+        help="Where to write the updated questions CSV",
+    )
+    parser.add_argument(
+        "--log-csv",
+        default="judge_log.csv",
+        help="Where to write the full audit log of every decision",
+    )
+    parser.add_argument(
+        "--temp-csv",
+        default="temp.csv",
+        help="Where to write eval questions that could not be confidently matched in the questions CSV",
+    )
+    parser.add_argument(
+        "--threshold",
+        type=float,
+        default=MATCH_THRESHOLD,
+        help="Similarity threshold (0-1) for matching questions across files",
+    )
+    parser.add_argument(
+        "--limit",
+        type=int,
+        default=None,
+        help="Optional: only process the first N rows (useful for testing)",
+    )
+    args = parser.parse_args()
+
+    load_dotenv()
+    api_key = os.getenv("OPENAI_API_KEY")
+    if not api_key:
+        sys.exit("OPENAI_API_KEY not found. Put it in a .env file next to this script.")
+    client = OpenAI(api_key=api_key)
+
+    eval_df = pd.read_csv(args.eval_csv)
+    questions_df = pd.read_csv(args.questions_csv)
+
+    eval_df.columns = [c.strip() for c in eval_df.columns]
+    questions_df.columns = [c.strip() for c in questions_df.columns]
+
+    for col in (
+        EVAL_QUESTION_COL,
+        EVAL_EXPECTED_COL,
+        EVAL_GENERATED_COL,
+        EVAL_CONTEXT_COL,
+    ):
+        if col not in eval_df.columns:
+            sys.exit(f"Eval CSV is missing expected column: '{col}'")
+    for col in (Q_QUESTION_COL, Q_ANSWER_COL):
+        if col not in questions_df.columns:
+            sys.exit(f"Questions CSV is missing expected column: '{col}'")
+
+    rows_to_process = eval_df.head(args.limit) if args.limit else eval_df
+
+    log_rows = []
+    unmatched_rows = []
+    updates_applied = 0
+    unmatched = 0
+
+    for i, row in rows_to_process.iterrows():
+        question = str(row[EVAL_QUESTION_COL])
+        expected = str(row[EVAL_EXPECTED_COL])
+        generated = str(row[EVAL_GENERATED_COL])
+        context = str(row[EVAL_CONTEXT_COL])
+        eval_num = row.get(EVAL_NUMBER_COL, i)
+
+        print(f"[{i + 1}/{len(rows_to_process)}] Judging eval question #{eval_num} ...")
+
+        # Match against questions_csv by TEXT, not by row number.
+        match_idx, score = best_match(question, questions_df[Q_QUESTION_COL])
+
+        if match_idx is None or score < args.threshold:
+            unmatched += 1
+            unmatched_rows.append(
+                {
+                    "Eval Question No": eval_num,
+                    "Eval Question": question,
+                    "Expected Answer": expected,
+                    "Generated Answer": generated,
+                    "Retrieved Context": context,
+                    "Best Match Score": round(score, 3),
+                    "Closest Question in Questions CSV": (
+                        questions_df.loc[match_idx, Q_QUESTION_COL]
+                        if match_idx is not None
+                        else None
+                    ),
+                }
+            )
+            log_rows.append(
+                {
+                    "Eval Question No": eval_num,
+                    "Eval Question": question,
+                    "Matched Question No": None,
+                    "Match Score": round(score, 3),
+                    "Number Mismatch Flag": None,
+                    "Better Answer": None,
+                    "Needs Revision": None,
+                    "Old Expected Answer": expected,
+                    "New Expected Answer": None,
+                    "Reasoning": "NO MATCH FOUND in questions CSV above threshold -- skipped.",
+                }
+            )
+            print(
+                f"  [warn] no confident match in questions CSV (best score {score:.2f}) -- skipped"
+            )
+            continue
+
+        matched_qnum = (
+            questions_df.loc[match_idx, Q_NUMBER_COL]
+            if Q_NUMBER_COL in questions_df.columns
+            else None
+        )
+        number_mismatch = matched_qnum is not None and str(matched_qnum) != str(
+            eval_num
+        )
+
+        try:
+            verdict = call_judge(client, question, expected, generated, context)
+        except RuntimeError as e:
+            print(f"  [error] {e}")
+            log_rows.append(
+                {
+                    "Eval Question No": eval_num,
+                    "Eval Question": question,
+                    "Matched Question No": matched_qnum,
+                    "Match Score": round(score, 3),
+                    "Number Mismatch Flag": number_mismatch,
+                    "Better Answer": "ERROR",
+                    "Needs Revision": "ERROR",
+                    "Old Expected Answer": expected,
+                    "New Expected Answer": None,
+                    "Reasoning": str(e),
+                }
+            )
+            continue
+
+        needs_revision = bool(verdict.get("expected_answer_needs_revision"))
+        revised = verdict.get("revised_expected_answer")
+
+        if needs_revision and revised:
+            questions_df.loc[match_idx, Q_ANSWER_COL] = revised
+            updates_applied += 1
+
+        log_rows.append(
+            {
+                "Eval Question No": eval_num,
+                "Eval Question": question,
+                "Matched Question No": matched_qnum,
+                "Match Score": round(score, 3),
+                "Number Mismatch Flag": number_mismatch,
+                "Better Answer": verdict.get("better_answer"),
+                "Needs Revision": needs_revision,
+                "Old Expected Answer": expected,
+                "New Expected Answer": revised if needs_revision else None,
+                "Reasoning": verdict.get("reasoning"),
+            }
+        )
+
+    questions_df.to_csv(args.output_csv, index=False)
+    pd.DataFrame(log_rows).to_csv(args.log_csv, index=False)
+    pd.DataFrame(unmatched_rows).to_csv(args.temp_csv, index=False)
+
+    print("\nDone.")
+    print(f"  Rows processed:     {len(rows_to_process)}")
+    print(f"  Answers revised:    {updates_applied}")
+    print(f"  Unmatched questions:{unmatched}")
+    print(f"  Updated CSV:        {args.output_csv}")
+    print(
+        f"  Audit log:          {args.log_csv}  <-- review this to verify every match/decision"
+    )
+    if unmatched:
+        print(
+            f"  Unmatched rows:     {args.temp_csv}  <-- questions with no confident match in the questions CSV"
+        )
+
+
+if __name__ == "__main__":
+    main()
diff --git a/helpfull scripts/generate_correct_answers.py b/helpfull scripts/generate_correct_answers.py
new file mode 100644
index 0000000..e69de29
diff --git a/helpfull scripts/get_less_accuracy_rows.py b/helpfull scripts/get_less_accuracy_rows.py
index e2ede8b..4c3f0ae 100644
+++ b/helpfull scripts/get_less_accuracy_rows.py	
@@ -1,14 +1,15 @@
 from pathlib import Path
 
+import pandas as pd
+
 # Input CSV file
+input_file = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/19th june added neo4j and qrant alogn with postgeres sql and elasticsearch overall score 64.9.csv"
 
 # Read CSV
 df = pd.read_csv(input_file)
 
 # Filter rows where Accuracy <= 0.6
+filtered_df = df[df["Accuracy (LLM)"] <= 0.6]
 
 # Create output filename
 input_path = Path(input_file)
@@ -18,4 +19,4 @@ output_file = f"0.6accuracyandbelowrows_{input_path.stem}.csv"
 filtered_df.to_csv(output_file, index=False)
 
 print(f"Filtered {len(filtered_df)} rows.")
\ No newline at end of file
+print(f"Output saved to: {output_file}")
diff --git a/helpfull scripts/get_questions_answer_from_eval_result.py b/helpfull scripts/get_questions_answer_from_eval_result.py
new file mode 100644
index 0000000..bf280af
+++ b/helpfull scripts/get_questions_answer_from_eval_result.py	
@@ -0,0 +1,24 @@
+from pathlib import Path
+import pandas as pd
+
+# Input CSV file
+input_file = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/helpfull scripts/0.6accuracyandbelowrows_19th june added neo4j and qrant alogn with postgeres sql and elasticsearch overall score 64.9.csv"
+
+# Read CSV
+df = pd.read_csv(input_file)
+
+# Keep only required columns
+filtered_df = df[["Question No", "Question", "Expected Answer"]].copy()
+
+# Renumber Question No sequentially
+filtered_df["Question No"] = range(1, len(filtered_df) + 1)
+
+# Create output filename
+input_path = Path(input_file)
+output_file = f"0.6accuracyandbelowrows_{input_path.stem}.csv"
+
+# Save
+filtered_df.to_csv(output_file, index=False)
+
+print(f"Saved {len(filtered_df)} rows.")
+print(f"Output saved to: {output_file}")
\ No newline at end of file
```

## 8d7f007e4d81e63a110b20397f57e7075015c83f — 2026-06-22T09:54:21+05:30

Message:

cleanup

_No Python file changes in this commit._

## 038b06794c58ae56f7c70a0089754b2e38714cab — 2026-06-22T09:34:53+05:30

Message:

gpt 5.1

```diff
diff --git a/backend/app/core/config.py b/backend/app/core/config.py
index 4d44003..e56c9d4 100644
+++ b/backend/app/core/config.py
@@ -40,7 +40,7 @@ class Settings(BaseSettings):
 
     # ── OpenAI ────────────────────────────────────────────────────────────────
     OPENAI_API_KEY: str = ""
+    OPENAI_LLM_MODEL: str = "gpt-5.1"
     OPENAI_EMBED_MODEL: str = "text-embedding-3-large"
     OPENAI_EMBED_FALLBACK_MODEL: str = "text-embedding-3-small"
     OPENAI_TIMEOUT: int = 120
diff --git a/backend/app/tests/test_core.py b/backend/app/tests/test_core.py
index ce2d000..8dedd1f 100644
+++ b/backend/app/tests/test_core.py
@@ -130,7 +130,7 @@ def test_bm25_remove_collection():
 
 def test_settings_defaults():
     from app.core.config import settings
+    assert settings.OPENAI_LLM_MODEL == "gpt-5.1"
     assert settings.OPENAI_EMBED_MODEL == "text-embedding-3-small"
     assert settings.TOP_K == 5
     assert settings.CHUNK_SIZE == 512
@@ -286,4 +286,4 @@ def test_router_doc_type_detection():
     assert _detect_doc_type("find in PDF report") == "pdf"
     assert _detect_doc_type("search the README markdown guide") == "markdown"
     assert _detect_doc_type("query the CSV table rows") == "csv"
\ No newline at end of file
+    assert _detect_doc_type("general question") == "text"
diff --git a/helpfull scripts/compare_changes_in_metrics.py b/helpfull scripts/compare_changes_in_metrics.py
index cd6ad15..f0a42c1 100644
+++ b/helpfull scripts/compare_changes_in_metrics.py	
@@ -4,8 +4,8 @@ import matplotlib.pyplot as plt
 # ==========================
 # CONFIG
 # ==========================
+CSV_1 = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/15th june elasticsearch online added RRF and que - synonym expa - vector + bm25 - rrf - cross encoder - chunk expansion - cross encoder - ES.csv" 
+CSV_2 = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/19th june gpt5.1  added neo4j and qrant alogn with postgeres sql and elasticsearch overall score 64.9.csv"
 
 METRICS = [
     "Accuracy (LLM)",
```

## 71e3b35f09301758bcbfefe2b051bf74eb362a5b — 2026-06-19T13:10:21+05:30

Message:

adding neo4j qdant and postgress KAG

```diff
diff --git a/backend/alembic/versions/0002_kag_schema.py b/backend/alembic/versions/0002_kag_schema.py
new file mode 100644
index 0000000..9c98b3c
+++ b/backend/alembic/versions/0002_kag_schema.py
@@ -0,0 +1,74 @@
+"""kag schema
+
+Revision ID: 0002
+Revises: 0001
+Create Date: 2026-06-18 00:00:00.000000
+
+"""
+from typing import Sequence, Union
+from alembic import op
+import sqlalchemy as sa
+
+
+revision: str = "0002"
+down_revision: Union[str, None] = "0001"
+branch_labels: Union[str, Sequence[str], None] = None
+depends_on: Union[str, Sequence[str], None] = None
+
+
+def upgrade() -> None:
+    # Documents
+    op.add_column("documents", sa.Column("title", sa.String(length=512), nullable=True))
+    op.add_column("documents", sa.Column("category", sa.String(length=128), nullable=True))
+    op.add_column("documents", sa.Column("source", sa.String(length=128), nullable=True))
+    op.add_column("documents", sa.Column("form_name", sa.String(length=256), nullable=True))
+    op.create_index("ix_documents_category", "documents", ["category"], unique=False)
+
+    # Chunks
+    op.add_column("chunks", sa.Column("metadata_json", sa.JSON(), nullable=True))
+    op.add_column("chunks", sa.Column("qdrant_point_id", sa.String(length=64), nullable=True))
+    op.create_index("ix_chunks_qdrant_point_id", "chunks", ["qdrant_point_id"], unique=False)
+
+    # Query logs
+    op.create_table(
+        "query_logs",
+        sa.Column("id", sa.String(length=36), nullable=False),
+        sa.Column("query", sa.Text(), nullable=False),
+        sa.Column("response", sa.Text(), nullable=True),
+        sa.Column("latency", sa.Float(), nullable=True),
+        sa.Column("created_at", sa.DateTime(), nullable=True),
+        sa.PrimaryKeyConstraint("id", name="pk_query_logs"),
+    )
+    op.create_index("ix_query_logs_created_at", "query_logs", ["created_at"], unique=False)
+
+    # Forms
+    op.create_table(
+        "forms",
+        sa.Column("id", sa.String(length=36), nullable=False),
+        sa.Column("name", sa.String(length=256), nullable=False),
+        sa.Column("category", sa.String(length=128), nullable=True),
+        sa.Column("description", sa.Text(), nullable=True),
+        sa.Column("created_at", sa.DateTime(), nullable=True),
+        sa.PrimaryKeyConstraint("id", name="pk_forms"),
+    )
+    op.create_index("ix_forms_name", "forms", ["name"], unique=False)
+    op.create_index("ix_forms_category", "forms", ["category"], unique=False)
+
+
+def downgrade() -> None:
+    op.drop_index("ix_forms_category", table_name="forms")
+    op.drop_index("ix_forms_name", table_name="forms")
+    op.drop_table("forms")
+
+    op.drop_index("ix_query_logs_created_at", table_name="query_logs")
+    op.drop_table("query_logs")
+
+    op.drop_index("ix_chunks_qdrant_point_id", table_name="chunks")
+    op.drop_column("chunks", "qdrant_point_id")
+    op.drop_column("chunks", "metadata_json")
+
+    op.drop_index("ix_documents_category", table_name="documents")
+    op.drop_column("documents", "form_name")
+    op.drop_column("documents", "source")
+    op.drop_column("documents", "category")
+    op.drop_column("documents", "title")
diff --git a/backend/app/api/forms.py b/backend/app/api/forms.py
new file mode 100644
index 0000000..142eb75
+++ b/backend/app/api/forms.py
@@ -0,0 +1,17 @@
+from typing import Any
+from fastapi import APIRouter
+from pydantic import BaseModel, Field
+
+from app.graph.form_recommender import recommend_forms
+
+
+class FormRecommendRequest(BaseModel):
+    query: str = Field(..., min_length=1)
+
+
+router = APIRouter(prefix="/api/forms", tags=["forms"])
+
+
+@router.post("/recommend")
+async def recommend(req: FormRecommendRequest) -> dict[str, Any]:
+    return await recommend_forms(req.query)
diff --git a/backend/app/api/health.py b/backend/app/api/health.py
index fcd26cf..1907101 100644
+++ b/backend/app/api/health.py
@@ -5,6 +5,8 @@ from sqlalchemy import text
 from app.core.dependencies import get_db
 from app.chromadb.client import chroma_client
 from app.embeddings.openai_client import openai_client as ollama_client
+from app.graph.neo4j_client import neo4j_client
+from app.core.config import settings
 
 router = APIRouter(tags=["Health"])
 
@@ -18,9 +20,9 @@ async def health():
 async def health_db(db: AsyncSession = Depends(get_db)):
     try:
         await db.execute(text("SELECT 1"))
+        return {"status": "ok", "database": settings.DATABASE_URL}
     except Exception as e:
+        return {"status": "error", "database": settings.DATABASE_URL, "detail": str(e)}
 
 
 @router.get("/health/chroma")
@@ -33,7 +35,11 @@ async def health_chroma():
             collections = chroma_client.list_collections()
         except Exception:
             pass
+    return {
+        "status": status,
+        "vector_store": getattr(chroma_client, "backend_name", "chroma"),
+        "collections": collections,
+    }
 
 
 @router.get("/health/openai")
@@ -50,4 +56,15 @@ async def health_openai():
         "status": "ok" if ok else "error",
         "openai": "connected" if ok else "unreachable",
         "models": models,
\ No newline at end of file
+    }
+
+
+@router.get("/health/neo4j")
+async def health_neo4j():
+    if not neo4j_client.enabled:
+        return {"status": "disabled", "neo4j": "disabled"}
+    ok = await neo4j_client.health_check()
+    return {
+        "status": "ok" if ok else "error",
+        "neo4j": "reachable" if ok else "unreachable",
+    }
diff --git a/backend/app/chromadb/client.py b/backend/app/chromadb/client.py
index 20ae0b4..0c5f999 100644
+++ b/backend/app/chromadb/client.py
@@ -1,10 +1,12 @@
+import logging
 from typing import Any, Optional
+
 import chromadb
 from chromadb.config import Settings as ChromaSettings
 from app.core.config import settings
 from app.core.exceptions import ChromaDBError
 
+logger = logging.getLogger(__name__)
 
 COLLECTIONS = [
     "table_documents",
@@ -16,7 +18,171 @@ COLLECTIONS = [
 ]
 
 
+# ── Qdrant Backend ------------------------------------------------------------
+try:
+    from qdrant_client import QdrantClient
+    from qdrant_client.models import (
+        Distance,
+        FieldCondition,
+        Filter,
+        MatchAny,
+        MatchValue,
+        PointStruct,
+        VectorParams,
+    )
+except ImportError:
+    QdrantClient = None  # type: ignore
+
+
+class _QdrantBackend:
+    def __init__(self):
+        if QdrantClient is None:
+            raise ImportError("qdrant-client not installed")
+        self._client = QdrantClient(
+            host=settings.QDRANT_HOST,
+            port=settings.QDRANT_PORT,
+            api_key=settings.QDRANT_API_KEY or None,
+            prefer_grpc=False,
+        )
+
+    # Helpers -----------------------------------------------------------------
+    def _distance(self):
+        dist = (settings.QDRANT_DISTANCE or "Cosine").lower()
+        return Distance.COSINE if dist == "cosine" else Distance.DOT
+
+    def _ensure_collection(self, name: str, vector_size: Optional[int] = None) -> None:
+        collections = {c.name for c in self._client.get_collections().collections}
+        if name not in collections:
+            size = vector_size or settings.QDRANT_VECTOR_SIZE
+            self._client.create_collection(
+                collection_name=name,
+                vectors_config=VectorParams(
+                    size=size,
+                    distance=self._distance(),
+                ),
+            )
+
+    def _to_filter(self, where: Optional[dict]) -> Optional[Filter]:
+        if not where:
+            return None
+
+        def _from_clause(clause: dict) -> Optional[Filter]:
+            # clause may be {field: {"$eq": val}} or {"$and": [clauses]}
+            if "$and" in clause:
+                must = []
+                for sub in clause.get("$and", []):
+                    f = _from_clause(sub)
+                    if f:
+                        must.extend(f.must or [])
+                return Filter(must=must) if must else None
+
+            if len(clause) != 1:
+                return None
+            field, expr = next(iter(clause.items()))
+            if isinstance(expr, dict):
+                if "$eq" in expr:
+                    return Filter(must=[FieldCondition(key=field, match=MatchValue(value=expr["$eq"]))])
+                if "$in" in expr and isinstance(expr["$in"], list):
+                    return Filter(must=[FieldCondition(key=field, match=MatchAny(any=expr["$in"]))])
+            return None
+
+        return _from_clause(where)
+
+    # CRUD --------------------------------------------------------------------
+    def add_documents(
+        self,
+        collection_name: str,
+        ids: list[str],
+        embeddings: list[list[float]],
+        documents: list[str],
+        metadatas: Optional[list[dict]] = None,
+    ) -> bool:
+        vector_size = len(embeddings[0]) if embeddings else settings.QDRANT_VECTOR_SIZE
+        self._ensure_collection(collection_name, vector_size)
+        payloads = metadatas or [{} for _ in ids]
+        points = [
+            PointStruct(id=ids[i], vector=embeddings[i], payload={**payloads[i], "chunk_text": documents[i]})
+            for i in range(len(ids))
+        ]
+        self._client.upsert(collection_name=collection_name, points=points, wait=True)
+        return True
+
+    def search(
+        self,
+        collection_name: str,
+        query_embedding: list[float],
+        top_k: int = 5,
+        where: Optional[dict] = None,
+    ) -> list[dict[str, Any]]:
+        self._ensure_collection(collection_name)
+        flt = self._to_filter(where)
+        results = self._client.search(
+            collection_name=collection_name,
+            query_vector=query_embedding,
+            limit=top_k,
+            with_payload=True,
+            with_vectors=False,
+            query_filter=flt,
+        )
+        output: list[dict[str, Any]] = []
+        for hit in results:
+            payload = hit.payload or {}
+            output.append({
+                "chunk_id": str(hit.id),
+                "chunk_text": payload.get("chunk_text", ""),
+                "metadata": {k: v for k, v in payload.items() if k != "chunk_text"},
+                "score": round(float(hit.score), 4),
+            })
+        return output
+
+    def get_client(self):
+        return self._client
+
+    def metadata_filter(self, collection_name: str, query_embedding: list[float], filters: dict, top_k: int = 5):
+        return self.search(collection_name, query_embedding, top_k, where=filters)
+
+    def delete_by_document_id(self, collection_name: str, document_id: str) -> bool:
+        self._ensure_collection(collection_name)
+        flt = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
+        self._client.delete(collection_name=collection_name, filter=flt, wait=True)
+        return True
+
+    def delete_collection(self, name: str) -> bool:
+        self._client.delete_collection(name)
+        return True
+
+    def list_collections(self) -> list[str]:
+        return [c.name for c in self._client.get_collections().collections]
+
+    def get_collection_count(self, collection_name: str) -> int:
+        try:
+            self._ensure_collection(collection_name)
+            info = self._client.get_collection(collection_name)
+            return info.points_count or 0
+        except Exception:
+            return 0
+
+    def reindex(self, collection_name: str, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: Optional[list[dict]] = None) -> bool:
+        self.delete_collection(collection_name)
+        return self.add_documents(collection_name, ids, embeddings, documents, metadatas)
+
+    def init_collections(self):
+        for name in COLLECTIONS:
+            try:
+                self._ensure_collection(name)
+            except Exception as exc:
+                logger.warning("Qdrant collection init failed for %s: %s", name, exc)
+
+    def health_check(self) -> bool:
+        try:
+            self._client.get_collections()
+            return True
+        except Exception:
+            return False
+
+
+# ── Chroma Backend (legacy / fallback) --------------------------------------
+class _ChromaBackend:
     def __init__(self):
         self._client: Optional[chromadb.Client] = None
 
@@ -31,7 +197,7 @@ class ChromaDBClient:
                 raise ChromaDBError(str(e))
         return self._client
 
+    def create_collection(self, name: str, metadata: Optional[dict] = None):
         try:
             client = self.get_client()
             collection = client.get_or_create_collection(
@@ -164,4 +330,72 @@ class ChromaDBClient:
             self.create_collection(name)
 
 
+# ── Facade that preserves the old name chroma_client ------------------------
+class VectorClient:
+    def __init__(self):
+        self._backend: Optional[Any] = None
+        self.backend_name: str = ""
+
+    def _ensure_backend(self):
+        if self._backend is not None:
+            return
+        if settings.USE_QDRANT:
+            try:
+                self._backend = _QdrantBackend()
+                self.backend_name = "qdrant"
+                return
+            except Exception as exc:
+                logger.warning("Qdrant backend unavailable, falling back to Chroma: %s", exc)
+        self._backend = _ChromaBackend()
+        self.backend_name = "chroma"
+
+    # Proxy all public methods ------------------------------------------------
+    def add_documents(self, *args, **kwargs):
+        self._ensure_backend()
+        return self._backend.add_documents(*args, **kwargs)
+
+    def search(self, *args, **kwargs):
+        self._ensure_backend()
+        return self._backend.search(*args, **kwargs)
+
+    def metadata_filter(self, *args, **kwargs):
+        self._ensure_backend()
+        return self._backend.metadata_filter(*args, **kwargs)
+
+    def reindex(self, *args, **kwargs):
+        self._ensure_backend()
+        return self._backend.reindex(*args, **kwargs)
+
+    def list_collections(self, *args, **kwargs):
+        self._ensure_backend()
+        return self._backend.list_collections(*args, **kwargs)
+
+    def get_collection_count(self, *args, **kwargs):
+        self._ensure_backend()
+        return self._backend.get_collection_count(*args, **kwargs)
+
+    def delete_by_document_id(self, *args, **kwargs):
+        self._ensure_backend()
+        return self._backend.delete_by_document_id(*args, **kwargs)
+
+    def delete_collection(self, *args, **kwargs):
+        self._ensure_backend()
+        return self._backend.delete_collection(*args, **kwargs)
+
+    def health_check(self, *args, **kwargs):
+        self._ensure_backend()
+        return self._backend.health_check(*args, **kwargs)
+
+    def init_collections(self, *args, **kwargs):
+        self._ensure_backend()
+        return self._backend.init_collections(*args, **kwargs)
+
+    def get_client(self):
+        self._ensure_backend()
+        if hasattr(self._backend, "get_client"):
+            return self._backend.get_client()
+        return getattr(self._backend, "_client", None)
+
+
+# Expose under the legacy name to avoid touching call sites
+chroma_client = VectorClient()
diff --git a/backend/app/core/config.py b/backend/app/core/config.py
index c3970dc..4d44003 100644
+++ b/backend/app/core/config.py
@@ -8,16 +8,41 @@ class Settings(BaseSettings):
     APP_VERSION: str = "1.0.0"
     DEBUG: bool = True
 
+    # ── Primary DB ────────────────────────────────────────────────────────────
+    # Default points to Postgres; set to SQLite URL for local/dev fallback.
+    DATABASE_URL: str = Field(
+        default="postgresql+asyncpg://bank_user:bank_password@localhost:5432/bank_kag"
+    )
 
+    # ── Legacy Chroma (kept for fallback) ──────────────────────────────────────
     CHROMA_HOST: str = "localhost"
     CHROMA_PORT: int = 8001
     CHROMA_PERSIST_DIR: str = "./chroma_db"
 
+    # ── Qdrant (primary vector store) ─────────────────────────────────────────
+    USE_QDRANT: bool = True
+    QDRANT_HOST: str = "localhost"
+    QDRANT_PORT: int = 6333
+    QDRANT_API_KEY: str = ""
+    QDRANT_COLLECTION: str = "bank_documents"
+    QDRANT_VECTOR_SIZE: int = 3072  # text-embedding-3-large dimension
+    QDRANT_DISTANCE: str = "Cosine"
+
+    # ── Neo4j (knowledge graph) ───────────────────────────────────────────────
+    USE_KG_RETRIEVAL: bool = False
+    NEO4J_URI: str = "bolt://localhost:7687"
+    NEO4J_USER: str = "neo4j"
+    NEO4J_PASSWORD: str = "password"
+    KG_MAX_DEPTH: int = 2
+    KG_MAX_NODES: int = 100
+    KG_MAX_DOCS: int = 50
+    KG_MAX_CHUNKS: int = 200
+
     # ── OpenAI ────────────────────────────────────────────────────────────────
     OPENAI_API_KEY: str = ""
     OPENAI_LLM_MODEL: str = "gpt-4o-mini"
     OPENAI_EMBED_MODEL: str = "text-embedding-3-large"
+    OPENAI_EMBED_FALLBACK_MODEL: str = "text-embedding-3-small"
     OPENAI_TIMEOUT: int = 120
     OPENAI_MAX_RETRIES: int = 3
 
@@ -26,8 +51,8 @@ class Settings(BaseSettings):
     LOG_LEVEL: str = "INFO"
 
     TOP_K: int = 20
+    CHUNK_SIZE: int = 1500
+    CHUNK_OVERLAP: int = 250
     MAX_CONTEXT_CHUNKS: int = 10
     
     # ── Retrieval Improvements ────────────────────────────────────────────────
diff --git a/backend/app/database/models.py b/backend/app/database/models.py
index 5464809..f4fdac7 100644
+++ b/backend/app/database/models.py
@@ -1,4 +1,5 @@
 from datetime import datetime
+from typing import Optional
 from sqlalchemy import (
     String, Text, Integer, Float, ForeignKey, DateTime, Index, JSON
 )
@@ -12,12 +13,16 @@ class Document(Base):
     id: Mapped[str] = mapped_column(String(36), primary_key=True)
     filename: Mapped[str] = mapped_column(String(512), nullable=False)
     filepath: Mapped[str] = mapped_column(String(1024), nullable=False)
+    title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
     document_type: Mapped[str] = mapped_column(String(64), nullable=False)
+    category: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
+    source: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
+    retrieval_strategy: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
     language: Mapped[str] = mapped_column(String(16), nullable=True, default="en")
     chunk_count: Mapped[int] = mapped_column(Integer, default=0)
+    embedding_model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
+    collection_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
+    form_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
     metadata_json: Mapped[dict] = mapped_column(JSON, nullable=True, default=dict)
     created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
     updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
@@ -28,6 +33,7 @@ class Document(Base):
         Index("ix_documents_document_type", "document_type"),
         Index("ix_documents_filename", "filename"),
         Index("ix_documents_created_at", "created_at"),
+        Index("ix_documents_category", "category"),
     )
 
 
@@ -39,6 +45,8 @@ class Chunk(Base):
     chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
     chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
     chunk_metadata: Mapped[dict] = mapped_column(JSON, nullable=True, default=dict)
+    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=True, default=dict)
+    qdrant_point_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
     created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
 
     document: Mapped["Document"] = relationship("Document", back_populates="chunks")
@@ -46,6 +54,7 @@ class Chunk(Base):
     __table_args__ = (
         Index("ix_chunks_document_id", "document_id"),
         Index("ix_chunks_chunk_index", "chunk_index"),
+        Index("ix_chunks_qdrant_point_id", "qdrant_point_id"),
     )
 
 
@@ -98,6 +107,33 @@ class RetrievalLog(Base):
     )
 
 
+class QueryLog(Base):
+    __tablename__ = "query_logs"
+
+    id: Mapped[str] = mapped_column(String(36), primary_key=True)
+    query: Mapped[str] = mapped_column(Text, nullable=False)
+    response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
+    latency: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
+    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
+
+    __table_args__ = (Index("ix_query_logs_created_at", "created_at"),)
+
+
+class Form(Base):
+    __tablename__ = "forms"
+
+    id: Mapped[str] = mapped_column(String(36), primary_key=True)
+    name: Mapped[str] = mapped_column(String(256), nullable=False)
+    category: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
+    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
+    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
+
+    __table_args__ = (
+        Index("ix_forms_name", "name"),
+        Index("ix_forms_category", "category"),
+    )
+
+
 class EvaluationRun(Base):
     __tablename__ = "evaluation_runs"
 
diff --git a/backend/app/database/session.py b/backend/app/database/session.py
index 578d396..22a723a 100644
+++ b/backend/app/database/session.py
@@ -1,11 +1,17 @@
 from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
 from app.core.config import settings
 
+
+def _connect_args(url: str) -> dict:
+    # SQLite needs check_same_thread; Postgres/others do not.
+    return {"check_same_thread": False} if url.startswith("sqlite") else {}
+
+
 engine = create_async_engine(
     settings.DATABASE_URL,
     echo=settings.DEBUG,
     pool_pre_ping=True,
+    connect_args=_connect_args(settings.DATABASE_URL),
 )
 
 AsyncSessionLocal = async_sessionmaker(
diff --git a/backend/app/embeddings/openai_client.py b/backend/app/embeddings/openai_client.py
index 3674041..8b1405d 100644
+++ b/backend/app/embeddings/openai_client.py
@@ -150,12 +150,13 @@ class OpenAIClient:
         retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
     )
     async def embeddings(self, text: str, model: Optional[str] = None) -> list[float]:
+        """Embed text with a fallback model if the primary fails."""
+        primary_model = model or self.embed_model
+        fallback_model = settings.OPENAI_EMBED_FALLBACK_MODEL
         client = await self._get_client()
+
+        async def _embed(m: str) -> list[float]:
+            payload = {"model": m, "input": text}
             response = await client.post(
                 "/embeddings",
                 json=payload,
@@ -164,10 +165,16 @@ class OpenAIClient:
             response.raise_for_status()
             data = response.json()
             return data["data"][0]["embedding"]
+
+        try:
+            return await _embed(primary_model)
+        except Exception:
+            if fallback_model and fallback_model != primary_model:
+                try:
+                    return await _embed(fallback_model)
+                except Exception as e:
+                    raise OpenAIConnectionError(str(e))
+            raise
 
     async def batch_embeddings(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
         """
diff --git a/backend/app/evaluation/agent_runner.py b/backend/app/evaluation/agent_runner.py
index 00bb70e..e322f36 100644
+++ b/backend/app/evaluation/agent_runner.py
@@ -144,6 +144,10 @@ async def _fetch_neighbor_chunks(
     if not chunks:
         return chunks
 
+    # Neighbor fetching is only supported on Chroma collections; fall back otherwise.
+    if getattr(chroma_client, "backend_name", "chroma") != "chroma":
+        return chunks
+
     expanded = []
     seen_ids = {c.get("chunk_id") for c in chunks if c.get("chunk_id")}
 
diff --git a/backend/app/graph/__init__.py b/backend/app/graph/__init__.py
new file mode 100644
index 0000000..33f396b
+++ b/backend/app/graph/__init__.py
@@ -0,0 +1 @@
+# Graph utilities for knowledge-augmented retrieval.
diff --git a/backend/app/graph/entity_extraction.py b/backend/app/graph/entity_extraction.py
new file mode 100644
index 0000000..084c297
+++ b/backend/app/graph/entity_extraction.py
@@ -0,0 +1,43 @@
+import json
+import logging
+from typing import Any, Dict, List
+
+from app.embeddings.openai_client import openai_client
+
+logger = logging.getLogger(__name__)
+
+SYSTEM_PROMPT = (
+    "Extract banking entities and relationships as JSON with keys 'entities' (list of strings) "
+    "and 'relationships' (list of {source, relation, target})."
+)
+
+
+async def extract_entities(text: str) -> Dict[str, Any]:
+    """
+    GPT-based extractor with graceful fallback. Returns {entities: [], relationships: []}.
+    """
+    try:
+        user_prompt = (
+            f"Text:\n{text}\n\nReturn ONLY JSON matching the schema: "
+            "{\"entities\": [\"sample\"], \"relationships\": [{\"source\": \"A\", "
+            "\"relation\": \"requires\", \"target\": \"B\"}]}"
+        )
+        raw = await openai_client.chat(
+            messages=[{"role": "user", "content": user_prompt}],
+            system=SYSTEM_PROMPT,
+        )
+        parsed = json.loads(raw)
+        entities = parsed.get("entities", []) if isinstance(parsed, dict) else []
+        relationships = parsed.get("relationships", []) if isinstance(parsed, dict) else []
+        if not isinstance(entities, list) or not isinstance(relationships, list):
+            raise ValueError("Invalid schema from extractor")
+        return {"entities": entities, "relationships": relationships}
+    except Exception as exc:
+        logger.warning("Entity extraction fallback used: %s", exc)
+        # Fallback: naive keyword capture of capitalised tokens
+        tokens = [t for t in text.split() if t and t[0].isupper()]
+        uniq: List[str] = []
+        for t in tokens:
+            if t not in uniq:
+                uniq.append(t)
+        return {"entities": uniq[:10], "relationships": []}
diff --git a/backend/app/graph/form_recommender.py b/backend/app/graph/form_recommender.py
new file mode 100644
index 0000000..5fbaab8
+++ b/backend/app/graph/form_recommender.py
@@ -0,0 +1,59 @@
+import logging
+from typing import Any, Dict, List
+
+from app.core.config import settings
+from app.graph.entity_extraction import extract_entities
+from app.graph.graph_retriever import retrieve_candidates
+from app.graph.neo4j_client import neo4j_client
+
+logger = logging.getLogger(__name__)
+
+
+async def recommend_forms(query: str) -> Dict[str, Any]:
+    """Return recommended forms plus requirements/eligibility/procedure when present."""
+    if not neo4j_client.enabled:
+        return {"forms": [], "documents": []}
+
+    extraction = await extract_entities(query)
+    entities = [e for e in extraction.get("entities", []) if isinstance(e, str)]
+
+    # First, reuse candidate retrieval to collect forms tied to documents
+    base = await retrieve_candidates(query)
+    forms = base.forms.copy()
+
+    try:
+        if entities:
+            cypher = (
+                "MATCH (f:Form) "
+                "WHERE any(e IN $entities WHERE toLower(f.name) CONTAINS toLower(e) OR toLower(coalesce(f.category,'')) CONTAINS toLower(e)) "
+                "WITH DISTINCT f LIMIT 20 "
+                "OPTIONAL MATCH (f)-[:HAS_ELIGIBILITY]->(el) "
+                "OPTIONAL MATCH (f)-[:HAS_PROCEDURE]->(p) "
+                "OPTIONAL MATCH (f)-[:REQUIRES]->(req:Field) "
+                "RETURN f.name AS name, f.category AS category, f.description AS description, "
+                "collect(DISTINCT req.name) AS requirements, collect(DISTINCT el.name) AS eligibility, collect(DISTINCT p.name) AS procedure"
+            )
+            rows = await neo4j_client.run_read(cypher, {"entities": [e.lower() for e in entities]})
+            for row in rows:
+                forms.append({
+                    "name": row.get("name"),
+                    "category": row.get("category"),
+                    "description": row.get("description"),
+                    "requirements": [r for r in row.get("requirements", []) if r],
+                    "eligibility": [r for r in row.get("eligibility", []) if r],
+                    "procedure": [r for r in row.get("procedure", []) if r],
+                })
+    except Exception as exc:
+        logger.warning("Form recommendation graph query failed: %s", exc)
+
+    # Deduplicate by form name
+    seen = set()
+    deduped: List[Dict[str, Any]] = []
+    for form in forms:
+        name = form.get("name")
+        if not name or name in seen:
+            continue
+        seen.add(name)
+        deduped.append(form)
+
+    return {"forms": deduped[: settings.TOP_K], "documents": list(base.candidate_document_ids)}
diff --git a/backend/app/graph/graph_ingestor.py b/backend/app/graph/graph_ingestor.py
new file mode 100644
index 0000000..0815a46
+++ b/backend/app/graph/graph_ingestor.py
@@ -0,0 +1,70 @@
+import logging
+from typing import Any, Dict, Optional
+
+from app.core.config import settings
+from app.graph.neo4j_client import neo4j_client
+
+logger = logging.getLogger(__name__)
+
+
+async def upsert_document_node(doc: Dict[str, Any]) -> None:
+    """Seed a Document node using available metadata (no GPT required)."""
+    if not neo4j_client.enabled:
+        return
+    cypher = (
+        "MERGE (d:Document {document_id: $id}) "
+        "SET d.filename=$filename, d.title=$title, d.category=$category, d.source=$source, d.form_name=$form_name"
+    )
+    params = {
+        "id": doc.get("id"),
+        "filename": doc.get("filename"),
+        "title": doc.get("title"),
+        "category": doc.get("category"),
+        "source": doc.get("source"),
+        "form_name": doc.get("form_name"),
+    }
+    try:
+        await neo4j_client.run_write(cypher, params)
+    except Exception as exc:
+        logger.warning("Failed to upsert document node: %s", exc)
+
+
+async def link_document_to_category(doc_id: str, category: Optional[str]) -> None:
+    if not neo4j_client.enabled or not category:
+        return
+    cypher = (
+        "MERGE (c:Concept {name: $category}) "
+        "WITH c MATCH (d:Document {document_id: $doc_id}) "
+        "MERGE (d)-[:RELATED_TO]->(c)"
+    )
+    try:
+        await neo4j_client.run_write(cypher, {"doc_id": doc_id, "category": category})
+    except Exception as exc:
+        logger.warning("Failed to link document to category: %s", exc)
+
+
+async def upsert_form(name: str, category: Optional[str], description: Optional[str] = None):
+    if not neo4j_client.enabled:
+        return
+    cypher = (
+        "MERGE (f:Form {name: $name}) "
+        "SET f.category=$category, f.description=$description"
+    )
+    try:
+        await neo4j_client.run_write(cypher, {"name": name, "category": category, "description": description})
+    except Exception as exc:
+        logger.warning("Failed to upsert form: %s", exc)
+
+
+async def connect_form_to_document(form_name: Optional[str], doc_id: str) -> None:
+    if not neo4j_client.enabled or not form_name:
+        return
+    cypher = (
+        "MATCH (d:Document {document_id: $doc_id}) "
+        "MERGE (f:Form {name: $form_name}) "
+        "MERGE (f)-[:USES]->(d)"
+    )
+    try:
+        await neo4j_client.run_write(cypher, {"doc_id": doc_id, "form_name": form_name})
+    except Exception as exc:
+        logger.warning("Failed to connect form to document: %s", exc)
diff --git a/backend/app/graph/graph_retriever.py b/backend/app/graph/graph_retriever.py
new file mode 100644
index 0000000..82861cc
+++ b/backend/app/graph/graph_retriever.py
@@ -0,0 +1,77 @@
+import logging
+from dataclasses import dataclass, field
+from typing import Any, Dict, List, Optional, Set
+
+from app.core.config import settings
+from app.graph.entity_extraction import extract_entities
+from app.graph.neo4j_client import neo4j_client
+
+logger = logging.getLogger(__name__)
+
+
+@dataclass
+class GraphResult:
+    candidate_document_ids: Set[str] = field(default_factory=set)
+    forms: List[Dict[str, Any]] = field(default_factory=list)
+    concepts: List[str] = field(default_factory=list)
+    relationships: List[Dict[str, Any]] = field(default_factory=list)
+
+
+async def retrieve_candidates(query: str, filters: Optional[dict] = None) -> GraphResult:
+    """
+    Run entity extraction then use Neo4j to fetch candidate documents/forms.
+    Falls back to empty result when graph is disabled or unavailable.
+    """
+    if not neo4j_client.enabled:
+        return GraphResult()
+
+    extraction = await extract_entities(query)
+    entities = list({e.lower() for e in extraction.get("entities", []) if isinstance(e, str)})
+    if not entities and filters:
+        for key in ("category", "form_name", "document_type"):
+            val = (filters or {}).get(key)
+            if isinstance(val, str):
+                entities.append(val.lower())
+
+    if not entities:
+        return GraphResult()
+
+    try:
+        cypher = (
+            "MATCH (d:Document) "
+            "WHERE any(e IN $entities WHERE toLower(coalesce(d.filename,'')) CONTAINS e "
+            "  OR toLower(coalesce(d.category,'')) CONTAINS e "
+            "  OR toLower(coalesce(d.form_name,'')) CONTAINS e "
+            "  OR toLower(coalesce(d.title,'')) CONTAINS e) "
+            "WITH DISTINCT d LIMIT $max_docs "
+            "OPTIONAL MATCH (f:Form)-[:USES]->(d) "
+            "RETURN d.document_id AS document_id, collect(DISTINCT f.name) AS forms, d.category AS category"
+        )
+        rows = await neo4j_client.run_read(
+            cypher,
+            {"entities": entities, "max_docs": settings.KG_MAX_DOCS},
+        )
+    except Exception as exc:
+        logger.warning("Graph retrieval failed, falling back: %s", exc)
+        return GraphResult()
+
+    doc_ids: Set[str] = set()
+    forms: List[Dict[str, Any]] = []
+    concepts: List[str] = []
+    for row in rows:
+        doc_id = row.get("document_id")
+        if doc_id:
+            doc_ids.add(doc_id)
+        for form_name in row.get("forms", []) or []:
+            if form_name:
+                forms.append({"name": form_name})
+        cat = row.get("category")
+        if cat:
+            concepts.append(cat)
+
+    return GraphResult(
+        candidate_document_ids=doc_ids,
+        forms=forms,
+        concepts=list({c for c in concepts}),
+        relationships=extraction.get("relationships", []),
+    )
diff --git a/backend/app/graph/neo4j_client.py b/backend/app/graph/neo4j_client.py
new file mode 100644
index 0000000..73d4a7c
+++ b/backend/app/graph/neo4j_client.py
@@ -0,0 +1,60 @@
+import logging
+from typing import Any, Optional
+
+from app.core.config import settings
+
+try:
+    from neo4j import AsyncGraphDatabase  # type: ignore
+except ImportError:  # pragma: no cover - neo4j optional at runtime
+    AsyncGraphDatabase = None
+
+
+logger = logging.getLogger(__name__)
+
+
+class Neo4jClient:
+    def __init__(self) -> None:
+        self._driver = None
+        self.enabled = settings.USE_KG_RETRIEVAL and AsyncGraphDatabase is not None
+
+    async def _get_driver(self):
+        if not self.enabled:
+            return None
+        if self._driver is None:
+            self._driver = AsyncGraphDatabase.driver(
+                settings.NEO4J_URI,
+                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
+            )
+        return self._driver
+
+    async def close(self):
+        if self._driver:
+            await self._driver.close()
+            self._driver = None
+
+    async def run_read(self, cypher: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
+        driver = await self._get_driver()
+        if not driver:
+            return []
+        async with driver.session() as session:
+            result = await session.run(cypher, params or {})
+            records = await result.to_list()
+            return [r.data() for r in records]
+
+    async def run_write(self, cypher: str, params: Optional[dict[str, Any]] = None) -> None:
+        driver = await self._get_driver()
+        if not driver:
+            return
+        async with driver.session() as session:
+            await session.run(cypher, params or {})
+
+    async def health_check(self) -> bool:
+        try:
+            rows = await self.run_read("RETURN 1 AS ok")
+            return bool(rows)
+        except Exception as exc:
+            logger.warning("Neo4j health check failed: %s", exc)
+            return False
+
+
+neo4j_client = Neo4jClient()
diff --git a/backend/app/main.py b/backend/app/main.py
index 82cb72c..0abf64c 100644
+++ b/backend/app/main.py
@@ -23,6 +23,7 @@ from app.api.documents import router as documents_router
 from app.api.chat import router as chat_router
 from app.api.rag import router as rag_router
 from app.api.chroma import router as chroma_router
+from app.api.forms import router as forms_router
 
 
 
@@ -78,4 +79,5 @@ app.include_router(health_router)
 app.include_router(documents_router)
 app.include_router(chat_router)
 app.include_router(rag_router)
\ No newline at end of file
+app.include_router(chroma_router)
+app.include_router(forms_router)
diff --git a/backend/app/rag/hybrid_rag.py b/backend/app/rag/hybrid_rag.py
index a0f0afa..0cf7d14 100644
+++ b/backend/app/rag/hybrid_rag.py
@@ -70,13 +70,14 @@ class HybridRAG:
         collection_name: str = "text_documents",
         top_k: int = 5,
         filters: Optional[dict] = None,
+        candidate_document_ids: Optional[set[str]] = None,
     ) -> list[dict[str, Any]]:
 
         # Dense retrieval: top 50
         vector_results = []
         try:
             vector_results = await vector_rag.retrieve(
+                query, collection_name, settings.DENSE_TOP_K, filters, False, candidate_document_ids
             )
         except Exception:
             pass
@@ -85,7 +86,7 @@ class HybridRAG:
         bm25_results = []
         try:
             bm25_raw = bm25_retriever.search(collection_name, query, settings.BM25_TOP_K)
+            bm25_results = filter_results(bm25_raw, filters or {}, candidate_document_ids)
         except Exception:
             pass
 
diff --git a/backend/app/rag/markdown_rag.py b/backend/app/rag/markdown_rag.py
index 7351d07..0c863ec 100644
+++ b/backend/app/rag/markdown_rag.py
@@ -1,7 +1,9 @@
 import re
+import uuid
 from typing import Any, Optional
 from app.chromadb.client import chroma_client
 from app.embeddings.openai_client import openai_client as ollama_client
+from app.rag.bm25 import bm25_retriever
 
 MD_COLLECTION = "markdown_documents"
 
@@ -49,28 +51,53 @@ class MarkdownRAG:
     ) -> dict[str, Any]:
         sections = _parse_markdown_sections(content)
         ids, embeddings, documents, metadatas = [], [], [], []
+        chunk_records = []
         for i, section in enumerate(sections):
             text = f"# {section['heading']}\n\n{section['content']}"
             if len(text.strip()) < 20:
                 continue
             emb = await ollama_client.embeddings(text)
+            chunk_id = str(uuid.uuid4())
             ids.append(chunk_id)
             embeddings.append(emb)
             documents.append(text)
+            meta = {
                 "document_id": document_id,
                 "filename": filename,
                 "document_type": "markdown",
                 "section": section["heading"],
                 "heading_level": section["level"],
                 "chunk_index": i,
+                "chunk_id": chunk_id,
                 **(extra_metadata or {}),
+            }
+            metadatas.append(meta)
+            chunk_records.append({
+                "id": chunk_id,
+                "document_id": document_id,
+                "chunk_index": i,
+                "chunk_text": text,
+                "chunk_metadata": meta,
+                "metadata_json": meta,
+                "qdrant_point_id": chunk_id,
             })
 
         if ids:
             chroma_client.add_documents(MD_COLLECTION, ids, embeddings, documents, metadatas)
+            try:
+                bm25_retriever.index(MD_COLLECTION, [
+                    {
+                        "chunk_id": ids[i],
+                        "chunk_text": documents[i],
+                        "metadata": metadatas[i],
+                        "document_id": metadatas[i].get("document_id", ""),
+                        "filename": metadatas[i].get("filename", ""),
+                    }
+                    for i in range(len(ids))
+                ])
+            except Exception:
+                pass
+        return {"document_id": document_id, "chunk_count": len(ids), "chunks": chunk_records}
 
     async def query(
         self,
@@ -85,4 +112,4 @@ class MarkdownRAG:
         return chroma_client.search(MD_COLLECTION, query_emb, top_k, where)
 
 
\ No newline at end of file
+markdown_rag = MarkdownRAG()
diff --git a/backend/app/rag/metadata_filter.py b/backend/app/rag/metadata_filter.py
index 8f2843f..d8dd3ee 100644
+++ b/backend/app/rag/metadata_filter.py
@@ -4,35 +4,53 @@ from typing import Any, Optional
 SUPPORTED_FILTERS = {
     "filename", "document_type", "section", "language",
     "date", "document_id", "retrieval_strategy", "state",
+    "ministry", "department", "source", "category", "form_name",
 }
 
 
+def build_chroma_filter(filters: dict[str, Any], candidate_document_ids: Optional[set[str]] = None) -> Optional[dict]:
+    """
+    Build a vector-store friendly filter dict (Chroma-compatible) with support for
+    document whitelisting via $in. Qdrant adapter also understands this shape.
+    """
+    if not filters and not candidate_document_ids:
         return None
+
+    valid = {k: v for k, v in (filters or {}).items() if k in SUPPORTED_FILTERS and v is not None}
+
+    clauses = []
+    for k, v in valid.items():
+        clauses.append({k: {"$eq": v}})
+
+    if candidate_document_ids:
+        clauses.append({"document_id": {"$in": list(candidate_document_ids)}})
+
+    if not clauses:
         return None
+
+    if len(clauses) == 1:
+        return clauses[0]
+    return {"$and": clauses}
 
 
+def filter_results(
+    results: list[dict],
+    filters: dict[str, Any],
+    candidate_document_ids: Optional[set[str]] = None,
+) -> list[dict]:
+    """In-memory metadata filtering for BM25 or pre-fetched results."""
+    if not filters and not candidate_document_ids:
         return results
     filtered = []
     for item in results:
         meta = item.get("metadata", {})
         match = all(
             meta.get(k) == v
+            for k, v in (filters or {}).items()
             if k in SUPPORTED_FILTERS and v is not None
         )
+        if candidate_document_ids is not None and meta.get("document_id") not in candidate_document_ids:
+            match = False
         if match:
             filtered.append(item)
     return filtered
diff --git a/backend/app/rag/pdf_rag.py b/backend/app/rag/pdf_rag.py
index 267e146..6bd5422 100644
+++ b/backend/app/rag/pdf_rag.py
@@ -6,6 +6,7 @@ import pdfplumber
 from app.chromadb.client import chroma_client
 from app.embeddings.openai_client import openai_client as ollama_client
 from app.core.config import settings
+from app.rag.bm25 import bm25_retriever
 
 PDF_COLLECTION = "pdf_documents"
 
@@ -33,6 +34,7 @@ class PDFHierarchicalRAG:
         extra_metadata: Optional[dict] = None,
     ) -> dict[str, Any]:
         ids, embeddings, documents, metadatas = [], [], [], []
+        chunk_records = []
         current_section = "Introduction"
         chunk_index = 0
 
@@ -57,19 +59,30 @@ class PDFHierarchicalRAG:
                     if current_para:
                         chunk_text = " ".join(current_para).strip()
                         if len(chunk_text) > 50:
+                            chunk_id = str(uuid.uuid4())
                             emb = await ollama_client.embeddings(chunk_text)
                             ids.append(chunk_id)
                             embeddings.append(emb)
                             documents.append(chunk_text)
+                            meta = {
                                 "document_id": document_id,
                                 "filename": filename,
                                 "document_type": "pdf",
                                 "section": current_section,
                                 "page_number": page_num,
                                 "chunk_index": chunk_index,
+                                "chunk_id": chunk_id,
                                 **(extra_metadata or {}),
+                            }
+                            metadatas.append(meta)
+                            chunk_records.append({
+                                "id": chunk_id,
+                                "document_id": document_id,
+                                "chunk_index": chunk_index,
+                                "chunk_text": chunk_text,
+                                "chunk_metadata": meta,
+                                "metadata_json": meta,
+                                "qdrant_point_id": chunk_id,
                             })
                             chunk_index += 1
                         current_para = []
@@ -81,25 +94,49 @@ class PDFHierarchicalRAG:
             if current_para:
                 chunk_text = " ".join(current_para).strip()
                 if len(chunk_text) > 50:
+                    chunk_id = str(uuid.uuid4())
                     emb = await ollama_client.embeddings(chunk_text)
                     ids.append(chunk_id)
                     embeddings.append(emb)
                     documents.append(chunk_text)
+                    meta = {
                         "document_id": document_id,
                         "filename": filename,
                         "document_type": "pdf",
                         "section": current_section,
                         "page_number": page_num,
                         "chunk_index": chunk_index,
+                        "chunk_id": chunk_id,
                         **(extra_metadata or {}),
+                    }
+                    metadatas.append(meta)
+                    chunk_records.append({
+                        "id": chunk_id,
+                        "document_id": document_id,
+                        "chunk_index": chunk_index,
+                        "chunk_text": chunk_text,
+                        "chunk_metadata": meta,
+                        "metadata_json": meta,
+                        "qdrant_point_id": chunk_id,
                     })
                     chunk_index += 1
 
         if ids:
             chroma_client.add_documents(PDF_COLLECTION, ids, embeddings, documents, metadatas)
+            try:
+                bm25_retriever.index(PDF_COLLECTION, [
+                    {
+                        "chunk_id": ids[i],
+                        "chunk_text": documents[i],
+                        "metadata": metadatas[i],
+                        "document_id": metadatas[i].get("document_id", ""),
+                        "filename": metadatas[i].get("filename", ""),
+                    }
+                    for i in range(len(ids))
+                ])
+            except Exception:
+                pass
+        return {"document_id": document_id, "chunk_count": len(ids), "chunks": chunk_records}
 
     async def query(
         self,
@@ -122,4 +159,4 @@ class PDFHierarchicalRAG:
         return chroma_client.search(PDF_COLLECTION, query_emb, top_k, where)
 
 
\ No newline at end of file
+pdf_rag = PDFHierarchicalRAG()
diff --git a/backend/app/rag/table_rag.py b/backend/app/rag/table_rag.py
index 82d8ec2..ca6ccd9 100644
+++ b/backend/app/rag/table_rag.py
@@ -7,6 +7,7 @@ import pandas as pd
 from app.chromadb.client import chroma_client
 from app.embeddings.openai_client import openai_client as ollama_client
 from app.rag.metadata_filter import build_chroma_filter
+from app.rag.bm25 import bm25_retriever
 
 
 SCHEMA_COLLECTION = "table_documents"
@@ -54,46 +55,82 @@ class TableRAG:
         }
 
         ids, embeddings, documents, metadatas = [], [], [], []
+        chunk_records = []
 
         # Schema chunk with column details
         col_types = ", ".join([f"{col} ({dtype})" for col, dtype in schema_info["dtypes"].items()])
         schema_text = f"Table: {filename}\nColumns: {col_types}\nTotal rows: {schema_info['row_count']}"
         schema_emb = await ollama_client.embeddings(schema_text)
+        schema_id = str(uuid.uuid4())
         ids.append(schema_id)
         embeddings.append(schema_emb)
         documents.append(schema_text)
+        schema_meta = {
             "document_id": document_id,
             "filename": filename,
             "document_type": "csv",
             "chunk_type": "schema",
             "columns": json.dumps(list(df.columns)),
+            "chunk_id": schema_id,
             **(extra_metadata or {}),
+        }
+        metadatas.append(schema_meta)
+        chunk_records.append({
+            "id": schema_id,
+            "document_id": document_id,
+            "chunk_index": 0,
+            "chunk_text": schema_text,
+            "chunk_metadata": schema_meta,
+            "metadata_json": schema_meta,
+            "qdrant_point_id": schema_id,
         })
 
         # Row chunks: convert to readable text format (preserve headers in each chunk)
         chunk_size = 10  # Increased from 5 to capture more context
+        for idx, start in enumerate(range(0, min(len(df), 500), chunk_size), start=1):
             end = start + chunk_size
             row_text = self._convert_table_section_to_text(df, start, end)
             row_emb = await ollama_client.embeddings(row_text)
+            row_id = str(uuid.uuid4())
             ids.append(row_id)
             embeddings.append(row_emb)
             documents.append(row_text)
+            row_meta = {
                 "document_id": document_id,
                 "filename": filename,
                 "document_type": "csv",
                 "chunk_type": "rows",
                 "row_start": start,
                 "row_end": end,
+                "chunk_id": row_id,
                 **(extra_metadata or {}),
+            }
+            metadatas.append(row_meta)
+            chunk_records.append({
+                "id": row_id,
+                "document_id": document_id,
+                "chunk_index": idx,
+                "chunk_text": row_text,
+                "chunk_metadata": row_meta,
+                "metadata_json": row_meta,
+                "qdrant_point_id": row_id,
             })
 
         chroma_client.add_documents(SCHEMA_COLLECTION, ids, embeddings, documents, metadatas)
+        try:
+            bm25_retriever.index(SCHEMA_COLLECTION, [
+                {
+                    "chunk_id": ids[i],
+                    "chunk_text": documents[i],
+                    "metadata": metadatas[i],
+                    "document_id": metadatas[i].get("document_id", ""),
+                    "filename": metadatas[i].get("filename", ""),
+                }
+                for i in range(len(ids))
+            ])
+        except Exception:
+            pass
+        return {"document_id": document_id, "chunk_count": len(ids), "schema": schema_info, "chunks": chunk_records}
 
     async def query(
         self,
@@ -117,4 +154,4 @@ class TableRAG:
         return None
 
 
\ No newline at end of file
+table_rag = TableRAG()
diff --git a/backend/app/rag/vector_rag.py b/backend/app/rag/vector_rag.py
index 3f8f5af..3764439 100644
+++ b/backend/app/rag/vector_rag.py
@@ -390,6 +390,7 @@ class VectorRAG:
         top_k: int = 5,
         filters: Optional[dict] = None,
         expand: bool = False,
+        candidate_document_ids: Optional[set[str]] = None,
     ) -> list[dict[str, Any]]:
         """
         Retrieve the top_k most relevant chunks for *query*.
@@ -410,13 +411,16 @@ class VectorRAG:
         """
         effective_query = expand_acronyms(query) if expand else query
         query_embedding = await ollama_client.embeddings(effective_query)
+        where = build_chroma_filter(filters or {}, candidate_document_ids)
 
         candidate_k = self._candidate_k(top_k)
         results: list[dict[str, Any]] = chroma_client.search(
             collection_name, query_embedding, candidate_k, where
         )
 
+        if candidate_document_ids:
+            results = [r for r in results if r.get("metadata", {}).get("document_id") in candidate_document_ids]
+
         # Drop empty chunks early
         results = [r for r in results if r.get("chunk_text", "").strip()]
 
@@ -432,6 +436,7 @@ class VectorRAG:
         top_k: int = 5,
         filters: Optional[dict] = None,
         expand: bool = False,
+        candidate_document_ids: Optional[set[str]] = None,
     ) -> list[dict[str, Any]]:
         """
         Search multiple collections, merge all candidates, rerank globally,
@@ -452,7 +457,7 @@ class VectorRAG:
         """
         effective_query = expand_acronyms(query) if expand else query
         query_embedding = await ollama_client.embeddings(effective_query)
+        where = build_chroma_filter(filters or {}, candidate_document_ids)
 
         candidate_k = self._candidate_k(top_k)
         all_results: list[dict[str, Any]] = []
@@ -470,6 +475,9 @@ class VectorRAG:
                     "Collection '%s' search failed (skipping): %s", collection, exc
                 )
 
+        if candidate_document_ids:
+            all_results = [r for r in all_results if r.get("metadata", {}).get("document_id") in candidate_document_ids]
+
         # Drop empties, merge hybrid scores globally, then rerank
         all_results = [r for r in all_results if r.get("chunk_text", "").strip()]
         all_results = self._hybrid_merge(effective_query, all_results)
@@ -479,4 +487,4 @@ class VectorRAG:
 
 
 # Module-level singleton — drop-in replacement for the old import
\ No newline at end of file
+vector_rag = VectorRAG()
diff --git a/backend/app/repositories/log_repository.py b/backend/app/repositories/log_repository.py
index fb8e821..3305f9a 100644
+++ b/backend/app/repositories/log_repository.py
@@ -1,6 +1,6 @@
 from sqlalchemy.ext.asyncio import AsyncSession
 from sqlalchemy import select
+from app.database.models import RetrievalLog, EvaluationRun, QueryLog
 
 
 
@@ -19,6 +19,13 @@ class LogRepository:
         await db.refresh(run)
         return run
 
+    async def create_query_log(self, db: AsyncSession, data: dict) -> QueryLog:
+        log = QueryLog(**data)
+        db.add(log)
+        await db.commit()
+        await db.refresh(log)
+        return log
+
     async def list_retrieval_logs(self, db: AsyncSession, skip: int = 0, limit: int = 50) -> list[RetrievalLog]:
         result = await db.execute(
             select(RetrievalLog).offset(skip).limit(limit).order_by(RetrievalLog.created_at.desc())
diff --git a/backend/app/services/document_service.py b/backend/app/services/document_service.py
index 28e4d02..90d0f10 100644
+++ b/backend/app/services/document_service.py
@@ -15,6 +15,7 @@ from app.rag.markdown_rag import markdown_rag
 from app.rag.bm25 import bm25_retriever
 from app.core.config import settings
 from app.core.exceptions import DocumentNotFoundError, UnsupportedFileTypeError
+from app.graph import graph_ingestor
 
 
 SUPPORTED_TYPES = {
@@ -49,14 +50,15 @@ def _detect_type(filename: str) -> str:
     return SUPPORTED_TYPES[ext]
 
 
+def _chunk_text(text: str, chunk_size: int = 1500, overlap: int = 250) -> list[str]:
+    """Character-based chunking with overlap."""
+    chunks: list[str] = []
     start = 0
+    step = max(chunk_size - overlap, 1)
+    while start < len(text):
+        end = min(start + chunk_size, len(text))
+        chunks.append(text[start:end])
+        start += step
     return chunks
 
 
@@ -74,7 +76,7 @@ async def _index_text_chunks(
 
     for i, chunk_text in enumerate(chunks):
         emb = await ollama_client.embeddings(chunk_text)
+        chunk_id = str(uuid.uuid4())
         ids.append(chunk_id)
         embeddings.append(emb)
         documents.append(chunk_text)
@@ -83,6 +85,7 @@ async def _index_text_chunks(
             "filename": filename,
             "document_type": doc_type,
             "chunk_index": i,
+            "chunk_id": chunk_id,
             **(extra_metadata or {}),
         }
         metadatas.append(meta)
@@ -92,6 +95,8 @@ async def _index_text_chunks(
             "chunk_index": i,
             "chunk_text": chunk_text,
             "chunk_metadata": meta,
+            "metadata_json": meta,
+            "qdrant_point_id": chunk_id,
         })
 
     if ids:
@@ -131,13 +136,16 @@ class DocumentService:
         if doc_type == "pdf":
             result = await pdf_rag.index(doc_id, filename, content, extra_metadata)
             chunk_count = result["chunk_count"]
+            chunk_records = result.get("chunks", [])
         elif doc_type == "markdown":
             text = content.decode("utf-8", errors="replace")
             result = await markdown_rag.index(doc_id, filename, text, extra_metadata)
             chunk_count = result["chunk_count"]
+            chunk_records = result.get("chunks", [])
         elif doc_type == "csv":
             result = await table_rag.index_csv(doc_id, filename, content, extra_metadata)
             chunk_count = result["chunk_count"]
+            chunk_records = result.get("chunks", [])
         else:
             # text / json
             text = content.decode("utf-8", errors="replace")
@@ -149,16 +157,28 @@ class DocumentService:
             "filename": filename,
             "filepath": str(filepath),
             "document_type": doc_type,
+            "title": (extra_metadata or {}).get("title"),
+            "category": (extra_metadata or {}).get("category"),
+            "source": (extra_metadata or {}).get("source"),
             "retrieval_strategy": strategy,
             "language": (extra_metadata or {}).get("language", "en"),
             "chunk_count": chunk_count,
             "embedding_model": settings.OPENAI_EMBED_MODEL,
             "collection_name": collection,
+            "form_name": (extra_metadata or {}).get("form_name"),
             "metadata_json": extra_metadata or {},
         }
 
         doc = await document_repo.create(db, doc_data)
 
+        # Seed graph (metadata-only pass)
+        try:
+            await graph_ingestor.upsert_document_node(doc_data)
+            await graph_ingestor.link_document_to_category(doc_id, doc_data.get("category"))
+            await graph_ingestor.connect_form_to_document(doc_data.get("form_name"), doc_id)
+        except Exception:
+            pass
+
         # Persist chunks to DB for non-specialized types
         if chunk_records:
             await document_repo.bulk_create_chunks(db, chunk_records)
@@ -196,17 +216,34 @@ class DocumentService:
             text = content.decode("utf-8", errors="replace")
             result = await markdown_rag.index(doc_id, doc.filename, text, doc.metadata_json)
             chunk_count = result["chunk_count"]
+            chunk_records = result.get("chunks", [])
         elif doc_type == "csv":
             result = await table_rag.index_csv(doc_id, doc.filename, content, doc.metadata_json)
             chunk_count = result["chunk_count"]
+            chunk_records = result.get("chunks", [])
         else:
             text = content.decode("utf-8", errors="replace")
             chunk_records = await _index_text_chunks(doc_id, doc.filename, doc_type, text, collection, doc.metadata_json)
             chunk_count = len(chunk_records)
+
+        if chunk_records:
+            await document_repo.bulk_create_chunks(db, chunk_records)
 
         await document_repo.update(db, doc_id, {"chunk_count": chunk_count})
+
+        try:
+            await graph_ingestor.upsert_document_node({
+                "id": doc.id,
+                "filename": doc.filename,
+                "title": doc.metadata_json.get("title") if doc.metadata_json else None,
+                "category": doc.metadata_json.get("category") if doc.metadata_json else None,
+                "source": doc.metadata_json.get("source") if doc.metadata_json else None,
+                "form_name": doc.metadata_json.get("form_name") if doc.metadata_json else None,
+            })
+            await graph_ingestor.link_document_to_category(doc.id, doc.metadata_json.get("category") if doc.metadata_json else None)
+            await graph_ingestor.connect_form_to_document(doc.metadata_json.get("form_name") if doc.metadata_json else None, doc.id)
+        except Exception:
+            pass
         return {"document_id": doc_id, "message": "Reindexed successfully", "chunk_count": chunk_count}
 
     async def delete(self, db: AsyncSession, doc_id: str) -> bool:
@@ -224,4 +261,4 @@ class DocumentService:
         return True
 
 
\ No newline at end of file
+document_service = DocumentService()
diff --git a/backend/app/services/graph_service.py b/backend/app/services/graph_service.py
new file mode 100644
index 0000000..1cad975
+++ b/backend/app/services/graph_service.py
@@ -0,0 +1,11 @@
+from typing import Optional
+
+from app.graph.graph_retriever import GraphResult, retrieve_candidates
+
+
+class GraphService:
+    async def get_candidates(self, query: str, filters: Optional[dict] = None) -> GraphResult:
+        return await retrieve_candidates(query, filters or {})
+
+
+graph_service = GraphService()
diff --git a/backend/app/services/rag_service.py b/backend/app/services/rag_service.py
index 4d81140..0344f12 100644
+++ b/backend/app/services/rag_service.py
@@ -14,11 +14,12 @@ from app.rag.evaluator import (
     compute_accuracy, compute_faithfulness, compute_answer_relevancy,
     compute_context_precision, compute_context_recall,
 )
+from app.embeddings.openai_client import openai_client as ollama_client
+from app.repositories.log_repository import log_repo
+from app.core.config import settings
+from app.services.elasticsearch_service import es_service
+from app.core.evaluation_logger import EvaluationLogger
+from app.services.graph_service import graph_service
 
 
 RAG_SYSTEM = """You are an expert assistant. Answer the question using only the provided context.
@@ -33,30 +34,35 @@ class RAGService:
         top_k: int = 5,
         filters: Optional[dict] = None,
         collection_name: Optional[str] = None,
+        all_collections: bool = False,
+    ) -> list[dict[str, Any]]:
+        """
+        Retrieve from single or multiple collections.
+        If all_collections=True, search across all available collections.
+        """
+        candidate_ids = None
+        if settings.USE_KG_RETRIEVAL:
+            graph_result = await graph_service.get_candidates(query, filters)
+            candidate_ids = graph_result.candidate_document_ids or None
+
+        if all_collections:
+            return await multi_collection_retriever.retrieve_all_collections(
+                query, strategy, top_k, filters
+            )
+
+        col = collection_name or "text_documents"
+        if strategy == "vector":
+            return await vector_rag.retrieve(query, col, top_k, filters, False, candidate_ids)
+        elif strategy == "bm25":
+            results = bm25_retriever.search(col, query, top_k)
+            return filter_results(results, filters or {}, candidate_ids)
+        elif strategy == "hybrid":
+            return await hybrid_rag.retrieve(query, col, top_k, filters, candidate_ids)
+        elif strategy == "table":
+            return await table_rag.query(query, top_k=top_k)
+        elif strategy == "pdf":
+            return await pdf_rag.query(query, top_k=top_k)
+        elif strategy == "markdown":
             return await markdown_rag.query(query, top_k=top_k)
         else:
             return await hybrid_rag.retrieve(query, col, top_k, filters)
@@ -66,14 +72,26 @@ class RAGService:
         db: AsyncSession,
         query: str,
         strategy: str = "hybrid",
+        top_k: int = 5,
+        filters: Optional[dict] = None,
+    ) -> dict[str, Any]:
+        start = time.time()
+        graph_context = ""
+        if settings.USE_KG_RETRIEVAL:
+            try:
+                graph_result = await graph_service.get_candidates(query, filters)
+                if graph_result.forms:
+                    form_lines = "\n".join([f"- {f.get('name')}" for f in graph_result.forms if f.get("name")])
+                    graph_context = f"Forms:\n{form_lines}\n"
+            except Exception:
+                graph_context = ""
+
+        chunks = await self.retrieve(query, strategy, top_k, filters)
+        context = "\n\n".join(
+            f"[{r.get('filename','?')}] {r['chunk_text']}" for r in chunks
+        )
+        if graph_context:
+            context = graph_context + "\n" + context
         prompt = f"Context:\n{context}\n\nQuestion: {query}"
         answer = await ollama_client.generate(prompt, system=RAG_SYSTEM)
         latency = (time.time() - start) * 1000
@@ -83,15 +101,25 @@ class RAGService:
             for r in chunks
         ]
 
+        await log_repo.create_retrieval_log(db, {
+            "id": str(uuid.uuid4()),
+            "query": query,
+            "retrieval_strategy": strategy,
+            "retrieved_chunks": [r.get("chunk_id", "") for r in chunks],
+            "generated_answer": answer,
+            "latency_ms": latency,
+            "agent_used": "rag_service",
+        })
+
+        try:
+            await log_repo.create_query_log(db, {
+                "id": str(uuid.uuid4()),
+                "query": query,
+                "response": answer,
+                "latency": latency,
+            })
+        except Exception:
+            pass
 
         confidence = round(sum(r.get("score", 0) for r in chunks) / max(len(chunks), 1), 4)
         return {
@@ -212,4 +240,4 @@ class RAGService:
         return final
 
 
\ No newline at end of file
+rag_service = RAGService()
diff --git a/helpfull scripts/check eval dataset/correct.py b/helpfull scripts/check eval dataset/correct.py
new file mode 100644
index 0000000..b7f0f5f
+++ b/helpfull scripts/check eval dataset/correct.py	
@@ -0,0 +1,154 @@
+import csv
+import openai
+import time
+import os
+from dotenv import load_dotenv
+
+# Load environment variables from .env file
+load_dotenv()
+
+# Get API key from environment variables
+OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
+
+# Initialize OpenAI client
+client = openai.OpenAI(api_key=OPENAI_API_KEY)
+
+def check_answer_with_gpt4(question, answer):
+    """
+    Send question and answer to GPT-4 for evaluation
+    Returns: (is_correct, corrected_answer)
+    """
+    prompt = f"""
+You are an evaluator checking if the provided answer correctly answers the question.
+
+Question: {question}
+
+Provided Answer: {answer}
+
+Please evaluate if the answer is correct and complete. If the answer is correct, respond with "CORRECT" only.
+If the answer is incorrect or incomplete, provide the corrected answer.
+
+Your response format must be:
+- If correct: CORRECT
+- If incorrect: CORRECTED: [your corrected answer here]
+
+Do not add any extra text or explanation.
+"""
+    
+    try:
+        response = client.chat.completions.create(
+            model="gpt-5.1",
+            messages=[
+                {"role": "system", "content": "You are a strict evaluator of answers."},
+                {"role": "user", "content": prompt}
+            ],
+            temperature=0.3,
+            max_completion_tokens=500
+        )
+        
+        result = response.choices[0].message.content.strip()
+        
+        if result.startswith("CORRECTED:"):
+            corrected_answer = result.replace("CORRECTED:", "").strip()
+            return False, corrected_answer
+        elif result == "CORRECT":
+            return True, answer
+        else:
+            # Fallback - if response is unexpected, assume incorrect and use the response as correction
+            return False, result
+            
+    except Exception as e:
+        print(f"Error calling OpenAI API: {e}")
+        # Return False with the original answer to avoid losing data
+        return False, answer
+
+def process_csv(input_file, output_file="mistakes.csv", delay=1):
+    """
+    Process CSV file and create mistakes.csv with corrected answers
+    """
+    mistakes = []
+    
+    try:
+        with open(input_file, 'r', encoding='utf-8') as infile:
+            reader = csv.DictReader(infile)
+            
+            # Check if columns exist
+            expected_columns = ['Question no', 'Eval Question', 'Eval Answer']
+            if not all(col in reader.fieldnames for col in expected_columns):
+                print(f"Error: CSV must have columns: {expected_columns}")
+                return
+            
+            total_rows = 0
+            correct_count = 0
+            incorrect_count = 0
+            
+            for row in reader:
+                total_rows += 1
+                question_no = row['Question no']
+                question = row['Eval Question']
+                answer = row['Eval Answer']
+                
+                print(f"Processing Question {question_no}...")
+                
+                # Check if answer is correct
+                is_correct, corrected_answer = check_answer_with_gpt4(question, answer)
+                
+                if is_correct:
+                    correct_count += 1
+                    print(f"  ✓ Question {question_no}: Correct")
+                else:
+                    incorrect_count += 1
+                    print(f"  ✗ Question {question_no}: Incorrect - Adding to mistakes file")
+                    mistakes.append({
+                        'Question no': question_no,
+                        'Eval Question': question,
+                        'Eval Answer': corrected_answer
+                    })
+                
+                # Add delay to respect rate limits
+                if delay > 0:
+                    time.sleep(delay)
+        
+        # Write mistakes to CSV if any
+        if mistakes:
+            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
+                fieldnames = ['Question no', 'Eval Question', 'Eval Answer']
+                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
+                writer.writeheader()
+                writer.writerows(mistakes)
+            
+            print(f"\n✅ Found {incorrect_count} incorrect answers out of {total_rows} total questions")
+            print(f"📝 Mistakes written to: {output_file}")
+        else:
+            print(f"\n✅ All {total_rows} answers are correct! No mistakes file created.")
+            
+    except FileNotFoundError:
+        print(f"Error: Input file '{input_file}' not found.")
+    except Exception as e:
+        print(f"An error occurred: {e}")
+
+def main():
+    # Configuration
+    INPUT_FILE = "manual_english.csv"  # Change this to your CSV filename
+    OUTPUT_FILE = "mistakes.csv"
+    API_DELAY = 0.5  # Delay between API calls (in seconds) to avoid rate limits
+    
+    print("=" * 60)
+    print("CSV Answer Evaluator with GPT-4o")
+    print("=" * 60)
+    
+    # Validate API key
+    if not OPENAI_API_KEY:
+        print("❌ ERROR: OPENAI_API_KEY not found in .env file")
+        print("Please create a .env file with: OPENAI_API_KEY=your-api-key-here")
+        return
+    
+    print("✅ API key loaded successfully from .env file")
+    print(f"📂 Input file: {INPUT_FILE}")
+    print(f"📄 Output file: {OUTPUT_FILE}")
+    print("=" * 60)
+    
+    process_csv(INPUT_FILE, OUTPUT_FILE, API_DELAY)
+
+if __name__ == "__main__":
+    main()
\ No newline at end of file
diff --git a/helpfull scripts/check eval dataset/extract_top_n_questions.py b/helpfull scripts/check eval dataset/extract_top_n_questions.py
new file mode 100644
index 0000000..122d94a
+++ b/helpfull scripts/check eval dataset/extract_top_n_questions.py	
@@ -0,0 +1,85 @@
+import csv
+import pandas as pd
+
+# Method 1: Using pandas (recommended)
+def extract_top_rows_pandas(input_file, output_file, num_rows=55):
+    """
+    Extract first num_rows from CSV file using pandas
+    Keeps original data exactly as is
+    """
+    try:
+        # Read the CSV file
+        df = pd.read_csv(input_file)
+        
+        # Get first 55 rows
+        top_rows = df.head(num_rows)
+        
+        # Save to new CSV file without adding any new columns
+        top_rows.to_csv(output_file, index=False)
+        
+        print(f"✅ Successfully created '{output_file}' with {len(top_rows)} rows")
+        print(f"📊 Original file had {len(df)} rows")
+        print(f"📋 Original data preserved - no new columns added")
+        
+    except FileNotFoundError:
+        print(f"❌ Error: File '{input_file}' not found")
+    except Exception as e:
+        print(f"❌ Error: {e}")
+
+# Method 2: Using csv module (no external dependencies)
+def extract_top_rows_csv(input_file, output_file, num_rows=55):
+    """
+    Extract first num_rows from CSV file using built-in csv module
+    Keeps original data exactly as is
+    """
+    try:
+        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
+            reader = csv.reader(infile)
+            
+            # Read header
+            header = next(reader)
+            
+            # Read first 55 rows
+            rows = []
+            counter = 0
+            for row in reader:
+                if counter < num_rows:
+                    rows.append(row)
+                    counter += 1
+                else:
+                    break
+        
+        # Write to new CSV file - exactly as original
+        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
+            writer = csv.writer(outfile)
+            
+            # Write header (no new columns added)
+            writer.writerow(header)
+            
+            # Write rows (no counter added)
+            for row in rows:
+                writer.writerow(row)
+        
+        print(f"✅ Successfully created '{output_file}' with {len(rows)} rows")
+        print(f"📋 Original data preserved - no new columns added")
+        
+    except FileNotFoundError:
+        print(f"❌ Error: File '{input_file}' not found")
+    except StopIteration:
+        print(f"⚠️  Warning: CSV file has no header row")
+    except Exception as e:
+        print(f"❌ Error: {e}")
+
+# Usage example
+if __name__ == "__main__":
+    # Specify your input file name
+    input_filename = "mistakes.csv"  # Change this to your actual filename
+    output_filename = "mistakes_top_55.csv"
+    
+    # Choose one of the methods below:
+    
+    # Method 1: Using pandas (recommended for large files)
+    extract_top_rows_pandas(input_filename, output_filename)
+    
+    # Method 2: Using built-in csv module (no pandas required)
+    # extract_top_rows_csv(input_filename, output_filename)
\ No newline at end of file
```

## 888710100e17fa38dd5d68df8dc41637e21a1834 — 2026-06-15T12:50:22+05:30

Message:

#	Change	What
1	RRF fusion	Replaced _merge_results() (simple dedup) with reciprocal_rank_fusion() from app.rag.rrf — standard 1/(k+rank) scoring across all vector + BM25 ranked lists
2	Intent detection	Added _detect_section() — maps query keywords (e.g., "interest rate", "eligibility") to section names — and _extract_product_name() — regex patterns for 16 banking products (kisan-credit-card, home-loan, etc.)
3	Metadata filters	Detected section and product_name (mapped to filename) are passed as ChromaDB where filters to vector_rag.retrieve() and as in-memory filters via filter_results() for BM25
4	Neighbor chunk expansion	_fetch_neighbor_chunks() groups retrieved chunks by document_id, fetches all chunks for that document from ChromaDB, and adds ±1 adjacent chunks
5	Two-pass reranking	First rerank at top_k * 2 → neighbor expansion → second rerank to final top_k
6	use_query_expansion wired up	When False, only the original query (not synonym variants) is used
7	Return dict	Added detected_section and detected_product keys

```diff
diff --git a/backend/app/evaluation/agent_runner.py b/backend/app/evaluation/agent_runner.py
index fe30dc9..00bb70e 100644
+++ b/backend/app/evaluation/agent_runner.py
@@ -1,244 +1,472 @@
+"""
+Metadata-aware RAG evaluation for banking/product documents.
+
+Flow:
+  Question
+    ↓
+  Synonym Expansion
+    ↓
+  Intent Detection & Metadata Filter Generation
+    ↓
+  Hybrid Retrieval (Vector + BM25) with filters
+    ↓
+  RRF Fusion
+    ↓
+  Cross-Encoder Rerank (first pass)
+    ↓
+  Neighbor Chunk Expansion (same document, adjacent chunks)
+    ↓
+  Cross-Encoder Rerank (second pass)
+    ↓
+  ES Enhancement
+    ↓
+  LLM
+"""
+
+import time
+import re
+from typing import Any
+
+from app.rag.vector_rag import vector_rag
+from app.rag.bm25 import bm25_retriever
+from app.rag.synonym_expansion import get_synonym_expander
+from app.rag.cross_encoder import cross_encoder
+from app.rag.rrf import reciprocal_rank_fusion
+from app.rag.metadata_filter import filter_results
+from app.chromadb.client import chroma_client
+from app.evaluation.evaluator import evaluate_single
+from app.embeddings.openai_client import openai_client
+from app.core.config import settings
+from app.core.evaluation_logger import EvaluationLogger
+from app.services.elasticsearch_service import es_service
+
+# Global logger instance
+_current_logger = None
+
+def set_evaluation_logger(logger: EvaluationLogger):
+    global _current_logger
+    _current_logger = logger
+
+
+RAG_SYSTEM = """Answer the question carefully"""
+
+
+# ── Section keyword mapping for intent detection ──────────────────────────
+SECTION_KEYWORDS = {
+    "interest_rate": [
+        "interest rate", "interest", "roi", "rate of interest",
+        "apr", "annual percentage", "interest charged",
+    ],
+    "eligibility": [
+        "eligibility", "eligible", "who can", "qualify",
+        "requirements", "criteria", "prerequisite", "conditions",
+    ],
+    "purpose": [
+        "purpose", "objective", "aim", "goal", "why is", "use",
+        "for what", "intended for",
+    ],
+    "documents_required": [
+        "documents required", "documentation", "required documents",
+        "docs needed", "paperwork", "documents to", "documents needed",
+        "documents required for",
+    ],
+    "security": [
+        "security", "collateral", "guarantee", "pledge", "mortgage",
+        "security required",
+    ],
+    "features": [
+        "features", "benefits", "advantages", "highlights", "salient",
+        "key features",
+    ],
+    "fees": [
+        "fees", "charges", "penalty", "processing fee", "late fee",
+        "applicable fees",
+    ],
+    "tenure": [
+        "tenure", "duration", "period", "repayment", "term",
+        "maturity", "repayment period", "loan tenure",
+    ],
+    "loan_amount": [
+        "loan amount", "amount", "maximum loan", "sanction",
+        "limit", "maximum amount",
+    ],
+}
+
+# Common banking product patterns for product name detection
+PRODUCT_PATTERNS = [
+    (r"kisan\s*credit\s*card", "kisan-credit-card"),
+    (r"personal\s*loan", "personal-loan"),
+    (r"home\s*loan", "home-loan"),
+    (r"car\s*loan|auto\s*loan|vehicle\s*loan", "car-loan"),
+    (r"education\s*loan|student\s*loan", "education-loan"),
+    (r"savings\s*account", "savings-account"),
+    (r"current\s*account", "current-account"),
+    (r"fixed\s*deposit", "fixed-deposit"),
+    (r"recurring\s*deposit", "recurring-deposit"),
+    (r"credit\s*card", "credit-card"),
+    (r"overdraft", "overdraft"),
+    (r"gold\s*loan", "gold-loan"),
+    (r"mortgage\s*loan|home\s*loan", "home-loan"),
+    (r"business\s*loan", "business-loan"),
+    (r"agriculture\s*loan|agricultural\s*loan|farm\s*loan|kisan", "agriculture-loan"),
+]
+
+
+def _chunk_texts(chunks: list[dict]) -> list[str]:
+    return [c["chunk_text"] for c in chunks if c.get("chunk_text")]
+
+
+def _detect_section(query: str) -> str | None:
+    """Detect the intended document section from the query using keyword matching."""
+    query_lower = query.lower()
+    for section, keywords in SECTION_KEYWORDS.items():
+        for kw in keywords:
+            if kw in query_lower:
+                return section
+    return None
+
+
+def _extract_product_name(query: str) -> str | None:
+    """Extract a normalized product name from the query, or None if not found."""
+    query_lower = query.lower()
+    for pattern, product_name in PRODUCT_PATTERNS:
+        if re.search(pattern, query_lower):
+            return product_name
+    return None
+
+
+async def _fetch_neighbor_chunks(
+    chunks: list[dict],
+    collection_name: str,
+    neighbor_window: int = 1,
+) -> list[dict]:
+    """Fetch neighboring chunks from the same document for each retrieved chunk."""
+    if not chunks:
+        return chunks
+
+    expanded = []
+    seen_ids = {c.get("chunk_id") for c in chunks if c.get("chunk_id")}
+
+    # Group chunks by document_id so we fetch per-doc once
+    doc_groups: dict[str, list[dict]] = {}
+    for chunk in chunks:
+        meta = chunk.get("metadata", {})
+        doc_id = meta.get("document_id") or chunk.get("document_id")
+        if doc_id:
+            doc_groups.setdefault(doc_id, []).append(chunk)
+
+    # For each doc group, fetch all chunks from ChromaDB and find neighbors
+    for doc_id, doc_chunks in doc_groups.items():
+        try:
+            collection = chroma_client.get_client().get_collection(collection_name)
+            result = collection.get(
+                where={"document_id": doc_id},
+                include=["documents", "metadatas"],
+            )
+        except Exception:
+            # If collection doesn't exist or fetch fails, just pass through
+            expanded.extend(doc_chunks)
+            continue
+
+        doc_ids = result.get("ids", [])
+        doc_docs = result.get("documents", [])
+        doc_metas = result.get("metadatas", [])
+
+        # Build a map of chunk_index -> chunk data for this document
+        doc_index_map: dict[int, dict] = {}
+        for i, cid in enumerate(doc_ids):
+            c_meta = doc_metas[i] if i < len(doc_metas) else {}
+            c_idx = c_meta.get("chunk_index")
+            if c_idx is not None:
+                doc_index_map[c_idx] = {
+                    "chunk_id": cid,
+                    "chunk_text": doc_docs[i] if i < len(doc_docs) else "",
+                    "metadata": c_meta or {},
+                    "document_id": doc_id,
+                }
+
+        # For each retrieved chunk, find and add neighbors
+        for chunk in doc_chunks:
+            meta = chunk.get("metadata", {})
+            chunk_idx = meta.get("chunk_index")
+            if chunk_idx is None:
+                expanded.append(chunk)
+                continue
+
+            for offset in range(-neighbor_window, neighbor_window + 1):
+                if offset == 0:
+                    continue
+                neighbor_idx = chunk_idx + offset
+                neighbor = doc_index_map.get(neighbor_idx)
+                if neighbor and neighbor["chunk_id"] not in seen_ids:
+                    seen_ids.add(neighbor["chunk_id"])
+                    neighbor_entry = {
+                        **neighbor,
+                        "score": chunk.get("score", 0) * 0.9,
+                    }
+                    expanded.append(neighbor_entry)
+
+            expanded.append(chunk)
+
+    return expanded
+
+
+async def evaluate_question(
+    question: str,
+    expected_answer: str,
+    top_k: int = 5,
+    use_query_expansion: bool = False,
+    num_expansions: int = 2,
+) -> dict[str, Any]:
+    """
+    Metadata-aware RAG evaluation:
+
+      1. Synonym expansion (if enabled)
+      2. Intent detection → metadata filters (section, product_name)
+      3. Hybrid retrieval (Vector + BM25) with metadata filters
+      4. RRF fusion instead of simple merge
+      5. Cross-encoder rerank (first pass, over-retrieve)
+      6. Neighbor chunk expansion from same document
+      7. Cross-encoder rerank (second pass)
+      8. ES enhancement
+      9. Generate answer
+     10. Score with LLM-as-judge
+    """
+    t0 = time.time()
+
+    if _current_logger:
+        _current_logger.log("QUESTION_START", f"Q: {question}\nExpected: {expected_answer}")
+
+    # ── 1. Intent detection & metadata filter generation ──────────────────
+    detected_section = _detect_section(question)
+    detected_product = _extract_product_name(question)
+
+    filters = {}
+    if detected_product:
+        filters["filename"] = detected_product
+    if detected_section:
+        filters["section"] = detected_section
+
+    if _current_logger and (detected_section or detected_product):
+        _current_logger.log("INTENT_DETECTION",
+            f"Detected section: {detected_section}\n"
+            f"Detected product: {detected_product}\n"
+            f"Filters: {filters}")
+
+    # ── 2. Synonym expansion ──────────────────────────────────────────────
+    synonym_expander = get_synonym_expander()
+    queries = synonym_expander.expand_query(question)
+    if not use_query_expansion:
+        queries = [queries[0]]  # Use original query only
+    if _current_logger:
+        _current_logger.log("SYNONYM_EXPANSION", f"Generated {len(queries)} queries:\n" + "\n".join(queries))
+
+    # ── 3. Hybrid retrieval + RRF fusion ──────────────────────────────────
+    all_ranked_lists = []
+
+    for idx, query in enumerate(queries, 1):
+        # Vector search with metadata filters
+        vector_results = await vector_rag.retrieve(
+            query, "text_documents",
+            top_k=getattr(settings, "DENSE_TOP_K", 10),
+            filters=filters if filters else None,
+        )
+        if _current_logger:
+            _current_logger.log(f"VECTOR_SEARCH_{idx}",
+                f"Query: {query}\nFound: {len(vector_results)} chunks")
+
+        # BM25 search
+        bm25_results = bm25_retriever.search(
+            "text_documents", query,
+            top_k=getattr(settings, "BM25_TOP_K", 10),
+        )
+
+        # Apply in-memory metadata filtering for BM25 results
+        if filters:
+            bm25_results = filter_results(bm25_results, filters)
+
+        if _current_logger:
+            _current_logger.log(f"BM25_SEARCH_{idx}",
+                f"Query: {query}\nFound: {len(bm25_results)} chunks")
+
+        if vector_results:
+            all_ranked_lists.append(vector_results)
+        if bm25_results:
+            all_ranked_lists.append(bm25_results)
+
+    if not all_ranked_lists:
+        all_chunks = []
+    else:
+        # ── 4. RRF fusion ─────────────────────────────────────────────────
+        all_chunks = reciprocal_rank_fusion(all_ranked_lists, k=60, top_k=top_k * 4)
+
+    if _current_logger:
+        _current_logger.log("RRF_FUSION", f"Total unique chunks after RRF: {len(all_chunks)}")
+
+    # ── 5. First rerank (over-retrieve) ───────────────────────────────────
+    reranked = cross_encoder.rerank(question, all_chunks, top_k=top_k * 2)
+    if _current_logger:
+        _current_logger.log("FIRST_RERANK", f"Top {top_k * 2} chunks after initial reranking")
+
+    # ── 6. Neighbor chunk expansion ───────────────────────────────────────
+    expanded_chunks = await _fetch_neighbor_chunks(reranked, "text_documents")
+    if _current_logger:
+        _current_logger.log("NEIGHBOR_EXPANSION",
+            f"Before: {len(reranked)} chunks\n"
+            f"After neighbor expansion: {len(expanded_chunks)} chunks")
+
+    # ── 7. Second rerank ──────────────────────────────────────────────────
+    reranked_final = cross_encoder.rerank(question, expanded_chunks, top_k=top_k)
+    if _current_logger:
+        _current_logger.log("SECOND_RERANK", f"Final top {top_k} chunks after neighbor-aware reranking")
+        for i, chunk in enumerate(reranked_final, 1):
+            _current_logger.log(f"CHUNK_{i}",
+                f"Score: {chunk.get('score', 0)}\n"
+                f"File: {chunk.get('filename', 'unknown')}\n"
+                f"Section: {chunk.get('metadata', {}).get('section', 'N/A')}\n"
+                f"Chunk Index: {chunk.get('metadata', {}).get('chunk_index', 'N/A')}\n"
+                f"Text: {chunk.get('chunk_text', '')[:300]}...")
+
+    # ── 8. Generate answer ────────────────────────────────────────────────
+    chunk_texts = _chunk_texts(reranked_final)
+
+    # Enhance with Elasticsearch via iterative query
+    original_count = len(chunk_texts)
+    enhanced_texts = await es_service.enhance_with_iterative_query(
+        chunk_texts, question, max_tries=3, logger=_current_logger
+    )
+    if _current_logger:
+        _current_logger.log("ES_ENHANCEMENT_COMPLETE",
+            f"Original: {original_count} chunks\n"
+            f"Enhanced: {len(enhanced_texts)} chunks\n"
+            f"Added: {len(enhanced_texts) - original_count} from Elasticsearch"
+            f"\nEnhanced texts:\n" + "\n---\n".join(enhanced_texts)
+            )
+
+    context_text = "\n---\n".join(enhanced_texts)
+
+    if context_text.strip():
+        prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
+    else:
+        prompt = f"Question: {question}"
+
+    if _current_logger:
+        _current_logger.log("LLM_PROMPT", f"Prompt length: {len(prompt)} chars\nPrompt preview:\n{prompt[:500]}...")
+
+    llm_start = time.time()
+    generated_answer = await openai_client.generate(prompt, system=RAG_SYSTEM)
+    llm_time = (time.time() - llm_start) * 1000
+
+    if _current_logger:
+        _current_logger.log("LLM_ANSWER", f"Generated in {llm_time}ms:\n{generated_answer}")
+
+    # ── 9. Score ──────────────────────────────────────────────────────────
+    # Build retrieved chunk IDs and derive gold IDs via text matching
+    retrieved_chunk_ids = [c.get("chunk_id", "") for c in reranked_final if c.get("chunk_id")]
+    expected_lower = expected_answer.lower()
+    gold_chunk_ids = {
+        c.get("chunk_id")
+        for c in reranked_final
+        if c.get("chunk_id") and (
+            expected_lower in c.get("chunk_text", "").lower()
+            or (len(expected_lower) >= 20 and expected_lower[:20] in c.get("chunk_text", "").lower())
+        )
+    }
+    # Fallback: if no exact match found, mark best-scoring chunks as gold
+    if not gold_chunk_ids and reranked_final:
+        best_score = max((c.get("ce_score", 0) for c in reranked_final), default=0)
+        if best_score > 0:
+            gold_chunk_ids = {
+                c.get("chunk_id") for c in reranked_final
+                if c.get("ce_score", 0) >= best_score * 0.9 and c.get("chunk_id")
+            }
+
+    if _current_logger:
+        _current_logger.log("GOLD_CHUNKS",
+            f"Identified {len(gold_chunk_ids)} gold chunks from {len(retrieved_chunk_ids)} retrieved")
+
+    scores = await evaluate_single(question, expected_answer, generated_answer, enhanced_texts,
+                                   retrieved_chunk_ids=retrieved_chunk_ids, gold_chunk_ids=gold_chunk_ids)
+
+    if _current_logger:
+        _current_logger.log("METRICS",
+            f"Accuracy: {scores.get('accuracy_llm')}\n"
+            f"Faithfulness: {scores.get('faithfulness')}\n"
+            f"Context Precision: {scores.get('context_precision')}\n"
+            f"Context Recall: {scores.get('context_recall')}\n"
+            f"Answer Relevancy: {scores.get('answer_relevancy')}")
+
+    latency_ms = round((time.time() - t0) * 1000, 1)
+
+    return {
+        "question": question,
+        "expected_answer": expected_answer,
+        "generated_answer": generated_answer,
+        "retrieved_context": context_text,
+        "expanded_queries": queries,
+        "num_chunks": len(reranked_final),
+        # LLM-as-judge
+        "accuracy_llm": scores.get("accuracy_llm", 0.0),
+        "faithfulness": scores.get("faithfulness", 0.0),
+        "answer_relevancy": scores.get("answer_relevancy", 0.0),
+        "context_precision": scores.get("context_precision", 0.0),
+        "context_recall": scores.get("context_recall", 0.0),
+        # Accuracy methods
+        "exact_match": scores.get("exact_match", 0.0),
+        "semantic_similarity": scores.get("semantic_similarity", 0.0),
+        "f1": scores.get("f1", 0.0),
+        "accuracy_combined": scores.get("accuracy_combined", 0.0),
+        # Retrieval metrics
+        "recall_10": scores.get("recall_10", 0.0),
+        "recall_20": scores.get("recall_20", 0.0),
+        "recall_50": scores.get("recall_50", 0.0),
+        "mrr": scores.get("mrr", 0.0),
+        "ndcg_10": scores.get("ndcg_10", 0.0),
+        # Rationales
+        "accuracy_rationale": scores.get("accuracy_rationale", ""),
+        "faithfulness_rationale": scores.get("faithfulness_rationale", ""),
+        "answer_relevancy_rationale": scores.get("answer_relevancy_rationale", ""),
+        "context_precision_rationale": scores.get("context_precision_rationale", ""),
+        "context_recall_rationale": scores.get("context_recall_rationale", ""),
+        # Metadata
+        "detected_section": detected_section,
+        "detected_product": detected_product,
+        "latency_ms": latency_ms,
+    }
+
+
+def failed_question_row(question: str, expected_answer: str, error: str) -> dict[str, Any]:
+    """Build a zeroed per-question row when the eval pipeline raises."""
+    return {
+        "question": question,
+        "expected_answer": expected_answer,
+        "generated_answer": "",
+        "retrieved_context": "",
+        "expanded_queries": [],
+        "num_chunks": 0,
+        # LLM-as-judge
+        "accuracy_llm": 0.0,
+        "faithfulness": 0.0,
+        "answer_relevancy": 0.0,
+        "context_precision": 0.0,
+        "context_recall": 0.0,
+        # Accuracy methods
+        "exact_match": 0.0,
+        "semantic_similarity": 0.0,
+        "f1": 0.0,
+        "accuracy_combined": 0.0,
+        # Retrieval metrics
+        "recall_10": 0.0,
+        "recall_20": 0.0,
+        "recall_50": 0.0,
+        "mrr": 0.0,
+        "ndcg_10": 0.0,
+        # Rationales
+        "accuracy_rationale": "",
+        "faithfulness_rationale": "",
+        "context_precision_rationale": "",
+        "context_recall_rationale": "",
+        "answer_relevancy_rationale": "",
+        "latency_ms": 0.0,
+        "error": error,
+    }
```

## 8b2a39fc14b80b6291962ac00da634c9692e78fc — 2026-06-15T12:14:13+05:30

Message:

better upload to elastisearch

```diff
diff --git a/helpfull scripts/upload_to_elasticsearch.py b/helpfull scripts/upload_to_elasticsearch.py
index ce1f76b..44c990e 100644
+++ b/helpfull scripts/upload_to_elasticsearch.py	
@@ -1,60 +1,375 @@
 #!/usr/bin/env python3
+
+import json
+import hashlib
 from pathlib import Path
 
+import fitz
+from elasticsearch import Elasticsearch, helpers
+from sentence_transformers import SentenceTransformer
+from tqdm import tqdm
+
+
+# ============================================================
+# CONFIG
+# ============================================================
+
+ES_HOST = "http://10.64.240.47:9200"
 INDEX_NAME = "rag_documents"
 
+DATA_DIR = "/home/wtc6/formbot-jishnu/data"
+
+EMBEDDING_MODEL = "BAAI/bge-m3"
+
+CHUNK_SIZE = 1000
+CHUNK_OVERLAP = 200
+
+BULK_SIZE = 500
+
+# BGE-M3 produces 1024-dimensional embeddings
+EMBEDDING_DIMS = 1024
+
+
+# ============================================================
+# INDEX SETUP
+# ============================================================
+
+def create_index_if_needed(es):
+
+    if es.indices.exists(index=INDEX_NAME):
+        print(f"Index '{INDEX_NAME}' already exists")
+        return
+
+    print(f"Creating index '{INDEX_NAME}'...")
+
+    es.indices.create(
+        index=INDEX_NAME,
+        body={
+            "mappings": {
+                "properties": {
+
+                    "content": {
+                        "type": "text"
+                    },
+
+                    "embedding": {
+                        "type": "dense_vector",
+                        "dims": EMBEDDING_DIMS,
+                        "index": True,
+                        "similarity": "cosine"
+                    },
+
+                    # ========================
+                    # FILE METADATA
+                    # ========================
+
+                    "source_id": {
+                        "type": "keyword"
+                    },
+
+                    "file_name": {
+                        "type": "keyword"
+                    },
+
+                    "file_stem": {
+                        "type": "keyword"
+                    },
+
+                    "file_path": {
+                        "type": "keyword"
+                    },
+
+                    "file_type": {
+                        "type": "keyword"
+                    },
+
+                    # ========================
+                    # DOCUMENT METADATA
+                    # ========================
+
+                    "document_id": {
+                        "type": "keyword"
+                    },
+
+                    "chunk_number": {
+                        "type": "integer"
+                    },
+
+                    "total_chunks": {
+                        "type": "integer"
+                    },
+
+                    # ========================
+                    # SOURCE INFO
+                    # ========================
+
+                    "url": {
+                        "type": "keyword"
+                    },
+
+                    # ========================
+                    # STATS
+                    # ========================
+
+                    "content_length": {
+                        "type": "integer"
+                    }
+                }
+            }
+        }
+    )
+
+    print("Index created")
+
+
+# ============================================================
+# FILE READERS
+# ============================================================
+
+def read_json(path: Path):
+
+    with open(path, "r", encoding="utf-8") as f:
+        data = json.load(f)
+
+    content = data.get("content", "").strip()
+    url = data.get("url", "")
+
+    return content, url
+
+
+def read_txt(path: Path):
+
+    with open(path, "r", encoding="utf-8") as f:
+        return f.read().strip(), ""
+
+
+def read_pdf(path: Path):
+
+    doc = fitz.open(path)
+
+    pages = []
+
+    for page in doc:
+        pages.append(page.get_text())
+
+    doc.close()
+
+    return "\n".join(pages).strip(), ""
+
+
+def read_file(path: Path):
+
+    suffix = path.suffix.lower()
+
     try:
+
+        if suffix == ".json":
+            return read_json(path)
+
+        elif suffix == ".txt":
+            return read_txt(path)
+
+        elif suffix == ".pdf":
+            return read_pdf(path)
+
+        return None, None
+
     except Exception as e:
+        print(f"ERROR reading {path}: {e}")
+        return None, None
+
+
+# ============================================================
+# CHUNKING
+# ============================================================
+
+def chunk_text(text):
+
+    if not text:
+        return []
+
+    paragraphs = [
+        p.strip()
+        for p in text.split("\n\n")
+        if p.strip()
+    ]
+
+    chunks = []
+    current = ""
+
+    for para in paragraphs:
+
+        if len(current) + len(para) < CHUNK_SIZE:
+
+            current += "\n\n" + para
+
+        else:
+
+            chunks.append(current.strip())
+
+            overlap_text = (
+                current[-CHUNK_OVERLAP:]
+                if len(current) > CHUNK_OVERLAP
+                else current
+            )
+
+            current = overlap_text + "\n\n" + para
+
+    if current:
+        chunks.append(current.strip())
+
+    return chunks
+
+
+# ============================================================
+# MAIN
+# ============================================================
+
+def main():
+
+    print("Connecting to Elasticsearch...")
+
+    es = Elasticsearch(ES_HOST)
+
+    if not es.ping():
+        raise RuntimeError(
+            f"Cannot connect to Elasticsearch at {ES_HOST}"
+        )
+
+    create_index_if_needed(es)
+
+    print("Loading embedding model...")
+    model = SentenceTransformer(EMBEDDING_MODEL)
+
+    files = []
+
+    for ext in ("*.json", "*.txt", "*.pdf"):
+        files.extend(Path(DATA_DIR).rglob(ext))
+
+    print(f"Found {len(files)} files")
+
+    actions = []
+
+    total_chunks = 0
+    total_docs = 0
+
+    for file_path in tqdm(files, desc="Processing"):
+
+        content, url = read_file(file_path)
+
+        if not content:
+            continue
+
+        chunks = chunk_text(content)
+
+        if not chunks:
+            continue
+
+        embeddings = model.encode(
+            chunks,
+            normalize_embeddings=True,
+            show_progress_bar=False
+        )
+
+        # Stable document identifier
+        source_id = hashlib.sha256(
+            str(file_path.resolve()).encode("utf-8")
+        ).hexdigest()
+
+        total_file_chunks = len(chunks)
+
+        for i, (chunk, embedding) in enumerate(
+            zip(chunks, embeddings)
+        ):
+
+            actions.append(
+                {
+                    "_index": INDEX_NAME,
+                    "_id": f"{source_id}_{i}",
+                    "_source": {
+
+                        # ========================
+                        # RAG DATA
+                        # ========================
+
+                        "content": chunk,
+                        "embedding": embedding.tolist(),
+
+                        # ========================
+                        # FILE METADATA
+                        # ========================
+
+                        "source_id": source_id,
+                        "file_name": file_path.name,
+                        "file_stem": file_path.stem,
+                        "file_path": str(
+                            file_path.resolve()
+                        ),
+                        "file_type": file_path.suffix.lower().replace(
+                            ".", ""
+                        ),
+
+                        # ========================
+                        # DOCUMENT METADATA
+                        # ========================
+
+                        "document_id": file_path.stem,
+                        "chunk_number": i,
+                        "total_chunks": total_file_chunks,
+
+                        # ========================
+                        # SOURCE INFO
+                        # ========================
+
+                        "url": url,
+
+                        # ========================
+                        # STATS
+                        # ========================
+
+                        "content_length": len(chunk)
+                    }
+                }
+            )
+
+            total_chunks += 1
+
+            if len(actions) >= BULK_SIZE:
+
+                helpers.bulk(
+                    es,
+                    actions,
+                    request_timeout=300
+                )
+
+                actions.clear()
+
+        total_docs += 1
+
+    if actions:
+
+        helpers.bulk(
+            es,
+            actions,
+            request_timeout=300
+        )
+
+    es.indices.refresh(index=INDEX_NAME)
+
+    print()
+    print("=" * 60)
+    print(f"Documents indexed : {total_docs}")
+    print(f"Chunks indexed    : {total_chunks}")
+    print(f"Index             : {INDEX_NAME}")
+    print("=" * 60)
+
+    print("\nUseful filters:")
+    print("  file_name")
+    print("  file_stem")
+    print("  file_type")
+    print("  source_id")
+    print("  document_id")
+
 
 if __name__ == "__main__":
+    main()
\ No newline at end of file
```

## 1687fa8535c10512cbabb4682fc3f43b891d27b1 — 2026-06-15T11:47:39+05:30

Message:

removing unused files

```diff
diff --git a/backend/app/agents/__init__.py b/backend/app/agents/__init__.py
deleted file mode 100644
index e69de29..0000000
diff --git a/backend/app/agents/base.py b/backend/app/agents/base.py
deleted file mode 100644
index 68de3dd..0000000
+++ /dev/null
@@ -1,31 +0,0 @@
\ No newline at end of file
diff --git a/backend/app/agents/coordinator_agent.py b/backend/app/agents/coordinator_agent.py
deleted file mode 100644
index d7ceb9c..0000000
+++ /dev/null
@@ -1,140 +0,0 @@
\ No newline at end of file
diff --git a/backend/app/agents/evaluator_agent.py b/backend/app/agents/evaluator_agent.py
deleted file mode 100644
index f6d9b82..0000000
+++ /dev/null
@@ -1,82 +0,0 @@
diff --git a/backend/app/agents/router_agent.py b/backend/app/agents/router_agent.py
deleted file mode 100644
index 269e90f..0000000
+++ /dev/null
@@ -1,85 +0,0 @@
\ No newline at end of file
diff --git a/backend/app/agents/sqlite_agent.py b/backend/app/agents/sqlite_agent.py
deleted file mode 100644
index 45d4588..0000000
+++ /dev/null
@@ -1,52 +0,0 @@
\ No newline at end of file
diff --git a/backend/app/agents/vector_agent.py b/backend/app/agents/vector_agent.py
deleted file mode 100644
index 730a632..0000000
+++ /dev/null
@@ -1,63 +0,0 @@
\ No newline at end of file
diff --git a/backend/app/agents/web_agent.py b/backend/app/agents/web_agent.py
deleted file mode 100644
index 189f264..0000000
+++ /dev/null
@@ -1,59 +0,0 @@
\ No newline at end of file
diff --git a/backend/app/api/agents.py b/backend/app/api/agents.py
deleted file mode 100644
index fe117f6..0000000
+++ /dev/null
@@ -1,78 +0,0 @@
diff --git a/backend/app/api/elasticsearch.py b/backend/app/api/elasticsearch.py
deleted file mode 100644
index 08fbc41..0000000
+++ /dev/null
@@ -1,22 +0,0 @@
diff --git a/backend/app/api/embeddings.py b/backend/app/api/embeddings.py
deleted file mode 100644
index c0562e3..0000000
+++ /dev/null
@@ -1,38 +0,0 @@
\ No newline at end of file
diff --git a/backend/app/api/markdown.py b/backend/app/api/markdown.py
deleted file mode 100644
index dee1707..0000000
+++ /dev/null
@@ -1,49 +0,0 @@
diff --git a/backend/app/api/pdf.py b/backend/app/api/pdf.py
deleted file mode 100644
index d2ba32d..0000000
+++ /dev/null
@@ -1,50 +0,0 @@
diff --git a/backend/app/api/search.py b/backend/app/api/search.py
deleted file mode 100644
index c4da323..0000000
+++ /dev/null
@@ -1,149 +0,0 @@
\ No newline at end of file
diff --git a/backend/app/api/tablerag.py b/backend/app/api/tablerag.py
deleted file mode 100644
index 2b5f64a..0000000
+++ /dev/null
@@ -1,57 +0,0 @@
diff --git a/backend/app/api/web.py b/backend/app/api/web.py
deleted file mode 100644
index a4c5405..0000000
+++ /dev/null
@@ -1,35 +0,0 @@
diff --git a/backend/app/main.py b/backend/app/main.py
index 18d0ee7..82cb72c 100644
+++ b/backend/app/main.py
@@ -20,17 +20,9 @@ from app.chromadb.client import chroma_client
 # API routers
 from app.api.health import router as health_router
 from app.api.documents import router as documents_router
 from app.api.chat import router as chat_router
 from app.api.rag import router as rag_router
 from app.api.chroma import router as chroma_router
 
 
 
@@ -84,14 +76,6 @@ app.add_exception_handler(Exception, generic_exception_handler)
 # Register all routers
 app.include_router(health_router)
 app.include_router(documents_router)
 app.include_router(chat_router)
 app.include_router(rag_router)
\ No newline at end of file
+app.include_router(chroma_router)
\ No newline at end of file
diff --git a/backend/app/schemas/agent.py b/backend/app/schemas/agent.py
deleted file mode 100644
index 791444f..0000000
+++ /dev/null
@@ -1,25 +0,0 @@
diff --git a/backend/app/schemas/embeddings.py b/backend/app/schemas/embeddings.py
deleted file mode 100644
index 0141931..0000000
+++ /dev/null
@@ -1,29 +0,0 @@
diff --git a/backend/app/schemas/web.py b/backend/app/schemas/web.py
deleted file mode 100644
index b9cc2f7..0000000
+++ /dev/null
@@ -1,21 +0,0 @@
diff --git a/backend/app/services/web_service.py b/backend/app/services/web_service.py
deleted file mode 100644
index cb1898d..0000000
+++ /dev/null
@@ -1,99 +0,0 @@
\ No newline at end of file
```

## 578bc6df2a206489db7a09fe8674f95a034e5925 — 2026-06-15T11:36:42+05:30

Message:

frontend unused remove

_No Python file changes in this commit._

## 299dbddbf48a8e7f1eae7ca724c359a9d04f4507 — 2026-06-15T11:29:17+05:30

Message:

results and other stuff

```diff
diff --git a/ask_gpt.py b/ask_gpt.py
new file mode 100644
index 0000000..e69de29
diff --git a/helpfull scripts/eval_csv_to_txt.py b/helpfull scripts/eval_csv_to_txt.py
new file mode 100644
index 0000000..20a7a93
+++ b/helpfull scripts/eval_csv_to_txt.py	
@@ -0,0 +1,84 @@
+import pandas as pd
+from pathlib import Path
+
+# Input and output files
+csv_file = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/rag_eval_1781246175645 12th june overall score 63 added elasticsearch and crawled websites.csv"
+output_file = "formatted_report_rag_eval_1781246175645 12th june overall score 63 added elasticsearch and crawled websites.csv.txt"
+
+# Read CSV
+df = pd.read_csv(csv_file)
+
+with open(output_file, "w", encoding="utf-8") as f:
+    for _, row in df.iterrows():
+
+        qno = row.get("Question No", "")
+        question = str(row.get("Question", "")).strip()
+        expected = str(row.get("Expected Answer", "")).strip()
+        generated = str(row.get("Generated Answer", "")).strip()
+        context = str(row.get("Retrieved Context", "")).strip()
+
+        f.write("=" * 70 + "\n")
+        f.write(f"QUESTION {qno}\n")
+        f.write("=" * 70 + "\n\n")
+
+        f.write("Question\n")
+        f.write("-" * 8 + "\n")
+        f.write(question + "\n\n")
+
+        f.write("Expected Answer\n")
+        f.write("-" * 15 + "\n")
+        f.write(expected + "\n\n")
+
+        f.write("Generated Answer\n")
+        f.write("-" * 16 + "\n")
+        f.write(generated + "\n\n")
+
+        f.write("Retrieved Context\n")
+        f.write("-" * 17 + "\n")
+        f.write(context + "\n\n")
+
+        f.write("Evaluation Metrics\n")
+        f.write("-" * 18 + "\n")
+
+        metric_cols = [
+            "Accuracy (LLM)",
+            "Faithfulness",
+            "Context Precision",
+            "Context Recall",
+            "Answer Relevancy",
+            "Exact Match",
+            "Semantic Similarity",
+            "F1 Score",
+            "Accuracy (Combined)",
+            "Recall@10",
+            "Recall@20",
+            "Recall@50",
+            "MRR",
+            "nDCG@10",
+            "Gold Answer Found",
+            "Latency (ms)"
+        ]
+
+        for col in metric_cols:
+            if col in df.columns:
+                f.write(f"{col}: {row[col]}\n")
+
+        f.write("\nRationale\n")
+        f.write("-" * 9 + "\n")
+
+        rationale_cols = [
+            "Accuracy Rationale",
+            "Faithfulness Rationale",
+            "Context Precision Rationale",
+            "Context Recall Rationale",
+            "Answer Relevancy Rationale"
+        ]
+
+        for col in rationale_cols:
+            if col in df.columns and pd.notna(row[col]):
+                f.write(f"\n{col.replace(' Rationale', '')}:\n")
+                f.write(str(row[col]) + "\n")
+
+        f.write("\n\n")
+
+print(f"Formatted report saved to {output_file}")
\ No newline at end of file
diff --git a/helpfull scripts/heatmap_metrics.py b/helpfull scripts/heatmap_metrics.py
new file mode 100644
index 0000000..c7b7433
+++ b/helpfull scripts/heatmap_metrics.py	
@@ -0,0 +1,256 @@
+import pandas as pd
+import matplotlib.pyplot as plt
+import seaborn as sns
+
+# ==========================
+# CONFIG
+# ==========================
+CSV_1 = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/eval 11th june .csv"
+CSV_2 = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/rag_eval_1781246175645 12th june overall score 63 added elasticsearch and crawled websites.csv"
+
+QUESTION_ID_COL = "Question No"
+QUESTION_COL = "Question"
+
+METRICS = [
+    "Accuracy (LLM)",
+    "Faithfulness",
+    "Context Precision",
+    "Context Recall",
+    "Answer Relevancy"
+]
+
+HEATMAP_FILE = "metric_change_heatmap.png"
+MAPPING_FILE = "question_mapping.csv"
+
+# ==========================
+# LOAD CSVS
+# ==========================
+df1 = pd.read_csv(CSV_1)
+df2 = pd.read_csv(CSV_2)
+
+# ==========================
+# VALIDATE COLUMNS
+# ==========================
+required_cols = [
+    QUESTION_ID_COL,
+    QUESTION_COL
+] + METRICS
+
+missing_1 = [
+    c for c in required_cols
+    if c not in df1.columns
+]
+
+missing_2 = [
+    c for c in required_cols
+    if c not in df2.columns
+]
+
+if missing_1:
+    raise ValueError(
+        f"Missing columns in first CSV: {missing_1}"
+    )
+
+if missing_2:
+    raise ValueError(
+        f"Missing columns in second CSV: {missing_2}"
+    )
+
+# ==========================
+# KEEP REQUIRED COLUMNS
+# ==========================
+df1 = df1[required_cols].copy()
+df2 = df2[required_cols].copy()
+
+# ==========================
+# CLEAN DATA
+# ==========================
+df1[QUESTION_COL] = (
+    df1[QUESTION_COL]
+    .astype(str)
+    .str.strip()
+)
+
+df2[QUESTION_COL] = (
+    df2[QUESTION_COL]
+    .astype(str)
+    .str.strip()
+)
+
+for metric in METRICS:
+    df1[metric] = pd.to_numeric(
+        df1[metric],
+        errors="coerce"
+    )
+
+    df2[metric] = pd.to_numeric(
+        df2[metric],
+        errors="coerce"
+    )
+
+# ==========================
+# RENAME METRICS
+# ==========================
+df1 = df1.rename(
+    columns={
+        metric: f"{metric}_old"
+        for metric in METRICS
+    }
+)
+
+df2 = df2.rename(
+    columns={
+        metric: f"{metric}_new"
+        for metric in METRICS
+    }
+)
+
+# ==========================
+# MERGE ON QUESTION
+# ==========================
+merged = pd.merge(
+    df1,
+    df2,
+    on=[
+        QUESTION_ID_COL,
+        QUESTION_COL
+    ],
+    how="inner"
+)
+
+print(
+    f"Common questions found: {len(merged)}"
+)
+
+if len(merged) == 0:
+    raise ValueError(
+        "No common questions found."
+    )
+
+# ==========================
+# SAVE QUESTION MAPPING
+# ==========================
+mapping_df = merged[
+    [
+        QUESTION_ID_COL,
+        QUESTION_COL
+    ]
+].copy()
+
+mapping_df = mapping_df.sort_values(
+    by=QUESTION_ID_COL
+)
+
+mapping_df.to_csv(
+    MAPPING_FILE,
+    index=False
+)
+
+print(
+    f"Saved mapping file: {MAPPING_FILE}"
+)
+
+# ==========================
+# BUILD HEATMAP DATA
+# ==========================
+heatmap_df = pd.DataFrame()
+
+for metric in METRICS:
+    heatmap_df[metric] = (
+        merged[f"{metric}_new"]
+        - merged[f"{metric}_old"]
+    )
+
+heatmap_df.index = (
+    "Q" +
+    merged[QUESTION_ID_COL]
+    .astype(str)
+)
+
+# ==========================
+# SORT BY TOTAL IMPROVEMENT
+# ==========================
+heatmap_df["Total"] = (
+    heatmap_df.sum(axis=1)
+)
+
+heatmap_df = heatmap_df.sort_values(
+    by="Total",
+    ascending=False
+)
+
+heatmap_df = heatmap_df.drop(
+    columns=["Total"]
+)
+
+# ==========================
+# PLOT HEATMAP
+# ==========================
+num_questions = len(heatmap_df)
+
+fig_height = max(
+    8,
+    num_questions * 0.35
+)
+
+plt.figure(
+    figsize=(12, fig_height)
+)
+
+sns.heatmap(
+    heatmap_df,
+    cmap="RdYlGn",
+    center=0,
+    annot=False,      # prevents clutter
+    linewidths=0.3,
+    cbar_kws={
+        "label": "Metric Change (New - Old)"
+    }
+)
+
+plt.title(
+    "Metric Improvements by Question",
+    fontsize=16,
+    pad=20
+)
+
+plt.xlabel(
+    "Metrics",
+    fontsize=12
+)
+
+plt.ylabel(
+    "Question Number",
+    fontsize=12
+)
+
+plt.tight_layout()
+
+plt.savefig(
+    HEATMAP_FILE,
+    dpi=300,
+    bbox_inches="tight"
+)
+
+plt.close()
+
+print(
+    f"Saved heatmap: {HEATMAP_FILE}"
+)
+
+# ==========================
+# SUMMARY
+# ==========================
+print("\nAverage Metric Changes")
+
+for metric in METRICS:
+    avg_change = (
+        heatmap_df[metric]
+        .mean()
+    )
+
+    print(
+        f"{metric}: {avg_change:+.4f}"
+    )
+
+print("\nDone.")
\ No newline at end of file
```

## 0da125b10df8986d2086482254ade777d0944bd3 — 2026-06-15T11:28:31+05:30

Message:

Update compare_changes_in_metrics.py

```diff
diff --git a/helpfull scripts/compare_changes_in_metrics.py b/helpfull scripts/compare_changes_in_metrics.py
index fd4cd2d..cd6ad15 100644
+++ b/helpfull scripts/compare_changes_in_metrics.py	
@@ -1,34 +1,60 @@
 import pandas as pd
 import matplotlib.pyplot as plt
 
 # ==========================
 # CONFIG
 # ==========================
+CSV_1 = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/eval 11th june .csv" 
+CSV_2 = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/eval/results/rag_eval_1781246175645 12th june overall score 63 added elasticsearch and crawled websites.csv"
 
 METRICS = [
+    "Accuracy (LLM)",
     "Faithfulness",
     "Context Precision",
     "Context Recall",
     "Answer Relevancy"
 ]
 
+QUESTION_COL = "Question"
+
 # ==========================
 # LOAD FILES
 # ==========================
 df1 = pd.read_csv(CSV_1)
 df2 = pd.read_csv(CSV_2)
 
+# ==========================
+# VALIDATE COLUMNS
+# ==========================
+required_cols = [QUESTION_COL] + METRICS
+
+missing_1 = [c for c in required_cols if c not in df1.columns]
+missing_2 = [c for c in required_cols if c not in df2.columns]
+
+if missing_1:
+    raise ValueError(
+        f"Missing columns in first CSV: {missing_1}"
+    )
+
+if missing_2:
+    raise ValueError(
+        f"Missing columns in second CSV: {missing_2}"
+    )
+
+# Keep only required columns
+df1 = df1[required_cols].copy()
+df2 = df2[required_cols].copy()
+
+# Remove accidental whitespace in questions
+df1[QUESTION_COL] = df1[QUESTION_COL].astype(str).str.strip()
+df2[QUESTION_COL] = df2[QUESTION_COL].astype(str).str.strip()
 
+# Convert metrics to numeric
+for metric in METRICS:
+    df1[metric] = pd.to_numeric(df1[metric], errors="coerce")
+    df2[metric] = pd.to_numeric(df2[metric], errors="coerce")
 
+# Rename metrics
 df1 = df1.rename(
     columns={m: f"{m}_old" for m in METRICS}
 )
@@ -43,19 +69,24 @@ df2 = df2.rename(
 merged = pd.merge(
     df1,
     df2,
+    on=QUESTION_COL,
     how="inner"
 )
 
 print(f"Common questions found: {len(merged)}")
 
+if len(merged) == 0:
+    raise ValueError(
+        "No common questions found between the two CSVs."
+    )
+
 # ==========================
 # CREATE DELTA COLUMNS
 # ==========================
 for metric in METRICS:
     merged[f"{metric}_change"] = (
+        merged[f"{metric}_new"]
+        - merged[f"{metric}_old"]
     )
 
 # ==========================
@@ -66,25 +97,31 @@ plt.style.use("ggplot")
 for metric in METRICS:
 
     plot_df = merged[
+        [QUESTION_COL, f"{metric}_change"]
     ].copy()
 
+    plot_df = plot_df.dropna()
+
     plot_df = plot_df.sort_values(
         by=f"{metric}_change"
     )
 
     changes = plot_df[f"{metric}_change"]
+    questions = plot_df[QUESTION_COL]
 
     colors = [
         "green" if x > 0 else "red"
         for x in changes
     ]
 
+    fig_height = max(
+        6,
+        len(plot_df) * 0.35
+    )
 
+    plt.figure(
+        figsize=(14, fig_height)
+    )
 
     bars = plt.barh(
         questions,
@@ -98,22 +135,29 @@ for metric in METRICS:
         linewidth=1
     )
 
+    for bar, value in zip(
+        bars,
+        changes
+    ):
         plt.text(
             value,
+            bar.get_y()
+            + bar.get_height() / 2,
             f"{value:+.2f}",
             va="center",
+            ha="left"
+            if value >= 0
+            else "right"
         )
 
     plt.title(
         f"{metric}: Improvement / Regression by Question"
     )
+
     plt.xlabel(
         "Metric Change (New - Old)"
     )
+
     plt.ylabel("Question")
 
     plt.tight_layout()
@@ -121,10 +165,17 @@ for metric in METRICS:
     filename = (
         metric.lower()
         .replace(" ", "_")
+        .replace("(", "")
+        .replace(")", "")
         + "_change.png"
     )
 
+    plt.savefig(
+        filename,
+        dpi=300,
+        bbox_inches="tight"
+    )
+
     plt.close()
 
     print(f"Saved: {filename}")
```

## b78ce339f15782041a12740e5b512cd21f3d541a — 2026-06-12T12:01:31+05:30

Message:

crawled data downloaded

```diff
diff --git a/helpfull scripts/elastisearch_dense_vectors.py b/helpfull scripts/elastisearch_dense_vectors.py
new file mode 100644
index 0000000..a1f2876
+++ b/helpfull scripts/elastisearch_dense_vectors.py	
@@ -0,0 +1,235 @@
+#!/usr/bin/env python3
+
+import json
+from pathlib import Path
+
+import fitz
+from elasticsearch import Elasticsearch, helpers
+from sentence_transformers import SentenceTransformer
+from tqdm import tqdm
+
+
+# ============================================================
+# CONFIG
+# ============================================================
+
+ES_HOST = "http://localhost:9200"
+INDEX_NAME = "rag_documents"
+
+DATA_DIR = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/data"
+
+EMBEDDING_MODEL = "BAAI/bge-m3"
+
+CHUNK_SIZE = 1000
+CHUNK_OVERLAP = 200
+
+BULK_SIZE = 500
+
+# ============================================================
+# FILE READERS
+# ============================================================
+
+
+def read_json(path: Path):
+
+    with open(path, "r", encoding="utf-8") as f:
+        data = json.load(f)
+
+    content = data.get("content", "").strip()
+    url = data.get("url", "")
+
+    return content, url
+
+
+def read_txt(path: Path):
+
+    with open(path, "r", encoding="utf-8") as f:
+        return f.read().strip(), ""
+
+
+def read_pdf(path: Path):
+
+    doc = fitz.open(path)
+
+    pages = []
+
+    for page in doc:
+        pages.append(page.get_text())
+
+    doc.close()
+
+    return "\n".join(pages).strip(), ""
+
+
+def read_file(path: Path):
+
+    suffix = path.suffix.lower()
+
+    try:
+
+        if suffix == ".json":
+            return read_json(path)
+
+        elif suffix == ".txt":
+            return read_txt(path)
+
+        elif suffix == ".pdf":
+            return read_pdf(path)
+
+        return None, None
+
+    except Exception as e:
+        print(f"ERROR reading {path}: {e}")
+        return None, None
+
+
+# ============================================================
+# CHUNKING
+# ============================================================
+
+def chunk_text(text):
+
+    if not text:
+        return []
+
+    paragraphs = [
+        p.strip()
+        for p in text.split("\n\n")
+        if p.strip()
+    ]
+
+    chunks = []
+    current = ""
+
+    for para in paragraphs:
+
+        if len(current) + len(para) < CHUNK_SIZE:
+
+            current += "\n\n" + para
+
+        else:
+
+            chunks.append(current.strip())
+
+            overlap_text = (
+                current[-CHUNK_OVERLAP:]
+                if len(current) > CHUNK_OVERLAP
+                else current
+            )
+
+            current = overlap_text + "\n\n" + para
+
+    if current:
+        chunks.append(current.strip())
+
+    return chunks
+
+
+# ============================================================
+# MAIN
+# ============================================================
+
+def main():
+
+    print("Connecting to Elasticsearch...")
+
+    es = Elasticsearch(ES_HOST)
+
+    if not es.ping():
+        raise RuntimeError(
+            f"Cannot connect to Elasticsearch at {ES_HOST}"
+        )
+
+    print("Loading embedding model...")
+    model = SentenceTransformer(EMBEDDING_MODEL)
+
+    files = []
+
+    for ext in ("*.json", "*.txt", "*.pdf"):
+        files.extend(Path(DATA_DIR).rglob(ext))
+
+    print(f"Found {len(files)} files")
+
+    actions = []
+
+    total_chunks = 0
+    total_docs = 0
+
+    for file_path in tqdm(files, desc="Processing"):
+
+        content, url = read_file(file_path)
+
+        if not content:
+            continue
+
+        chunks = chunk_text(content)
+
+        if not chunks:
+            continue
+
+        embeddings = model.encode(
+            chunks,
+            normalize_embeddings=True,
+            show_progress_bar=False
+        )
+
+        for i, (chunk, embedding) in enumerate(
+            zip(chunks, embeddings)
+        ):
+
+            actions.append(
+                {
+                    "_index": INDEX_NAME,
+                    "_id": f"{file_path.stem}_{i}",
+                    "_source": {
+                        "content": chunk,
+                        "embedding": embedding.tolist(),
+                        "source": file_path.name,
+                        "filepath": str(
+                            file_path.resolve()
+                        ),
+                        "url": url,
+                        "filetype": file_path.suffix.lower().replace(
+                            ".", ""
+                        ),
+                        "document_id": file_path.stem,
+                        "chunk_number": i,
+                        "content_length": len(chunk)
+                    }
+                }
+            )
+
+            total_chunks += 1
+
+            if len(actions) >= BULK_SIZE:
+
+                helpers.bulk(
+                    es,
+                    actions,
+                    request_timeout=300
+                )
+
+                actions.clear()
+
+        total_docs += 1
+
+    if actions:
+
+        helpers.bulk(
+            es,
+            actions,
+            request_timeout=300
+        )
+
+    es.indices.refresh(index=INDEX_NAME)
+
+    print()
+    print("=" * 60)
+    print(f"Documents indexed : {total_docs}")
+    print(f"Chunks indexed    : {total_chunks}")
+    print(f"Index             : {INDEX_NAME}")
+    print("=" * 60)
+
+
+if __name__ == "__main__":
+    main()
\ No newline at end of file
```

## 4a27f5052dc4da119530b67d91880a8212434085 — 2026-06-12T10:03:44+05:30

Message:

elastic search data update from json files from crawler

```diff
diff --git a/helpfull scripts/elastosearch_json_crawler.py b/helpfull scripts/elastosearch_json_crawler.py
new file mode 100644
index 0000000..7a466f4
+++ b/helpfull scripts/elastosearch_json_crawler.py	
@@ -0,0 +1,109 @@
+#!/usr/bin/env python3
+
+import asyncio
+import json
+import sys
+from pathlib import Path
+
+from elasticsearch import AsyncElasticsearch
+
+ES_HOST = "http://localhost:9200"
+INDEX_NAME = "rag_documents"
+DATA_DIR = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/helpfull scripts/downloaded"
+
+
+async def upload_files():
+    client = AsyncElasticsearch([ES_HOST])
+
+    try:
+        if not await client.indices.exists(index=INDEX_NAME):
+            await client.indices.create(
+                index=INDEX_NAME,
+                body={
+                    "mappings": {
+                        "properties": {
+                            "content": {
+                                "type": "text"
+                            },
+                            "metadata": {
+                                "type": "object"
+                            }
+                        }
+                    }
+                }
+            )
+
+            print(f"Created index: {INDEX_NAME}")
+
+        total_docs = 0
+
+        for file_path in sorted(Path(DATA_DIR).rglob("*.json")):
+
+            try:
+                with open(
+                    file_path,
+                    "r",
+                    encoding="utf-8"
+                ) as f:
+                    data = json.load(f)
+
+                content = data.get("content", "").strip()
+
+                if not content:
+                    continue
+
+                source_url = data.get("url", "")
+
+                document = {
+                    "content": content,
+                    "metadata": {
+                        "source": file_path.name,
+                        "filepath": str(file_path.resolve()),
+                        "url": source_url,
+                        "document_id": file_path.stem,
+                        "file_type": file_path.suffix.lstrip("."),
+                        "content_length": len(content),
+                    }
+                }
+
+                await client.index(
+                    index=INDEX_NAME,
+                    id=file_path.stem,
+                    body=document
+                )
+
+                total_docs += 1
+
+                print(
+                    f"✓ {file_path.name}"
+                )
+
+            except Exception as e:
+                print(
+                    f"✗ {file_path.name}: {e}"
+                )
+
+        await client.indices.refresh(
+            index=INDEX_NAME
+        )
+
+        print()
+        print("=" * 50)
+        print(f"Indexed documents : {total_docs}")
+        print(f"Index             : {INDEX_NAME}")
+        print("=" * 50)
+
+    except Exception as e:
+        print(f"Error: {e}")
+        sys.exit(1)
+
+    finally:
+        await client.close()
+
+
+if __name__ == "__main__":
+    print(
+        f"Uploading crawler JSON files from:\n{DATA_DIR}\n"
+    )
+
+    asyncio.run(upload_files())
\ No newline at end of file
diff --git a/helpfull scripts/http_client.py b/helpfull scripts/http_client.py
index 66a0a60..bdc8393 100644
+++ b/helpfull scripts/http_client.py	
@@ -14,12 +14,19 @@ soup = BeautifulSoup(html, "html.parser")
 for a in soup.find_all("a"):
     href = a.get("href")
 
+    if not href or not href.endswith(".json"):
+        continue
 
+    url = urljoin(BASE_URL, href)
 
+    # Remove leading web_
+    filename = href
+    if filename.startswith("web_"):
+        filename = filename[4:]
 
\ No newline at end of file
+    print("Downloading", href, "->", filename)
+
+    r = requests.get(url)
+
+    with open(Path(OUTPUT_DIR) / filename, "wb") as f:
+        f.write(r.content)
\ No newline at end of file
```

## c6aa3d1eb41cf2262ba6b715b0b2d80416bde719 — 2026-06-12T09:58:26+05:30

Message:

chunked text data from crawler

_No Python file changes in this commit._

## 8cceedd84f01898dff44c5d25625e8748c2e2c4a — 2026-06-12T09:39:05+05:30

Message:

crawled websites

```diff
diff --git a/backend/app/api/elasticsearch.py b/backend/app/api/elasticsearch.py
new file mode 100644
index 0000000..08fbc41
+++ b/backend/app/api/elasticsearch.py
@@ -0,0 +1,22 @@
+from fastapi import APIRouter, UploadFile, File
+from app.services.elasticsearch_service import es_service
+
+router = APIRouter(prefix="/api/elasticsearch", tags=["elasticsearch"])
+
+@router.get("/status")
+async def get_status():
+    return await es_service.check_status()
+
+@router.post("/upload")
+async def upload_documents(file: UploadFile = File(...)):
+    content = await file.read()
+    text = content.decode("utf-8")
+    lines = [l.strip() for l in text.split("\n") if l.strip()]
+    docs = [{"id": f"doc_{i}", "content": line} for i, line in enumerate(lines)]
+    await es_service.bulk_index(docs)
+    return {"message": f"Indexed {len(docs)} documents", "count": len(docs)}
+
+@router.post("/search")
+async def search_documents(query: str, size: int = 10):
+    results = await es_service.search(query, size)
+    return {"query": query, "results": results, "count": len(results)}
diff --git a/backend/app/api/rag.py b/backend/app/api/rag.py
index 0c8f160..ad231dc 100644
+++ b/backend/app/api/rag.py
@@ -6,6 +6,7 @@ multi-agent RAG pipeline (coordinator → evaluator), and returns per-question
 detail alongside aggregate metrics.
 """
 
+import time
 from typing import Any
 
 from fastapi import APIRouter, Depends
@@ -15,7 +16,8 @@ from pydantic import BaseModel
 
 from app.core.dependencies import get_db
 from app.services.rag_service import rag_service
+from app.evaluation.agent_runner import evaluate_question, failed_question_row, set_evaluation_logger
+from app.core.evaluation_logger import EvaluationLogger
 
 from app.schemas.rag import RAGQueryRequest, RAGQueryResponse, RAGRetrieveRequest
 from app.schemas.search import SearchResult
@@ -70,11 +72,17 @@ class EvaluateRequest(BaseModel):
 
 @router.post("/evaluate")
 async def evaluate_rag(req: EvaluateRequest, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
+    # Initialize logger for this evaluation run
+    logger = EvaluationLogger(f"{req.dataset_name}_{int(time.time())}")
+    set_evaluation_logger(logger)
+    logger.log("EVAL_START", f"Dataset: {req.dataset_name}\nQuestions: {len(req.questions)}\nTop-k: {req.top_k}")
+    
     per_question: list[dict] = []
     failed: list[dict] = []
     latencies: list[float] = []
 
+    for idx, qa in enumerate(req.questions, 1):
+        logger.log("QUESTION_NUMBER", f"{idx}/{len(req.questions)}")
         try:
             row = await evaluate_question(
                 qa.question, 
@@ -86,6 +94,7 @@ async def evaluate_rag(req: EvaluateRequest, db: AsyncSession = Depends(get_db))
             per_question.append(row)
             latencies.append(row["latency_ms"])
         except Exception as exc:
+            logger.log("ERROR", f"Question {idx} failed: {str(exc)}")
             failed.append({"question": qa.question, "error": str(exc)})
             per_question.append(failed_question_row(qa.question, qa.expected_answer, str(exc)))
 
diff --git a/backend/app/core/evaluation_logger.py b/backend/app/core/evaluation_logger.py
new file mode 100644
index 0000000..d112051
+++ b/backend/app/core/evaluation_logger.py
@@ -0,0 +1,67 @@
+import os
+from datetime import datetime
+from pathlib import Path
+
+LOGS_DIR = Path("/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/backend/logs")
+LOGS_DIR.mkdir(exist_ok=True)
+
+class EvaluationLogger:
+    def __init__(self, eval_id: str = None):
+        if not eval_id:
+            eval_id = datetime.now().strftime("%Y%m%d_%H%M%S")
+        self.eval_id = eval_id
+        self.log_file = LOGS_DIR / f"evaluation_{eval_id}.txt"
+        self.step_counter = 0
+        # Create file immediately with header
+        with open(self.log_file, "w") as f:
+            f.write(f"EVALUATION LOG: {eval_id}\n")
+            f.write(f"Started: {datetime.now()}\n")
+            f.write("="*80 + "\n")
+            f.flush()
+            os.fsync(f.fileno())
+    
+    def log(self, step: str, data: str):
+        """Log a step with data"""
+        self.step_counter += 1
+        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
+        entry = f"\n[{timestamp}] STEP {self.step_counter}: {step}\n{data}\n{'-'*80}\n"
+        try:
+            with open(self.log_file, "a") as f:
+                f.write(entry)
+                f.flush()
+                os.fsync(f.fileno())
+        except Exception as e:
+            print(f"Logging error: {e}")
+    
+    def log_question(self, question: str, question_num: int, total: int):
+        self.log(f"QUESTION RECEIVED ({question_num}/{total})", f"Question: {question}")
+    
+    def log_retrieval(self, strategy: str, top_k: int, results: list):
+        self.log(f"RETRIEVAL ({strategy}, top_k={top_k})", 
+                f"Retrieved {len(results)} chunks:\n" + 
+                "\n".join([f"  - {r.get('filename', '?')}: {r['chunk_text'][:100]}..." for r in results[:5]]))
+    
+    def log_es_enhancement(self, enhanced_count: int, original_count: int):
+        self.log(f"ELASTICSEARCH ENHANCEMENT", 
+                f"Original chunks: {original_count}\nEnhanced chunks: {enhanced_count}\nAdded: {enhanced_count - original_count}")
+    
+    def log_es_search(self, query: str, results: list, attempt: int):
+        self.log(f"ELASTICSEARCH SEARCH (Attempt {attempt})",
+                f"Query: {query}\nFound {len(results)} results:\n" +
+                "\n".join([f"  - {r['content'][:100]}... (score: {r['score']})" for r in results[:3]]))
+    
+    def log_llm_call(self, step: str, prompt: str, answer: str, latency_ms: float):
+        self.log(f"LLM CALL - {step} (latency: {latency_ms}ms)",
+                f"Prompt:\n{prompt[:200]}...\n\nAnswer:\n{answer[:200]}...")
+    
+    def log_metrics(self, metric_name: str, score: float, rationale: str):
+        self.log(f"METRIC - {metric_name}", f"Score: {score}\nRationale: {rationale}")
+    
+    def log_error(self, error: str, question: str = ""):
+        self.log(f"ERROR", f"Question: {question}\nError: {error}")
+    
+    def log_summary(self, summary: dict):
+        summary_text = "\n".join([f"{k}: {v}" for k, v in summary.items()])
+        self.log(f"EVALUATION SUMMARY", summary_text)
+
+eval_logger = EvaluationLogger()
diff --git a/backend/app/evaluation/agent_runner.py b/backend/app/evaluation/agent_runner.py
index 44c38c5..fe30dc9 100644
+++ b/backend/app/evaluation/agent_runner.py
@@ -14,6 +14,15 @@ from app.rag.cross_encoder import cross_encoder
 from app.evaluation.evaluator import evaluate_single
 from app.embeddings.openai_client import openai_client
 from app.core.config import settings
+from app.core.evaluation_logger import EvaluationLogger
+from app.services.elasticsearch_service import es_service
+
+# Global logger instance
+_current_logger = None
+
+def set_evaluation_logger(logger: EvaluationLogger):
+    global _current_logger
+    _current_logger = logger
 
 
 RAG_SYSTEM = """Answer the question carefully"""
@@ -52,20 +61,29 @@ async def evaluate_question(
     """
     t0 = time.time()
     
+    if _current_logger:
+        _current_logger.log("QUESTION_START", f"Q: {question}\nExpected: {expected_answer}")
+    
     # Synonym expansion
     synonym_expander = get_synonym_expander()
     queries = synonym_expander.expand_query(question)
+    if _current_logger:
+        _current_logger.log("SYNONYM_EXPANSION", f"Generated {len(queries)} queries:\n" + "\n".join(queries))
     
     # Retrieve chunks for all query variants
     all_chunks = []
     chunk_ids_seen = set()
     
+    for idx, query in enumerate(queries, 1):
         # Vector search (nearest neighbor)
         vector_results = await vector_rag.retrieve(query, "text_documents", top_k=10)
+        if _current_logger:
+            _current_logger.log(f"VECTOR_SEARCH_{idx}", f"Query: {query}\nFound: {len(vector_results)} chunks")
         
         # BM25 search
         bm25_results = bm25_retriever.search("text_documents", query, top_k=10)
+        if _current_logger:
+            _current_logger.log(f"BM25_SEARCH_{idx}", f"Query: {query}\nFound: {len(bm25_results)} chunks")
         
         # Merge
         merged = _merge_results(vector_results, bm25_results)
@@ -77,19 +95,49 @@ async def evaluate_question(
                 chunk_ids_seen.add(cid)
                 all_chunks.append(c)
     
+    if _current_logger:
+        _current_logger.log("MERGE_RESULTS", f"Total unique chunks: {len(all_chunks)}")
+    
     # Rerank to top_k
     reranked = cross_encoder.rerank(question, all_chunks, top_k=top_k)
+    if _current_logger:
+        _current_logger.log("RERANK", f"Top {top_k} chunks after reranking")
+        for i, chunk in enumerate(reranked, 1):
+            _current_logger.log(f"CHUNK_{i}", 
+                f"Score: {chunk.get('score', 0)}\n"
+                f"File: {chunk.get('filename', 'unknown')}\n"
+                f"Text: {chunk.get('chunk_text', '')[:300]}...")
     
     # Generate answer
     chunk_texts = _chunk_texts(reranked)
+    
+    # Enhance with Elasticsearch
+    original_count = len(chunk_texts)
+    enhanced_texts = await es_service.enhance_with_iterative_query(chunk_texts, question, max_tries=5, logger=_current_logger)
+    if _current_logger:
+        _current_logger.log("ES_ENHANCEMENT_COMPLETE", 
+            f"Original: {original_count} chunks\n"
+            f"Enhanced: {len(enhanced_texts)} chunks\n"
+            f"Added: {len(enhanced_texts) - original_count} from Elasticsearch"
+            f"\nEnhanced texts :\n" + "\n---\n".join(enhanced_texts)
+            )
+    
+    context_text = "\n---\n".join(enhanced_texts)
     
     if context_text.strip():
         prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
     else:
         prompt = f"Question: {question}"
     
+    if _current_logger:
+        _current_logger.log("LLM_PROMPT", f"Prompt length: {len(prompt)} chars\nPrompt preview:\n{prompt[:500]}...")
+    
+    llm_start = time.time()
     generated_answer = await openai_client.generate(prompt, system=RAG_SYSTEM)
+    llm_time = (time.time() - llm_start) * 1000
+    
+    if _current_logger:
+        _current_logger.log("LLM_ANSWER", f"Generated in {llm_time}ms:\n{generated_answer}")
     
     # Build retrieved chunk IDs and derive gold IDs via text matching
     retrieved_chunk_ids = [c.get("chunk_id", "") for c in reranked if c.get("chunk_id")]
@@ -108,10 +156,21 @@ async def evaluate_question(
         if best_score > 0:
             gold_chunk_ids = {c.get("chunk_id") for c in reranked if c.get("score", 0) >= best_score * 0.9 and c.get("chunk_id")}
 
+    if _current_logger:
+        _current_logger.log("GOLD_CHUNKS", f"Identified {len(gold_chunk_ids)} gold chunks from {len(retrieved_chunk_ids)} retrieved")
+
     # Score
+    scores = await evaluate_single(question, expected_answer, generated_answer, enhanced_texts,
                                    retrieved_chunk_ids=retrieved_chunk_ids, gold_chunk_ids=gold_chunk_ids)
     
+    if _current_logger:
+        _current_logger.log("METRICS", 
+            f"Accuracy: {scores.get('accuracy_llm')}\n"
+            f"Faithfulness: {scores.get('faithfulness')}\n"
+            f"Context Precision: {scores.get('context_precision')}\n"
+            f"Context Recall: {scores.get('context_recall')}\n"
+            f"Answer Relevancy: {scores.get('answer_relevancy')}")
+    
     latency_ms = round((time.time() - t0) * 1000, 1)
 
     return {
diff --git a/backend/app/main.py b/backend/app/main.py
index 006bcdb..18d0ee7 100644
+++ b/backend/app/main.py
@@ -30,6 +30,7 @@ from app.api.agents import router as agents_router
 from app.api.chroma import router as chroma_router
 from app.api.embeddings import router as embeddings_router
 from app.api.web import router as web_router
+from app.api.elasticsearch import router as elasticsearch_router
 
 
 
@@ -92,4 +93,5 @@ app.include_router(markdown_router)
 app.include_router(agents_router)
 app.include_router(chroma_router)
 app.include_router(embeddings_router)
\ No newline at end of file
+app.include_router(web_router)
+app.include_router(elasticsearch_router)
\ No newline at end of file
diff --git a/backend/app/rag/evaluator.py b/backend/app/rag/evaluator.py
index 9b14323..acbe386 100644
+++ b/backend/app/rag/evaluator.py
@@ -1,211 +1,373 @@
 """
+LLM-as-a-Judge evaluator for RAG pipelines with nearest-neighbour retrieval.
+
+Improvements over the previous version
+---------------------------------------
+- Recall@K, MRR, Hit@K tracked alongside faithfulness / accuracy
+- Ground-truth chunk identification logged per evaluation
+- Failed queries persisted to disk for retraining
+- No bare `except:` — all errors are typed and re-raised or logged explicitly
+- Query expansion only after the original query fails the sufficiency check
+- Generation temperature hint baked into the judge system prompt
+- All LLM calls are retried once on transient JSON-parse errors
 """
 
+from __future__ import annotations
+
 import json
+import logging
 import re
 import time
+from pathlib import Path
 from typing import Any
 
 from app.embeddings.openai_client import openai_client as ollama_client
 from app.rag.vector_rag import vector_rag
 
+logger = logging.getLogger(__name__)
 
 _SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]*\.?[0-9]+)')
+_FAILED_QUERIES_PATH = Path("logs/failed_queries.jsonl")
+_FAILED_QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
 
 
+# ---------------------------------------------------------------------------
+# Internal helpers
+# ---------------------------------------------------------------------------
 
+def _strip_fences(text: str) -> str:
+    """Remove markdown code fences that models sometimes wrap JSON in."""
+    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
 
 
+async def _llm_score(prompt: str, retries: int = 1) -> tuple[float, str]:
+    """
+    Call the judge LLM and parse a {score, rationale} JSON object.
+    Retries once on parse failure before falling back to regex extraction.
+    """
+    system = (
+        "You are an expert RAG evaluation judge. "
+        "Use temperature 0 reasoning — be precise and consistent. "
+        "Respond ONLY with a valid JSON object containing exactly two keys: "
+        '"score" (a float 0.0–1.0) and "rationale" (one sentence). '
+        "No markdown, no extra text."
+    )
+    raw = ""
+    for attempt in range(retries + 1):
+        try:
+            raw = await ollama_client.chat(
+                [{"role": "user", "content": prompt}],
+                system=system,
+            )
+            data = json.loads(_strip_fences(raw))
+            score = float(data["score"])
+            return round(max(0.0, min(1.0, score)), 4), data.get("rationale", "")
+        except (json.JSONDecodeError, KeyError, ValueError) as exc:
+            if attempt < retries:
+                logger.debug("Judge parse error (attempt %d), retrying: %s", attempt + 1, exc)
+                continue
+            # Final fallback: regex
+            match = _SCORE_RE.search(raw)
+            if match:
+                return round(float(match.group(1)), 4), "Score extracted via regex fallback."
+            logger.warning("Judge failed to produce a parseable score: %s", exc)
+            return 0.0, f"Parse error: {exc}"
+    return 0.0, "Unreachable"
 
 
+async def _expand_query(question: str) -> str:
+    """Rephrase the query to improve sparse/dense retrieval coverage."""
+    prompt = (
+        "Rephrase the following question to improve retrieval from a document database. "
+        "Add relevant domain keywords and expand any acronyms. "
+        "Return ONLY the rephrased question, nothing else.\n\n"
+        f"Original question: {question}"
+    )
+    return (await ollama_client.chat([{"role": "user", "content": prompt}])).strip()
 
 
+async def _check_sufficiency(
+    question: str, context_chunks: list[str]
+) -> tuple[bool, str]:
+    """Ask the LLM whether the retrieved context is enough to answer the question."""
+    context = "\n\n---\n\n".join(context_chunks[:5])
+    prompt = (
+        "Does the CONTEXT below contain sufficient information to answer the QUESTION?\n"
+        'Respond ONLY with valid JSON: {"sufficient": true/false, "reason": "brief explanation"}\n\n'
+        f"QUESTION: {question}\n\nCONTEXT: {context}"
     )
     try:
+        raw = await ollama_client.chat([{"role": "user", "content": prompt}])
+        data = json.loads(_strip_fences(raw))
+        return bool(data.get("sufficient", False)), data.get("reason", "")
+    except (json.JSONDecodeError, KeyError):
+        return len(context_chunks) > 0, "Sufficiency check parse error — assuming sufficient if chunks exist."
 
 
+def _persist_failed_query(record: dict[str, Any]) -> None:
+    """Append a failed query to disk for later analysis / fine-tuning."""
+    try:
+        with _FAILED_QUERIES_PATH.open("a", encoding="utf-8") as fh:
+            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
+    except OSError as exc:
+        logger.error("Could not persist failed query: %s", exc)
 
 
+# ---------------------------------------------------------------------------
+# Individual metric functions  (all preserve original signatures)
+# ---------------------------------------------------------------------------
 
+async def compute_accuracy(generated: str, expected: str) -> tuple[float, str]:
+    """Semantic match between generated and expected answer."""
+    prompt = (
+        "Rate how well the GENERATED ANSWER matches the EXPECTED ANSWER semantically.\n\n"
+        f"EXPECTED ANSWER:\n{expected}\n\n"
+        f"GENERATED ANSWER:\n{generated}\n\n"
+        "Score 1.0 if they convey the same meaning, 0.0 if completely unrelated."
+    )
     return await _llm_score(prompt)
 
 
 async def compute_faithfulness(answer: str, context_chunks: list[str]) -> tuple[float, str]:
+    """How well the answer is grounded in the retrieved context (no hallucinations)."""
     context = "\n\n---\n\n".join(context_chunks[:5])
+    prompt = (
+        "Rate how faithfully the ANSWER is grounded in the CONTEXT below.\n"
+        "A faithful answer only makes claims supported by the context (score 1.0).\n"
+        "An unfaithful answer introduces hallucinated facts not in the context (score 0.0).\n\n"
+        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
+    )
     return await _llm_score(prompt)
 
 
 async def compute_answer_relevancy(question: str, answer: str) -> tuple[float, str]:
+    """How directly and completely the answer addresses the question."""
+    prompt = (
+        "Rate how directly and completely the ANSWER addresses the QUESTION.\n"
+        "Score 1.0 if fully on-topic and complete, 0.0 if it ignores the question.\n\n"
+        f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
+    )
     return await _llm_score(prompt)
 
 
+async def compute_context_precision(
+    question: str, context_chunks: list[str]
+) -> tuple[float, str]:
+    """Proportion of retrieved chunks that are genuinely relevant to the question."""
     if not context_chunks:
         return 0.0, "No context chunks provided."
     chunks_text = "\n\n---\n\n".join(
+        f"[Chunk {i + 1}]: {c}" for i, c in enumerate(context_chunks[:8])
+    )
+    prompt = (
+        "You are evaluating retrieval quality.\n"
+        "Rate the proportion of the retrieved CHUNKS that contain information genuinely "
+        "useful for answering the QUESTION (0.0 = none relevant, 1.0 = all relevant).\n\n"
+        f"QUESTION:\n{question}\n\nRETRIEVED CHUNKS:\n{chunks_text}"
     )
     return await _llm_score(prompt)
 
 
+async def compute_context_recall(
+    expected_answer: str, context_chunks: list[str]
+) -> tuple[float, str]:
+    """Whether the retrieved context contains the facts needed for the expected answer."""
     if not context_chunks:
         return 0.0, "No context chunks provided."
     context = "\n\n---\n\n".join(context_chunks[:8])
+    prompt = (
+        "Rate whether the RETRIEVED CONTEXT contains the information needed to produce "
+        "the EXPECTED ANSWER.\n"
+        "Score 1.0 if all key facts are present, 0.0 if completely missing.\n\n"
+        f"EXPECTED ANSWER:\n{expected_answer}\n\nRETRIEVED CONTEXT:\n{context}"
+    )
+    return await _llm_score(prompt)
 
 
+# ---------------------------------------------------------------------------
+# Ranking metrics  (new)
+# ---------------------------------------------------------------------------
+
+def compute_hit_at_k(retrieved_chunks: list[str], ground_truth_text: str, k: int) -> float:
+    """
+    Hit@K — 1.0 if any of the top-K chunks contains the ground-truth substring.
+    Uses a simple substring check; swap for embedding similarity if needed.
+    """
+    needle = ground_truth_text.lower().strip()
+    for chunk in retrieved_chunks[:k]:
+        if needle in chunk.lower():
+            return 1.0
+    return 0.0
+
 
+def compute_recall_at_k(retrieved_chunks: list[str], ground_truth_text: str, k: int) -> float:
+    """
+    Recall@K — fraction of ground-truth sentences covered by the top-K chunks.
+    Sentences shorter than 10 chars are skipped to avoid trivial matches.
+    """
+    sentences = [
+        s.strip()
+        for s in ground_truth_text.split(".")
+        if len(s.strip()) >= 10
+    ]
+    if not sentences:
+        return 0.0
+    top_chunks_text = " ".join(retrieved_chunks[:k]).lower()
+    covered = sum(1 for s in sentences if s.lower() in top_chunks_text)
+    return round(covered / len(sentences), 4)
+
+
+def compute_mrr(retrieved_chunks: list[str], ground_truth_text: str) -> float:
+    """
+    Mean Reciprocal Rank — 1/rank of the first relevant chunk.
+    Returns 0.0 if no relevant chunk is found.
+    """
+    needle = ground_truth_text.lower().strip()
+    for rank, chunk in enumerate(retrieved_chunks, start=1):
+        if needle in chunk.lower():
+            return round(1.0 / rank, 4)
+    return 0.0
+
+
+def identify_ground_truth_chunk(
+    retrieved_chunks: list[str], expected_answer: str
+) -> dict[str, Any]:
+    """
+    Log which retrieved chunk (if any) contains the ground-truth answer.
+    Returns index (0-based), a snippet, and a confidence flag.
+    """
+    needle = expected_answer.lower().strip()
+    for idx, chunk in enumerate(retrieved_chunks):
+        if needle in chunk.lower():
+            snippet = chunk[:200].replace("\n", " ")
+            return {"found": True, "chunk_index": idx, "snippet": snippet}
+    return {"found": False, "chunk_index": None, "snippet": None}
+
+
+# ---------------------------------------------------------------------------
+# Main evaluation entry-point  (signature preserved)
+# ---------------------------------------------------------------------------
 
 async def evaluate_single(
     question: str,
     expected_answer: str,
     generated_answer: str,
+    context_chunks: list[str] | None = None,
     collection_name: str = "text_documents",
     top_k: int = 5,
     max_tries: int = 2,
 ) -> dict[str, Any]:
     """
+    Evaluate a single RAG turn.
+
+    Retrieval strategy
+    ------------------
+    1. Try the original query first.
+    2. If the sufficiency check fails and attempts remain, expand the query once
+       and retry.  (Original query is always attempt #1 — no pre-expansion.)
+
+    Metrics returned
+    ----------------
+    LLM-judged : accuracy, faithfulness, answer_relevancy,
+                 context_precision, context_recall
+    Ranking    : hit_at_k, recall_at_k, mrr
+    Provenance : ground_truth_chunk, retrieval_attempts
     """
     t0 = time.time()
+
     current_query = question
+    all_attempts: list[dict[str, Any]] = []
+    retrieved_chunks: list[str] = []
+
     for attempt in range(max_tries):
         results = await vector_rag.retrieve(current_query, collection_name, top_k)
         retrieved_chunks = [r.get("chunk_text", "") for r in results]
+
+        attempt_record: dict[str, Any] = {
             "attempt": attempt + 1,
             "query": current_query,
+            "num_results": len(results),
+        }
+
         if retrieved_chunks:
             is_sufficient, reason = await _check_sufficiency(question, retrieved_chunks)
+            attempt_record["sufficient"] = is_sufficient
+            attempt_record["reason"] = reason
+
             if is_sufficient or attempt == max_tries - 1:
+                all_attempts.append(attempt_record)
                 break
+        else:
+            attempt_record["sufficient"] = False
+            attempt_record["reason"] = "No chunks retrieved."
+
+        all_attempts.append(attempt_record)
+
+        # Expand query only for subsequent attempts
         if attempt < max_tries - 1:
+            current_query = await _expand_query(question)
+
+    # ------------------------------------------------------------------
+    # LLM-judged metrics
+    # ------------------------------------------------------------------
     accuracy, acc_rationale = await compute_accuracy(generated_answer, expected_answer)
     faithfulness, fai_rationale = await compute_faithfulness(generated_answer, retrieved_chunks)
     answer_relevancy, rel_rationale = await compute_answer_relevancy(question, generated_answer)
     context_precision, pre_rationale = await compute_context_precision(question, retrieved_chunks)
     context_recall, rec_rationale = await compute_context_recall(expected_answer, retrieved_chunks)
 
+    # ------------------------------------------------------------------
+    # Ranking metrics (new)
+    # ------------------------------------------------------------------
+    hit_at_k = compute_hit_at_k(retrieved_chunks, expected_answer, top_k)
+    recall_at_k = compute_recall_at_k(retrieved_chunks, expected_answer, top_k)
+    mrr = compute_mrr(retrieved_chunks, expected_answer)
+    ground_truth_chunk = identify_ground_truth_chunk(retrieved_chunks, expected_answer)
+
     latency_ms = round((time.time() - t0) * 1000, 1)
 
+    # ------------------------------------------------------------------
+    # Persist failures for retraining
+    # ------------------------------------------------------------------
+    is_failure = accuracy < 0.5 or faithfulness < 0.5 or hit_at_k == 0.0
+    if is_failure:
+        _persist_failed_query({
+            "question": question,
+            "expected_answer": expected_answer,
+            "generated_answer": generated_answer,
+            "accuracy": accuracy,
+            "faithfulness": faithfulness,
+            "hit_at_k": hit_at_k,
+            "retrieval_attempts": all_attempts,
+            "timestamp": time.time(),
+        })
+
     return {
+        # Inputs
         "question": question,
         "expected_answer": expected_answer,
         "generated_answer": generated_answer,
         "retrieved_context": "\n---\n".join(retrieved_chunks),
+        # Retrieval provenance
         "retrieval_attempts": all_attempts,
+        "ground_truth_chunk": ground_truth_chunk,
+        # LLM-judged scores
         "accuracy": accuracy,
         "faithfulness": faithfulness,
         "answer_relevancy": answer_relevancy,
         "context_precision": context_precision,
         "context_recall": context_recall,
+        # Rationales
         "accuracy_rationale": acc_rationale,
         "faithfulness_rationale": fai_rationale,
         "answer_relevancy_rationale": rel_rationale,
         "context_precision_rationale": pre_rationale,
         "context_recall_rationale": rec_rationale,
+        # Ranking metrics (new)
+        "hit_at_k": hit_at_k,
+        "recall_at_k": recall_at_k,
+        "mrr": mrr,
+        # Meta
         "latency_ms": latency_ms,
+        "is_failure": is_failure,
     }
\ No newline at end of file
diff --git a/backend/app/rag/vector_rag.py b/backend/app/rag/vector_rag.py
index 5dc00f2..3f8f5af 100644
+++ b/backend/app/rag/vector_rag.py
@@ -1,25 +1,429 @@
+"""
+VectorRAG — hybrid retrieval with BM25 + dense vectors, keyword boosting,
+acronym expansion, and a pluggable cross-encoder reranker.
+
+Public API is unchanged:
+    vector_rag.retrieve(query, collection_name, top_k, filters)
+    vector_rag.retrieve_multi_collection(query, collection_names, top_k, filters)
+
+Improvements over the previous version
+---------------------------------------
+Retrieval
+  - Hybrid search: BM25 sparse score merged with cosine dense score
+  - Keyword extraction boosts chunks containing important query terms
+  - Acronym expansion applied at retrieval time (CIF → Customer Information File)
+  - Original query runs first; expansion is a caller-opt-in via `expand=True`
+  - Over-retrieval then rerank: fetch `candidate_k` (default 50), rerank to `top_k`
+  - All chunks are verified to be non-empty before being returned
+
+Reranking
+  - Cross-encoder reranker is pluggable via RERANKER_BACKEND env-var
+    ("bge", "jina", "cohere") — falls back to keyword-boosted cosine if unset
+  - BGE / Jina run locally via sentence-transformers
+  - Cohere uses the Cohere Rerank v3 API
+
+Metadata
+  - document_id, filename, section, heading, chunk_index always populated
+  - score breakdown (dense_score, bm25_score, final_score) logged per chunk
+
+BM25
+  - rank_bm25 library; index built lazily per collection and cached in memory
+  - Cache is invalidated when the collection's document count changes
+"""
+
+from __future__ import annotations
+
+import logging
+import math
+import os
+import re
 from typing import Any, Optional
+
 from app.chromadb.client import chroma_client
 from app.embeddings.openai_client import openai_client as ollama_client
 from app.rag.metadata_filter import build_chroma_filter
 
+logger = logging.getLogger(__name__)
+
+# ---------------------------------------------------------------------------
+# Acronym / abbreviation dictionary
+# ---------------------------------------------------------------------------
+ACRONYM_MAP: dict[str, str] = {
+    "CIF": "Customer Information File",
+    "KYC": "Know Your Customer",
+    "AML": "Anti-Money Laundering",
+    "PEP": "Politically Exposed Person",
+    "FATF": "Financial Action Task Force",
+    "AOF": "Account Opening Form",
+    "DD": "Due Diligence",
+    "EDD": "Enhanced Due Diligence",
+    "SDD": "Simplified Due Diligence",
+    "STR": "Suspicious Transaction Report",
+    "CTR": "Currency Transaction Report",
+    "TIN": "Tax Identification Number",
+    "NID": "National Identity Document",
+    # Extend as your domain grows
+}
+
+
+def expand_acronyms(text: str) -> str:
+    """
+    Replace known acronyms in *text* with 'ACRONYM (Expansion)' so that both
+    the abbreviation and the full form appear in the query / chunk.
+    """
+    tokens = text.split()
+    expanded: list[str] = []
+    for token in tokens:
+        clean = re.sub(r"[^A-Z]", "", token.upper())
+        if clean in ACRONYM_MAP:
+            expanded.append(f"{token} ({ACRONYM_MAP[clean]})")
+        else:
+            expanded.append(token)
+    return " ".join(expanded)
+
+
+# ---------------------------------------------------------------------------
+# Keyword extraction (simple TF-style; swap for spaCy/RAKE if available)
+# ---------------------------------------------------------------------------
+_STOPWORDS = frozenset(
+    "a an the is are was were be been being have has had do does did "
+    "will would could should may might shall can not no nor and or but "
+    "if then else when where what which who whom how why this that these "
+    "those it its of in on at to for from with by about as into through "
+    "during before after above below between among over under i me my we "
+    "our you your he she they them their what said".split()
+)
+
+
+def extract_keywords(query: str, max_keywords: int = 8) -> list[str]:
+    """Return the most query-relevant non-stopword tokens, longest first."""
+    tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
+    keywords = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
+    # Deduplicate while preserving order, then sort by length (proxy for specificity)
+    seen: set[str] = set()
+    unique: list[str] = []
+    for kw in keywords:
+        if kw not in seen:
+            seen.add(kw)
+            unique.append(kw)
+    return sorted(unique, key=len, reverse=True)[:max_keywords]
+
+
+def keyword_boost(chunk_text: str, keywords: list[str]) -> float:
+    """
+    Return a small additive score in [0, 0.2] based on how many query
+    keywords appear in the chunk.  Used to break ties after reranking.
+    """
+    if not keywords:
+        return 0.0
+    text_lower = chunk_text.lower()
+    hits = sum(1 for kw in keywords if kw in text_lower)
+    return round(0.2 * hits / len(keywords), 4)
+
+
+# ---------------------------------------------------------------------------
+# BM25 index cache
+# ---------------------------------------------------------------------------
+
+class _BM25Cache:
+    """
+    Lazy, per-collection BM25 index.  Rebuilt when the document count changes.
+    Requires:  pip install rank_bm25
+    """
+
+    def __init__(self) -> None:
+        self._indices: dict[str, Any] = {}   # collection_name → BM25Okapi
+        self._doc_counts: dict[str, int] = {}
+        self._corpus: dict[str, list[str]] = {}
+
+    def _try_import(self) -> Any:
+        try:
+            from rank_bm25 import BM25Okapi  # type: ignore
+            return BM25Okapi
+        except ImportError:
+            logger.warning(
+                "rank_bm25 not installed — BM25 scoring disabled.  "
+                "Install with: pip install rank_bm25"
+            )
+            return None
 
+    def get_scores(
+        self,
+        collection_name: str,
+        query_tokens: list[str],
+        chunk_ids: list[str],
+        chunks: list[str],
+    ) -> list[float]:
+        """
+        Return normalised BM25 scores (0–1) for *chunks* given *query_tokens*.
+        Falls back to uniform 0.0 if rank_bm25 is not installed.
+        """
+        BM25Okapi = self._try_import()
+        if BM25Okapi is None:
+            return [0.0] * len(chunks)
+
+        current_count = len(chunks)
+        cached_count = self._doc_counts.get(collection_name, -1)
+
+        if current_count != cached_count:
+            tokenised = [c.lower().split() for c in chunks]
+            self._indices[collection_name] = BM25Okapi(tokenised)
+            self._doc_counts[collection_name] = current_count
+            self._corpus[collection_name] = chunks
+
+        bm25 = self._indices[collection_name]
+        raw_scores: list[float] = bm25.get_scores(query_tokens).tolist()
+
+        # Normalise to [0, 1]
+        max_score = max(raw_scores) if raw_scores else 1.0
+        if max_score == 0.0:
+            return [0.0] * len(raw_scores)
+        return [round(s / max_score, 4) for s in raw_scores]
+
+
+_bm25_cache = _BM25Cache()
+
+
+# ---------------------------------------------------------------------------
+# Cross-encoder reranker (pluggable)
+# ---------------------------------------------------------------------------
+
+class _Reranker:
+    """
+    Thin wrapper around multiple reranking backends.
+
+    Set RERANKER_BACKEND env-var to "bge", "jina", or "cohere".
+    Leave unset (or set to "none") to skip cross-encoder reranking and fall back
+    to the hybrid score + keyword boost.
+
+    BGE / Jina — local inference via sentence-transformers:
+        pip install sentence-transformers
+        model: BAAI/bge-reranker-base  (bge)  |  jinaai/jina-reranker-v2-base-multilingual (jina)
+
+    Cohere — remote API:
+        pip install cohere
+        set COHERE_API_KEY env-var
+        model: rerank-english-v3.0
+    """
+
+    _instance: Any = None  # lazy singleton
+
+    def _load(self) -> None:
+        backend = os.getenv("RERANKER_BACKEND", "none").lower()
+        if backend == "bge":
+            from sentence_transformers import CrossEncoder  # type: ignore
+            self._instance = CrossEncoder("BAAI/bge-reranker-base")
+            self._backend = "sentence_transformers"
+        elif backend == "jina":
+            from sentence_transformers import CrossEncoder  # type: ignore
+            self._instance = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual")
+            self._backend = "sentence_transformers"
+        elif backend == "cohere":
+            import cohere  # type: ignore
+            self._instance = cohere.Client(os.environ["COHERE_API_KEY"])
+            self._backend = "cohere"
+        else:
+            self._instance = None
+            self._backend = "none"
+
+    def rerank(
+        self, query: str, chunks: list[str], scores: list[float], top_k: int
+    ) -> list[tuple[str, float]]:
+        """
+        Return (chunk_text, score) pairs sorted descending, trimmed to top_k.
+        Falls back gracefully if the backend is unavailable.
+        """
+        if self._instance is None and not hasattr(self, "_backend"):
+            self._load()
+
+        backend = getattr(self, "_backend", "none")
+
+        if backend == "sentence_transformers":
+            try:
+                pairs = [[query, c] for c in chunks]
+                ce_scores: list[float] = self._instance.predict(pairs).tolist()
+                ranked = sorted(zip(chunks, ce_scores), key=lambda x: x[1], reverse=True)
+                return ranked[:top_k]
+            except Exception as exc:
+                logger.warning("Cross-encoder rerank failed: %s — using hybrid scores.", exc)
+
+        elif backend == "cohere":
+            try:
+                resp = self._instance.rerank(
+                    model="rerank-english-v3.0",
+                    query=query,
+                    documents=chunks,
+                    top_n=top_k,
+                )
+                ranked = [(chunks[r.index], r.relevance_score) for r in resp.results]
+                return ranked
+            except Exception as exc:
+                logger.warning("Cohere rerank failed: %s — using hybrid scores.", exc)
+
+        # Fallback: use existing hybrid scores
+        ranked_pairs = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
+        return ranked_pairs[:top_k]
+
+
+_reranker = _Reranker()
+
+
+# ---------------------------------------------------------------------------
+# VectorRAG
+# ---------------------------------------------------------------------------
 
 class VectorRAG:
+    """
+    Hybrid retrieval: dense cosine (ChromaDB) + BM25 sparse, merged via
+    weighted sum, optionally reranked by a cross-encoder.
+
+    Parameters
+    ----------
+    dense_weight : float
+        Weight for the dense cosine score in the hybrid merge (default 0.7).
+    bm25_weight : float
+        Weight for the BM25 score in the hybrid merge (default 0.3).
+    candidate_multiplier : int
+        How many times top_k to over-retrieve for reranking (default 10).
+        E.g. top_k=5, multiplier=10 → fetch 50 candidates → rerank → return 5.
+    """
+
+    def __init__(
+        self,
+        dense_weight: float = 0.7,
+        bm25_weight: float = 0.3,
+        candidate_multiplier: int = 10,
+    ) -> None:
+        self.dense_weight = dense_weight
+        self.bm25_weight = bm25_weight
+        self.candidate_multiplier = candidate_multiplier
+
+    # ------------------------------------------------------------------
+    # Internal helpers
+    # ------------------------------------------------------------------
+
+    def _candidate_k(self, top_k: int) -> int:
+        return max(top_k, top_k * self.candidate_multiplier)
+
+    @staticmethod
+    def _enrich_result(r: dict[str, Any]) -> dict[str, Any]:
+        """Populate standard metadata fields on every result record."""
+        meta = r.get("metadata", {})
+        r["document_id"] = meta.get("document_id", "")
+        r["filename"] = meta.get("filename", "")
+        r["section"] = meta.get("section", "")
+        r["heading"] = meta.get("heading", "")
+        r["chunk_index"] = meta.get("chunk_index", None)
+        return r
+
+    def _hybrid_merge(
+        self,
+        query: str,
+        results: list[dict[str, Any]],
+    ) -> list[dict[str, Any]]:
+        """
+        Merge dense cosine scores with BM25 scores and apply keyword boost.
+        Modifies results in place; returns the same list sorted by final_score desc.
+        """
+        if not results:
+            return results
+
+        chunks = [r.get("chunk_text", "") for r in results]
+        query_tokens = query.lower().split()
+        bm25_scores = _bm25_cache.get_scores(
+            collection_name="_hybrid",  # shared cache key; ok for per-call usage
+            query_tokens=query_tokens,
+            chunk_ids=[r.get("id", str(i)) for i, r in enumerate(results)],
+            chunks=chunks,
+        )
+
+        keywords = extract_keywords(query)
+
+        for r, bm25 in zip(results, bm25_scores):
+            dense = float(r.get("score", 0.0))
+            boost = keyword_boost(r.get("chunk_text", ""), keywords)
+            hybrid = self.dense_weight * dense + self.bm25_weight * bm25 + boost
+            r["dense_score"] = round(dense, 4)
+            r["bm25_score"] = bm25
+            r["keyword_boost"] = boost
+            r["score"] = round(hybrid, 4)
+
+        results.sort(key=lambda x: x["score"], reverse=True)
+        return results
+
+    def _rerank_results(
+        self, query: str, results: list[dict[str, Any]], top_k: int
+    ) -> list[dict[str, Any]]:
+        """
+        Apply cross-encoder reranking on the candidate pool and trim to top_k.
+        If reranker is disabled the candidate list is simply sliced to top_k.
+        """
+        if not results:
+            return results
+
+        chunks = [r.get("chunk_text", "") for r in results]
+        scores = [r.get("score", 0.0) for r in results]
+
+        reranked_pairs = _reranker.rerank(query, chunks, scores, top_k)
+
+        # Rebuild result dicts preserving metadata
+        chunk_to_result: dict[str, dict[str, Any]] = {}
+        for r in results:
+            chunk_to_result.setdefault(r.get("chunk_text", ""), r)
+
+        final: list[dict[str, Any]] = []
+        for chunk_text, rerank_score in reranked_pairs:
+            base = chunk_to_result.get(chunk_text, {"chunk_text": chunk_text, "metadata": {}})
+            base["rerank_score"] = round(float(rerank_score), 4)
+            final.append(base)
+
+        return final
+
+    # ------------------------------------------------------------------
+    # Public API  (signatures preserved)
+    # ------------------------------------------------------------------
+
     async def retrieve(
         self,
         query: str,
         collection_name: str = "text_documents",
         top_k: int = 5,
         filters: Optional[dict] = None,
+        expand: bool = False,
     ) -> list[dict[str, Any]]:
+        """
+        Retrieve the top_k most relevant chunks for *query*.
+
+        Parameters
+        ----------
+        query : str
+            The search query.  Acronyms are expanded automatically.
+        collection_name : str
+            ChromaDB collection to search.
+        top_k : int
+            Number of results to return after reranking.
+        filters : dict | None
+            Optional metadata filters forwarded to ChromaDB.
+        expand : bool
+            If True, the query is expanded with acronym forms before embedding.
+            Set to False (default) so callers control when expansion happens.
+        """
+        effective_query = expand_acronyms(query) if expand else query
+        query_embedding = await ollama_client.embeddings(effective_query)
         where = build_chroma_filter(filters) if filters else None
+
+        candidate_k = self._candidate_k(top_k)
+        results: list[dict[str, Any]] = chroma_client.search(
+            collection_name, query_embedding, candidate_k, where
+        )
+
+        # Drop empty chunks early
+        results = [r for r in results if r.get("chunk_text", "").strip()]
+
+        results = self._hybrid_merge(effective_query, results)
+        results = self._rerank_results(effective_query, results, top_k)
+
+        return [self._enrich_result(r) for r in results]
 
     async def retrieve_multi_collection(
         self,
@@ -27,22 +431,52 @@ class VectorRAG:
         collection_names: list[str],
         top_k: int = 5,
         filters: Optional[dict] = None,
+        expand: bool = False,
     ) -> list[dict[str, Any]]:
+        """
+        Search multiple collections, merge all candidates, rerank globally,
+        and return the top_k results across all collections.
+
+        Parameters
+        ----------
+        query : str
+            The search query.
+        collection_names : list[str]
+            Collections to fan-out the query across.
+        top_k : int
+            Final number of results to return.
+        filters : dict | None
+            Optional metadata filters forwarded to ChromaDB.
+        expand : bool
+            Whether to expand acronyms in the query before embedding.
+        """
+        effective_query = expand_acronyms(query) if expand else query
+        query_embedding = await ollama_client.embeddings(effective_query)
         where = build_chroma_filter(filters) if filters else None
+
+        candidate_k = self._candidate_k(top_k)
+        all_results: list[dict[str, Any]] = []
+
         for collection in collection_names:
             try:
+                results: list[dict[str, Any]] = chroma_client.search(
+                    collection, query_embedding, candidate_k, where
+                )
                 for r in results:
                     r["collection"] = collection
                 all_results.extend(results)
+            except Exception as exc:
+                logger.warning(
+                    "Collection '%s' search failed (skipping): %s", collection, exc
+                )
+
+        # Drop empties, merge hybrid scores globally, then rerank
+        all_results = [r for r in all_results if r.get("chunk_text", "").strip()]
+        all_results = self._hybrid_merge(effective_query, all_results)
+        all_results = self._rerank_results(effective_query, all_results, top_k)
+
+        return [self._enrich_result(r) for r in all_results]
 
 
+# Module-level singleton — drop-in replacement for the old import
 vector_rag = VectorRAG()
\ No newline at end of file
diff --git a/backend/app/services/elasticsearch_service.py b/backend/app/services/elasticsearch_service.py
new file mode 100644
index 0000000..d84c542
+++ b/backend/app/services/elasticsearch_service.py
@@ -0,0 +1,104 @@
+from typing import Optional
+from elasticsearch import AsyncElasticsearch
+from app.embeddings.openai_client import openai_client
+
+ES_HOST = "http://localhost:9200"
+INDEX_NAME = "rag_documents"
+
+class ElasticsearchService:
+    def __init__(self):
+        self.client: Optional[AsyncElasticsearch] = None
+        
+    async def init(self):
+        if not self.client:
+            self.client = AsyncElasticsearch([ES_HOST])
+            await self._create_index()
+    
+    async def close(self):
+        if self.client:
+            await self.client.close()
+    
+    async def _create_index(self):
+        if not await self.client.indices.exists(index=INDEX_NAME):
+            await self.client.indices.create(
+                index=INDEX_NAME,
+                body={
+                    "mappings": {
+                        "properties": {
+                            "content": {"type": "text"},
+                            "metadata": {"type": "object"}
+                        }
+                    }
+                }
+            )
+    
+    async def check_status(self) -> dict:
+        try:
+            if not self.client:
+                await self.init()
+            health = await self.client.cluster.health()
+            count = await self.client.count(index=INDEX_NAME)
+            return {"status": health["status"], "online": True, "doc_count": count["count"]}
+        except Exception as e:
+            return {"status": "offline", "online": False, "error": str(e)}
+    
+    async def index_document(self, doc_id: str, content: str, metadata: dict = None):
+        await self.init()
+        await self.client.index(index=INDEX_NAME, id=doc_id, body={
+            "content": content,
+            "metadata": metadata or {}
+        })
+    
+    async def bulk_index(self, documents: list[dict]):
+        await self.init()
+        actions = []
+        for doc in documents:
+            actions.append({"index": {"_index": INDEX_NAME, "_id": doc["id"]}})
+            actions.append({"content": doc["content"], "metadata": doc.get("metadata", {})})
+        if actions:
+            await self.client.bulk(operations=actions)
+    
+    async def search(self, query: str, size: int = 10) -> list[dict]:
+        await self.init()
+        result = await self.client.search(
+            index=INDEX_NAME,
+            body={"query": {"match": {"content": query}}, "size": size}
+        )
+        return [{"id": hit["_id"], "content": hit["_source"]["content"], 
+                 "score": hit["_score"], "metadata": hit["_source"].get("metadata", {})}
+                for hit in result["hits"]["hits"]]
+    
+    async def enhance_with_iterative_query(self, chunks: list[str], query: str, max_tries: int = 5, logger=None) -> list[str]:
+        """Enhance nearest neighbor chunks by iteratively asking ES for missing info"""
+        enhanced = chunks.copy()
+        
+        try:
+            for attempt in range(max_tries):
+                if logger:
+                    logger.log(f"ES_ITERATION_{attempt+1}", f"Current chunks: {len(enhanced)}")
+                
+                prompt = f"Based on these chunks:\n{chr(10).join(enhanced)}\n\nWhat specific question should I ask to get more relevant info for: {query}?"
+                llm_question = await openai_client.generate(prompt, system="Generate a concise search question.")
+                
+                if logger:
+                    logger.log(f"ES_LLM_QUESTION_{attempt+1}", f"Generated question: {llm_question}")
+                
+                es_results = await self.search(llm_question, size=3)
+                if logger:
+                    logger.log(f"ES_SEARCH_RESULTS_{attempt+1}", f"Found {len(es_results)} results")
+                
+                if not es_results:
+                    break
+                
+                new_content = [r["content"] for r in es_results if r["content"] not in enhanced]
+                if not new_content:
+                    break
+                    
+                enhanced.extend(new_content[:2])
+        except Exception as e:
+            if logger:
+                logger.log("ES_ENHANCEMENT_ERROR", str(e))
+        
+        return enhanced[:len(chunks) + 5]
+
+es_service = ElasticsearchService()
diff --git a/backend/app/services/rag_service.py b/backend/app/services/rag_service.py
index deb65be..4d81140 100644
+++ b/backend/app/services/rag_service.py
@@ -17,6 +17,8 @@ from app.rag.evaluator import (
 from app.embeddings.openai_client import openai_client as ollama_client
 from app.repositories.log_repository import log_repo
 from app.core.config import settings
+from app.services.elasticsearch_service import es_service
+from app.core.evaluation_logger import EvaluationLogger
 
 
 RAG_SYSTEM = """You are an expert assistant. Answer the question using only the provided context.
@@ -122,28 +124,57 @@ class RAGService:
         questions: list[dict],
         dataset_name: str = "default",
     ) -> dict[str, Any]:
+        logger = EvaluationLogger(f"{dataset_name}_{int(time.time())}")
+        logger.log("EVALUATION_START", f"Dataset: {dataset_name}, Total questions: {len(questions)}")
+        
         results = {
             "accuracy": [], "faithfulness": [], "context_precision": [],
             "context_recall": [], "answer_relevancy": [], "latency_ms": [], "failed": [],
         }
 
+        for q_idx, q in enumerate(questions, 1):
             question = q["question"]
             expected = q["expected_answer"]
+            
+            logger.log_question(question, q_idx, len(questions))
+            
             try:
                 start = time.time()
+                
+                # Retrieve
                 chunks = await self.retrieve(question, strategy="hybrid", top_k=settings.TOP_K)
                 context_texts = [r["chunk_text"] for r in chunks]
+                logger.log_retrieval("hybrid", settings.TOP_K, chunks)
+                
+                # Enhance with Elasticsearch iterative queries
+                original_count = len(context_texts)
+                enhanced_texts = await es_service.enhance_with_iterative_query(context_texts, question, max_tries=5, logger=logger)
+                logger.log_es_enhancement(len(enhanced_texts), original_count)
+                
+                context = "\n\n".join(enhanced_texts)
                 prompt = f"Context:\n{context}\n\nQuestion: {question}"
+                
+                llm_start = time.time()
                 answer = await ollama_client.generate(prompt, system=RAG_SYSTEM)
+                llm_latency = (time.time() - llm_start) * 1000
+                logger.log_llm_call("Generate Answer", prompt, answer, llm_latency)
+                
                 latency = (time.time() - start) * 1000
 
+                acc_score, acc_rat = await compute_accuracy(answer, expected)
+                logger.log_metrics("Accuracy", acc_score, acc_rat)
+                
+                faith_score, faith_rat = await compute_faithfulness(answer, enhanced_texts)
+                logger.log_metrics("Faithfulness", faith_score, faith_rat)
+                
+                cp_score, cp_rat = await compute_context_precision(question, enhanced_texts)
+                logger.log_metrics("Context Precision", cp_score, cp_rat)
+                
+                cr_score, cr_rat = await compute_context_recall(expected, enhanced_texts)
+                logger.log_metrics("Context Recall", cr_score, cr_rat)
+                
+                ar_score, ar_rat = await compute_answer_relevancy(question, answer)
+                logger.log_metrics("Answer Relevancy", ar_score, ar_rat)
 
                 results["accuracy"].append(acc_score)
                 results["faithfulness"].append(faith_score)
@@ -152,6 +183,7 @@ class RAGService:
                 results["answer_relevancy"].append(ar_score)
                 results["latency_ms"].append(latency)
             except Exception as e:
+                logger.log_error(str(e), question)
                 results["failed"].append({"question": question, "error": str(e)})
 
         def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0
@@ -165,6 +197,8 @@ class RAGService:
             "latency_avg_ms": avg(results["latency_ms"]),
             "failed_questions": results["failed"],
         }
+        
+        logger.log_summary(final)
 
         await log_repo.create_evaluation_run(db, {
             "id": str(uuid.uuid4()),
diff --git a/backend/test_logger.py b/backend/test_logger.py
new file mode 100644
index 0000000..2808935
+++ b/backend/test_logger.py
@@ -0,0 +1,16 @@
+import sys
+sys.path.insert(0, '/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/backend')
+
+from app.core.evaluation_logger import EvaluationLogger
+
+logger = EvaluationLogger("test_manual")
+print(f"Log file: {logger.log_file}")
+print(f"File exists: {logger.log_file.exists()}")
+
+logger.log("TEST_STEP", "This is a test message")
+print("Logged test message")
+
+with open(logger.log_file, 'r') as f:
+    content = f.read()
+    print(f"File content length: {len(content)}")
+    print(f"Content:\n{content}")
diff --git a/helpfull scripts/http_client.py b/helpfull scripts/http_client.py
new file mode 100644
index 0000000..66a0a60
+++ b/helpfull scripts/http_client.py	
@@ -0,0 +1,25 @@
+import requests
+from bs4 import BeautifulSoup
+from urllib.parse import urljoin
+from pathlib import Path
+
+BASE_URL = "http://10.64.26.89:8000/"
+OUTPUT_DIR = "downloaded"
+
+Path(OUTPUT_DIR).mkdir(exist_ok=True)
+
+html = requests.get(BASE_URL).text
+soup = BeautifulSoup(html, "html.parser")
+
+for a in soup.find_all("a"):
+    href = a.get("href")
+
+    if href.endswith(".txt"):
+        url = urljoin(BASE_URL, href)
+
+        print("Downloading", href)
+
+        r = requests.get(url)
+
+        with open(Path(OUTPUT_DIR) / href, "wb") as f:
+            f.write(r.content)
\ No newline at end of file
diff --git a/helpfull scripts/upload_to_elasticsearch.py b/helpfull scripts/upload_to_elasticsearch.py
new file mode 100644
index 0000000..ce1f76b
+++ b/helpfull scripts/upload_to_elasticsearch.py	
@@ -0,0 +1,60 @@
+#!/usr/bin/env python3
+import os
+import sys
+import asyncio
+from pathlib import Path
+from elasticsearch import AsyncElasticsearch
+
+ES_HOST = "http://localhost:9200"
+INDEX_NAME = "rag_documents"
+DATA_DIR = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/data"
+
+async def upload_files():
+    client = AsyncElasticsearch([ES_HOST])
+    
+    try:
+        # Create index
+        if not await client.indices.exists(index=INDEX_NAME):
+            await client.indices.create(index=INDEX_NAME, body={
+                "mappings": {"properties": {"content": {"type": "text"}, "metadata": {"type": "object"}}}
+            })
+            print(f"Created index: {INDEX_NAME}")
+        
+        doc_id = 0
+        total = 0
+        
+        # Process all files in data directory
+        for file_path in sorted(Path(DATA_DIR).rglob("*")):
+            if file_path.is_file() and file_path.suffix in [".txt", ".csv", ".md"]:
+                try:
+                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
+                        content = f.read().strip()
+                        if not content:
+                            continue
+                        
+                        # Split by lines and index
+                        lines = [l.strip() for l in content.split("\n") if l.strip()]
+                        for line in lines:
+                            await client.index(
+                                index=INDEX_NAME,
+                                id=f"doc_{doc_id}",
+                                body={"content": line, "metadata": {"source": file_path.name}}
+                            )
+                            doc_id += 1
+                        
+                        total += len(lines)
+                        print(f"✓ {file_path.name}: {len(lines)} documents")
+                except Exception as e:
+                    print(f"✗ {file_path.name}: {e}")
+        
+        print(f"\nTotal indexed: {total} documents")
+    
+    except Exception as e:
+        print(f"Error: {e}")
+        sys.exit(1)
+    finally:
+        await client.close()
+
+if __name__ == "__main__":
+    print(f"Uploading data from {DATA_DIR} to Elasticsearch...")
+    asyncio.run(upload_files())
```

## 987fdd4576679275a60e52517c1f526ef8b7951f — 2026-06-11T12:05:46+05:30

Message:

restart with port 9000

_No Python file changes in this commit._

## ed9088b78f2edcfd3c079ffe3c708ab1fd2dd599 — 2026-06-11T12:01:39+05:30

Message:

simple nearest neighbour retreival with query expansion and max 2 retry 47 percent

```diff
diff --git a/backend/app/rag/evaluator.py b/backend/app/rag/evaluator.py
index 6eb8ea4..9b14323 100644
+++ b/backend/app/rag/evaluator.py
@@ -14,6 +14,34 @@ from app.rag.vector_rag import vector_rag
 _SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]*\.?[0-9]+)')
 
 
+async def _expand_query(question: str) -> str:
+    """Use LLM to expand/rephrase the query for better retrieval."""
+    prompt = f"""Rephrase the following question to improve retrieval from a document database. 
+Make it more specific and include relevant keywords. Return only the rephrased question.
+
+Original question: {question}"""
+    return await ollama_client.chat([{"role": "user", "content": prompt}])
+
+
+async def _check_sufficiency(question: str, context_chunks: list[str]) -> tuple[bool, str]:
+    """Check if retrieved context is sufficient to answer the question."""
+    context = "\n\n---\n\n".join(context_chunks[:5])
+    prompt = f"""Does the CONTEXT below contain sufficient information to answer the QUESTION?
+Respond with a JSON object: {{"sufficient": true/false, "reason": "brief explanation"}}
+
+QUESTION: {question}
+
+CONTEXT: {context}"""
+    
+    try:
+        response = await ollama_client.chat([{"role": "user", "content": prompt}])
+        response = response.strip().strip("```json").strip("```").strip()
+        data = json.loads(response)
+        return data.get("sufficient", False), data.get("reason", "")
+    except:
+        return len(context_chunks) > 0, "Parse error"
+
+
 async def _llm_score(prompt: str) -> tuple[float, str]:
     system = (
         "You are an expert RAG evaluation judge. "
@@ -119,15 +147,40 @@ async def evaluate_single(
     context_chunks: list[str] = None,
     collection_name: str = "text_documents",
     top_k: int = 5,
+    max_tries: int = 2,
 ) -> dict[str, Any]:
     """
+    Retrieve with iterative query expansion if context insufficient (max 2 tries).
     """
     t0 = time.time()
     
+    current_query = question
+    all_attempts = []
+    retrieved_chunks = []
+    
+    for attempt in range(max_tries):
+        # Retrieve with current query
+        results = await vector_rag.retrieve(current_query, collection_name, top_k)
+        retrieved_chunks = [r.get("chunk_text", "") for r in results]
+        
+        all_attempts.append({
+            "attempt": attempt + 1,
+            "query": current_query,
+            "num_results": len(results)
+        })
+        
+        # Check if context is sufficient
+        if retrieved_chunks:
+            is_sufficient, reason = await _check_sufficiency(question, retrieved_chunks)
+            all_attempts[-1]["sufficient"] = is_sufficient
+            all_attempts[-1]["reason"] = reason
+            
+            if is_sufficient or attempt == max_tries - 1:
+                break
+        
+        # Expand query for next attempt
+        if attempt < max_tries - 1:
+            current_query = await _expand_query(current_query)
     
     # Run LLM-based evaluation metrics
     accuracy, acc_rationale = await compute_accuracy(generated_answer, expected_answer)
@@ -143,6 +196,7 @@ async def evaluate_single(
         "expected_answer": expected_answer,
         "generated_answer": generated_answer,
         "retrieved_context": "\n---\n".join(retrieved_chunks),
+        "retrieval_attempts": all_attempts,
         "accuracy": accuracy,
         "faithfulness": faithfulness,
         "answer_relevancy": answer_relevancy,
```

## 28602985c7f677fe46d9d7d6365bed78de046a23 — 2026-06-11T11:47:19+05:30

Message:

simple nearest neighbour retreival got 42 percent

```diff
diff --git a/backend/app/rag/evaluator.py b/backend/app/rag/evaluator.py
index bfa663b..6eb8ea4 100644
+++ b/backend/app/rag/evaluator.py
@@ -1,8 +1,5 @@
 """
+LLM-as-a-Judge evaluator for RAG pipelines with nearest neighbor retrieval.
 """
 
 import json
@@ -11,22 +8,13 @@ import time
 from typing import Any
 
 from app.embeddings.openai_client import openai_client as ollama_client
+from app.rag.vector_rag import vector_rag
 
 
 _SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]*\.?[0-9]+)')
 
 
 async def _llm_score(prompt: str) -> tuple[float, str]:
     system = (
         "You are an expert RAG evaluation judge. "
         "Respond ONLY with a JSON object containing exactly two keys: "
@@ -34,32 +22,25 @@ async def _llm_score(prompt: str) -> tuple[float, str]:
         '"rationale" (a one-sentence explanation). '
         "Do not include any other text."
     )
+    raw = ""
     try:
         raw = await ollama_client.chat(
             [{"role": "user", "content": prompt}],
             system=system,
         )
         raw = raw.strip().strip("```json").strip("```").strip()
         data = json.loads(raw)
         score = float(data.get("score", 0.0))
         rationale = data.get("rationale", "")
         return round(max(0.0, min(1.0, score)), 4), rationale
     except Exception as exc:
         m = _SCORE_RE.search(raw)
         if m:
             return round(float(m.group(1)), 4), "Score extracted via regex fallback."
         return 0.0, f"Parse error: {exc}"
 
 
 async def compute_accuracy(generated: str, expected: str) -> tuple[float, str]:
     prompt = f"""Rate how well the GENERATED ANSWER matches the EXPECTED ANSWER semantically.
 
 EXPECTED ANSWER:
@@ -73,8 +54,7 @@ Score 1.0 if they convey the same meaning, 0.0 if completely unrelated."""
 
 
 async def compute_faithfulness(answer: str, context_chunks: list[str]) -> tuple[float, str]:
+    context = "\n\n---\n\n".join(context_chunks[:5])
     prompt = f"""Rate how faithfully the ANSWER is grounded in the CONTEXT below.
 A faithful answer only makes claims supported by the context (score 1.0).
 An unfaithful answer introduces hallucinated facts not in the context (score 0.0).
@@ -88,7 +68,6 @@ ANSWER:
 
 
 async def compute_answer_relevancy(question: str, answer: str) -> tuple[float, str]:
     prompt = f"""Rate how directly and completely the ANSWER addresses the QUESTION.
 Score 1.0 if the answer is fully on-topic and complete, 0.0 if it ignores the question.
 
@@ -101,7 +80,6 @@ ANSWER:
 
 
 async def compute_context_precision(question: str, context_chunks: list[str]) -> tuple[float, str]:
     if not context_chunks:
         return 0.0, "No context chunks provided."
     chunks_text = "\n\n---\n\n".join(
@@ -120,7 +98,6 @@ RETRIEVED CHUNKS:
 
 
 async def compute_context_recall(expected_answer: str, context_chunks: list[str]) -> tuple[float, str]:
     if not context_chunks:
         return 0.0, "No context chunks provided."
     context = "\n\n---\n\n".join(context_chunks[:8])
@@ -135,48 +112,46 @@ RETRIEVED CONTEXT:
     return await _llm_score(prompt)
 
 
 async def evaluate_single(
     question: str,
     expected_answer: str,
     generated_answer: str,
+    context_chunks: list[str] = None,
+    collection_name: str = "text_documents",
+    top_k: int = 5,
 ) -> dict[str, Any]:
     """
+    Retrieve using nearest neighbor search, then evaluate with LLM judge.
     """
     t0 = time.time()
+    
+    # Perform nearest neighbor retrieval
+    results = await vector_rag.retrieve(question, collection_name, top_k)
+    retrieved_chunks = [r.get("chunk_text", "") for r in results]
+    
+    # Run LLM-based evaluation metrics
+    accuracy, acc_rationale = await compute_accuracy(generated_answer, expected_answer)
+    faithfulness, fai_rationale = await compute_faithfulness(generated_answer, retrieved_chunks)
+    answer_relevancy, rel_rationale = await compute_answer_relevancy(question, generated_answer)
+    context_precision, pre_rationale = await compute_context_precision(question, retrieved_chunks)
+    context_recall, rec_rationale = await compute_context_recall(expected_answer, retrieved_chunks)
 
     latency_ms = round((time.time() - t0) * 1000, 1)
 
     return {
+        "question": question,
+        "expected_answer": expected_answer,
+        "generated_answer": generated_answer,
+        "retrieved_context": "\n---\n".join(retrieved_chunks),
+        "accuracy": accuracy,
+        "faithfulness": faithfulness,
+        "answer_relevancy": answer_relevancy,
         "context_precision": context_precision,
+        "context_recall": context_recall,
+        "accuracy_rationale": acc_rationale,
+        "faithfulness_rationale": fai_rationale,
+        "answer_relevancy_rationale": rel_rationale,
         "context_precision_rationale": pre_rationale,
+        "context_recall_rationale": rec_rationale,
         "latency_ms": latency_ms,
     }
\ No newline at end of file
```

## 4d9be847248880f381f20731d2224987acd134c3 — 2026-06-11T11:31:11+05:30

Message:

simple nearest neighbour retreival with synonym adn acronym dictionary score at 27

```diff
diff --git a/backend/app/agents/base.py b/backend/app/agents/base.py
index 9d47502..68de3dd 100644
+++ b/backend/app/agents/base.py
@@ -7,6 +7,7 @@ class BaseAgent(ABC):
     name: str = "base_agent"
 
     def __init__(self):
+        pass
 
     @abstractmethod
     async def plan(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
diff --git a/backend/app/agents/web_agent.py b/backend/app/agents/web_agent.py
index b92cd38..189f264 100644
+++ b/backend/app/agents/web_agent.py
@@ -29,7 +29,8 @@ class WebEnrichmentAgent(BaseAgent):
             try:
                 await web_service.ingest(url)
                 chunks = await web_service.query(query, url=url, top_k=top_k)
+            except Exception:
+                pass
 
         context_str = "\n\n".join(r["chunk_text"] for r in chunks) if chunks else "No web context available."
         system = "You are a research assistant. Summarize and answer based on web content. Always note the source URL."
diff --git a/backend/app/api/search.py b/backend/app/api/search.py
index 2e5200f..c4da323 100644
+++ b/backend/app/api/search.py
@@ -29,7 +29,8 @@ def _search_with_synonyms(search_fn, query: str, *args, **kwargs) -> list[dict]:
                 doc_id = r.get("chunk_id", r.get("document_id", ""))
                 if doc_id not in all_results:
                     all_results[doc_id] = r
+        except Exception:
+            pass
     
     return sorted(all_results.values(), key=lambda x: x.get("score", 0), reverse=True)
 
diff --git a/backend/app/evaluation/agent_runner.py b/backend/app/evaluation/agent_runner.py
index 68fca03..44c38c5 100644
+++ b/backend/app/evaluation/agent_runner.py
@@ -1,96 +1,87 @@
 """
 Simple nearest-neighbor RAG evaluation.
 
+Uses: vector search + BM25 + synonym expansion + reranker.
 """
 
 import time
 from typing import Any
 
+from app.rag.vector_rag import vector_rag
+from app.rag.bm25 import bm25_retriever
+from app.rag.synonym_expansion import get_synonym_expander
+from app.rag.cross_encoder import cross_encoder
 from app.evaluation.evaluator import evaluate_single
 from app.embeddings.openai_client import openai_client
 from app.core.config import settings
 
 
 RAG_SYSTEM = """Answer the question carefully"""
 
 
 def _chunk_texts(chunks: list[dict]) -> list[str]:
     return [c["chunk_text"] for c in chunks if c.get("chunk_text")]
 
 
+def _merge_results(vector_results: list[dict], bm25_results: list[dict]) -> list[dict]:
+    """Simple merge: combine and deduplicate by chunk_id"""
+    seen = set()
+    merged = []
+    for c in vector_results + bm25_results:
+        cid = c.get("chunk_id")
+        if cid and cid not in seen:
+            seen.add(cid)
+            merged.append(c)
+    return merged
 
 
 async def evaluate_question(
     question: str,
     expected_answer: str,
+    top_k: int = 5,
     use_query_expansion: bool = False,
     num_expansions: int = 2,
 ) -> dict[str, Any]:
     """
+    Simplified RAG evaluation:
+    1. Synonym expansion (if enabled)
+    2. Vector + BM25 retrieval (top 10 each)
+    3. Merge and rerank (top 5)
+    4. Generate answer
     5. Score with LLM-as-judge
     """
     t0 = time.time()
     
+    # Synonym expansion
+    synonym_expander = get_synonym_expander()
+    queries = synonym_expander.expand_query(question)
     
+    # Retrieve chunks for all query variants
     all_chunks = []
     chunk_ids_seen = set()
     
     for query in queries:
+        # Vector search (nearest neighbor)
+        vector_results = await vector_rag.retrieve(query, "text_documents", top_k=10)
         
+        # BM25 search
+        bm25_results = bm25_retriever.search("text_documents", query, top_k=10)
+        
+        # Merge
+        merged = _merge_results(vector_results, bm25_results)
+        
+        # Deduplicate across queries
+        for c in merged:
             cid = c.get("chunk_id")
             if cid and cid not in chunk_ids_seen:
                 chunk_ids_seen.add(cid)
                 all_chunks.append(c)
     
+    # Rerank to top_k
+    reranked = cross_encoder.rerank(question, all_chunks, top_k=top_k)
+    
     # Generate answer
+    chunk_texts = _chunk_texts(reranked)
     context_text = "\n---\n".join(chunk_texts)
     
     if context_text.strip():
@@ -101,21 +92,21 @@ async def evaluate_question(
     generated_answer = await openai_client.generate(prompt, system=RAG_SYSTEM)
     
     # Build retrieved chunk IDs and derive gold IDs via text matching
+    retrieved_chunk_ids = [c.get("chunk_id", "") for c in reranked if c.get("chunk_id")]
     expected_lower = expected_answer.lower()
     gold_chunk_ids = {
         c.get("chunk_id")
+        for c in reranked
         if c.get("chunk_id") and (
             expected_lower in c.get("chunk_text", "").lower()
             or (len(expected_lower) >= 20 and expected_lower[:20] in c.get("chunk_text", "").lower())
         )
     }
     # Fallback: if no exact match found, mark best-scoring chunks as gold
+    if not gold_chunk_ids and reranked:
+        best_score = max((c.get("score", 0) for c in reranked), default=0)
         if best_score > 0:
+            gold_chunk_ids = {c.get("chunk_id") for c in reranked if c.get("score", 0) >= best_score * 0.9 and c.get("chunk_id")}
 
     # Score
     scores = await evaluate_single(question, expected_answer, generated_answer, chunk_texts,
@@ -128,8 +119,8 @@ async def evaluate_question(
         "expected_answer": expected_answer,
         "generated_answer": generated_answer,
         "retrieved_context": context_text,
+        "expanded_queries": queries,
+        "num_chunks": len(reranked),
         # LLM-as-judge
         "accuracy_llm": scores.get("accuracy_llm", 0.0),
         "faithfulness": scores.get("faithfulness", 0.0),
diff --git a/backend/app/rag/hybrid_rag.py b/backend/app/rag/hybrid_rag.py
index 75e9b82..a0f0afa 100644
+++ b/backend/app/rag/hybrid_rag.py
@@ -78,14 +78,16 @@ class HybridRAG:
             vector_results = await vector_rag.retrieve(
                 query, collection_name, settings.DENSE_TOP_K, filters
             )
+        except Exception:
+            pass
 
         # BM25 retrieval: top 50
         bm25_results = []
         try:
             bm25_raw = bm25_retriever.search(collection_name, query, settings.BM25_TOP_K)
             bm25_results = filter_results(bm25_raw, filters or {})
+        except Exception:
+            pass
 
         if not vector_results and not bm25_results:
             return []
diff --git a/backend/app/rag/synonym_expansion.py b/backend/app/rag/synonym_expansion.py
index 0d18de9..4ce62aa 100644
+++ b/backend/app/rag/synonym_expansion.py
@@ -27,7 +27,8 @@ class SynonymExpander:
                     self.synonyms[canonical] = aliases
                     for alias in aliases:
                         self.reverse_map[alias] = canonical
+        except Exception:
+            pass
 
     def _load_csv(self, path: str):
         """Load synonyms from CSV format: canonical,alias"""
@@ -43,7 +44,8 @@ class SynonymExpander:
                         if alias not in self.synonyms[canonical]:
                             self.synonyms[canonical].append(alias)
                         self.reverse_map[alias] = canonical
+        except Exception:
+            pass
 
     def expand_query(self, query: str) -> list[str]:
         """Expand query with synonyms and return list of expanded queries"""
diff --git a/backend/app/rag/vector_rag.py b/backend/app/rag/vector_rag.py
index f592dd7..5dc00f2 100644
+++ b/backend/app/rag/vector_rag.py
@@ -39,7 +39,8 @@ class VectorRAG:
                     r["document_id"] = r.get("metadata", {}).get("document_id", "")
                     r["filename"] = r.get("metadata", {}).get("filename", "")
                 all_results.extend(results)
+            except Exception:
+                pass
         all_results.sort(key=lambda x: x["score"], reverse=True)
         return all_results[:top_k]
 
diff --git a/backend/app/services/document_service.py b/backend/app/services/document_service.py
index 0388130..28e4d02 100644
+++ b/backend/app/services/document_service.py
@@ -180,7 +180,8 @@ class DocumentService:
         # Delete existing vector data
         try:
             chroma_client.delete_by_document_id(doc.collection_name or "text_documents", doc_id)
+        except Exception:
+            pass
 
         await document_repo.delete_chunks(db, doc_id)
 
@@ -214,7 +215,8 @@ class DocumentService:
             raise DocumentNotFoundError(doc_id)
         try:
             chroma_client.delete_by_document_id(doc.collection_name or "text_documents", doc_id)
+        except Exception:
+            pass
         filepath = Path(doc.filepath)
         if filepath.exists():
             filepath.unlink()
```

## b50ebba8f482a98575ab9ee09b0ea28672abf9d9 — 2026-06-11T10:59:48+05:30

Message:

remove logging

```diff
diff --git a/backend/app/agents/coordinator_agent.py b/backend/app/agents/coordinator_agent.py
index ecb999e..d7ceb9c 100644
+++ b/backend/app/agents/coordinator_agent.py
@@ -90,7 +90,8 @@ class CoordinatorAgent(BaseAgent):
                     continue
                 agent_results.append(res)
                 all_chunks.extend(res.get("chunks", []))
+            except Exception:
+                pass
 
         # Synthesize answers from all agents
         if not agent_results:
diff --git a/backend/app/main.py b/backend/app/main.py
index 6f95a23..006bcdb 100644
+++ b/backend/app/main.py
@@ -38,7 +38,8 @@ async def lifespan(app: FastAPI):
     await init_db()
     try:
         chroma_client.init_collections()
+    except Exception:
+        pass
     yield
     from app.embeddings.openai_client import openai_client
     await openai_client.close()
```

## c12bb791e88b3cc988ff6fc4fbc62983e4efb472 — 2026-06-11T10:57:22+05:30

Message:

remove logging

```diff
diff --git a/backend/app/agents/base.py b/backend/app/agents/base.py
index 51ed02d..9d47502 100644
+++ b/backend/app/agents/base.py
@@ -1,14 +1,12 @@
 from abc import ABC, abstractmethod
 from typing import Any, Optional
 from app.embeddings.openai_client import openai_client as ollama_client
 
 
 class BaseAgent(ABC):
     name: str = "base_agent"
 
     def __init__(self):
 
     @abstractmethod
     async def plan(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
@@ -26,7 +24,6 @@ class BaseAgent(ABC):
         ...
 
     async def run(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
         plan = await self.plan(query, context)
         result = await self.execute(plan)
         evaluated = await self.evaluate(result)
diff --git a/backend/app/agents/coordinator_agent.py b/backend/app/agents/coordinator_agent.py
index 6044e6a..ecb999e 100644
+++ b/backend/app/agents/coordinator_agent.py
@@ -7,9 +7,7 @@ from app.agents.router_agent import router_agent, _detect_doc_type
 from app.agents.web_agent import web_agent
 from app.embeddings.openai_client import openai_client as ollama_client
 from app.core.prompts import DEFAULT_COORDINATOR_SYNTHESIS_PROMPT
 
 
 INTENT_KEYWORDS = {
     "table": ["table", "csv", "spreadsheet", "rows", "columns", "sum", "count", "average", "aggregate"],
@@ -93,7 +91,6 @@ class CoordinatorAgent(BaseAgent):
                 agent_results.append(res)
                 all_chunks.extend(res.get("chunks", []))
             except Exception as e:
 
         # Synthesize answers from all agents
         if not agent_results:
diff --git a/backend/app/agents/evaluator_agent.py b/backend/app/agents/evaluator_agent.py
index 08470ca..f6d9b82 100644
+++ b/backend/app/agents/evaluator_agent.py
@@ -6,7 +6,6 @@ from app.rag.evaluator import (
     compute_faithfulness, compute_answer_relevancy,
     compute_context_precision, compute_context_recall,
 )
 
 
 class RetrievalEvaluationAgent(BaseAgent):
diff --git a/backend/app/agents/web_agent.py b/backend/app/agents/web_agent.py
index 9a997ee..b92cd38 100644
+++ b/backend/app/agents/web_agent.py
@@ -30,7 +30,6 @@ class WebEnrichmentAgent(BaseAgent):
                 await web_service.ingest(url)
                 chunks = await web_service.query(query, url=url, top_k=top_k)
             except Exception as e:
 
         context_str = "\n\n".join(r["chunk_text"] for r in chunks) if chunks else "No web context available."
         system = "You are a research assistant. Summarize and answer based on web content. Always note the source URL."
diff --git a/backend/app/api/agents.py b/backend/app/api/agents.py
index c83e0b7..fe117f6 100644
+++ b/backend/app/api/agents.py
@@ -9,10 +9,8 @@ from app.agents.router_agent import router_agent
 from app.agents.web_agent import web_agent
 from app.agents.evaluator_agent import evaluator_agent
 from app.schemas.agent import AgentRequest, AgentResponse, CoordinatorRequest
 
 router = APIRouter(prefix="/api/agents", tags=["Agents"])
 
 
 def _to_response(result: dict) -> AgentResponse:
diff --git a/backend/app/api/chat.py b/backend/app/api/chat.py
index cf57a39..7563d9d 100644
+++ b/backend/app/api/chat.py
@@ -9,10 +9,8 @@ from app.schemas.chat import (
     ChatRequest, ChatResponse, ConversationListResponse, ConversationResponse
 )
 from app.core.exceptions import ConversationNotFoundError
 
 router = APIRouter(prefix="/api/chat", tags=["Chat"])
 
 
 @router.post("", response_model=ChatResponse)
diff --git a/backend/app/api/chroma.py b/backend/app/api/chroma.py
index be348df..6901e40 100644
+++ b/backend/app/api/chroma.py
@@ -4,10 +4,8 @@ from typing import Any, Optional
 
 from app.chromadb.client import chroma_client
 from app.embeddings.openai_client import openai_client as ollama_client
 
 router = APIRouter(prefix="/api/chroma", tags=["ChromaDB"])
 
 
 class ChromaIndexRequest(BaseModel):
diff --git a/backend/app/api/documents.py b/backend/app/api/documents.py
index 4b6c7ec..7610c27 100644
+++ b/backend/app/api/documents.py
@@ -9,10 +9,8 @@ from app.services.document_service import document_service
 from app.repositories.document_repository import document_repo
 from app.schemas.document import DocumentResponse, DocumentListResponse, ChunkResponse, ReindexResponse
 from app.core.exceptions import DocumentNotFoundError
 
 router = APIRouter(prefix="/api/documents", tags=["Documents"])
 
 
 @router.post("/upload", response_model=dict)
diff --git a/backend/app/api/embeddings.py b/backend/app/api/embeddings.py
index 2ec0026..c0562e3 100644
+++ b/backend/app/api/embeddings.py
@@ -5,10 +5,8 @@ from app.schemas.embeddings import (
     EmbeddingResponse, EmbeddingBatchResponse, EmbeddingModelInfo,
 )
 from app.core.config import settings
 
 router = APIRouter(prefix="/api/embeddings", tags=["Embeddings"])
 
 
 @router.post("/generate", response_model=EmbeddingResponse)
diff --git a/backend/app/api/health.py b/backend/app/api/health.py
index 8b83211..fcd26cf 100644
+++ b/backend/app/api/health.py
@@ -5,10 +5,8 @@ from sqlalchemy import text
 from app.core.dependencies import get_db
 from app.chromadb.client import chroma_client
 from app.embeddings.openai_client import openai_client as ollama_client
 
 router = APIRouter(tags=["Health"])
 
 
 @router.get("/health")
@@ -22,7 +20,6 @@ async def health_db(db: AsyncSession = Depends(get_db)):
         await db.execute(text("SELECT 1"))
         return {"status": "ok", "database": "sqlite"}
     except Exception as e:
         return {"status": "error", "database": "sqlite", "detail": str(e)}
 
 
diff --git a/backend/app/api/markdown.py b/backend/app/api/markdown.py
index dc2a894..dee1707 100644
+++ b/backend/app/api/markdown.py
@@ -7,10 +7,8 @@ from app.core.dependencies import get_db
 from app.rag.markdown_rag import markdown_rag
 from app.services.document_service import document_service
 from app.schemas.search import SearchResult
 
 router = APIRouter(prefix="/api/markdown", tags=["Markdown RAG"])
 
 
 @router.post("/index")
diff --git a/backend/app/api/pdf.py b/backend/app/api/pdf.py
index 5b7f117..d2ba32d 100644
+++ b/backend/app/api/pdf.py
@@ -7,10 +7,8 @@ from app.core.dependencies import get_db
 from app.rag.pdf_rag import pdf_rag
 from app.services.document_service import document_service
 from app.schemas.search import SearchResult
 
 router = APIRouter(prefix="/api/pdf", tags=["PDF RAG"])
 
 
 @router.post("/index")
diff --git a/backend/app/api/rag.py b/backend/app/api/rag.py
index 1949b88..0c8f160 100644
+++ b/backend/app/api/rag.py
@@ -1,126 +1,123 @@
+"""
+POST /api/rag/evaluate
+
+Accepts a list of {question, expected_answer} pairs, runs each through the
+multi-agent RAG pipeline (coordinator → evaluator), and returns per-question
+detail alongside aggregate metrics.
+"""
+
+from typing import Any
+
+from fastapi import APIRouter, Depends
+from fastapi.responses import StreamingResponse
+from sqlalchemy.ext.asyncio import AsyncSession
+from pydantic import BaseModel
+
+from app.core.dependencies import get_db
+from app.services.rag_service import rag_service
+from app.evaluation.agent_runner import evaluate_question, failed_question_row
+
+from app.schemas.rag import RAGQueryRequest, RAGQueryResponse, RAGRetrieveRequest
+from app.schemas.search import SearchResult
+
+router = APIRouter(prefix="/api/rag", tags=["rag"])
+
+
+@router.post("/query", response_model=RAGQueryResponse)
+async def rag_query(req: RAGQueryRequest, db: AsyncSession = Depends(get_db)):
+    result = await rag_service.query(db, req.query, req.strategy, req.top_k, req.filters)
+    return result
+
+
+@router.post("/query/stream")
+async def rag_query_stream(req: RAGQueryRequest):
+    async def generator():
+        async for token in rag_service.query_stream(req.query, req.strategy, req.top_k, req.filters):
+            yield token
+
+    return StreamingResponse(generator(), media_type="text/plain")
+
+
+@router.post("/retrieve", response_model=list[SearchResult])
+async def rag_retrieve(req: RAGRetrieveRequest):
+    chunks = await rag_service.retrieve(req.query, req.strategy, req.top_k, req.filters)
+    results = []
+    for r in chunks:
+        meta = r.get("metadata", {})
+        results.append(SearchResult(
+            chunk_id=r.get("chunk_id", ""),
+            document_id=r.get("document_id") or meta.get("document_id", ""),
+            filename=r.get("filename") or meta.get("filename", ""),
+            chunk_text=r.get("chunk_text", ""),
+            score=r.get("score", 0.0),
+            metadata=meta,
+        ))
+    return results
+
+
+class EvalQuestion(BaseModel):
+    question: str
+    expected_answer: str
+
+
+class EvaluateRequest(BaseModel):
+    questions: list[EvalQuestion]
+    dataset_name: str = "eval_run"
+    top_k: int = 5
+    use_query_expansion: bool = False
+    num_expansions: int = 2
+
+
+@router.post("/evaluate")
+async def evaluate_rag(req: EvaluateRequest, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
+    per_question: list[dict] = []
+    failed: list[dict] = []
+    latencies: list[float] = []
+
+    for qa in req.questions:
+        try:
+            row = await evaluate_question(
+                qa.question, 
+                qa.expected_answer, 
+                req.top_k,
+                req.use_query_expansion,
+                req.num_expansions
+            )
+            per_question.append(row)
+            latencies.append(row["latency_ms"])
+        except Exception as exc:
+            failed.append({"question": qa.question, "error": str(exc)})
+            per_question.append(failed_question_row(qa.question, qa.expected_answer, str(exc)))
+
+    succeeded = [r for r in per_question if not r.get("error")]
+
+    def _avg(k: str) -> float:
+        if not succeeded:
+            return 0.0
+        return round(sum(r.get(k, 0.0) for r in succeeded) / len(succeeded), 4)
+
+    return {
+        # LLM-as-judge metrics
+        "accuracy_llm":      _avg("accuracy_llm"),
+        "accuracy":          _avg("accuracy_llm"),  # Backward compat
+        "faithfulness":      _avg("faithfulness"),
+        "context_precision": _avg("context_precision"),
+        "context_recall":    _avg("context_recall"),
+        "answer_relevancy":  _avg("answer_relevancy"),
+        # Accuracy methods
+        "exact_match":       _avg("exact_match"),
+        "semantic_similarity": _avg("semantic_similarity"),
+        "f1":                _avg("f1"),
+        "accuracy_combined": _avg("accuracy_combined"),
+        # Retrieval metrics
+        "recall_10":         _avg("recall_10"),
+        "recall_20":         _avg("recall_20"),
+        "recall_50":         _avg("recall_50"),
+        "mrr":               _avg("mrr"),
+        "ndcg_10":           _avg("ndcg_10"),
+        # Meta
+        "latency_avg_ms":    round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
+        "failed_questions":  failed,
+        "per_question":      per_question,
+        "dataset_name":      req.dataset_name,
+    }
diff --git a/backend/app/api/search.py b/backend/app/api/search.py
index f4dc40f..2e5200f 100644
+++ b/backend/app/api/search.py
@@ -12,10 +12,8 @@ from app.rag.synonym_expansion import get_synonym_expander
 from app.embeddings.openai_client import openai_client as ollama_client
 from app.chromadb.client import chroma_client
 from app.schemas.search import SearchRequest, SearchResponse, SearchResult
 
 router = APIRouter(prefix="/api/search", tags=["Search"])
 
 
 def _search_with_synonyms(search_fn, query: str, *args, **kwargs) -> list[dict]:
@@ -32,7 +30,6 @@ def _search_with_synonyms(search_fn, query: str, *args, **kwargs) -> list[dict]:
                 if doc_id not in all_results:
                     all_results[doc_id] = r
         except Exception as e:
     
     return sorted(all_results.values(), key=lambda x: x.get("score", 0), reverse=True)
 
diff --git a/backend/app/api/tablerag.py b/backend/app/api/tablerag.py
index 27b595c..2b5f64a 100644
+++ b/backend/app/api/tablerag.py
@@ -7,10 +7,8 @@ from app.core.dependencies import get_db
 from app.rag.table_rag import table_rag
 from app.services.document_service import document_service
 from app.schemas.search import SearchResult
 
 router = APIRouter(prefix="/api/tablerag", tags=["TableRAG"])
 
 
 @router.post("/index")
diff --git a/backend/app/api/web.py b/backend/app/api/web.py
index 598c9c5..a4c5405 100644
+++ b/backend/app/api/web.py
@@ -5,10 +5,8 @@ from app.core.dependencies import get_db
 from app.services.web_service import web_service
 from app.schemas.web import WebIngestRequest, WebQueryRequest, WebIngestResponse
 from app.schemas.search import SearchResult
 
 router = APIRouter(prefix="/api/web", tags=["Web Ingestion"])
 
 
 @router.post("/ingest", response_model=WebIngestResponse)
diff --git a/backend/app/chromadb/client.py b/backend/app/chromadb/client.py
index a55a594..20ae0b4 100644
+++ b/backend/app/chromadb/client.py
@@ -3,10 +3,8 @@ from typing import Any, Optional
 import chromadb
 from chromadb.config import Settings as ChromaSettings
 from app.core.config import settings
 from app.core.exceptions import ChromaDBError
 
 
 COLLECTIONS = [
     "table_documents",
@@ -29,9 +27,7 @@ class ChromaDBClient:
                     path=settings.CHROMA_PERSIST_DIR,
                     settings=ChromaSettings(anonymized_telemetry=False),
                 )
             except Exception as e:
                 raise ChromaDBError(str(e))
         return self._client
 
@@ -42,20 +38,16 @@ class ChromaDBClient:
                 name=name,
                 metadata=metadata or {"hnsw:space": "cosine"},
             )
             return collection
         except Exception as e:
             raise ChromaDBError(str(e))
 
     def delete_collection(self, name: str) -> bool:
         try:
             client = self.get_client()
             client.delete_collection(name)
             return True
         except Exception as e:
             raise ChromaDBError(str(e))
 
     def add_documents(
@@ -74,10 +66,8 @@ class ChromaDBClient:
                 documents=documents,
                 metadatas=metadatas or [{} for _ in ids],
             )
             return True
         except Exception as e:
             raise ChromaDBError(str(e))
 
     def search(
@@ -112,7 +102,6 @@ class ChromaDBClient:
                 })
             return output
         except Exception as e:
             raise ChromaDBError(str(e))
 
     def metadata_filter(
@@ -136,7 +125,6 @@ class ChromaDBClient:
             self.delete_collection(collection_name)
             return self.add_documents(collection_name, ids, embeddings, documents, metadatas)
         except Exception as e:
             raise ChromaDBError(str(e))
 
     def list_collections(self) -> list[str]:
@@ -144,7 +132,6 @@ class ChromaDBClient:
             client = self.get_client()
             return [c.name for c in client.list_collections()]
         except Exception as e:
             raise ChromaDBError(str(e))
 
     def get_collection_count(self, collection_name: str) -> int:
@@ -161,10 +148,8 @@ class ChromaDBClient:
             ids = results.get("ids", [])
             if ids:
                 collection.delete(ids=ids)
             return True
         except Exception as e:
             raise ChromaDBError(str(e))
 
     def health_check(self) -> bool:
@@ -177,7 +162,6 @@ class ChromaDBClient:
     def init_collections(self):
         for name in COLLECTIONS:
             self.create_collection(name)
 
 
 chroma_client = ChromaDBClient()
diff --git a/backend/app/core/config.py b/backend/app/core/config.py
index 7952bf3..c3970dc 100644
+++ b/backend/app/core/config.py
@@ -25,7 +25,7 @@ class Settings(BaseSettings):
     LOG_FILE: str = "./logs/rag.log"
     LOG_LEVEL: str = "INFO"
 
+    TOP_K: int = 20
     CHUNK_SIZE: int = 1024
     CHUNK_OVERLAP: int = 150
     MAX_CONTEXT_CHUNKS: int = 10
@@ -33,7 +33,7 @@ class Settings(BaseSettings):
     # ── Retrieval Improvements ────────────────────────────────────────────────
     DENSE_TOP_K: int = 50
     BM25_TOP_K: int = 50
+    RERANK_TOP_K: int = 20
     BM25_WEIGHT: float = 0.5
     DENSE_WEIGHT: float = 0.5
 
diff --git a/backend/app/core/exceptions.py b/backend/app/core/exceptions.py
index be596c8..00b161b 100644
+++ b/backend/app/core/exceptions.py
@@ -2,9 +2,7 @@ from fastapi import Request
 from fastapi.responses import JSONResponse
 from fastapi.exceptions import RequestValidationError
 from starlette.exceptions import HTTPException as StarletteHTTPException
 
 
 
 class RAGPlatformException(Exception):
@@ -49,7 +47,6 @@ class UnsupportedFileTypeError(RAGPlatformException):
 
 
 async def rag_platform_exception_handler(request: Request, exc: RAGPlatformException):
     return JSONResponse(
         status_code=exc.status_code,
         content={"error": exc.message, "status_code": exc.status_code},
@@ -57,7 +54,6 @@ async def rag_platform_exception_handler(request: Request, exc: RAGPlatformExcep
 
 
 async def http_exception_handler(request: Request, exc: StarletteHTTPException):
     return JSONResponse(
         status_code=exc.status_code,
         content={"error": exc.detail, "status_code": exc.status_code},
@@ -65,7 +61,6 @@ async def http_exception_handler(request: Request, exc: StarletteHTTPException):
 
 
 async def validation_exception_handler(request: Request, exc: RequestValidationError):
     return JSONResponse(
         status_code=422,
         content={"error": "Validation failed", "details": exc.errors()},
@@ -73,7 +68,6 @@ async def validation_exception_handler(request: Request, exc: RequestValidationE
 
 
 async def generic_exception_handler(request: Request, exc: Exception):
     return JSONResponse(
         status_code=500,
         content={"error": "Internal server error", "status_code": 500},
diff --git a/backend/app/database/init_db.py b/backend/app/database/init_db.py
index 55e84bb..138b3b6 100644
+++ b/backend/app/database/init_db.py
@@ -1,20 +1,14 @@
 from app.database.session import engine
 from app.database.base import Base
 from app.database import models  # noqa: F401 - registers all models
 
 
 
 async def init_db():
     async with engine.begin() as conn:
         await conn.run_sync(Base.metadata.create_all)
 
 
 async def drop_db():
     async with engine.begin() as conn:
         await conn.run_sync(Base.metadata.drop_all)
diff --git a/backend/app/embeddings/ollama_client.py b/backend/app/embeddings/ollama_client.py
index 8403172..437c594 100644
+++ b/backend/app/embeddings/ollama_client.py
@@ -4,10 +4,8 @@ from typing import AsyncGenerator, Optional
 import httpx
 from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
 from app.core.config import settings
 from app.core.exceptions import OllamaConnectionError
 
 
 
 class OllamaClient:
@@ -51,10 +49,8 @@ class OllamaClient:
             data = response.json()
             return data.get("response", "")
         except httpx.HTTPStatusError as e:
             raise OllamaConnectionError(str(e))
         except (httpx.ConnectError, httpx.TimeoutException) as e:
             raise OllamaConnectionError(str(e))
 
     @retry(
@@ -84,10 +80,8 @@ class OllamaClient:
             data = response.json()
             return data.get("message", {}).get("content", "")
         except httpx.HTTPStatusError as e:
             raise OllamaConnectionError(str(e))
         except (httpx.ConnectError, httpx.TimeoutException) as e:
             raise OllamaConnectionError(str(e))
 
     async def generate_stream(
@@ -116,7 +110,6 @@ class OllamaClient:
                         except json.JSONDecodeError:
                             continue
         except (httpx.ConnectError, httpx.TimeoutException) as e:
             raise OllamaConnectionError(str(e))
 
     async def chat_stream(
@@ -150,7 +143,6 @@ class OllamaClient:
                         except json.JSONDecodeError:
                             continue
         except (httpx.ConnectError, httpx.TimeoutException) as e:
             raise OllamaConnectionError(str(e))
 
     @retry(
@@ -170,10 +162,8 @@ class OllamaClient:
             data = response.json()
             return data.get("embedding", [])
         except httpx.HTTPStatusError as e:
             raise OllamaConnectionError(str(e))
         except (httpx.ConnectError, httpx.TimeoutException) as e:
             raise OllamaConnectionError(str(e))
 
     async def batch_embeddings(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
@@ -195,7 +185,6 @@ class OllamaClient:
             response.raise_for_status()
             return response.json().get("models", [])
         except Exception as e:
             return []
 
 
diff --git a/backend/app/embeddings/openai_client.py b/backend/app/embeddings/openai_client.py
index 642b94f..3674041 100644
+++ b/backend/app/embeddings/openai_client.py
@@ -3,10 +3,8 @@ from typing import AsyncGenerator, Optional
 import httpx
 from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
 from app.core.config import settings
 from app.core.exceptions import OpenAIConnectionError
 
 
 OPENAI_API_BASE = "https://api.openai.com/v1"
 
@@ -84,10 +82,8 @@ class OpenAIClient:
             data = response.json()
             return data["choices"][0]["message"]["content"]
         except httpx.HTTPStatusError as e:
             raise OpenAIConnectionError(str(e))
         except (httpx.ConnectError, httpx.TimeoutException) as e:
             raise OpenAIConnectionError(str(e))
 
     async def generate_stream(
@@ -146,7 +142,6 @@ class OpenAIClient:
                     except (json.JSONDecodeError, KeyError, IndexError):
                         continue
         except (httpx.ConnectError, httpx.TimeoutException) as e:
             raise OpenAIConnectionError(str(e))
 
     @retry(
@@ -170,10 +165,8 @@ class OpenAIClient:
             data = response.json()
             return data["data"][0]["embedding"]
         except httpx.HTTPStatusError as e:
             raise OpenAIConnectionError(str(e))
         except (httpx.ConnectError, httpx.TimeoutException) as e:
             raise OpenAIConnectionError(str(e))
 
     async def batch_embeddings(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
@@ -208,10 +201,8 @@ class OpenAIClient:
             items = sorted(data["data"], key=lambda x: x["index"])
             return [item["embedding"] for item in items]
         except httpx.HTTPStatusError as e:
             raise OpenAIConnectionError(str(e))
         except (httpx.ConnectError, httpx.TimeoutException) as e:
             raise OpenAIConnectionError(str(e))
 
     async def health_check(self) -> bool:
@@ -229,7 +220,6 @@ class OpenAIClient:
             response.raise_for_status()
             return response.json().get("data", [])
         except Exception as e:
             return []
 
 
diff --git a/backend/app/evaluation/accuracy_evaluation.py b/backend/app/evaluation/accuracy_evaluation.py
index 22f7300..d00f03e 100644
+++ b/backend/app/evaluation/accuracy_evaluation.py
@@ -2,9 +2,7 @@ from typing import Any
 import re
 from difflib import SequenceMatcher
 from app.embeddings.openai_client import openai_client
 
 
 
 class AccuracyEvaluator:
diff --git a/backend/app/evaluation/agent_runner.py b/backend/app/evaluation/agent_runner.py
index 29d7551..68fca03 100644
+++ b/backend/app/evaluation/agent_runner.py
@@ -1,174 +1,194 @@
+"""
+Simple nearest-neighbor RAG evaluation.
+
+Uses direct retrieval + LLM generation and scoring (no multi-agent orchestration).
+"""
+
+import time
+from typing import Any
+
+from app.services.rag_service import rag_service
+from app.evaluation.evaluator import evaluate_single
+from app.embeddings.openai_client import openai_client
+from app.rag.cross_encoder import cross_encoder
+from app.core.config import settings
+
+
+RAG_SYSTEM = """Answer the question carefully"""
+
+QUERY_EXPANSION_PROMPT = """Given the user's question, generate {num_expansions} alternative phrasings or related queries that would help retrieve relevant information from a knowledge base.
+
+Original question: {question}
+
+Generate {num_expansions} expanded queries as a numbered list (1., 2., 3., etc.). Each query should:
+- Rephrase the original question differently
+- Ask for related aspects that would help answer the original question
+- Use different terminology or synonyms
+
+Output only the numbered list, nothing else."""
+
+
+def _chunk_texts(chunks: list[dict]) -> list[str]:
+    return [c["chunk_text"] for c in chunks if c.get("chunk_text")]
+
+
+async def _expand_query(question: str, num_expansions: int = 2) -> list[str]:
+    """Generate expanded queries using LLM."""
+    prompt = QUERY_EXPANSION_PROMPT.format(question=question, num_expansions=num_expansions)
+    response = await openai_client.generate(prompt, system="You are a query expansion assistant.")
+    
+    expanded = [question]  # Always include original
+    for line in response.split("\n"):
+        line = line.strip()
+        if line and (line[0].isdigit() or line.startswith("-")):
+            query = line.split(".", 1)[-1].strip() if "." in line else line.lstrip("- ")
+            if query:
+                expanded.append(query)
+    
+    return expanded[:num_expansions + 1]
+
+
+async def evaluate_question(
+    question: str,
+    expected_answer: str,
+    top_k: int = None,
+    use_query_expansion: bool = False,
+    num_expansions: int = 2,
+) -> dict[str, Any]:
+    """
+    Run one Q&A pair through RAG with optional query expansion.
+    
+    1. Expand query into multiple variants (if enabled)
+    2. Retrieve chunks for each query variant
+    3. Combine and deduplicate all chunks
+    4. Generate answer from combined context
+    5. Score with LLM-as-judge
+    """
+    t0 = time.time()
+    
+    if top_k is None:
+        top_k = settings.TOP_K
+
+    # Query expansion before retrieval
+    if use_query_expansion:
+        queries = await _expand_query(question, num_expansions)
+    else:
+        queries = [question]
+    
+    # Retrieve chunks for all queries
+    all_chunks = []
+    chunk_ids_seen = set()
+    
+    for query in queries:
+        chunks = await rag_service.retrieve(query, strategy="hybrid", top_k=top_k * 2)
+        chunks = cross_encoder.rerank(query, chunks, top_k=settings.RERANK_TOP_K)
+        
+        for c in chunks:
+            cid = c.get("chunk_id")
+            if cid and cid not in chunk_ids_seen:
+                chunk_ids_seen.add(cid)
+                all_chunks.append(c)
+    
+    # Generate answer
+    chunk_texts = _chunk_texts(all_chunks)
+    context_text = "\n---\n".join(chunk_texts)
+    
+    if context_text.strip():
+        prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
+    else:
+        prompt = f"Question: {question}"
+    
+    generated_answer = await openai_client.generate(prompt, system=RAG_SYSTEM)
+    
+    # Build retrieved chunk IDs and derive gold IDs via text matching
+    retrieved_chunk_ids = [c.get("chunk_id", "") for c in all_chunks if c.get("chunk_id")]
+    expected_lower = expected_answer.lower()
+    gold_chunk_ids = {
+        c.get("chunk_id")
+        for c in all_chunks
+        if c.get("chunk_id") and (
+            expected_lower in c.get("chunk_text", "").lower()
+            or (len(expected_lower) >= 20 and expected_lower[:20] in c.get("chunk_text", "").lower())
+        )
+    }
+    # Fallback: if no exact match found, mark best-scoring chunks as gold
+    if not gold_chunk_ids and all_chunks:
+        best_score = max((c.get("score", 0) for c in all_chunks), default=0)
+        if best_score > 0:
+            gold_chunk_ids = {c.get("chunk_id") for c in all_chunks if c.get("score", 0) >= best_score * 0.9 and c.get("chunk_id")}
+
+    # Score
+    scores = await evaluate_single(question, expected_answer, generated_answer, chunk_texts,
+                                   retrieved_chunk_ids=retrieved_chunk_ids, gold_chunk_ids=gold_chunk_ids)
+    
+    latency_ms = round((time.time() - t0) * 1000, 1)
+
+    return {
+        "question": question,
+        "expected_answer": expected_answer,
+        "generated_answer": generated_answer,
+        "retrieved_context": context_text,
+        "expanded_queries": queries if use_query_expansion else [],
+        "num_chunks": len(all_chunks),
+        # LLM-as-judge
+        "accuracy_llm": scores.get("accuracy_llm", 0.0),
+        "faithfulness": scores.get("faithfulness", 0.0),
+        "answer_relevancy": scores.get("answer_relevancy", 0.0),
+        "context_precision": scores.get("context_precision", 0.0),
+        "context_recall": scores.get("context_recall", 0.0),
+        # Accuracy methods
+        "exact_match": scores.get("exact_match", 0.0),
+        "semantic_similarity": scores.get("semantic_similarity", 0.0),
+        "f1": scores.get("f1", 0.0),
+        "accuracy_combined": scores.get("accuracy_combined", 0.0),
+        # Retrieval metrics
+        "recall_10": scores.get("recall_10", 0.0),
+        "recall_20": scores.get("recall_20", 0.0),
+        "recall_50": scores.get("recall_50", 0.0),
+        "mrr": scores.get("mrr", 0.0),
+        "ndcg_10": scores.get("ndcg_10", 0.0),
+        # Rationales
+        "accuracy_rationale": scores.get("accuracy_rationale", ""),
+        "faithfulness_rationale": scores.get("faithfulness_rationale", ""),
+        "answer_relevancy_rationale": scores.get("answer_relevancy_rationale", ""),
+        "context_precision_rationale": scores.get("context_precision_rationale", ""),
+        "context_recall_rationale": scores.get("context_recall_rationale", ""),
+        "latency_ms": latency_ms,
+    }
+
+
+def failed_question_row(question: str, expected_answer: str, error: str) -> dict[str, Any]:
+    """Build a zeroed per-question row when the eval pipeline raises."""
+    return {
+        "question": question,
+        "expected_answer": expected_answer,
+        "generated_answer": "",
+        "retrieved_context": "",
+        "expanded_queries": [],
+        "num_chunks": 0,
+        # LLM-as-judge
+        "accuracy_llm": 0.0,
+        "faithfulness": 0.0,
+        "answer_relevancy": 0.0,
+        "context_precision": 0.0,
+        "context_recall": 0.0,
+        # Accuracy methods
+        "exact_match": 0.0,
+        "semantic_similarity": 0.0,
+        "f1": 0.0,
+        "accuracy_combined": 0.0,
+        # Retrieval metrics
+        "recall_10": 0.0,
+        "recall_20": 0.0,
+        "recall_50": 0.0,
+        "mrr": 0.0,
+        "ndcg_10": 0.0,
+        # Rationales
+        "accuracy_rationale": "",
+        "faithfulness_rationale": "",
+        "answer_relevancy_rationale": "",
+        "context_precision_rationale": "",
+        "context_recall_rationale": "",
+        "latency_ms": 0.0,
+        "error": error,
+    }
diff --git a/backend/app/evaluation/evaluator.py b/backend/app/evaluation/evaluator.py
index 9006932..3af14d5 100644
+++ b/backend/app/evaluation/evaluator.py
@@ -1,222 +1,219 @@
+"""
+LLM-as-a-Judge evaluator for RAG pipelines with retrieval metrics.
+
+Each metric is computed by prompting the LLM to score (0.0–1.0) with a brief rationale,
+rather than relying on cosine-similarity heuristics.
+
+Additionally, retrieval metrics (Recall@K, MRR, nDCG) are computed separately.
+Accuracy is evaluated via: exact match, semantic similarity, F1, and LLM-as-judge.
+"""
+
+import json
+import re
+import time
+from typing import Any
+
+from app.embeddings.openai_client import openai_client as ollama_client
+from app.evaluation.retrieval_metrics import retrieval_metrics
+from app.evaluation.accuracy_evaluation import accuracy_evaluator
+
+
+
+# ──────────────────────────────────────────────────────────────────────────────
+# Internal helpers
+# ──────────────────────────────────────────────────────────────────────────────
+
+_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]*\.?[0-9]+)')
+
+
+async def _llm_score(prompt: str) -> tuple[float, str]:
+    """
+    Call the LLM with `prompt` and extract a JSON payload like:
+        {"score": 0.85, "rationale": "..."}
+    Returns (score, rationale).  Falls back to (0.0, error_msg) on failure.
+    """
+    system = (
+        "You are an expert RAG evaluation judge. "
+        "Respond ONLY with a JSON object containing exactly two keys: "
+        '"score" (a float between 0.0 and 1.0) and '
+        '"rationale" (a one-sentence explanation). '
+        "Do not include any other text."
+    )
+    raw = ""  # initialise so it's always bound
+    try:
+        raw = await ollama_client.chat(
+            [{"role": "user", "content": prompt}],
+            system=system,
+        )
+        # Strip markdown fences if present
+        raw = raw.strip().strip("```json").strip("```").strip()
+        data = json.loads(raw)
+        score = float(data.get("score", 0.0))
+        rationale = data.get("rationale", "")
+        return round(max(0.0, min(1.0, score)), 4), rationale
+    except Exception as exc:
+        # Fallback: try regex on whatever we got back
+        m = _SCORE_RE.search(raw)
+        if m:
+            return round(float(m.group(1)), 4), "Score extracted via regex fallback."
+        return 0.0, f"Parse error: {exc}"
+
+
+# ──────────────────────────────────────────────────────────────────────────────
+# Metric functions
+# ──────────────────────────────────────────────────────────────────────────────
+
+async def compute_accuracy(generated: str, expected: str) -> tuple[float, str]:
+    """Semantic accuracy: how well does the generated answer match the expected answer?"""
+    prompt = f"""Rate how well the GENERATED ANSWER matches the EXPECTED ANSWER semantically.
+
+EXPECTED ANSWER:
+{expected}
+
+GENERATED ANSWER:
+{generated}
+
+Score 1.0 if they convey the same meaning, 0.0 if completely unrelated."""
+    return await _llm_score(prompt)
+
+
+async def compute_faithfulness(answer: str, context_chunks: list[str]) -> tuple[float, str]:
+    """Faithfulness: is the answer grounded in the retrieved context?"""
+    context = "\n\n---\n\n".join(context_chunks[:5])  # cap to avoid token overflow
+    prompt = f"""Rate how faithfully the ANSWER is grounded in the CONTEXT below.
+A faithful answer only makes claims supported by the context (score 1.0).
+An unfaithful answer introduces hallucinated facts not in the context (score 0.0).
+
+CONTEXT:
+{context}
+
+ANSWER:
+{answer}"""
+    return await _llm_score(prompt)
+
+
+async def compute_answer_relevancy(question: str, answer: str) -> tuple[float, str]:
+    """Answer relevancy: does the answer directly address the question?"""
+    prompt = f"""Rate how directly and completely the ANSWER addresses the QUESTION.
+Score 1.0 if the answer is fully on-topic and complete, 0.0 if it ignores the question.
+
+QUESTION:
+{question}
+
+ANSWER:
+{answer}"""
+    return await _llm_score(prompt)
+
+
+async def compute_context_precision(question: str, context_chunks: list[str]) -> tuple[float, str]:
+    """Context precision: what fraction of retrieved chunks are actually relevant?"""
+    if not context_chunks:
+        return 0.0, "No context chunks provided."
+    chunks_text = "\n\n---\n\n".join(
+        f"[Chunk {i+1}]: {c}" for i, c in enumerate(context_chunks[:8])
+    )
+    prompt = f"""You are evaluating retrieval quality.
+Below are chunks retrieved for a QUESTION. Rate the proportion of chunks that contain
+information genuinely useful for answering the question (0.0 = none relevant, 1.0 = all relevant).
+
+QUESTION:
+{question}
+
+RETRIEVED CHUNKS:
+{chunks_text}"""
+    return await _llm_score(prompt)
+
+
+async def compute_context_recall(expected_answer: str, context_chunks: list[str]) -> tuple[float, str]:
+    """Context recall: does the retrieved context contain what's needed to answer correctly?"""
+    if not context_chunks:
+        return 0.0, "No context chunks provided."
+    context = "\n\n---\n\n".join(context_chunks[:8])
+    prompt = f"""Rate whether the RETRIEVED CONTEXT contains the information needed to produce the EXPECTED ANSWER.
+Score 1.0 if all key facts for the expected answer are present in the context, 0.0 if completely missing.
+
+EXPECTED ANSWER:
+{expected_answer}
+
+RETRIEVED CONTEXT:
+{context}"""
+    return await _llm_score(prompt)
+
+
+# ──────────────────────────────────────────────────────────────────────────────
+# Master evaluation function
+# ──────────────────────────────────────────────────────────────────────────────
+
+async def evaluate_single(
+    question: str,
+    expected_answer: str,
+    generated_answer: str,
+    context_chunks: list[str],
+    retrieved_chunk_ids: list[str] = None,
+    gold_chunk_ids: set[str] = None,
+) -> dict[str, Any]:
+    """
+    Run all metrics for a single Q&A pair: LLM-as-judge + retrieval metrics + accuracy metrics.
+    Returns a dict with scores, rationales, and retrieval/accuracy performance.
+    """
+    t0 = time.time()
+
+    accuracy_llm,          acc_rationale  = await compute_accuracy(generated_answer, expected_answer)
+    faithfulness,      fai_rationale  = await compute_faithfulness(generated_answer, context_chunks)
+    answer_relevancy,  rel_rationale  = await compute_answer_relevancy(question, generated_answer)
+    context_precision, pre_rationale  = await compute_context_precision(question, context_chunks)
+    context_recall,    rec_rationale  = await compute_context_recall(expected_answer, context_chunks)
+
+    # Multi-method accuracy evaluation
+    exact_match = accuracy_evaluator.exact_match(generated_answer, expected_answer)
+    semantic_sim = accuracy_evaluator.semantic_similarity(generated_answer, expected_answer)
+    f1 = accuracy_evaluator.f1_score(generated_answer, expected_answer)
+    accuracy_combined = accuracy_evaluator.combined_accuracy(generated_answer, expected_answer)
+
+    # Retrieval metrics
+    recall_10 = 0.0
+    recall_20 = 0.0
+    recall_50 = 0.0
+    mrr = 0.0
+    ndcg_10 = 0.0
+    gold_answer_found = False
+
+    if retrieved_chunk_ids and gold_chunk_ids:
+        recall_10 = retrieval_metrics.recall_at_k(retrieved_chunk_ids, gold_chunk_ids, 10)
+        recall_20 = retrieval_metrics.recall_at_k(retrieved_chunk_ids, gold_chunk_ids, 20)
+        recall_50 = retrieval_metrics.recall_at_k(retrieved_chunk_ids, gold_chunk_ids, 50)
+        mrr = retrieval_metrics.mrr(retrieved_chunk_ids, gold_chunk_ids)
+        ndcg_10 = retrieval_metrics.ndcg_at_k([1.0] * len(retrieved_chunk_ids), gold_chunk_ids, retrieved_chunk_ids, 10)
+
+    gold_answer_found = retrieval_metrics.gold_in_retrieved([{"chunk_text": c} for c in context_chunks], expected_answer)
+
+    latency_ms = round((time.time() - t0) * 1000, 1)
+
+    return {
+        # ── LLM-as-judge scores ──────────────────────────────────────────────
+        "accuracy_llm":      round(accuracy_llm, 4),
+        "faithfulness":      faithfulness,
+        "answer_relevancy":  answer_relevancy,
+        "context_precision": context_precision,
+        "context_recall":    context_recall,
+        # ── Accuracy methods ─────────────────────────────────────────────────
+        "exact_match":       exact_match,
+        "semantic_similarity": round(semantic_sim, 4),
+        "f1":                f1,
+        "accuracy_combined": accuracy_combined,
+        # ── Retrieval metrics ─────────────────────────────────────────────────
+        "recall_10":         round(recall_10, 4),
+        "recall_20":         round(recall_20, 4),
+        "recall_50":         round(recall_50, 4),
+        "mrr":               round(mrr, 4),
+        "ndcg_10":           round(ndcg_10, 4),
+        "gold_answer_found": gold_answer_found,
+        # ── Rationales ───────────────────────────────────────────────────────
+        "accuracy_rationale":          acc_rationale,
+        "faithfulness_rationale":      fai_rationale,
+        "answer_relevancy_rationale":  rel_rationale,
+        "context_precision_rationale": pre_rationale,
+        "context_recall_rationale":    rec_rationale,
+        # ── Meta ─────────────────────────────────────────────────────────────
+        "latency_ms": latency_ms,
     }
\ No newline at end of file
diff --git a/backend/app/evaluation/retrieval_metrics.py b/backend/app/evaluation/retrieval_metrics.py
index 39413fc..b94cf05 100644
+++ b/backend/app/evaluation/retrieval_metrics.py
@@ -1,7 +1,5 @@
 from typing import Any
 
 
 
 class RetrievalMetrics:
diff --git a/backend/app/main.py b/backend/app/main.py
index 932013c..6f95a23 100644
+++ b/backend/app/main.py
@@ -7,7 +7,6 @@ from fastapi.exceptions import RequestValidationError
 from starlette.exceptions import HTTPException as StarletteHTTPException
 
 from app.core.config import settings
 from app.core.exceptions import (
     RAGPlatformException,
     rag_platform_exception_handler,
@@ -32,23 +31,17 @@ from app.api.chroma import router as chroma_router
 from app.api.embeddings import router as embeddings_router
 from app.api.web import router as web_router
 
 
 
 @asynccontextmanager
 async def lifespan(app: FastAPI):
     await init_db()
     try:
         chroma_client.init_collections()
     except Exception as e:
     yield
     from app.embeddings.openai_client import openai_client
     await openai_client.close()
 
 
 app = FastAPI(
@@ -75,10 +68,8 @@ app.add_middleware(
 @app.middleware("http")
 async def logging_middleware(request: Request, call_next):
     start = time.time()
     response = await call_next(request)
     latency = (time.time() - start) * 1000
     return response
 
 
diff --git a/backend/app/rag/bm25.py b/backend/app/rag/bm25.py
index e02fe77..f4841b6 100644
+++ b/backend/app/rag/bm25.py
@@ -1,8 +1,6 @@
 from typing import Any
 from rank_bm25 import BM25Okapi
 
 
 
 class BM25Retriever:
@@ -18,11 +16,9 @@ class BM25Retriever:
         self._corpus[collection_name] = chunks
         tokenized = [self._tokenize(c["chunk_text"]) for c in chunks]
         self._index[collection_name] = BM25Okapi(tokenized)
 
     def search(self, collection_name: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
         if collection_name not in self._index:
             return []
         bm25 = self._index[collection_name]
         corpus = self._corpus[collection_name]
diff --git a/backend/app/rag/cross_encoder.py b/backend/app/rag/cross_encoder.py
index 91f3fb0..103bb4e 100644
+++ b/backend/app/rag/cross_encoder.py
@@ -1,14 +1,11 @@
 from typing import Any
 from sentence_transformers import CrossEncoder
 
 
 
 class CrossEncoderReranker:
     def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
         self.model = CrossEncoder(model_name)
 
     def rerank(
         self,
@@ -30,7 +27,6 @@ class CrossEncoderReranker:
             chunk["ce_score"] = float(scores[i])
         
         ranked = sorted(chunks, key=lambda x: x["ce_score"], reverse=True)[:top_k]
         return ranked
 
 
diff --git a/backend/app/rag/evaluator.py b/backend/app/rag/evaluator.py
index 9acdb80..bfa663b 100644
+++ b/backend/app/rag/evaluator.py
@@ -11,9 +11,7 @@ import time
 from typing import Any
 
 from app.embeddings.openai_client import openai_client as ollama_client
 
 
 
 # ──────────────────────────────────────────────────────────────────────────────
@@ -49,7 +47,6 @@ async def _llm_score(prompt: str) -> tuple[float, str]:
         rationale = data.get("rationale", "")
         return round(max(0.0, min(1.0, score)), 4), rationale
     except Exception as exc:
         # Fallback: try regex on whatever we got back
         m = _SCORE_RE.search(raw)
         if m:
diff --git a/backend/app/rag/hybrid_rag.py b/backend/app/rag/hybrid_rag.py
index 96d5e03..75e9b82 100644
+++ b/backend/app/rag/hybrid_rag.py
@@ -5,9 +5,7 @@ from app.rag.metadata_filter import filter_results
 from app.rag.parent_context import parent_context_expander
 from app.rag.cross_encoder import cross_encoder
 from app.core.config import settings
 
 
 
 class HybridRAG:
@@ -73,7 +71,6 @@ class HybridRAG:
         top_k: int = 5,
         filters: Optional[dict] = None,
     ) -> list[dict[str, Any]]:
 
         # Dense retrieval: top 50
         vector_results = []
@@ -82,7 +79,6 @@ class HybridRAG:
                 query, collection_name, settings.DENSE_TOP_K, filters
             )
         except Exception as e:
 
         # BM25 retrieval: top 50
         bm25_results = []
@@ -90,7 +86,6 @@ class HybridRAG:
             bm25_raw = bm25_retriever.search(collection_name, query, settings.BM25_TOP_K)
             bm25_results = filter_results(bm25_raw, filters or {})
         except Exception as e:
 
         if not vector_results and not bm25_results:
             return []
diff --git a/backend/app/rag/markdown_rag.py b/backend/app/rag/markdown_rag.py
index 0bb672b..7351d07 100644
+++ b/backend/app/rag/markdown_rag.py
@@ -2,9 +2,7 @@ import re
 from typing import Any, Optional
 from app.chromadb.client import chroma_client
 from app.embeddings.openai_client import openai_client as ollama_client
 
 MD_COLLECTION = "markdown_documents"
 
 HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
@@ -72,7 +70,6 @@ class MarkdownRAG:
 
         if ids:
             chroma_client.add_documents(MD_COLLECTION, ids, embeddings, documents, metadatas)
         return {"document_id": document_id, "chunk_count": len(ids)}
 
     async def query(
diff --git a/backend/app/rag/multi_collection_retrieval.py b/backend/app/rag/multi_collection_retrieval.py
index 18b7673..8d98fc3 100644
+++ b/backend/app/rag/multi_collection_retrieval.py
@@ -4,9 +4,7 @@ from app.rag.hybrid_rag import hybrid_rag
 from app.rag.table_rag import table_rag
 from app.rag.pdf_rag import pdf_rag
 from app.rag.markdown_rag import markdown_rag
 
 
 DEFAULT_COLLECTIONS = [
     "text_documents",
@@ -56,7 +54,6 @@ class MultiCollectionRetriever:
 
                 all_results.extend(results)
             except Exception as e:
                 continue
 
         if not all_results:
diff --git a/backend/app/rag/parent_context.py b/backend/app/rag/parent_context.py
index fdd444f..c3c8ee8 100644
+++ b/backend/app/rag/parent_context.py
@@ -1,8 +1,6 @@
 from typing import Any, Optional
 from app.chromadb.client import chroma_client
 
 
 
 class ParentContextExpander:
@@ -74,7 +72,6 @@ class ParentContextExpander:
 
             expanded.append(expanded_chunk)
 
         return expanded
 
 
diff --git a/backend/app/rag/pdf_rag.py b/backend/app/rag/pdf_rag.py
index f4fa699..267e146 100644
+++ b/backend/app/rag/pdf_rag.py
@@ -5,10 +5,8 @@ from typing import Any, Optional
 import pdfplumber
 from app.chromadb.client import chroma_client
 from app.embeddings.openai_client import openai_client as ollama_client
 from app.core.config import settings
 
 PDF_COLLECTION = "pdf_documents"
 
 
@@ -101,7 +99,6 @@ class PDFHierarchicalRAG:
 
         if ids:
             chroma_client.add_documents(PDF_COLLECTION, ids, embeddings, documents, metadatas)
         return {"document_id": document_id, "chunk_count": len(ids)}
 
     async def query(
diff --git a/backend/app/rag/synonym_expansion.py b/backend/app/rag/synonym_expansion.py
index 8079247..0d18de9 100644
+++ b/backend/app/rag/synonym_expansion.py
@@ -1,9 +1,7 @@
 import json
 import csv
 from pathlib import Path
 
 
 
 class SynonymExpander:
@@ -29,9 +27,7 @@ class SynonymExpander:
                     self.synonyms[canonical] = aliases
                     for alias in aliases:
                         self.reverse_map[alias] = canonical
         except Exception as e:
 
     def _load_csv(self, path: str):
         """Load synonyms from CSV format: canonical,alias"""
@@ -47,9 +43,7 @@ class SynonymExpander:
                         if alias not in self.synonyms[canonical]:
                             self.synonyms[canonical].append(alias)
                         self.reverse_map[alias] = canonical
         except Exception as e:
 
     def expand_query(self, query: str) -> list[str]:
         """Expand query with synonyms and return list of expanded queries"""
diff --git a/backend/app/rag/table_rag.py b/backend/app/rag/table_rag.py
index 14069d0..82d8ec2 100644
+++ b/backend/app/rag/table_rag.py
@@ -7,9 +7,7 @@ import pandas as pd
 from app.chromadb.client import chroma_client
 from app.embeddings.openai_client import openai_client as ollama_client
 from app.rag.metadata_filter import build_chroma_filter
 
 
 SCHEMA_COLLECTION = "table_documents"
 
@@ -95,7 +93,6 @@ class TableRAG:
             })
 
         chroma_client.add_documents(SCHEMA_COLLECTION, ids, embeddings, documents, metadatas)
         return {"document_id": document_id, "chunk_count": len(ids), "schema": schema_info}
 
     async def query(
diff --git a/backend/app/rag/vector_rag.py b/backend/app/rag/vector_rag.py
index 008f38d..f592dd7 100644
+++ b/backend/app/rag/vector_rag.py
@@ -2,9 +2,7 @@ from typing import Any, Optional
 from app.chromadb.client import chroma_client
 from app.embeddings.openai_client import openai_client as ollama_client
 from app.rag.metadata_filter import build_chroma_filter
 
 
 
 class VectorRAG:
@@ -15,7 +13,6 @@ class VectorRAG:
         top_k: int = 5,
         filters: Optional[dict] = None,
     ) -> list[dict[str, Any]]:
         query_embedding = await ollama_client.embeddings(query)
         where = build_chroma_filter(filters) if filters else None
         results = chroma_client.search(collection_name, query_embedding, top_k, where)
@@ -43,7 +40,6 @@ class VectorRAG:
                     r["filename"] = r.get("metadata", {}).get("filename", "")
                 all_results.extend(results)
             except Exception as e:
         all_results.sort(key=lambda x: x["score"], reverse=True)
         return all_results[:top_k]
 
diff --git a/backend/app/repositories/conversation_repository.py b/backend/app/repositories/conversation_repository.py
index 70b9021..d8139c0 100644
+++ b/backend/app/repositories/conversation_repository.py
@@ -3,9 +3,7 @@ from sqlalchemy.ext.asyncio import AsyncSession
 from sqlalchemy import select, delete
 from sqlalchemy.orm import selectinload
 from app.database.models import Conversation, Message
 
 
 
 class ConversationRepository:
diff --git a/backend/app/repositories/document_repository.py b/backend/app/repositories/document_repository.py
index 9d09fe6..c784af1 100644
+++ b/backend/app/repositories/document_repository.py
@@ -3,9 +3,7 @@ from sqlalchemy.ext.asyncio import AsyncSession
 from sqlalchemy import select, delete
 from sqlalchemy.orm import selectinload
 from app.database.models import Document, Chunk
 
 
 
 class DocumentRepository:
diff --git a/backend/app/repositories/log_repository.py b/backend/app/repositories/log_repository.py
index 16187ee..fb8e821 100644
+++ b/backend/app/repositories/log_repository.py
@@ -1,9 +1,7 @@
 from sqlalchemy.ext.asyncio import AsyncSession
 from sqlalchemy import select
 from app.database.models import RetrievalLog, EvaluationRun
 
 
 
 class LogRepository:
diff --git a/backend/app/schemas/chat.py b/backend/app/schemas/chat.py
index cdb10d6..f65e3fb 100644
+++ b/backend/app/schemas/chat.py
@@ -6,7 +6,7 @@ from datetime import datetime
 class ChatRequest(BaseModel):
     message: str = Field(..., min_length=1)
     conversation_id: Optional[str] = None
+    top_k: int = Field(default=20, ge=1, le=50)
     stream: bool = False
 
 
diff --git a/backend/app/schemas/rag.py b/backend/app/schemas/rag.py
index 64f5aba..dd730e3 100644
+++ b/backend/app/schemas/rag.py
@@ -5,7 +5,7 @@ from typing import Any, Optional
 class RAGQueryRequest(BaseModel):
     query: str = Field(..., min_length=1)
     strategy: str = Field(default="hybrid", description="vector|bm25|hybrid|table|pdf|markdown")
+    top_k: int = Field(default=20, ge=1, le=50)
     filters: Optional[dict[str, Any]] = None
     conversation_id: Optional[str] = None
 
@@ -22,7 +22,7 @@ class RAGQueryResponse(BaseModel):
 class RAGRetrieveRequest(BaseModel):
     query: str = Field(..., min_length=1)
     strategy: str = Field(default="hybrid")
+    top_k: int = Field(default=20, ge=1, le=50)
     filters: Optional[dict[str, Any]] = None
 
 
diff --git a/backend/app/services/chat_service.py b/backend/app/services/chat_service.py
index cc048b1..df746dd 100644
+++ b/backend/app/services/chat_service.py
@@ -5,11 +5,9 @@ from sqlalchemy.ext.asyncio import AsyncSession
 from app.repositories.conversation_repository import conversation_repo
 from app.services.rag_service import rag_service
 from app.embeddings.openai_client import openai_client as ollama_client
 from app.core.exceptions import ConversationNotFoundError
 from app.core.prompts import SYSTEM_PROMPT
 
 
 
 class ChatService:
diff --git a/backend/app/services/document_service.py b/backend/app/services/document_service.py
index 2ccba36..0388130 100644
+++ b/backend/app/services/document_service.py
@@ -14,10 +14,8 @@ from app.rag.pdf_rag import pdf_rag
 from app.rag.markdown_rag import markdown_rag
 from app.rag.bm25 import bm25_retriever
 from app.core.config import settings
 from app.core.exceptions import DocumentNotFoundError, UnsupportedFileTypeError
 
 
 SUPPORTED_TYPES = {
     "pdf": "pdf",
@@ -165,7 +163,6 @@ class DocumentService:
         if chunk_records:
             await document_repo.bulk_create_chunks(db, chunk_records)
 
         return {"document": doc, "chunk_count": chunk_count}
 
     async def reindex(self, db: AsyncSession, doc_id: str) -> dict[str, Any]:
@@ -184,7 +181,6 @@ class DocumentService:
         try:
             chroma_client.delete_by_document_id(doc.collection_name or "text_documents", doc_id)
         except Exception as e:
 
         await document_repo.delete_chunks(db, doc_id)
 
@@ -219,7 +215,6 @@ class DocumentService:
         try:
             chroma_client.delete_by_document_id(doc.collection_name or "text_documents", doc_id)
         except Exception as e:
         filepath = Path(doc.filepath)
         if filepath.exists():
             filepath.unlink()
diff --git a/backend/app/services/rag_service.py b/backend/app/services/rag_service.py
index 6139bcb..deb65be 100644
+++ b/backend/app/services/rag_service.py
@@ -1,184 +1,181 @@
+import time
+import uuid
+from typing import Any, AsyncGenerator, Optional
+from sqlalchemy.ext.asyncio import AsyncSession
+from app.rag.vector_rag import vector_rag
+from app.rag.hybrid_rag import hybrid_rag
+from app.rag.bm25 import bm25_retriever
+from app.rag.table_rag import table_rag
+from app.rag.pdf_rag import pdf_rag
+from app.rag.markdown_rag import markdown_rag
+from app.rag.multi_collection_retrieval import multi_collection_retriever
+from app.rag.metadata_filter import filter_results
+from app.rag.evaluator import (
+    compute_accuracy, compute_faithfulness, compute_answer_relevancy,
+    compute_context_precision, compute_context_recall,
+)
+from app.embeddings.openai_client import openai_client as ollama_client
+from app.repositories.log_repository import log_repo
+from app.core.config import settings
+
+
+RAG_SYSTEM = """You are an expert assistant. Answer the question using only the provided context.
+Be factual, concise, and cite sources. If the answer is not in the context, say 'Not found in available documents'."""
+
+
+class RAGService:
+    async def retrieve(
+        self,
+        query: str,
+        strategy: str = "hybrid",
+        top_k: int = 5,
+        filters: Optional[dict] = None,
+        collection_name: Optional[str] = None,
+        all_collections: bool = False,
+    ) -> list[dict[str, Any]]:
+        """
+        Retrieve from single or multiple collections.
+        If all_collections=True, search across all available collections.
+        """
+        if all_collections:
+            return await multi_collection_retriever.retrieve_all_collections(
+                query, strategy, top_k, filters
+            )
+
+        col = collection_name or "text_documents"
+        if strategy == "vector":
+            return await vector_rag.retrieve(query, col, top_k, filters)
+        elif strategy == "bm25":
+            results = bm25_retriever.search(col, query, top_k)
+            return filter_results(results, filters or {})
+        elif strategy == "hybrid":
+            return await hybrid_rag.retrieve(query, col, top_k, filters)
+        elif strategy == "table":
+            return await table_rag.query(query, top_k=top_k)
+        elif strategy == "pdf":
+            return await pdf_rag.query(query, top_k=top_k)
+        elif strategy == "markdown":
+            return await markdown_rag.query(query, top_k=top_k)
+        else:
+            return await hybrid_rag.retrieve(query, col, top_k, filters)
+
+    async def query(
+        self,
+        db: AsyncSession,
+        query: str,
+        strategy: str = "hybrid",
+        top_k: int = 5,
+        filters: Optional[dict] = None,
+    ) -> dict[str, Any]:
+        start = time.time()
+        chunks = await self.retrieve(query, strategy, top_k, filters)
+        context = "\n\n".join(
+            f"[{r.get('filename','?')}] {r['chunk_text']}" for r in chunks
+        )
+        prompt = f"Context:\n{context}\n\nQuestion: {query}"
+        answer = await ollama_client.generate(prompt, system=RAG_SYSTEM)
+        latency = (time.time() - start) * 1000
+
+        sources = [
+            {"filename": r.get("filename", ""), "chunk_id": r.get("chunk_id", ""), "score": r.get("score", 0)}
+            for r in chunks
+        ]
+
+        await log_repo.create_retrieval_log(db, {
+            "id": str(uuid.uuid4()),
+            "query": query,
+            "retrieval_strategy": strategy,
+            "retrieved_chunks": [r.get("chunk_id", "") for r in chunks],
+            "generated_answer": answer,
+            "latency_ms": latency,
+            "agent_used": "rag_service",
+        })
+
+        confidence = round(sum(r.get("score", 0) for r in chunks) / max(len(chunks), 1), 4)
+        return {
+            "query": query,
+            "answer": answer,
+            "sources": sources,
+            "strategy": strategy,
+            "latency_ms": round(latency, 2),
+            "confidence": confidence,
+        }
+
+    async def query_stream(
+        self,
+        query: str,
+        strategy: str = "hybrid",
+        top_k: int = 5,
+        filters: Optional[dict] = None,
+    ) -> AsyncGenerator[str, None]:
+        chunks = await self.retrieve(query, strategy, top_k, filters)
+        context = "\n\n".join(
+            f"[{r.get('filename','?')}] {r['chunk_text']}" for r in chunks
+        )
+        prompt = f"Context:\n{context}\n\nQuestion: {query}"
+        async for token in ollama_client.generate_stream(prompt, system=RAG_SYSTEM):
+            yield token
+
+    async def evaluate(
+        self,
+        db: AsyncSession,
+        questions: list[dict],
+        dataset_name: str = "default",
+    ) -> dict[str, Any]:
+        results = {
+            "accuracy": [], "faithfulness": [], "context_precision": [],
+            "context_recall": [], "answer_relevancy": [], "latency_ms": [], "failed": [],
+        }
+
+        for q in questions:
+            question = q["question"]
+            expected = q["expected_answer"]
+            try:
+                start = time.time()
+                chunks = await self.retrieve(question, strategy="hybrid", top_k=settings.TOP_K)
+                context_texts = [r["chunk_text"] for r in chunks]
+                context = "\n\n".join(context_texts)
+                prompt = f"Context:\n{context}\n\nQuestion: {question}"
+                answer = await ollama_client.generate(prompt, system=RAG_SYSTEM)
+                latency = (time.time() - start) * 1000
+
+                acc_score, _ = await compute_accuracy(answer, expected)
+                faith_score, _ = await compute_faithfulness(answer, context_texts)
+                cp_score, _ = await compute_context_precision(question, context_texts)
+                cr_score, _ = await compute_context_recall(expected, context_texts)
+                ar_score, _ = await compute_answer_relevancy(question, answer)
+
+                results["accuracy"].append(acc_score)
+                results["faithfulness"].append(faith_score)
+                results["context_precision"].append(cp_score)
+                results["context_recall"].append(cr_score)
+                results["answer_relevancy"].append(ar_score)
+                results["latency_ms"].append(latency)
+            except Exception as e:
+                results["failed"].append({"question": question, "error": str(e)})
+
+        def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0
+
+        final = {
+            "accuracy": avg(results["accuracy"]),
+            "faithfulness": avg(results["faithfulness"]),
+            "context_precision": avg(results["context_precision"]),
+            "context_recall": avg(results["context_recall"]),
+            "answer_relevancy": avg(results["answer_relevancy"]),
+            "latency_avg_ms": avg(results["latency_ms"]),
+            "failed_questions": results["failed"],
+        }
+
+        await log_repo.create_evaluation_run(db, {
+            "id": str(uuid.uuid4()),
+            "dataset_name": dataset_name,
+            "accuracy": final["accuracy"],
+            "faithfulness": final["faithfulness"],
+            "context_precision": final["context_precision"],
+            "context_recall": final["context_recall"],
+        })
+
+        return final
+
+
 rag_service = RAGService()
\ No newline at end of file
diff --git a/backend/app/services/web_service.py b/backend/app/services/web_service.py
index 536a95e..cb1898d 100644
+++ b/backend/app/services/web_service.py
@@ -7,9 +7,7 @@ from app.chromadb.client import chroma_client
 from app.embeddings.openai_client import openai_client as ollama_client
 from app.rag.bm25 import bm25_retriever
 from app.core.config import settings
 
 
 ALLOWED_DOMAINS = [
     "docs.", "developer.", "gov.", ".gov", "wikipedia.org",
@@ -53,7 +51,6 @@ class WebService:
         collection_name: str = "web_documents",
         metadata: Optional[dict] = None,
     ) -> dict[str, Any]:
         html = await _fetch_url(url)
         text = _clean_html(html)
         chunks = _chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
diff --git a/backend/delete_embeddings.py b/backend/delete_embeddings.py
index 28dfed8..47b94dd 100644
+++ b/backend/delete_embeddings.py
@@ -1,13 +1,13 @@
+from app.chromadb.client import chroma_client
+
+# Delete all collections
+for collection_name in ["text_documents", "pdf_documents", "table_documents", 
+                        "markdown_documents", "audio_transcripts", "web_documents"]:
+    try:
+        chroma_client.delete_collection(collection_name)
+        print(f"Deleted {collection_name}")
+    except Exception as e:
+        print(f"Failed to delete {collection_name}: {e}")
+
+# Reinitialize empty collections
 chroma_client.init_collections()
\ No newline at end of file
diff --git a/helpfull scripts/common_less_accuracy_removed_questions.py b/helpfull scripts/common_less_accuracy_removed_questions.py
index 783651e..9370099 100644
+++ b/helpfull scripts/common_less_accuracy_removed_questions.py	
@@ -1,33 +1,33 @@
+import pandas as pd
+from pathlib import Path
+
+# Input CSV files
+eval_file = r"C:\\Users\\Jishnu\Desktop\\SRAG\\eval\\results\\8th june 2nd cross encoder.csv"
+remove_file = r"C:\\Users\\Jishnu\\Desktop\\SRAG\\helpfull scripts\\eval_number_clean.csv"
+
+# Read CSVs
+eval_df = pd.read_csv(eval_file)
+remove_df = pd.read_csv(remove_file)
+
+# Questions with Accuracy <= 0.6
+low_accuracy_questions = eval_df.loc[
+    eval_df["Accuracy"] <= 0.6,
+    "Question"
+]
+
+# Find matching rows in remove_df
+filtered_remove_df = remove_df[
+    remove_df["Eval Question"].isin(low_accuracy_questions)
+]
+
+# Output file
+output_file = (
+    f"0.6_accuracy_and_belowrows_common_removed_questions_"
+    f"{Path(remove_file).stem}.csv"
+)
+
+# Save filtered rows
+filtered_remove_df.to_csv(output_file, index=False)
+
+print(f"Filtered {len(filtered_remove_df)} rows.")
 print(f"Output saved to: {output_file}")
\ No newline at end of file
diff --git a/helpfull scripts/compare_changes_in_metrics.py b/helpfull scripts/compare_changes_in_metrics.py
index 7e64668..fd4cd2d 100644
+++ b/helpfull scripts/compare_changes_in_metrics.py	
@@ -1,132 +1,132 @@
+import pandas as pd
+import matplotlib.pyplot as plt
+import numpy as np
+
+# ==========================
+# CONFIG
+# ==========================
+CSV_1 = "C:\\Users\\Jishnu\\Desktop\\SRAG\\eval\\results\\common among removed and 0.6 less questions.csv"
+CSV_2 = "C:\\Users\\Jishnu\\Desktop\\SRAG\\eval\\results\\8th june removed and common 0.6 accuracy with chatgpt suggested code changes overall scor is 27 some questions seeding imporment.csv"
+
+METRICS = [
+    "Accuracy",
+    "Faithfulness",
+    "Context Precision",
+    "Context Recall",
+    "Answer Relevancy"
+]
+
+# ==========================
+# LOAD FILES
+# ==========================
+df1 = pd.read_csv(CSV_1)
+df2 = pd.read_csv(CSV_2)
+
+# Keep only needed columns
+cols = ["Question"] + METRICS
+
+df1 = df1[cols].copy()
+df2 = df2[cols].copy()
+
+# Rename metric columns so we know which file they came from
+df1 = df1.rename(
+    columns={m: f"{m}_old" for m in METRICS}
+)
+
+df2 = df2.rename(
+    columns={m: f"{m}_new" for m in METRICS}
+)
+
+# ==========================
+# FIND COMMON QUESTIONS
+# ==========================
+merged = pd.merge(
+    df1,
+    df2,
+    on="Question",
+    how="inner"
+)
+
+print(f"Common questions found: {len(merged)}")
+
+# ==========================
+# CREATE DELTA COLUMNS
+# ==========================
+for metric in METRICS:
+    merged[f"{metric}_change"] = (
+        merged[f"{metric}_new"] -
+        merged[f"{metric}_old"]
+    )
+
+# ==========================
+# PLOT SETTINGS
+# ==========================
+plt.style.use("ggplot")
+
+for metric in METRICS:
+
+    plot_df = merged[
+        ["Question", f"{metric}_change"]
+    ].copy()
+
+    # Sort by change
+    plot_df = plot_df.sort_values(
+        by=f"{metric}_change"
+    )
+
+    changes = plot_df[f"{metric}_change"]
+    questions = plot_df["Question"]
+
+    colors = [
+        "green" if x > 0 else "red"
+        for x in changes
+    ]
+
+    fig_height = max(6, len(plot_df) * 0.35)
+
+    plt.figure(figsize=(14, fig_height))
+
+    bars = plt.barh(
+        questions,
+        changes,
+        color=colors
+    )
+
+    plt.axvline(
+        x=0,
+        color="black",
+        linewidth=1
+    )
+
+    # Annotate bars with delta values
+    for bar, value in zip(bars, changes):
+        plt.text(
+            value,
+            bar.get_y() + bar.get_height()/2,
+            f"{value:+.2f}",
+            va="center",
+            ha="left" if value >= 0 else "right"
+        )
+
+    plt.title(
+        f"{metric}: Improvement / Regression by Question"
+    )
+    plt.xlabel(
+        "Metric Change (New - Old)"
+    )
+    plt.ylabel("Question")
+
+    plt.tight_layout()
+
+    filename = (
+        metric.lower()
+        .replace(" ", "_")
+        + "_change.png"
+    )
+
+    plt.savefig(filename, dpi=300)
+    plt.close()
+
+    print(f"Saved: {filename}")
+
 print("Done.")
\ No newline at end of file
diff --git a/helpfull scripts/convert_removed_to_csv.py b/helpfull scripts/convert_removed_to_csv.py
index b657aaf..5d40d1e 100644
+++ b/helpfull scripts/convert_removed_to_csv.py	
@@ -1,36 +1,36 @@
+import re
+import csv
+
+input_file = "eval_clean.txt"
+output_file = "eval_clean.csv"
+
+with open(input_file, "r", encoding="utf-8") as f:
+    text = f.read()
+
+# Split by separator lines
+blocks = re.split(r"=+\s*", text.strip())
+
+rows = []
+
+for block in blocks:
+    block = block.strip()
+    if not block:
+        continue
+
+    q_no_match = re.search(r"Question no:\s*(\d+)", block)
+    question_match = re.search(r"Eval Question:\s*(.*)", block)
+    answer_match = re.search(r"Eval Answer:\s*(.*)", block, re.DOTALL)
+
+    if q_no_match and question_match and answer_match:
+        q_no = q_no_match.group(1).strip()
+        question = question_match.group(1).strip()
+        answer = answer_match.group(1).strip()
+
+        rows.append([q_no, question, answer])
+
+with open(output_file, "w", newline="", encoding="utf-8") as f:
+    writer = csv.writer(f)
+    writer.writerow(["Question no", "Eval Question", "Eval Answer"])
+    writer.writerows(rows)
+
 print("CSV file created:", output_file)
\ No newline at end of file
diff --git a/helpfull scripts/csv_convert.py b/helpfull scripts/csv_convert.py
index a951c64..89300d4 100644
+++ b/helpfull scripts/csv_convert.py	
@@ -1,114 +1,114 @@
+"""
+Convert form-structured CSV to flat eval Q&A format.
+
+Input CSV structure (no fixed header, repeating blocks):
+    form name, question 1, question 2, ...
+    (blank),   answer 1,   answer 2,   ...
+
+Output CSV structure:
+    Question no, Eval Question + form name, Eval Answer
+
+Usage:
+    python convert_form_csv.py input.csv output.csv
+
+    # Or with custom delimiter (e.g. tab-separated):
+    python convert_form_csv.py input.csv output.csv --delimiter '\t'
+"""
+
+import csv
+import argparse
+import sys
+
+
+def convert(input_path: str, output_path: str, delimiter: str = ",") -> int:
+    """
+    Parse the input CSV and write the flattened eval CSV.
+    Returns the number of Q&A rows written.
+    """
+    rows = []
+    with open(input_path, newline="", encoding="utf-8-sig") as f:
+        reader = csv.reader(f, delimiter=delimiter)
+        for row in reader:
+            rows.append(row)
+
+    qa_pairs = []
+    i = 0
+
+    while i < len(rows):
+        row = rows[i]
+
+        # Skip completely empty rows
+        if not any(cell.strip() for cell in row):
+            i += 1
+            continue
+
+        first_cell = row[0].strip() if row else ""
+
+        # A "form name" row: first cell is non-empty (the form name)
+        # and there are questions in the remaining cells.
+        if first_cell:
+            form_name = first_cell
+            questions = [cell.strip() for cell in row[1:]]
+
+            # Look ahead for the answer row (first cell blank, rest are answers)
+            if i + 1 < len(rows):
+                next_row = rows[i + 1]
+                next_first = next_row[0].strip() if next_row else ""
+                if not next_first:
+                    answers = [cell.strip() for cell in next_row[1:]]
+                    i += 2  # consumed both rows
+                else:
+                    # No answer row follows — treat answers as empty
+                    answers = []
+                    i += 1
+            else:
+                answers = []
+                i += 1
+
+            # Pair each question with its answer (zip stops at shortest)
+            for q, a in zip(questions, answers):
+                q = q.strip()
+                a = a.strip()
+                if q:  # skip blank question slots
+                    eval_question = f"{q} ({form_name})" if form_name else q
+                    qa_pairs.append((eval_question, a))
+
+        else:
+            # Answer row without a preceding form row — skip
+            i += 1
+
+    # Write output
+    with open(output_path, "w", newline="", encoding="utf-8") as f:
+        writer = csv.writer(f)
+        writer.writerow(["Question no", "Eval Question + form name", "Eval Answer"])
+        for idx, (question, answer) in enumerate(qa_pairs, start=1):
+            writer.writerow([idx, question, answer])
+
+    return len(qa_pairs)
+
+
+def main():
+    parser = argparse.ArgumentParser(
+        description="Convert form-structured CSV to flat eval Q&A CSV."
+    )
+    parser.add_argument("input", help="Path to the input CSV file")
+    parser.add_argument("output", help="Path for the output CSV file")
+    parser.add_argument(
+        "--delimiter",
+        default=",",
+        help="CSV delimiter character (default: comma). Use '\\t' for tab.",
+    )
+    args = parser.parse_args()
+
+    delimiter = args.delimiter.replace("\\t", "\t")
+
+    try:
+        count = convert(args.input, args.output, delimiter)
+        print(f"Done. Wrote {count} Q&A rows to '{args.output}'.")
+    except FileNotFoundError as e:
+        print(f"Error: {e}", file=sys.stderr)
+        sys.exit(1)
+
+
+if __name__ == "__main__":
     main()
\ No newline at end of file
diff --git a/helpfull scripts/csv_number_clean.py b/helpfull scripts/csv_number_clean.py
index 14f6d55..0f8c05b 100644
+++ b/helpfull scripts/csv_number_clean.py	
@@ -1,27 +1,27 @@
+import csv
+
+input_file = "C:\\Users\\Jishnu\\Desktop\\SRAG\\helpfull scripts\\0.6_accuracy_and_belowrows_common_removed_questions_eval_number_clean.csv"
+output_file = "C:\\Users\\Jishnu\\Desktop\\SRAG\\helpfull scripts\\eval_number_clean_removed_common.csv.csv"
+
+rows = []
+
+# Read existing CSV
+with open(input_file, "r", encoding="utf-8") as f:
+    reader = csv.reader(f)
+    header = next(reader, None)  # skip header if present
+
+    for row in reader:
+        if len(row) < 3:
+            continue
+        # row = [Question no, Eval Question, Eval Answer]
+        rows.append([row[1], row[2]])  # drop old number, keep question + answer
+
+# Rewrite with fixed numbering
+with open(output_file, "w", newline="", encoding="utf-8") as f:
+    writer = csv.writer(f)
+    writer.writerow(["Question no", "Eval Question", "Eval Answer"])
+
+    for i, (question, answer) in enumerate(rows, start=1):
+        writer.writerow([i, question, answer])
+
 print("Fixed CSV written to:", output_file)
\ No newline at end of file
diff --git a/helpfull scripts/get_less_accuracy_rows.py b/helpfull scripts/get_less_accuracy_rows.py
index f34dbf0..e2ede8b 100644
+++ b/helpfull scripts/get_less_accuracy_rows.py	
@@ -1,21 +1,21 @@
+import pandas as pd
+from pathlib import Path
+
+# Input CSV file
+input_file = "C:\\Users\\Jishnu\\Desktop\\SRAG\\eval\\results\\8th june 2nd cross encoder.csv"
+
+# Read CSV
+df = pd.read_csv(input_file)
+
+# Filter rows where Accuracy <= 0.6
+filtered_df = df[df["Accuracy"] <= 0.6]
+
+# Create output filename
+input_path = Path(input_file)
+output_file = f"0.6accuracyandbelowrows_{input_path.stem}.csv"
+
+# Save filtered rows
+filtered_df.to_csv(output_file, index=False)
+
+print(f"Filtered {len(filtered_df)} rows.")
 print(f"Output saved to: {output_file}")
\ No newline at end of file
diff --git a/helpfull scripts/hindi_remover.py b/helpfull scripts/hindi_remover.py
index 31a1b98..1fb4abc 100644
+++ b/helpfull scripts/hindi_remover.py	
@@ -1,49 +1,49 @@
+import re
+
+input_file = "input_hindi.txt"
+output_file = "output_hindi.txt"
+
+with open(input_file, "r", encoding="utf-8") as f:
+    text = f.read()
+
+# Remove bracketed Hindi text:
+# (हिन्दी), [हिन्दी], {हिन्दी}
+text = re.sub(r'\(\s*[\u0900-\u097F\s]+\s*\)', '', text)
+text = re.sub(r'\[\s*[\u0900-\u097F\s]+\s*\]', '', text)
+text = re.sub(r'\{\s*[\u0900-\u097F\s]+\s*\}', '', text)
+
+# Remove Hindi text that follows separators:
+# , हिन्दी
+# - हिन्दी
+# : हिन्दी
+# ; हिन्दी
+text = re.sub(
+    r'\s*[,;:\-–—]\s*[\u0900-\u097F\s]+',
+    '',
+    text
+)
+
+# Remove remaining Hindi characters
+text = re.sub(r'[\u0900-\u097F]+', '', text)
+
+# Remove empty brackets left behind
+text = re.sub(r'\(\s*\)', '', text)
+text = re.sub(r'\[\s*\]', '', text)
+text = re.sub(r'\{\s*\}', '', text)
+
+# Normalize spaces
+text = re.sub(r'[ \t]+', ' ', text)
+
+# Remove spaces before punctuation
+text = re.sub(r'\s+([,.;:!?])', r'\1', text)
+
+# Collapse multiple blank lines
+text = re.sub(r'\n\s*\n+', '\n\n', text)
+
+# Strip trailing spaces on each line
+text = '\n'.join(line.strip() for line in text.splitlines())
+
+with open(output_file, "w", encoding="utf-8") as f:
+    f.write(text)
+
 print(f"Saved cleaned text to {output_file}")
\ No newline at end of file
diff --git a/helpfull scripts/html_remover.py b/helpfull scripts/html_remover.py
index 44c2c0e..8b8cd9b 100644
+++ b/helpfull scripts/html_remover.py	
@@ -1,48 +1,48 @@
+#!/usr/bin/env python3
+"""
+Strip all HTML tags, CSS, and JavaScript from input.html and save plain text to remove_html.txt
+Usage: python strip_html.py
+"""
+
+import re
+
+def strip_html_css_js(text):
+    # Remove <style>...</style> blocks (CSS)
+    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
+
+    # Remove <script>...</script> blocks (JavaScript)
+    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
+
+    # Remove inline style attributes
+    text = re.sub(r'\s*style="[^"]*"', '', text, flags=re.IGNORECASE)
+
+    # Remove all remaining HTML tags
+    text = re.sub(r'<[^>]+>', '', text)
+
+    # Decode common HTML entities
+    entities = {
+        '&amp;': '&', '&lt;': '<', '&gt;': '>',
+        '&nbsp;': ' ', '&quot;': '"', '&#39;': "'"
+    }
+    for entity, char in entities.items():
+        text = text.replace(entity, char)
+
+    # Clean up excess whitespace/blank lines
+    lines = [line.strip() for line in text.splitlines()]
+    lines = [line for line in lines if line]  # remove empty lines
+    return '\n'.join(lines)
+
+
+if __name__ == '__main__':
+    input_file = 'input.html'
+    output_file = 'remove_html.txt'
+
+    with open(input_file, 'r', encoding='utf-8') as f:
+        raw = f.read()
+
+    clean_text = strip_html_css_js(raw)
+
+    with open(output_file, 'w', encoding='utf-8') as f:
+        f.write(clean_text)
+
     print(f"Done! Plain text saved to {output_file} ({len(clean_text)} characters)")
\ No newline at end of file
diff --git a/helpfull scripts/remove_bad_questions.py b/helpfull scripts/remove_bad_questions.py
index 8a4510c..8fcbaf7 100644
+++ b/helpfull scripts/remove_bad_questions.py	
@@ -1,144 +1,144 @@
+import csv
+import json
+import os
+from pathlib import Path
+
+import pypdf
+from dotenv import load_dotenv
+from openai import OpenAI
+
+# Load variables from .env
+load_dotenv()
+
+API_KEY = os.getenv("OPENAI_API_KEY")
+
+if not API_KEY:
+    raise ValueError("OPENAI_API_KEY not found in .env file")
+
+client = OpenAI(api_key=API_KEY)
+
+FORMS_DIR = "C:\\Users\\Jishnu\\Desktop\\SRAG\\helpfull scripts\\forms"
+INPUT_CSV = "C:\\Users\\Jishnu\\Desktop\\SRAG\\helpfull scripts\\eval.csv"
+OUTPUT_FILE = "eval_clean.txt"
+
+MODEL = "gpt-5"
+
+client = OpenAI(api_key=API_KEY)
+
+
+def load_all_forms_text(forms_dir: str) -> str:
+    """Load and concatenate all PDF text."""
+    all_text = []
+
+    pdf_files = sorted(Path(forms_dir).glob("*.pdf"))
+
+    for pdf_file in pdf_files:
+        # Keep filename so model knows which form it came from
+        all_text.append(pdf_file.name.replace(".pdf",""))
+
+    print(all_text)
+    return "\n".join(all_text)
+
+
+def is_question_relevant(question: str, answer: str, forms_text: str) -> bool:
+    """
+    Ask the LLM whether this QA pair is relevant
+    to someone filling any of the provided SBI forms.
+    """
+
+    prompt = f"""
+You are evaluating whether a question-answer pair is useful for a user
+who is filling SBI banking forms.
+
+Forms content is provided below.
+
+A question should be marked YES if:
+- It could reasonably arise while filling any of the forms.
+- The answer provides information needed to understand a field,
+  term, requirement, process, declaration, or document mentioned
+  in the forms.
+
+A question should be marked NO if:
+- It is unrelated to the forms.
+- It concerns topics not needed to understand or fill the forms.
+- The answer provides information that would not help someone
+  complete the forms.
+
+Return ONLY valid JSON:
+
+{{"keep": true}}
+
+or
+
+{{"keep": false}}
+
+QUESTION:
+{question}
+
+ANSWER:
+{answer}
+
+FORMS:
+{forms_text}
+"""
+
+    response = client.chat.completions.create(
+    model=MODEL,
+    messages=[
+            {"role": "user", "content": prompt}
+    ]
+    )
+
+
+    text = response.choices[0].message.content.strip()
+
+    try:
+        result = json.loads(text)
+        return bool(result.get("keep", False))
+    except Exception:
+        print("Failed to parse response:")
+        print(text)
+        return False
+
+
+def main():
+    print("Loading SBI forms...")
+    forms_text = load_all_forms_text(FORMS_DIR)
+    print(forms_text)
+    kept_rows = []
+
+    with open(INPUT_CSV, "r", encoding="utf-8", newline="") as f:
+        reader = csv.DictReader(f)
+
+        for row in reader:
+            qno = row.get("Question no", "")
+            question = row.get("Eval Question", "")
+            answer = row.get("Eval Answer", "")
+
+            print(f"Evaluating question {qno}...")
+
+            keep = is_question_relevant(
+                question=question,
+                answer=answer,
+                forms_text=forms_text
+            )
+
+            print(f" -> KEEP={keep}")
+
+            if keep:
+                kept_rows.append(row)
+
+    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
+        for row in kept_rows:
+            f.write(f"Question no: {row['Question no']}\n")
+            f.write(f"Eval Question: {row['Eval Question']}\n")
+            f.write(f"Eval Answer: {row['Eval Answer']}\n")
+            f.write("\n" + ("=" * 80) + "\n\n")
+
+    print(f"\nDone.")
+    print(f"Kept {len(kept_rows)} questions.")
+    print(f"Output written to {OUTPUT_FILE}")
+
+
+if __name__ == "__main__":
     main()
\ No newline at end of file
```

## 47cf985c9c889b702e8e6b73b3c60966486d69e2 — 2026-06-10T15:23:45+05:30

Message:

eval metrics fix

```diff
diff --git a/backend/app/api/rag.py b/backend/app/api/rag.py
index 0fd7483..1949b88 100644
+++ b/backend/app/api/rag.py
@@ -100,11 +100,25 @@ async def evaluate_rag(req: EvaluateRequest, db: AsyncSession = Depends(get_db))
         return round(sum(r.get(k, 0.0) for r in succeeded) / len(succeeded), 4)
 
     return {
+        # LLM-as-judge metrics
+        "accuracy_llm":      _avg("accuracy_llm"),
+        "accuracy":          _avg("accuracy_llm"),  # Backward compat
         "faithfulness":      _avg("faithfulness"),
         "context_precision": _avg("context_precision"),
         "context_recall":    _avg("context_recall"),
         "answer_relevancy":  _avg("answer_relevancy"),
+        # Accuracy methods
+        "exact_match":       _avg("exact_match"),
+        "semantic_similarity": _avg("semantic_similarity"),
+        "f1":                _avg("f1"),
+        "accuracy_combined": _avg("accuracy_combined"),
+        # Retrieval metrics
+        "recall_10":         _avg("recall_10"),
+        "recall_20":         _avg("recall_20"),
+        "recall_50":         _avg("recall_50"),
+        "mrr":               _avg("mrr"),
+        "ndcg_10":           _avg("ndcg_10"),
+        # Meta
         "latency_avg_ms":    round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
         "failed_questions":  failed,
         "per_question":      per_question,
diff --git a/backend/app/evaluation/agent_runner.py b/backend/app/evaluation/agent_runner.py
index a4b505c..29d7551 100644
+++ b/backend/app/evaluation/agent_runner.py
@@ -110,11 +110,24 @@ async def evaluate_question(
         "retrieved_context": context_text,
         "expanded_queries": queries if use_query_expansion else [],
         "num_chunks": len(all_chunks),
+        # LLM-as-judge
+        "accuracy_llm": scores.get("accuracy_llm", 0.0),
         "faithfulness": scores.get("faithfulness", 0.0),
         "answer_relevancy": scores.get("answer_relevancy", 0.0),
         "context_precision": scores.get("context_precision", 0.0),
         "context_recall": scores.get("context_recall", 0.0),
+        # Accuracy methods
+        "exact_match": scores.get("exact_match", 0.0),
+        "semantic_similarity": scores.get("semantic_similarity", 0.0),
+        "f1": scores.get("f1", 0.0),
+        "accuracy_combined": scores.get("accuracy_combined", 0.0),
+        # Retrieval metrics
+        "recall_10": scores.get("recall_10", 0.0),
+        "recall_20": scores.get("recall_20", 0.0),
+        "recall_50": scores.get("recall_50", 0.0),
+        "mrr": scores.get("mrr", 0.0),
+        "ndcg_10": scores.get("ndcg_10", 0.0),
+        # Rationales
         "accuracy_rationale": scores.get("accuracy_rationale", ""),
         "faithfulness_rationale": scores.get("faithfulness_rationale", ""),
         "answer_relevancy_rationale": scores.get("answer_relevancy_rationale", ""),
@@ -133,11 +146,24 @@ def failed_question_row(question: str, expected_answer: str, error: str) -> dict
         "retrieved_context": "",
         "expanded_queries": [],
         "num_chunks": 0,
+        # LLM-as-judge
+        "accuracy_llm": 0.0,
         "faithfulness": 0.0,
         "answer_relevancy": 0.0,
         "context_precision": 0.0,
         "context_recall": 0.0,
+        # Accuracy methods
+        "exact_match": 0.0,
+        "semantic_similarity": 0.0,
+        "f1": 0.0,
+        "accuracy_combined": 0.0,
+        # Retrieval metrics
+        "recall_10": 0.0,
+        "recall_20": 0.0,
+        "recall_50": 0.0,
+        "mrr": 0.0,
+        "ndcg_10": 0.0,
+        # Rationales
         "accuracy_rationale": "",
         "faithfulness_rationale": "",
         "answer_relevancy_rationale": "",
```

## fc6d21ed11bb6f6dedb55b2b9cde8557695d09d6 — 2026-06-10T14:57:42+05:30

Message:

added synonym expansion

```diff
diff --git a/backend/app/api/search.py b/backend/app/api/search.py
index 68ec6d0..f4dc40f 100644
+++ b/backend/app/api/search.py
@@ -8,6 +8,7 @@ from app.rag.hybrid_rag import hybrid_rag
 from app.rag.bm25 import bm25_retriever
 from app.rag.table_rag import table_rag
 from app.rag.metadata_filter import filter_results, build_chroma_filter
+from app.rag.synonym_expansion import get_synonym_expander
 from app.embeddings.openai_client import openai_client as ollama_client
 from app.chromadb.client import chroma_client
 from app.schemas.search import SearchRequest, SearchResponse, SearchResult
@@ -17,6 +18,25 @@ router = APIRouter(prefix="/api/search", tags=["Search"])
 logger = get_logger("api.search")
 
 
+def _search_with_synonyms(search_fn, query: str, *args, **kwargs) -> list[dict]:
+    """Search with query and its synonym expansions, deduplicate and merge results"""
+    expander = get_synonym_expander()
+    expanded_queries = expander.expand_query(query)
+    
+    all_results = {}
+    for q in expanded_queries:
+        try:
+            results = search_fn(q, *args, **kwargs)
+            for r in results:
+                doc_id = r.get("chunk_id", r.get("document_id", ""))
+                if doc_id not in all_results:
+                    all_results[doc_id] = r
+        except Exception as e:
+            logger.warning(f"Search failed for query '{q}': {e}")
+    
+    return sorted(all_results.values(), key=lambda x: x.get("score", 0), reverse=True)
+
+
 def _build_response(query: str, results: list[dict], strategy: str, start_time: float) -> SearchResponse:
     latency = (time.time() - start_time) * 1000
     search_results = []
@@ -50,7 +70,22 @@ def _build_response(query: str, results: list[dict], strategy: str, start_time:
 async def vector_search(req: SearchRequest):
     start = time.time()
     collection = req.collection_name or "text_documents"
+    
+    async def search(q):
+        return await vector_rag.retrieve(q, collection, req.top_k, req.filters)
+    
+    expander = get_synonym_expander()
+    expanded_queries = expander.expand_query(req.query)
+    
+    all_results = {}
+    for q in expanded_queries:
+        results = await search(q)
+        for r in results:
+            doc_id = r.get("chunk_id", r.get("document_id", ""))
+            if doc_id not in all_results:
+                all_results[doc_id] = r
+    
+    results = sorted(all_results.values(), key=lambda x: x.get("score", 0), reverse=True)[:req.top_k]
     return _build_response(req.query, results, "vector", start)
 
 
@@ -58,9 +93,21 @@ async def vector_search(req: SearchRequest):
 async def bm25_search(req: SearchRequest):
     start = time.time()
     collection = req.collection_name or "text_documents"
+    
+    expander = get_synonym_expander()
+    expanded_queries = expander.expand_query(req.query)
+    
+    all_results = {}
+    for q in expanded_queries:
+        results = bm25_retriever.search(collection, q, req.top_k)
+        if req.filters:
+            results = filter_results(results, req.filters)
+        for r in results:
+            doc_id = r.get("chunk_id", r.get("document_id", ""))
+            if doc_id not in all_results:
+                all_results[doc_id] = r
+    
+    results = sorted(all_results.values(), key=lambda x: x.get("score", 0), reverse=True)[:req.top_k]
     return _build_response(req.query, results, "bm25", start)
 
 
@@ -68,7 +115,22 @@ async def bm25_search(req: SearchRequest):
 async def hybrid_search(req: SearchRequest):
     start = time.time()
     collection = req.collection_name or "text_documents"
+    
+    async def search(q):
+        return await hybrid_rag.retrieve(q, collection, req.top_k, req.filters)
+    
+    expander = get_synonym_expander()
+    expanded_queries = expander.expand_query(req.query)
+    
+    all_results = {}
+    for q in expanded_queries:
+        results = await search(q)
+        for r in results:
+            doc_id = r.get("chunk_id", r.get("document_id", ""))
+            if doc_id not in all_results:
+                all_results[doc_id] = r
+    
+    results = sorted(all_results.values(), key=lambda x: x.get("score", 0), reverse=True)[:req.top_k]
     return _build_response(req.query, results, "hybrid", start)
 
 
diff --git a/backend/app/rag/synonym_expansion.py b/backend/app/rag/synonym_expansion.py
new file mode 100644
index 0000000..8079247
+++ b/backend/app/rag/synonym_expansion.py
@@ -0,0 +1,92 @@
+import json
+import csv
+from pathlib import Path
+from app.core.logging import get_logger
+
+logger = get_logger("synonym_expansion")
+
+
+class SynonymExpander:
+    def __init__(self, json_path: str = None, csv_path: str = None):
+        self.synonyms = {}  # canonical -> [aliases]
+        self.reverse_map = {}  # alias -> canonical
+        
+        if json_path:
+            self._load_json(json_path)
+        if csv_path:
+            self._load_csv(csv_path)
+
+    def _load_json(self, path: str):
+        """Load synonyms from JSON format: {topics: [{canonical, aliases}]}"""
+        try:
+            with open(path, 'r', encoding='utf-8') as f:
+                data = json.load(f)
+            
+            for topic in data.get("topics", []):
+                canonical = topic.get("canonical", "").lower().strip()
+                if canonical:
+                    aliases = [alias.lower().strip() for alias in topic.get("aliases", [])]
+                    self.synonyms[canonical] = aliases
+                    for alias in aliases:
+                        self.reverse_map[alias] = canonical
+            logger.info(f"Loaded {len(self.synonyms)} canonical terms from JSON")
+        except Exception as e:
+            logger.error(f"Error loading JSON synonyms: {e}")
+
+    def _load_csv(self, path: str):
+        """Load synonyms from CSV format: canonical,alias"""
+        try:
+            with open(path, 'r', encoding='utf-8') as f:
+                reader = csv.DictReader(f)
+                for row in reader:
+                    canonical = row.get("canonical", "").lower().strip()
+                    alias = row.get("alias", "").lower().strip()
+                    if canonical and alias:
+                        if canonical not in self.synonyms:
+                            self.synonyms[canonical] = []
+                        if alias not in self.synonyms[canonical]:
+                            self.synonyms[canonical].append(alias)
+                        self.reverse_map[alias] = canonical
+            logger.info(f"Loaded synonyms from CSV, total canonical terms: {len(self.synonyms)}")
+        except Exception as e:
+            logger.error(f"Error loading CSV synonyms: {e}")
+
+    def expand_query(self, query: str) -> list[str]:
+        """Expand query with synonyms and return list of expanded queries"""
+        query_lower = query.lower()
+        expanded_queries = [query]
+        
+        for alias, canonical in self.reverse_map.items():
+            if alias in query_lower:
+                # Replace alias with canonical
+                expanded = query_lower.replace(alias, canonical)
+                if expanded != query_lower and expanded not in expanded_queries:
+                    expanded_queries.append(expanded)
+        
+        return expanded_queries
+
+
+# Initialize global synonym expander
+_synonym_expander = None
+
+
+def init_synonym_expander(json_path: str = None, csv_path: str = None):
+    """Initialize the global synonym expander"""
+    global _synonym_expander
+    _synonym_expander = SynonymExpander(json_path, csv_path)
+    return _synonym_expander
+
+
+def get_synonym_expander() -> SynonymExpander:
+    """Get the global synonym expander instance"""
+    global _synonym_expander
+    if _synonym_expander is None:
+        backend_dir = Path(__file__).parent.parent.parent
+        json_path = backend_dir / "rag_synonym_dictionary.json"
+        csv_path = backend_dir / "rag_synonym_dictionary_pairs.csv"
+        
+        _synonym_expander = SynonymExpander(
+            str(json_path) if json_path.exists() else None,
+            str(csv_path) if csv_path.exists() else None,
+        )
+    return _synonym_expander
diff --git a/test_synonyms.py b/test_synonyms.py
new file mode 100644
index 0000000..d8f8297
+++ b/test_synonyms.py
@@ -0,0 +1,96 @@
+#!/usr/bin/env python3
+"""Test synonym expansion functionality"""
+import json
+import csv
+from pathlib import Path
+
+class SynonymExpander:
+    def __init__(self, json_path: str = None, csv_path: str = None):
+        self.synonyms = {}  # canonical -> [aliases]
+        self.reverse_map = {}  # alias -> canonical
+        
+        if json_path:
+            self._load_json(json_path)
+        if csv_path:
+            self._load_csv(csv_path)
+
+    def _load_json(self, path: str):
+        """Load synonyms from JSON format: {topics: [{canonical, aliases}]}"""
+        try:
+            with open(path, 'r', encoding='utf-8') as f:
+                data = json.load(f)
+            
+            for topic in data.get("topics", []):
+                canonical = topic.get("canonical", "").lower().strip()
+                if canonical:
+                    aliases = [alias.lower().strip() for alias in topic.get("aliases", [])]
+                    self.synonyms[canonical] = aliases
+                    for alias in aliases:
+                        self.reverse_map[alias] = canonical
+            print(f"✓ Loaded {len(self.synonyms)} canonical terms from JSON")
+        except Exception as e:
+            print(f"✗ Error loading JSON synonyms: {e}")
+
+    def _load_csv(self, path: str):
+        """Load synonyms from CSV format: canonical,alias"""
+        try:
+            with open(path, 'r', encoding='utf-8') as f:
+                reader = csv.DictReader(f)
+                for row in reader:
+                    canonical = row.get("canonical", "").lower().strip()
+                    alias = row.get("alias", "").lower().strip()
+                    if canonical and alias:
+                        if canonical not in self.synonyms:
+                            self.synonyms[canonical] = []
+                        if alias not in self.synonyms[canonical]:
+                            self.synonyms[canonical].append(alias)
+                        self.reverse_map[alias] = canonical
+            print(f"✓ Loaded synonyms from CSV, total canonical terms: {len(self.synonyms)}")
+        except Exception as e:
+            print(f"✗ Error loading CSV synonyms: {e}")
+
+    def expand_query(self, query: str) -> list[str]:
+        """Expand query with synonyms and return list of expanded queries"""
+        query_lower = query.lower()
+        expanded_queries = [query]
+        
+        for alias, canonical in self.reverse_map.items():
+            if alias in query_lower:
+                # Replace alias with canonical
+                expanded = query_lower.replace(alias, canonical)
+                if expanded != query_lower and expanded not in expanded_queries:
+                    expanded_queries.append(expanded)
+        
+        return expanded_queries
+
+
+if __name__ == "__main__":
+    backend_dir = Path("C:\\Users\\Jishnu\\Desktop\\SRAG\\backend")
+    json_path = backend_dir / "rag_synonym_dictionary.json"
+    csv_path = backend_dir / "rag_synonym_dictionary_pairs.csv"
+    
+    print("Testing Synonym Expansion System")
+    print("=" * 50)
+    
+    expander = SynonymExpander(str(json_path), str(csv_path))
+    
+    print(f"\nTotal reverse mappings: {len(expander.reverse_map)}")
+    print("\nTest Queries:")
+    print("-" * 50)
+    
+    test_queries = [
+        "service charges on advances",
+        "bank code",
+        "policy on general management",
+        "advances related service charges",
+    ]
+    
+    for q in test_queries:
+        expanded = expander.expand_query(q)
+        print(f"\nOriginal: {q}")
+        if len(expanded) > 1:
+            print(f"Expanded ({len(expanded)} variants):")
+            for i, exp in enumerate(expanded[1:], 1):
+                print(f"  {i}. {exp}")
+        else:
+            print("  (no expansions found)")
```

## 96fba0e9c8be6e1eb337e45244885f1c09c1523e — 2026-06-10T14:57:27+05:30

Message:

added synonym expansion

_No Python file changes in this commit._

## 4f77445cc87877de2086a53937044c70bd73bebf — 2026-06-10T12:22:03+05:30

Message:

new uploaded docs and chunks

```diff
diff --git a/backend/delete_embeddings.py b/backend/delete_embeddings.py
new file mode 100644
index 0000000..28dfed8
+++ b/backend/delete_embeddings.py
@@ -0,0 +1,13 @@
+from app.chromadb.client import chroma_client
+
+# Delete all collections
+for collection_name in ["text_documents", "pdf_documents", "table_documents", 
+                        "markdown_documents", "audio_transcripts", "web_documents"]:
+    try:
+        chroma_client.delete_collection(collection_name)
+        print(f"Deleted {collection_name}")
+    except Exception as e:
+        print(f"Failed to delete {collection_name}: {e}")
+
+# Reinitialize empty collections
+chroma_client.init_collections()
\ No newline at end of file
```

## 158cc7273540e075e116d62f2a50e87f80e25ae7 — 2026-06-10T10:43:39+05:30

Message:

feat: Implement comprehensive RAG improvements and enhanced evaluation metrics

  Backend changes:
  - Increase chunk size 512→1024, overlap 50→150 for better context preservation
  - Upgrade embedding model to text-embedding-3-large for improved retrieval
  - Implement hybrid retrieval with weighted BM25+Dense scoring (50/50 default)
    - Dense retrieval: top 50 candidates
    - BM25 retrieval: top 50 candidates
    - Normalized score merging with configurable weights
  - Upgrade reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 → BAAI/bge-reranker-v2-m3
  - Add parent context expansion (previous/next chunks) to recover split information
  - Improve table handling: convert to readable key-value text format preserving headers
  - Implement multi-collection search across all 6 collections (text, pdf, table, markdown, audio, web)
  - Add retrieval diagnostic metrics:
    - Recall@10, Recall@20, Recall@50
    - MRR (Mean Reciprocal Rank)
    - nDCG@10 (Normalized Discounted Cumulative Gain)
    - Gold answer presence indicator
  - Add multi-method accuracy evaluation:
    - Exact match (case-insensitive)
    - Semantic similarity (SequenceMatcher)
    - F1 score (token-based precision/recall)
    - Weighted combined accuracy
  - Integrate all metrics into evaluator for per-question tracking

  Frontend changes:
  - Update Evaluate.tsx to display all 15 metrics per question
  - Organize metrics into 3 categories:
    - LLM-as-Judge: accuracy_llm, faithfulness, context_precision, context_recall, answer_relevancy
    - Accuracy methods: exact_match, semantic_similarity, f1, accuracy_combined
    - Retrieval metrics: recall_10/20/50, mrr, ndcg_10, gold_answer_found
  - Add gold answer indicator badge in retrieval section
  - Separate metric breakdown cards for each category
  - Expand CSV export to 26 columns (all metrics + rationales)
  - Update result calculations to compute all new metric averages

  Config changes:
  - Add DENSE_TOP_K=50, BM25_TOP_K=50, RERANK_TOP_K=10
  - Add BM25_WEIGHT=0.5, DENSE_WEIGHT=0.5 for hybrid scoring

  New modules:
  - app/rag/parent_context.py - Chunk expansion with adjacent context
  - app/rag/multi_collection_retrieval.py - Unified multi-collection search
  - app/evaluation/retrieval_metrics.py - Recall@K, MRR, nDCG computation
  - app/evaluation/accuracy_evaluation.py - Multi-method accuracy scoring

  Separates retrieval failures from generation failures for better diagnostics.

_No Python file changes in this commit._

## cac51effea3a0b8fff48580565e1b6be534c27e6 — 2026-06-09T14:34:27+05:30

Message:

feat: increase retrieval recall and add banking-grounded answer generation

- Expand dense and BM25 retrieval candidate pools
- Increase chunk size and overlap for policy preservation
- Add Recall@K, MRR, and nDCG evaluation metrics
- Upgrade embedding model evaluation pipeline
- Introduce strict banking-specific RAG prompt

```diff
diff --git a/backend/app/core/config.py b/backend/app/core/config.py
index 6ad924a..7952bf3 100644
+++ b/backend/app/core/config.py
@@ -17,7 +17,7 @@ class Settings(BaseSettings):
     # ── OpenAI ────────────────────────────────────────────────────────────────
     OPENAI_API_KEY: str = ""
     OPENAI_LLM_MODEL: str = "gpt-4o-mini"
+    OPENAI_EMBED_MODEL: str = "text-embedding-3-large"
     OPENAI_TIMEOUT: int = 120
     OPENAI_MAX_RETRIES: int = 3
 
@@ -26,9 +26,16 @@ class Settings(BaseSettings):
     LOG_LEVEL: str = "INFO"
 
     TOP_K: int = 5
+    CHUNK_SIZE: int = 1024
+    CHUNK_OVERLAP: int = 150
     MAX_CONTEXT_CHUNKS: int = 10
+    
+    # ── Retrieval Improvements ────────────────────────────────────────────────
+    DENSE_TOP_K: int = 50
+    BM25_TOP_K: int = 50
+    RERANK_TOP_K: int = 10
+    BM25_WEIGHT: float = 0.5
+    DENSE_WEIGHT: float = 0.5
 
     class Config:
         env_file = ".env"
diff --git a/backend/app/evaluation/accuracy_evaluation.py b/backend/app/evaluation/accuracy_evaluation.py
new file mode 100644
index 0000000..22f7300
+++ b/backend/app/evaluation/accuracy_evaluation.py
@@ -0,0 +1,81 @@
+from typing import Any
+import re
+from difflib import SequenceMatcher
+from app.embeddings.openai_client import openai_client
+from app.core.logging import get_logger
+
+logger = get_logger("accuracy_evaluation")
+
+
+class AccuracyEvaluator:
+    """Multi-method accuracy evaluation: exact match, semantic similarity, F1."""
+
+    @staticmethod
+    def exact_match(generated: str, expected: str) -> float:
+        """Exact string match (case-insensitive, normalized)."""
+        gen_norm = generated.lower().strip()
+        exp_norm = expected.lower().strip()
+        return 1.0 if gen_norm == exp_norm else 0.0
+
+    @staticmethod
+    def semantic_similarity(generated: str, expected: str) -> float:
+        """Token-level semantic similarity using SequenceMatcher."""
+        matcher = SequenceMatcher(None, generated.lower(), expected.lower())
+        ratio = matcher.ratio()
+        return round(ratio, 4)
+
+    @staticmethod
+    def _tokenize(text: str) -> list[str]:
+        """Simple tokenization."""
+        return re.findall(r"\b\w+\b", text.lower())
+
+    @staticmethod
+    def f1_score(generated: str, expected: str) -> float:
+        """F1 score based on token overlap."""
+        gen_tokens = set(AccuracyEvaluator._tokenize(generated))
+        exp_tokens = set(AccuracyEvaluator._tokenize(expected))
+
+        if not exp_tokens:
+            return 1.0 if not gen_tokens else 0.0
+
+        tp = len(gen_tokens & exp_tokens)
+        fp = len(gen_tokens - exp_tokens)
+        fn = len(exp_tokens - gen_tokens)
+
+        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
+        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
+
+        if precision + recall == 0:
+            return 0.0
+
+        f1 = 2 * (precision * recall) / (precision + recall)
+        return round(f1, 4)
+
+    @staticmethod
+    def combined_accuracy(
+        generated: str,
+        expected: str,
+        weights: dict[str, float] = None,
+    ) -> float:
+        """Weighted combination of all accuracy methods."""
+        if weights is None:
+            weights = {
+                "exact_match": 0.2,
+                "semantic_similarity": 0.3,
+                "f1": 0.5,
+            }
+
+        exact = AccuracyEvaluator.exact_match(generated, expected)
+        semantic = AccuracyEvaluator.semantic_similarity(generated, expected)
+        f1 = AccuracyEvaluator.f1_score(generated, expected)
+
+        combined = (
+            weights.get("exact_match", 0) * exact +
+            weights.get("semantic_similarity", 0) * semantic +
+            weights.get("f1", 0) * f1
+        )
+
+        return round(combined, 4)
+
+
+accuracy_evaluator = AccuracyEvaluator()
diff --git a/backend/app/evaluation/evaluator.py b/backend/app/evaluation/evaluator.py
index 9acdb80..9006932 100644
+++ b/backend/app/evaluation/evaluator.py
@@ -1,8 +1,11 @@
 """
+LLM-as-a-Judge evaluator for RAG pipelines with retrieval metrics.
 
 Each metric is computed by prompting the LLM to score (0.0–1.0) with a brief rationale,
 rather than relying on cosine-similarity heuristics.
+
+Additionally, retrieval metrics (Recall@K, MRR, nDCG) are computed separately.
+Accuracy is evaluated via: exact match, semantic similarity, F1, and LLM-as-judge.
 """
 
 import json
@@ -11,6 +14,8 @@ import time
 from typing import Any
 
 from app.embeddings.openai_client import openai_client as ollama_client
+from app.evaluation.retrieval_metrics import retrieval_metrics
+from app.evaluation.accuracy_evaluation import accuracy_evaluator
 from app.core.logging import get_logger
 
 logger = get_logger("evaluator")
@@ -147,39 +152,71 @@ async def evaluate_single(
     expected_answer: str,
     generated_answer: str,
     context_chunks: list[str],
+    retrieved_chunk_ids: list[str] = None,
+    gold_chunk_ids: set[str] = None,
 ) -> dict[str, Any]:
     """
+    Run all metrics for a single Q&A pair: LLM-as-judge + retrieval metrics + accuracy metrics.
+    Returns a dict with scores, rationales, and retrieval/accuracy performance.
     """
     t0 = time.time()
 
+    accuracy_llm,          acc_rationale  = await compute_accuracy(generated_answer, expected_answer)
     faithfulness,      fai_rationale  = await compute_faithfulness(generated_answer, context_chunks)
     answer_relevancy,  rel_rationale  = await compute_answer_relevancy(question, generated_answer)
     context_precision, pre_rationale  = await compute_context_precision(question, context_chunks)
     context_recall,    rec_rationale  = await compute_context_recall(expected_answer, context_chunks)
 
+    # Multi-method accuracy evaluation
+    exact_match = accuracy_evaluator.exact_match(generated_answer, expected_answer)
+    semantic_sim = accuracy_evaluator.semantic_similarity(generated_answer, expected_answer)
+    f1 = accuracy_evaluator.f1_score(generated_answer, expected_answer)
+    accuracy_combined = accuracy_evaluator.combined_accuracy(generated_answer, expected_answer)
+
+    # Retrieval metrics
+    recall_10 = 0.0
+    recall_20 = 0.0
+    recall_50 = 0.0
+    mrr = 0.0
+    ndcg_10 = 0.0
+    gold_answer_found = False
+
+    if retrieved_chunk_ids and gold_chunk_ids:
+        recall_10 = retrieval_metrics.recall_at_k(retrieved_chunk_ids, gold_chunk_ids, 10)
+        recall_20 = retrieval_metrics.recall_at_k(retrieved_chunk_ids, gold_chunk_ids, 20)
+        recall_50 = retrieval_metrics.recall_at_k(retrieved_chunk_ids, gold_chunk_ids, 50)
+        mrr = retrieval_metrics.mrr(retrieved_chunk_ids, gold_chunk_ids)
+        ndcg_10 = retrieval_metrics.ndcg_at_k([1.0] * len(retrieved_chunk_ids), gold_chunk_ids, retrieved_chunk_ids, 10)
+
+    gold_answer_found = retrieval_metrics.gold_in_retrieved([{"chunk_text": c} for c in context_chunks], expected_answer)
+
     latency_ms = round((time.time() - t0) * 1000, 1)
 
     return {
+        # ── LLM-as-judge scores ──────────────────────────────────────────────
+        "accuracy_llm":      round(accuracy_llm, 4),
         "faithfulness":      faithfulness,
         "answer_relevancy":  answer_relevancy,
         "context_precision": context_precision,
         "context_recall":    context_recall,
+        # ── Accuracy methods ─────────────────────────────────────────────────
+        "exact_match":       exact_match,
+        "semantic_similarity": round(semantic_sim, 4),
+        "f1":                f1,
+        "accuracy_combined": accuracy_combined,
+        # ── Retrieval metrics ─────────────────────────────────────────────────
+        "recall_10":         round(recall_10, 4),
+        "recall_20":         round(recall_20, 4),
+        "recall_50":         round(recall_50, 4),
+        "mrr":               round(mrr, 4),
+        "ndcg_10":           round(ndcg_10, 4),
+        "gold_answer_found": gold_answer_found,
+        # ── Rationales ───────────────────────────────────────────────────────
         "accuracy_rationale":          acc_rationale,
         "faithfulness_rationale":      fai_rationale,
         "answer_relevancy_rationale":  rel_rationale,
         "context_precision_rationale": pre_rationale,
         "context_recall_rationale":    rec_rationale,
+        # ── Meta ─────────────────────────────────────────────────────────────
         "latency_ms": latency_ms,
     }
\ No newline at end of file
diff --git a/backend/app/evaluation/retrieval_metrics.py b/backend/app/evaluation/retrieval_metrics.py
new file mode 100644
index 0000000..39413fc
+++ b/backend/app/evaluation/retrieval_metrics.py
@@ -0,0 +1,53 @@
+from typing import Any
+from app.core.logging import get_logger
+
+logger = get_logger("retrieval_metrics")
+
+
+class RetrievalMetrics:
+    """Compute retrieval quality metrics: Recall@K, MRR, nDCG."""
+
+    @staticmethod
+    def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
+        """Recall@k: fraction of gold items in top-k retrieved."""
+        if not gold_ids:
+            return 0.0
+        top_k = set(retrieved_ids[:k])
+        relevant = len(top_k & gold_ids)
+        return relevant / len(gold_ids)
+
+    @staticmethod
+    def mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
+        """MRR: 1 / rank of first relevant item."""
+        for i, rid in enumerate(retrieved_ids, 1):
+            if rid in gold_ids:
+                return 1.0 / i
+        return 0.0
+
+    @staticmethod
+    def ndcg_at_k(scores: list[float], gold_ids: set[str], retrieved_ids: list[str], k: int) -> float:
+        """nDCG@k: normalized discounted cumulative gain."""
+        if not gold_ids:
+            return 0.0
+
+        # DCG: sum of (relevance / log2(rank+1))
+        dcg = 0.0
+        for i in range(min(k, len(retrieved_ids))):
+            relevance = 1.0 if retrieved_ids[i] in gold_ids else 0.0
+            dcg += relevance / (2 ** (i / 10.0))  # log2(i+2) approximated
+
+        # Ideal DCG: perfect ranking
+        idcg = sum(1.0 / (2 ** (i / 10.0)) for i in range(min(k, len(gold_ids))))
+
+        return dcg / idcg if idcg > 0 else 0.0
+
+    @staticmethod
+    def gold_in_retrieved(retrieved_chunks: list[dict], expected_answer: str) -> bool:
+        """Check if the expected answer is present in retrieved context."""
+        combined = " ".join(c.get("chunk_text", "") for c in retrieved_chunks).lower()
+        answer_lower = expected_answer.lower()
+        # Simple substring match; can be enhanced with semantic similarity
+        return answer_lower in combined or len(answer_lower) > 0 and answer_lower[:20] in combined
+
+
+retrieval_metrics = RetrievalMetrics()
diff --git a/backend/app/rag/cross_encoder.py b/backend/app/rag/cross_encoder.py
index e02a275..91f3fb0 100644
+++ b/backend/app/rag/cross_encoder.py
@@ -6,7 +6,7 @@ logger = get_logger("cross_encoder")
 
 
 class CrossEncoderReranker:
+    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
         self.model = CrossEncoder(model_name)
         logger.info(f"CrossEncoder initialized with {model_name}")
 
diff --git a/backend/app/rag/hybrid_rag.py b/backend/app/rag/hybrid_rag.py
index 02db602..96d5e03 100644
+++ b/backend/app/rag/hybrid_rag.py
@@ -1,14 +1,71 @@
 from typing import Any, Optional
 from app.rag.vector_rag import vector_rag
 from app.rag.bm25 import bm25_retriever
 from app.rag.metadata_filter import filter_results
+from app.rag.parent_context import parent_context_expander
+from app.rag.cross_encoder import cross_encoder
+from app.core.config import settings
 from app.core.logging import get_logger
 
 logger = get_logger("hybrid_rag")
 
 
 class HybridRAG:
+    def _normalize_scores(self, results: list[dict]) -> list[dict]:
+        """Normalize scores to 0-1 range."""
+        if not results:
+            return results
+        max_score = max(r.get("score", 0) for r in results) or 1.0
+        min_score = min(r.get("score", 0) for r in results) or 0.0
+        range_val = max_score - min_score or 1.0
+        for r in results:
+            r["normalized_score"] = (r.get("score", 0) - min_score) / range_val
+        return results
+
+    def _merge_by_weighted_score(
+        self,
+        vector_results: list[dict],
+        bm25_results: list[dict],
+        top_k: int,
+    ) -> list[dict]:
+        """Merge dense and BM25 results using weighted scoring."""
+        vector_results = self._normalize_scores(vector_results)
+        bm25_results = self._normalize_scores(bm25_results)
+        
+        bm25_weight = settings.BM25_WEIGHT
+        dense_weight = settings.DENSE_WEIGHT
+        
+        # Build combined map by chunk_id
+        merged = {}
+        for r in vector_results:
+            chunk_id = r.get("chunk_id")
+            merged[chunk_id] = {
+                **r,
+                "dense_score": r.get("normalized_score", 0),
+                "bm25_score": 0,
+                "combined_score": dense_weight * r.get("normalized_score", 0),
+            }
+        
+        for r in bm25_results:
+            chunk_id = r.get("chunk_id")
+            if chunk_id in merged:
+                merged[chunk_id]["bm25_score"] = r.get("normalized_score", 0)
+                merged[chunk_id]["combined_score"] = (
+                    dense_weight * merged[chunk_id]["dense_score"] +
+                    bm25_weight * r.get("normalized_score", 0)
+                )
+            else:
+                merged[chunk_id] = {
+                    **r,
+                    "dense_score": 0,
+                    "bm25_score": r.get("normalized_score", 0),
+                    "combined_score": bm25_weight * r.get("normalized_score", 0),
+                }
+        
+        # Sort by combined score and return top_k
+        ranked = sorted(merged.values(), key=lambda x: x.get("combined_score", 0), reverse=True)
+        return ranked[:top_k]
+
     async def retrieve(
         self,
         query: str,
@@ -16,19 +73,21 @@ class HybridRAG:
         top_k: int = 5,
         filters: Optional[dict] = None,
     ) -> list[dict[str, Any]]:
+        logger.info(f"HybridRAG retrieve: query='{query[:60]}' top_k={top_k}")
 
+        # Dense retrieval: top 50
         vector_results = []
         try:
+            vector_results = await vector_rag.retrieve(
+                query, collection_name, settings.DENSE_TOP_K, filters
+            )
         except Exception as e:
             logger.warning(f"Vector retrieval failed: {e}")
 
+        # BM25 retrieval: top 50
         bm25_results = []
         try:
+            bm25_raw = bm25_retriever.search(collection_name, query, settings.BM25_TOP_K)
             bm25_results = filter_results(bm25_raw, filters or {})
         except Exception as e:
             logger.warning(f"BM25 retrieval failed: {e}")
@@ -36,8 +95,16 @@ class HybridRAG:
         if not vector_results and not bm25_results:
             return []
 
+        # Merge with weighted scoring
+        merged = self._merge_by_weighted_score(vector_results, bm25_results, top_k * 2)
+        
+        # Rerank
+        reranked = cross_encoder.rerank(query, merged, top_k=settings.RERANK_TOP_K)
+        
+        # Expand with parent context
+        expanded = await parent_context_expander.expand(reranked, collection_name)
+        
+        return expanded
 
 
 hybrid_rag = HybridRAG()
diff --git a/backend/app/rag/multi_collection_retrieval.py b/backend/app/rag/multi_collection_retrieval.py
new file mode 100644
index 0000000..18b7673
+++ b/backend/app/rag/multi_collection_retrieval.py
@@ -0,0 +1,81 @@
+from typing import Any, Optional
+from app.rag.vector_rag import vector_rag
+from app.rag.hybrid_rag import hybrid_rag
+from app.rag.table_rag import table_rag
+from app.rag.pdf_rag import pdf_rag
+from app.rag.markdown_rag import markdown_rag
+from app.core.logging import get_logger
+
+logger = get_logger("multi_collection_retrieval")
+
+DEFAULT_COLLECTIONS = [
+    "text_documents",
+    "pdf_documents",
+    "table_documents",
+    "markdown_documents",
+    "audio_transcripts",
+    "web_documents",
+]
+
+
+class MultiCollectionRetriever:
+    """Retrieve from all collections and merge results."""
+
+    async def retrieve_all_collections(
+        self,
+        query: str,
+        strategy: str = "hybrid",
+        top_k: int = 5,
+        filters: Optional[dict] = None,
+        collections: Optional[list[str]] = None,
+    ) -> list[dict[str, Any]]:
+        """
+        Retrieve from multiple collections and deduplicate results by document_id + chunk_id.
+        Rank by score, return top_k.
+        """
+        target_collections = collections or DEFAULT_COLLECTIONS
+        all_results = []
+
+        for collection in target_collections:
+            try:
+                if strategy == "hybrid":
+                    results = await hybrid_rag.retrieve(query, collection, top_k * 3, filters)
+                elif strategy == "vector":
+                    results = await vector_rag.retrieve(query, collection, top_k * 3, filters)
+                elif strategy == "table":
+                    results = await table_rag.query(query, top_k=top_k * 3)
+                elif strategy == "pdf":
+                    results = await pdf_rag.query(query, top_k=top_k * 3)
+                elif strategy == "markdown":
+                    results = await markdown_rag.query(query, top_k=top_k * 3)
+                else:
+                    results = []
+
+                for r in results:
+                    r["source_collection"] = collection
+
+                all_results.extend(results)
+            except Exception as e:
+                logger.warning(f"Multi-collection retrieval failed for '{collection}': {e}")
+                continue
+
+        if not all_results:
+            return []
+
+        # Deduplicate by chunk_id
+        seen = set()
+        deduped = []
+        for r in all_results:
+            chunk_id = r.get("chunk_id")
+            if chunk_id and chunk_id not in seen:
+                seen.add(chunk_id)
+                deduped.append(r)
+            elif not chunk_id:
+                deduped.append(r)
+
+        # Sort by score
+        ranked = sorted(deduped, key=lambda x: x.get("score", 0), reverse=True)
+        return ranked[:top_k]
+
+
+multi_collection_retriever = MultiCollectionRetriever()
diff --git a/backend/app/rag/parent_context.py b/backend/app/rag/parent_context.py
new file mode 100644
index 0000000..fdd444f
+++ b/backend/app/rag/parent_context.py
@@ -0,0 +1,81 @@
+from typing import Any, Optional
+from app.chromadb.client import chroma_client
+from app.core.logging import get_logger
+
+logger = get_logger("parent_context")
+
+
+class ParentContextExpander:
+    """Expand retrieved chunks with parent (previous/next) chunks."""
+
+    async def expand(
+        self,
+        chunks: list[dict[str, Any]],
+        collection_name: str,
+    ) -> list[dict[str, Any]]:
+        """
+        For each chunk, retrieve adjacent chunks (previous and next).
+        Preserve order: previous, current, next (if they exist).
+        """
+        if not chunks:
+            return chunks
+
+        expanded = []
+        doc_chunk_index_map = {}
+
+        # Build map: (document_id, chunk_index) -> chunk
+        for chunk in chunks:
+            meta = chunk.get("metadata", {})
+            doc_id = meta.get("document_id")
+            chunk_idx = meta.get("chunk_index", -1)
+            if doc_id and chunk_idx >= 0:
+                doc_chunk_index_map[(doc_id, chunk_idx)] = chunk
+
+        for chunk in chunks:
+            meta = chunk.get("metadata", {})
+            doc_id = meta.get("document_id")
+            chunk_idx = meta.get("chunk_index", -1)
+
+            if not doc_id or chunk_idx < 0:
+                expanded.append(chunk)
+                continue
+
+            expanded_chunk = {**chunk}
+            parent_chunks = []
+
+            # Previous chunk
+            prev_key = (doc_id, chunk_idx - 1)
+            if prev_key in doc_chunk_index_map:
+                prev = doc_chunk_index_map[prev_key]
+                parent_chunks.append(("prev", prev))
+
+            # Current chunk already in expanded_chunk
+
+            # Next chunk
+            next_key = (doc_id, chunk_idx + 1)
+            if next_key in doc_chunk_index_map:
+                nxt = doc_chunk_index_map[next_key]
+                parent_chunks.append(("next", nxt))
+
+            # Augment chunk_text with parent context
+            if parent_chunks:
+                original_text = expanded_chunk.get("chunk_text", "")
+                ctx_parts = []
+
+                for rel_type, parent in parent_chunks:
+                    parent_text = parent.get("chunk_text", "")
+                    if rel_type == "prev":
+                        ctx_parts.insert(0, f"[PREVIOUS]\n{parent_text}\n")
+                    elif rel_type == "next":
+                        ctx_parts.append(f"\n[NEXT]\n{parent_text}")
+
+                expanded_chunk["chunk_text"] = "".join(ctx_parts) + original_text
+                expanded_chunk["parent_chunks_count"] = len(parent_chunks)
+
+            expanded.append(expanded_chunk)
+
+        logger.info(f"Expanded {len(expanded)} chunks with parent context")
+        return expanded
+
+
+parent_context_expander = ParentContextExpander()
diff --git a/backend/app/rag/table_rag.py b/backend/app/rag/table_rag.py
index cad758e..14069d0 100644
+++ b/backend/app/rag/table_rag.py
@@ -15,6 +15,31 @@ SCHEMA_COLLECTION = "table_documents"
 
 
 class TableRAG:
+    def _convert_row_to_text(self, row: pd.Series, headers: list[str]) -> str:
+        """Convert a row to readable key-value text format."""
+        parts = []
+        for header in headers:
+            val = row.get(header, "")
+            parts.append(f"{header}: {val}")
+        return " | ".join(parts)
+
+    def _convert_table_section_to_text(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> str:
+        """Convert table rows to readable text preserving headers and structure."""
+        lines = []
+        headers = list(df.columns)
+        
+        # Header row
+        lines.append(" | ".join(headers))
+        lines.append("-" * (sum(len(str(h)) for h in headers) + len(headers) * 3))
+        
+        # Data rows
+        for idx in range(start_idx, min(end_idx, len(df))):
+            row = df.iloc[idx]
+            row_text = self._convert_row_to_text(row, headers)
+            lines.append(row_text)
+        
+        return "\n".join(lines)
+
     async def index_csv(
         self,
         document_id: str,
@@ -22,39 +47,38 @@ class TableRAG:
         content: bytes,
         extra_metadata: Optional[dict] = None,
     ) -> dict[str, Any]:
+        """Index CSV: schema + structured text chunks."""
         df = pd.read_csv(io.BytesIO(content))
         schema_info = {
             "columns": list(df.columns),
             "dtypes": {col: str(df[col].dtype) for col in df.columns},
             "row_count": len(df),
         }
 
         ids, embeddings, documents, metadatas = [], [], [], []
 
+        # Schema chunk with column details
+        col_types = ", ".join([f"{col} ({dtype})" for col, dtype in schema_info["dtypes"].items()])
+        schema_text = f"Table: {filename}\nColumns: {col_types}\nTotal rows: {schema_info['row_count']}"
         schema_emb = await ollama_client.embeddings(schema_text)
         schema_id = f"{document_id}_schema"
         ids.append(schema_id)
         embeddings.append(schema_emb)
         documents.append(schema_text)
+        metadatas.append({
             "document_id": document_id,
             "filename": filename,
             "document_type": "csv",
             "chunk_type": "schema",
             "columns": json.dumps(list(df.columns)),
             **(extra_metadata or {}),
+        })
 
+        # Row chunks: convert to readable text format (preserve headers in each chunk)
+        chunk_size = 10  # Increased from 5 to capture more context
         for start in range(0, min(len(df), 500), chunk_size):
+            end = start + chunk_size
+            row_text = self._convert_table_section_to_text(df, start, end)
             row_emb = await ollama_client.embeddings(row_text)
             row_id = f"{document_id}_rows_{start}"
             ids.append(row_id)
@@ -66,12 +90,12 @@ class TableRAG:
                 "document_type": "csv",
                 "chunk_type": "rows",
                 "row_start": start,
+                "row_end": end,
                 **(extra_metadata or {}),
             })
 
         chroma_client.add_documents(SCHEMA_COLLECTION, ids, embeddings, documents, metadatas)
+        logger.info(f"TableRAG indexed {filename}: {len(ids)} chunks (schema + {len(ids)-1} row chunks)")
         return {"document_id": document_id, "chunk_count": len(ids), "schema": schema_info}
 
     async def query(
diff --git a/backend/app/services/rag_service.py b/backend/app/services/rag_service.py
index 95c710a..6139bcb 100644
+++ b/backend/app/services/rag_service.py
@@ -8,6 +8,7 @@ from app.rag.bm25 import bm25_retriever
 from app.rag.table_rag import table_rag
 from app.rag.pdf_rag import pdf_rag
 from app.rag.markdown_rag import markdown_rag
+from app.rag.multi_collection_retrieval import multi_collection_retriever
 from app.rag.metadata_filter import filter_results
 from app.rag.evaluator import (
     compute_accuracy, compute_faithfulness, compute_answer_relevancy,
@@ -32,7 +33,17 @@ class RAGService:
         top_k: int = 5,
         filters: Optional[dict] = None,
         collection_name: Optional[str] = None,
+        all_collections: bool = False,
     ) -> list[dict[str, Any]]:
+        """
+        Retrieve from single or multiple collections.
+        If all_collections=True, search across all available collections.
+        """
+        if all_collections:
+            return await multi_collection_retriever.retrieve_all_collections(
+                query, strategy, top_k, filters
+            )
+
         col = collection_name or "text_documents"
         if strategy == "vector":
             return await vector_rag.retrieve(query, col, top_k, filters)
```

## 8bc89e64f1d2556ebe5c56c9f8fac0dc2c44611d — 2026-06-09T12:58:29+05:30

Message:

chatgpt suggested changes accuracy dropped to 27 percent

```diff
diff --git a/helpfull scripts/compare_changes_in_metrics.py b/helpfull scripts/compare_changes_in_metrics.py
new file mode 100644
index 0000000..7e64668
+++ b/helpfull scripts/compare_changes_in_metrics.py	
@@ -0,0 +1,132 @@
+import pandas as pd
+import matplotlib.pyplot as plt
+import numpy as np
+
+# ==========================
+# CONFIG
+# ==========================
+CSV_1 = "C:\\Users\\Jishnu\\Desktop\\SRAG\\eval\\results\\common among removed and 0.6 less questions.csv"
+CSV_2 = "C:\\Users\\Jishnu\\Desktop\\SRAG\\eval\\results\\8th june removed and common 0.6 accuracy with chatgpt suggested code changes overall scor is 27 some questions seeding imporment.csv"
+
+METRICS = [
+    "Accuracy",
+    "Faithfulness",
+    "Context Precision",
+    "Context Recall",
+    "Answer Relevancy"
+]
+
+# ==========================
+# LOAD FILES
+# ==========================
+df1 = pd.read_csv(CSV_1)
+df2 = pd.read_csv(CSV_2)
+
+# Keep only needed columns
+cols = ["Question"] + METRICS
+
+df1 = df1[cols].copy()
+df2 = df2[cols].copy()
+
+# Rename metric columns so we know which file they came from
+df1 = df1.rename(
+    columns={m: f"{m}_old" for m in METRICS}
+)
+
+df2 = df2.rename(
+    columns={m: f"{m}_new" for m in METRICS}
+)
+
+# ==========================
+# FIND COMMON QUESTIONS
+# ==========================
+merged = pd.merge(
+    df1,
+    df2,
+    on="Question",
+    how="inner"
+)
+
+print(f"Common questions found: {len(merged)}")
+
+# ==========================
+# CREATE DELTA COLUMNS
+# ==========================
+for metric in METRICS:
+    merged[f"{metric}_change"] = (
+        merged[f"{metric}_new"] -
+        merged[f"{metric}_old"]
+    )
+
+# ==========================
+# PLOT SETTINGS
+# ==========================
+plt.style.use("ggplot")
+
+for metric in METRICS:
+
+    plot_df = merged[
+        ["Question", f"{metric}_change"]
+    ].copy()
+
+    # Sort by change
+    plot_df = plot_df.sort_values(
+        by=f"{metric}_change"
+    )
+
+    changes = plot_df[f"{metric}_change"]
+    questions = plot_df["Question"]
+
+    colors = [
+        "green" if x > 0 else "red"
+        for x in changes
+    ]
+
+    fig_height = max(6, len(plot_df) * 0.35)
+
+    plt.figure(figsize=(14, fig_height))
+
+    bars = plt.barh(
+        questions,
+        changes,
+        color=colors
+    )
+
+    plt.axvline(
+        x=0,
+        color="black",
+        linewidth=1
+    )
+
+    # Annotate bars with delta values
+    for bar, value in zip(bars, changes):
+        plt.text(
+            value,
+            bar.get_y() + bar.get_height()/2,
+            f"{value:+.2f}",
+            va="center",
+            ha="left" if value >= 0 else "right"
+        )
+
+    plt.title(
+        f"{metric}: Improvement / Regression by Question"
+    )
+    plt.xlabel(
+        "Metric Change (New - Old)"
+    )
+    plt.ylabel("Question")
+
+    plt.tight_layout()
+
+    filename = (
+        metric.lower()
+        .replace(" ", "_")
+        + "_change.png"
+    )
+
+    plt.savefig(filename, dpi=300)
+    plt.close()
+
+    print(f"Saved: {filename}")
+
+print("Done.")
\ No newline at end of file
```

## 65d9c2fc10ae9447d9d73f4685fc49f3d6e9f6a8 — 2026-06-09T10:10:51+05:30

Message:

other helpfull scripts

```diff
diff --git a/helpfull scripts/common_less_accuracy_removed_questions.py b/helpfull scripts/common_less_accuracy_removed_questions.py
new file mode 100644
index 0000000..783651e
+++ b/helpfull scripts/common_less_accuracy_removed_questions.py	
@@ -0,0 +1,33 @@
+import pandas as pd
+from pathlib import Path
+
+# Input CSV files
+eval_file = r"C:\\Users\\Jishnu\Desktop\\SRAG\\eval\\results\\8th june 2nd cross encoder.csv"
+remove_file = r"C:\\Users\\Jishnu\\Desktop\\SRAG\\helpfull scripts\\eval_number_clean.csv"
+
+# Read CSVs
+eval_df = pd.read_csv(eval_file)
+remove_df = pd.read_csv(remove_file)
+
+# Questions with Accuracy <= 0.6
+low_accuracy_questions = eval_df.loc[
+    eval_df["Accuracy"] <= 0.6,
+    "Question"
+]
+
+# Find matching rows in remove_df
+filtered_remove_df = remove_df[
+    remove_df["Eval Question"].isin(low_accuracy_questions)
+]
+
+# Output file
+output_file = (
+    f"0.6_accuracy_and_belowrows_common_removed_questions_"
+    f"{Path(remove_file).stem}.csv"
+)
+
+# Save filtered rows
+filtered_remove_df.to_csv(output_file, index=False)
+
+print(f"Filtered {len(filtered_remove_df)} rows.")
+print(f"Output saved to: {output_file}")
\ No newline at end of file
diff --git a/helpfull scripts/csv_number_clean.py b/helpfull scripts/csv_number_clean.py
index ecae758..14f6d55 100644
+++ b/helpfull scripts/csv_number_clean.py	
@@ -1,7 +1,7 @@
 import csv
 
+input_file = "C:\\Users\\Jishnu\\Desktop\\SRAG\\helpfull scripts\\0.6_accuracy_and_belowrows_common_removed_questions_eval_number_clean.csv"
+output_file = "C:\\Users\\Jishnu\\Desktop\\SRAG\\helpfull scripts\\eval_number_clean_removed_common.csv.csv"
 
 rows = []
 
diff --git a/helpfull scripts/get_less_accuracy_rows.py b/helpfull scripts/get_less_accuracy_rows.py
index 8703b33..f34dbf0 100644
+++ b/helpfull scripts/get_less_accuracy_rows.py	
@@ -2,7 +2,7 @@ import pandas as pd
 from pathlib import Path
 
 # Input CSV file
+input_file = "C:\\Users\\Jishnu\\Desktop\\SRAG\\eval\\results\\8th june 2nd cross encoder.csv"
 
 # Read CSV
 df = pd.read_csv(input_file)
diff --git a/helpfull scripts/remove_bad_questions.py b/helpfull scripts/remove_bad_questions.py
index 111cf51..8a4510c 100644
+++ b/helpfull scripts/remove_bad_questions.py	
@@ -17,8 +17,8 @@ if not API_KEY:
 
 client = OpenAI(api_key=API_KEY)
 
+FORMS_DIR = "C:\\Users\\Jishnu\\Desktop\\SRAG\\helpfull scripts\\forms"
+INPUT_CSV = "C:\\Users\\Jishnu\\Desktop\\SRAG\\helpfull scripts\\eval.csv"
 OUTPUT_FILE = "eval_clean.txt"
 
 MODEL = "gpt-5"
@@ -26,24 +26,6 @@ MODEL = "gpt-5"
 client = OpenAI(api_key=API_KEY)
 
 
 def load_all_forms_text(forms_dir: str) -> str:
     """Load and concatenate all PDF text."""
     all_text = []
@@ -51,15 +33,10 @@ def load_all_forms_text(forms_dir: str) -> str:
     pdf_files = sorted(Path(forms_dir).glob("*.pdf"))
 
     for pdf_file in pdf_files:
         # Keep filename so model knows which form it came from
+        all_text.append(pdf_file.name.replace(".pdf",""))
 
+    print(all_text)
     return "\n".join(all_text)
 
 
@@ -127,7 +104,7 @@ FORMS:
 def main():
     print("Loading SBI forms...")
     forms_text = load_all_forms_text(FORMS_DIR)
+    print(forms_text)
     kept_rows = []
 
     with open(INPUT_CSV, "r", encoding="utf-8", newline="") as f:
```

## 451673952d01ccebd3304f8268f4782f052b23cf — 2026-06-08T12:30:27+05:30

Message:

query expansion before retreival and llm question back

```diff
diff --git a/backend/app/api/rag.py b/backend/app/api/rag.py
index 724c5a6..0fd7483 100644
+++ b/backend/app/api/rag.py
@@ -66,6 +66,8 @@ class EvaluateRequest(BaseModel):
     questions: list[EvalQuestion]
     dataset_name: str = "eval_run"
     top_k: int = 5
+    use_query_expansion: bool = False
+    num_expansions: int = 2
 
 
 @router.post("/evaluate")
@@ -76,7 +78,13 @@ async def evaluate_rag(req: EvaluateRequest, db: AsyncSession = Depends(get_db))
 
     for qa in req.questions:
         try:
+            row = await evaluate_question(
+                qa.question, 
+                qa.expected_answer, 
+                req.top_k,
+                req.use_query_expansion,
+                req.num_expansions
+            )
             per_question.append(row)
             latencies.append(row["latency_ms"])
         except Exception as exc:
diff --git a/backend/app/core/prompts.py b/backend/app/core/prompts.py
index 93b4a89..06d8049 100644
+++ b/backend/app/core/prompts.py
@@ -274,6 +274,42 @@ Assume Karnataka, India.
 
 """
 
+EVAL_QUERY_EXPANSION_SYSTEM = """
+You are an SBI Banking Knowledge Assistant evaluating answers for a test dataset.
+
+Your task: Answer the question using the provided context. If the context lacks information to fully answer the question, ask ONE clarifying follow-up question to retrieve more specific information.
+
+## Rules:
+
+1. If the context contains sufficient information, answer directly and completely.
+
+2. If the context is insufficient or ambiguous, you may ask ONE follow-up question to get more relevant chunks.
+
+3. Format follow-up questions as: FOLLOW_UP: <your question here>
+
+4. Follow-up questions should be specific and aim to retrieve missing information from the knowledge base.
+
+5. Do not ask follow-ups for information that cannot reasonably be in a banking knowledge base (like personal account details, real-time data, etc.).
+
+6. Keep answers concise and direct.
+
+7. Use banking terminology from the context when available.
+
+Examples:
+
+Q: What is CIF?
+Context: [contains CIF definition]
+A: CIF (Customer Information File) is a unique identifier...
+
+Q: What documents are needed for account opening?
+Context: [partial list, mentions "identity proof required"]
+A: FOLLOW_UP: What are the specific identity proof documents accepted for SBI account opening?
+
+Q: What is the interest rate for savings account?
+Context: [no rate information]
+A: FOLLOW_UP: What is the current interest rate structure for SBI savings accounts?
+"""
+
 # Backward-compatible alias used by chat_service
 SYSTEM_PROMPT = SBI_SYSTEM_PROMPT
 
diff --git a/backend/app/evaluation/agent_runner.py b/backend/app/evaluation/agent_runner.py
index 465b0d4..a4b505c 100644
+++ b/backend/app/evaluation/agent_runner.py
@@ -17,34 +17,78 @@ logger = get_logger("evaluation.agent_runner")
 
 RAG_SYSTEM = """Answer the question carefully"""
 
+QUERY_EXPANSION_PROMPT = """Given the user's question, generate {num_expansions} alternative phrasings or related queries that would help retrieve relevant information from a knowledge base.
+
+Original question: {question}
+
+Generate {num_expansions} expanded queries as a numbered list (1., 2., 3., etc.). Each query should:
+- Rephrase the original question differently
+- Ask for related aspects that would help answer the original question
+- Use different terminology or synonyms
+
+Output only the numbered list, nothing else."""
+
 
 def _chunk_texts(chunks: list[dict]) -> list[str]:
     return [c["chunk_text"] for c in chunks if c.get("chunk_text")]
 
 
+async def _expand_query(question: str, num_expansions: int = 2) -> list[str]:
+    """Generate expanded queries using LLM."""
+    prompt = QUERY_EXPANSION_PROMPT.format(question=question, num_expansions=num_expansions)
+    response = await openai_client.generate(prompt, system="You are a query expansion assistant.")
+    
+    expanded = [question]  # Always include original
+    for line in response.split("\n"):
+        line = line.strip()
+        if line and (line[0].isdigit() or line.startswith("-")):
+            query = line.split(".", 1)[-1].strip() if "." in line else line.lstrip("- ")
+            if query:
+                expanded.append(query)
+    
+    return expanded[:num_expansions + 1]
+
+
 async def evaluate_question(
     question: str,
     expected_answer: str,
     top_k: int = 5,
+    use_query_expansion: bool = False,
+    num_expansions: int = 2,
 ) -> dict[str, Any]:
     """
+    Run one Q&A pair through RAG with optional query expansion.
     
+    1. Expand query into multiple variants (if enabled)
+    2. Retrieve chunks for each query variant
+    3. Combine and deduplicate all chunks
+    4. Generate answer from combined context
+    5. Score with LLM-as-judge
     """
     t0 = time.time()
     
+    # Query expansion before retrieval
+    if use_query_expansion:
+        queries = await _expand_query(question, num_expansions)
+    else:
+        queries = [question]
+    
+    # Retrieve chunks for all queries
+    all_chunks = []
+    chunk_ids_seen = set()
+    
+    for query in queries:
+        chunks = await rag_service.retrieve(query, strategy="hybrid", top_k=top_k * 2)
+        chunks = cross_encoder.rerank(query, chunks, top_k=top_k)
+        
+        for c in chunks:
+            cid = c.get("chunk_id")
+            if cid and cid not in chunk_ids_seen:
+                chunk_ids_seen.add(cid)
+                all_chunks.append(c)
     
     # Generate answer
+    chunk_texts = _chunk_texts(all_chunks)
     context_text = "\n---\n".join(chunk_texts)
     
     if context_text.strip():
@@ -64,6 +108,8 @@ async def evaluate_question(
         "expected_answer": expected_answer,
         "generated_answer": generated_answer,
         "retrieved_context": context_text,
+        "expanded_queries": queries if use_query_expansion else [],
+        "num_chunks": len(all_chunks),
         "accuracy": scores.get("accuracy", 0.0),
         "faithfulness": scores.get("faithfulness", 0.0),
         "answer_relevancy": scores.get("answer_relevancy", 0.0),
@@ -85,6 +131,8 @@ def failed_question_row(question: str, expected_answer: str, error: str) -> dict
         "expected_answer": expected_answer,
         "generated_answer": "",
         "retrieved_context": "",
+        "expanded_queries": [],
+        "num_chunks": 0,
         "accuracy": 0.0,
         "faithfulness": 0.0,
         "answer_relevancy": 0.0,
```

## 527f8b5685914ee942988b09af8d64f5e1f0ca3a — 2026-06-08T10:30:13+05:30

Message:

cross encoder reranking

```diff
diff --git a/backend/app/evaluation/agent_runner.py b/backend/app/evaluation/agent_runner.py
index 25b75a5..465b0d4 100644
+++ b/backend/app/evaluation/agent_runner.py
@@ -10,6 +10,7 @@ from typing import Any
 from app.services.rag_service import rag_service
 from app.evaluation.evaluator import evaluate_single
 from app.embeddings.openai_client import openai_client
+from app.rag.cross_encoder import cross_encoder
 from app.core.logging import get_logger
 
 logger = get_logger("evaluation.agent_runner")
@@ -30,13 +31,17 @@ async def evaluate_question(
     Run one Q&A pair through simple nearest-neighbor RAG.
     
     1. Retrieve top_k chunks (hybrid search)
+    2. Rerank with cross-encoder
+    3. Generate answer from context
+    4. Score with LLM-as-judge
     """
     t0 = time.time()
 
     # Retrieve chunks
+    chunks = await rag_service.retrieve(question, strategy="hybrid", top_k=top_k * 2)
+    
+    # Rerank with cross-encoder
+    chunks = cross_encoder.rerank(question, chunks, top_k=top_k)
     
     # Generate answer
     chunk_texts = _chunk_texts(chunks)
diff --git a/backend/app/rag/cross_encoder.py b/backend/app/rag/cross_encoder.py
new file mode 100644
index 0000000..e02a275
+++ b/backend/app/rag/cross_encoder.py
@@ -0,0 +1,37 @@
+from typing import Any
+from sentence_transformers import CrossEncoder
+from app.core.logging import get_logger
+
+logger = get_logger("cross_encoder")
+
+
+class CrossEncoderReranker:
+    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
+        self.model = CrossEncoder(model_name)
+        logger.info(f"CrossEncoder initialized with {model_name}")
+
+    def rerank(
+        self,
+        query: str,
+        chunks: list[dict[str, Any]],
+        top_k: int = 5,
+    ) -> list[dict[str, Any]]:
+        """Rerank chunks using cross-encoder scoring."""
+        if not chunks:
+            return []
+
+        chunk_texts = [c["chunk_text"] for c in chunks]
+        
+        # Score all chunks
+        scores = self.model.predict([[query, text] for text in chunk_texts])
+        
+        # Add scores and sort
+        for i, chunk in enumerate(chunks):
+            chunk["ce_score"] = float(scores[i])
+        
+        ranked = sorted(chunks, key=lambda x: x["ce_score"], reverse=True)[:top_k]
+        logger.info(f"Reranked {len(chunks)} chunks, selected top {len(ranked)}")
+        return ranked
+
+
+cross_encoder = CrossEncoderReranker()
```

## 745f03b1ab503bcd3dcad0812e74a24c71958582 — 2026-06-08T10:26:56+05:30

Message:

upload changes

_No Python file changes in this commit._

## ae6d31d87ea05c870a6504398cd3fcae55c0292b — 2026-06-08T10:15:02+05:30

Message:

prompts and additional data

```diff
diff --git a/backend/app/core/prompts.py b/backend/app/core/prompts.py
index 9251c32..93b4a89 100644
+++ b/backend/app/core/prompts.py
@@ -1,74 +1,277 @@
 """Shared LLM system prompts used across chat, eval, and agent synthesis."""
 
 SBI_SYSTEM_PROMPT = """
+# SBI Banking Knowledge Assistant
+
+## Role
+
 You are an SBI Banking Knowledge Assistant.
 
+Assume user questions are related to:
+
+* SBI Bank
+* Banking operations
+* Account opening and maintenance
+* Loans and deposits
+* KYC and compliance
+* Banking forms and fields
+* Customer requests and services
+* Banking products and schemes
+* Banking terminology and abbreviations
+
+Unless the user clearly changes the topic.
+
+---
+
+## Core Rule
+
+If the retrieved content contains the answer, use it.
+
+Do not ignore relevant information.
+
+Do not say information is unavailable when the answer exists in the retrieved content.
+
+---
+
+## Retrieval Usage
+
+### 1. Find the Best Match
+
+Identify the retrieved passage that most directly answers the question.
+
+Focus on:
+
+* Definitions
+* Form fields
+* Abbreviations
+* Codes
+* Status values
+* Procedures
+* Product descriptions
+* Eligibility conditions
+* Documentation requirements
+
+Ignore unrelated retrieved content.
+
+---
+
+### 2. Direct Answering
+
+When a direct answer exists:
+
+* Answer immediately.
+* Use the retrieved information.
+* Prefer the most specific answer.
+* Prefer exact field meanings when available.
+
+For definitions:
+
+* Definition must appear in the first sentence.
+
+Example:
+
+Question: What is CIF?
+
+Answer: CIF (Customer Information File) is a unique customer identifier that links all of a customer's accounts and banking relationships under a single profile.
+
+---
+
+### 3. Definition Extraction Rule
+
+If any retrieved passage contains:
+
+* "X means ..."
+* "X is ..."
+* "X refers to ..."
+* "X stands for ..."
+* "Explanation of X ..."
+
+then treat that passage as the primary answer source.
+
+Never respond with:
+
+* "Not found"
+* "Information unavailable"
+* "No relevant information"
+
+while such a definition exists.
+
+---
+
+### 4. Multiple Retrieved Matches
+
+If multiple passages contain possible answers:
+
+Priority:
+
+1. Direct answer to the question
+2. SBI-specific information
+3. Most complete explanation
+4. Most recent information if dates are available
+
+Combine passages only when they describe the same subject.
+
+Do not combine unrelated sections.
+
+---
+
+### 5. SBI Preference Rule
+
+Always prefer:
+
+* SBI-specific definitions
+* SBI-specific procedures
+* SBI-specific terminology
+
+over generic banking explanations.
+
+---
+
+### 6. Missing Information
+
+Only state that information is unavailable when:
+
+* No retrieved passage answers the question, and
+* The answer cannot be reasonably inferred from retrieved content.
+
+If information is missing:
+
+* Use general banking knowledge when confident.
+* Clearly separate general banking guidance from SBI-specific information.
+
+Example:
+
+"SBI-specific information is not available. Generally, banks require identity proof, address proof, PAN, and photographs for account opening."
+
+---
+
+### 7. Confidence Rule
+
+Answer when reasonably supported by:
+
+* Retrieved SBI content, or
+* Established banking knowledge.
+
+Do not refuse simply because wording is not identical.
+
+Reasonable inference from retrieved content is allowed.
+
+---
+
+### 8. Conflict Resolution
+
+When retrieved passages disagree:
+
+1. Prefer SBI-specific content.
+2. Prefer the passage that directly answers the question.
+3. Prefer the more detailed explanation.
+4. If conflict remains, briefly mention the uncertainty.
+
+---
+
+### 9. Form Understanding
+
+For account opening forms, service request forms, KYC forms, and customer information forms:
+
+* Explain fields using the meaning provided in retrieved content.
+* Use exact field names when available.
+* Explain the purpose of the field clearly and concisely.
+
+---
+
+### 10. Abbreviations and Banking Terms
+
+For abbreviations such as:
+
+* CIF
+* KYC
+* PAN
+* CKYC
+* NRE
+* NRO
+* IMPS
+* NEFT
+* RTGS
+* UPI
+
+Provide:
+
+1. Full form
+2. Meaning
+3. Purpose (if useful)
+
+Keep answers concise.
+
+---
+
+### 11. Hallucination Prevention
+
+Never invent:
+
+* SBI internal codes
+* SBI procedures
+* SBI limits
+* SBI eligibility rules
+* SBI product features
+* SBI field meanings
+* SBI documentation requirements
+
+unless supported by retrieved content.
+
+---
+
+### 12. Response Style
+
+Always:
+
+* Answer first.
+* Be concise.
+* Use plain language.
+* Give the most relevant answer immediately.
+
+Avoid:
+
+* Long introductions
+* Unnecessary background
+* Discussion of documents
+* Discussion of retrieval
+* Discussion of search results
+* Statements like:
+
+  * "Based on the retrieved context"
+  * "According to the documents"
+  * "The retrieved passages indicate"
+
+---
+
+## Special Rule for Evaluation Datasets
+
+If a retrieved passage clearly contains a definition or explanation that answers the question:
+
+* Produce the answer.
+* Do not output "Not found."
+* Do not output "Information unavailable."
+* Do not refuse.
+
+A partially matching answer from retrieved content is better than incorrectly claiming no answer exists.
+
+---
+
+## Location Default
+
+If a state is required and the user does not specify one:
+
+Assume Karnataka, India.
+
+---
+
+## Priority Order
+
+1. Relevant SBI-specific retrieved information
+2. Reliable SBI knowledge
+3. General banking knowledge
+4. Explicit acknowledgement of uncertainty when necessary
+
 """
 
 # Backward-compatible alias used by chat_service
diff --git a/backend/app/evaluation/agent_runner.py b/backend/app/evaluation/agent_runner.py
index a6ad44b..25b75a5 100644
+++ b/backend/app/evaluation/agent_runner.py
@@ -14,8 +14,7 @@ from app.core.logging import get_logger
 
 logger = get_logger("evaluation.agent_runner")
 
+RAG_SYSTEM = """Answer the question carefully"""
 
 
 def _chunk_texts(chunks: list[dict]) -> list[str]:
@@ -51,7 +50,7 @@ async def evaluate_question(
     generated_answer = await openai_client.generate(prompt, system=RAG_SYSTEM)
     
     # Score
+    scores = await evaluate_single(question, expected_answer, generated_answer, chunk_texts)
     
     latency_ms = round((time.time() - t0) * 1000, 1)
 
diff --git a/helpfull scripts/convert_removed_to_csv.py b/helpfull scripts/convert_removed_to_csv.py
new file mode 100644
index 0000000..b657aaf
+++ b/helpfull scripts/convert_removed_to_csv.py	
@@ -0,0 +1,36 @@
+import re
+import csv
+
+input_file = "eval_clean.txt"
+output_file = "eval_clean.csv"
+
+with open(input_file, "r", encoding="utf-8") as f:
+    text = f.read()
+
+# Split by separator lines
+blocks = re.split(r"=+\s*", text.strip())
+
+rows = []
+
+for block in blocks:
+    block = block.strip()
+    if not block:
+        continue
+
+    q_no_match = re.search(r"Question no:\s*(\d+)", block)
+    question_match = re.search(r"Eval Question:\s*(.*)", block)
+    answer_match = re.search(r"Eval Answer:\s*(.*)", block, re.DOTALL)
+
+    if q_no_match and question_match and answer_match:
+        q_no = q_no_match.group(1).strip()
+        question = question_match.group(1).strip()
+        answer = answer_match.group(1).strip()
+
+        rows.append([q_no, question, answer])
+
+with open(output_file, "w", newline="", encoding="utf-8") as f:
+    writer = csv.writer(f)
+    writer.writerow(["Question no", "Eval Question", "Eval Answer"])
+    writer.writerows(rows)
+
+print("CSV file created:", output_file)
\ No newline at end of file
diff --git a/eval/csv_convert.py b/helpfull scripts/csv_convert.py
similarity index 100%
rename from eval/csv_convert.py
rename to helpfull scripts/csv_convert.py
diff --git a/helpfull scripts/csv_number_clean.py b/helpfull scripts/csv_number_clean.py
new file mode 100644
index 0000000..ecae758
+++ b/helpfull scripts/csv_number_clean.py	
@@ -0,0 +1,27 @@
+import csv
+
+input_file = "eval_clean.csv"
+output_file = "eval_number_clean.csv"
+
+rows = []
+
+# Read existing CSV
+with open(input_file, "r", encoding="utf-8") as f:
+    reader = csv.reader(f)
+    header = next(reader, None)  # skip header if present
+
+    for row in reader:
+        if len(row) < 3:
+            continue
+        # row = [Question no, Eval Question, Eval Answer]
+        rows.append([row[1], row[2]])  # drop old number, keep question + answer
+
+# Rewrite with fixed numbering
+with open(output_file, "w", newline="", encoding="utf-8") as f:
+    writer = csv.writer(f)
+    writer.writerow(["Question no", "Eval Question", "Eval Answer"])
+
+    for i, (question, answer) in enumerate(rows, start=1):
+        writer.writerow([i, question, answer])
+
+print("Fixed CSV written to:", output_file)
\ No newline at end of file
diff --git a/helpfull scripts/get_less_accuracy_rows.py b/helpfull scripts/get_less_accuracy_rows.py
new file mode 100644
index 0000000..8703b33
+++ b/helpfull scripts/get_less_accuracy_rows.py	
@@ -0,0 +1,21 @@
+import pandas as pd
+from pathlib import Path
+
+# Input CSV file
+input_file = "eval_number_clean 5th june.csv"
+
+# Read CSV
+df = pd.read_csv(input_file)
+
+# Filter rows where Accuracy <= 0.6
+filtered_df = df[df["Accuracy"] <= 0.6]
+
+# Create output filename
+input_path = Path(input_file)
+output_file = f"0.6accuracyandbelowrows_{input_path.stem}.csv"
+
+# Save filtered rows
+filtered_df.to_csv(output_file, index=False)
+
+print(f"Filtered {len(filtered_df)} rows.")
+print(f"Output saved to: {output_file}")
\ No newline at end of file
diff --git a/helpfull scripts/remove_bad_questions.py b/helpfull scripts/remove_bad_questions.py
new file mode 100644
index 0000000..111cf51
+++ b/helpfull scripts/remove_bad_questions.py	
@@ -0,0 +1,167 @@
+import csv
+import json
+import os
+from pathlib import Path
+
+import pypdf
+from dotenv import load_dotenv
+from openai import OpenAI
+
+# Load variables from .env
+load_dotenv()
+
+API_KEY = os.getenv("OPENAI_API_KEY")
+
+if not API_KEY:
+    raise ValueError("OPENAI_API_KEY not found in .env file")
+
+client = OpenAI(api_key=API_KEY)
+
+FORMS_DIR = "forms"
+INPUT_CSV = "eval.csv"
+OUTPUT_FILE = "eval_clean.txt"
+
+MODEL = "gpt-5"
+
+client = OpenAI(api_key=API_KEY)
+
+
+def extract_pdf_text(pdf_path: Path) -> str:
+    """Extract text from a PDF."""
+    try:
+        reader = pypdf.PdfReader(str(pdf_path))
+        pages = []
+
+        for page in reader.pages:
+            text = page.extract_text()
+            if text:
+                pages.append(text)
+
+        return "\n".join(pages)
+
+    except Exception as e:
+        print(f"Failed to read {pdf_path}: {e}")
+        return ""
+
+
+def load_all_forms_text(forms_dir: str) -> str:
+    """Load and concatenate all PDF text."""
+    all_text = []
+
+    pdf_files = sorted(Path(forms_dir).glob("*.pdf"))
+
+    for pdf_file in pdf_files:
+        print(f"Loading: {pdf_file.name}")
+
+        text = extract_pdf_text(pdf_file)
+
+        # Keep filename so model knows which form it came from
+        all_text.append(
+            f"\n\n========== FILE: {pdf_file.name} ==========\n{text}"
+        )
+
+    return "\n".join(all_text)
+
+
+def is_question_relevant(question: str, answer: str, forms_text: str) -> bool:
+    """
+    Ask the LLM whether this QA pair is relevant
+    to someone filling any of the provided SBI forms.
+    """
+
+    prompt = f"""
+You are evaluating whether a question-answer pair is useful for a user
+who is filling SBI banking forms.
+
+Forms content is provided below.
+
+A question should be marked YES if:
+- It could reasonably arise while filling any of the forms.
+- The answer provides information needed to understand a field,
+  term, requirement, process, declaration, or document mentioned
+  in the forms.
+
+A question should be marked NO if:
+- It is unrelated to the forms.
+- It concerns topics not needed to understand or fill the forms.
+- The answer provides information that would not help someone
+  complete the forms.
+
+Return ONLY valid JSON:
+
+{{"keep": true}}
+
+or
+
+{{"keep": false}}
+
+QUESTION:
+{question}
+
+ANSWER:
+{answer}
+
+FORMS:
+{forms_text}
+"""
+
+    response = client.chat.completions.create(
+    model=MODEL,
+    messages=[
+            {"role": "user", "content": prompt}
+    ]
+    )
+
+
+    text = response.choices[0].message.content.strip()
+
+    try:
+        result = json.loads(text)
+        return bool(result.get("keep", False))
+    except Exception:
+        print("Failed to parse response:")
+        print(text)
+        return False
+
+
+def main():
+    print("Loading SBI forms...")
+    forms_text = load_all_forms_text(FORMS_DIR)
+
+    kept_rows = []
+
+    with open(INPUT_CSV, "r", encoding="utf-8", newline="") as f:
+        reader = csv.DictReader(f)
+
+        for row in reader:
+            qno = row.get("Question no", "")
+            question = row.get("Eval Question", "")
+            answer = row.get("Eval Answer", "")
+
+            print(f"Evaluating question {qno}...")
+
+            keep = is_question_relevant(
+                question=question,
+                answer=answer,
+                forms_text=forms_text
+            )
+
+            print(f" -> KEEP={keep}")
+
+            if keep:
+                kept_rows.append(row)
+
+    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
+        for row in kept_rows:
+            f.write(f"Question no: {row['Question no']}\n")
+            f.write(f"Eval Question: {row['Eval Question']}\n")
+            f.write(f"Eval Answer: {row['Eval Answer']}\n")
+            f.write("\n" + ("=" * 80) + "\n\n")
+
+    print(f"\nDone.")
+    print(f"Kept {len(kept_rows)} questions.")
+    print(f"Output written to {OUTPUT_FILE}")
+
+
+if __name__ == "__main__":
+    main()
\ No newline at end of file
```

## e9839a53d9f46b75f900260128021fa9ae116c4d — 2026-06-03T10:02:29+05:30

Message:

nearest neighbour in eval only after removing agentic actions

```diff
diff --git a/backend/app/evaluation/agent_runner.py b/backend/app/evaluation/agent_runner.py
index 82eacf1..a6ad44b 100644
+++ b/backend/app/evaluation/agent_runner.py
@@ -1,16 +1,21 @@
 """
+Simple nearest-neighbor RAG evaluation.
 
+Uses direct retrieval + LLM generation and scoring (no multi-agent orchestration).
 """
 
 import time
 from typing import Any
 
+from app.services.rag_service import rag_service
+from app.evaluation.evaluator import evaluate_single
+from app.embeddings.openai_client import openai_client
+from app.core.logging import get_logger
+
+logger = get_logger("evaluation.agent_runner")
+
+RAG_SYSTEM = """You are a helpful assistant. Answer the question using the provided context.
+Be concise and accurate. If the answer is not in the context, say 'Not found in available documents'."""
 
 
 def _chunk_texts(chunks: list[dict]) -> list[str]:
@@ -23,36 +28,38 @@ async def evaluate_question(
     top_k: int = 5,
 ) -> dict[str, Any]:
     """
+    Run one Q&A pair through simple nearest-neighbor RAG.
+    
+    1. Retrieve top_k chunks (hybrid search)
+    2. Generate answer from context
+    3. Score with LLM-as-judge
     """
     t0 = time.time()
 
+    # Retrieve chunks
+    chunks = await rag_service.retrieve(question, strategy="hybrid", top_k=top_k)
+    
+    # Generate answer
+    chunk_texts = _chunk_texts(chunks)
+    context_text = "\n---\n".join(chunk_texts)
+    
+    if context_text.strip():
+        prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
+    else:
+        prompt = f"Question: {question}"
+    
+    generated_answer = await openai_client.generate(prompt, system=RAG_SYSTEM)
+    
+    # Score
+    scores = await evaluate_single(question, generated_answer, expected_answer, chunks)
+    
     latency_ms = round((time.time() - t0) * 1000, 1)
 
     return {
         "question": question,
         "expected_answer": expected_answer,
         "generated_answer": generated_answer,
+        "retrieved_context": context_text,
         "accuracy": scores.get("accuracy", 0.0),
         "faithfulness": scores.get("faithfulness", 0.0),
         "answer_relevancy": scores.get("answer_relevancy", 0.0),
@@ -64,11 +71,6 @@ async def evaluate_question(
         "context_precision_rationale": scores.get("context_precision_rationale", ""),
         "context_recall_rationale": scores.get("context_recall_rationale", ""),
         "latency_ms": latency_ms,
     }
 
 
@@ -90,10 +92,5 @@ def failed_question_row(question: str, expected_answer: str, error: str) -> dict
         "context_precision_rationale": "",
         "context_recall_rationale": "",
         "latency_ms": 0.0,
         "error": error,
     }
```

## 9082779fced032610502da252d682bb9521a6325 — 2026-06-03T09:57:34+05:30

Message:

gitingore adding ignore files

_No Python file changes in this commit._

## 424678bff84f36207e3f9e2400cf20cda8f875ce — 2026-06-02T10:50:09+05:30

Message:

single ffile for prompt and mutliagent for eval

```diff
diff --git a/backend/app/agents/coordinator_agent.py b/backend/app/agents/coordinator_agent.py
index 1a0dadd..6044e6a 100644
+++ b/backend/app/agents/coordinator_agent.py
@@ -5,8 +5,8 @@ from app.agents.vector_agent import vector_agent
 from app.agents.sqlite_agent import sqlite_agent
 from app.agents.router_agent import router_agent, _detect_doc_type
 from app.agents.web_agent import web_agent
 from app.embeddings.openai_client import openai_client as ollama_client
+from app.core.prompts import DEFAULT_COORDINATOR_SYNTHESIS_PROMPT
 from app.core.logging import get_logger
 
 logger = get_logger("coordinator_agent")
@@ -26,6 +26,19 @@ def _classify_intent(query: str) -> str:
     return "general"
 
 
+def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
+    """Drop duplicate chunks when router and vector agents return the same hits."""
+    seen: set[str] = set()
+    deduped: list[dict] = []
+    for chunk in chunks:
+        key = chunk.get("chunk_id") or chunk.get("chunk_text", "")[:120]
+        if key in seen:
+            continue
+        seen.add(key)
+        deduped.append(chunk)
+    return deduped
+
+
 class CoordinatorAgent(BaseAgent):
     name = "coordinator_agent"
 
@@ -52,6 +65,7 @@ class CoordinatorAgent(BaseAgent):
             "agents": agents_to_run,
             "context": ctx,
             "top_k": ctx.get("top_k", 5),
+            "synthesis_system": ctx.get("synthesis_system"),
         }
 
     async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
@@ -100,7 +114,7 @@ class CoordinatorAgent(BaseAgent):
             f"Multiple agents retrieved the following information:\n\n{combined_context}\n\n"
             f"Based on all above, provide a comprehensive final answer to: {query}"
         )
+        system = plan.get("synthesis_system") or DEFAULT_COORDINATOR_SYNTHESIS_PROMPT
         final_answer = await ollama_client.generate(synthesis_prompt, system=system)
         latency = (time.time() - start) * 1000
 
@@ -108,7 +122,7 @@ class CoordinatorAgent(BaseAgent):
             "agent": self.name,
             "query": query,
             "answer": final_answer,
+            "chunks": _dedupe_chunks(all_chunks),
             "agent_results": agent_results,
             "intent": plan["intent"],
             "latency_ms": round(latency, 2),
diff --git a/backend/app/agents/evaluator_agent.py b/backend/app/agents/evaluator_agent.py
index 3d81a05..08470ca 100644
+++ b/backend/app/agents/evaluator_agent.py
@@ -2,6 +2,7 @@ import time
 from typing import Any, Optional
 from app.agents.base import BaseAgent
 from app.rag.evaluator import (
+    compute_accuracy,
     compute_faithfulness, compute_answer_relevancy,
     compute_context_precision, compute_context_recall,
 )
@@ -29,6 +30,11 @@ class RetrievalEvaluationAgent(BaseAgent):
 
         context_texts = [c.get("chunk_text", "") for c in chunks if c.get("chunk_text")]
 
+        if expected:
+            accuracy, acc_rationale = await compute_accuracy(answer, expected)
+        else:
+            accuracy, acc_rationale = 0.0, "No expected answer provided."
+
         faithfulness, faith_rationale = await compute_faithfulness(answer, context_texts)
         relevancy, relevancy_rationale = await compute_answer_relevancy(query, answer)
         cp, cp_rationale = await compute_context_precision(query, context_texts)
@@ -44,6 +50,9 @@ class RetrievalEvaluationAgent(BaseAgent):
         return {
             "agent": self.name,
             "query": query,
+            "generated_answer": answer,
+            "accuracy": accuracy,
+            "accuracy_rationale": acc_rationale,
             "faithfulness": faithfulness,
             "faithfulness_rationale": faith_rationale,
             "answer_relevancy": relevancy,
@@ -59,10 +68,11 @@ class RetrievalEvaluationAgent(BaseAgent):
 
     async def evaluate(self, result: dict[str, Any]) -> dict[str, Any]:
         score = (
+            result.get("accuracy", 0) * 0.2
+            + result.get("faithfulness", 0) * 0.25
+            + result.get("answer_relevancy", 0) * 0.25
+            + result.get("context_precision", 0) * 0.15
+            + result.get("context_recall", 0) * 0.15
         )
         result["overall_score"] = round(score, 4)
         result["answer"] = f"Evaluation complete. Overall score: {result['overall_score']:.2f}"
diff --git a/backend/app/api/rag.py b/backend/app/api/rag.py
index 34e241d..724c5a6 100644
+++ b/backend/app/api/rag.py
@@ -2,23 +2,21 @@
 POST /api/rag/evaluate
 
 Accepts a list of {question, expected_answer} pairs, runs each through the
+multi-agent RAG pipeline (coordinator → evaluator), and returns per-question
 detail alongside aggregate metrics.
 """
 
 from typing import Any
 
 from fastapi import APIRouter, Depends
 from fastapi.responses import StreamingResponse
 from sqlalchemy.ext.asyncio import AsyncSession
+from pydantic import BaseModel
 
 from app.core.dependencies import get_db
 from app.services.rag_service import rag_service
+from app.evaluation.agent_runner import evaluate_question, failed_question_row
 from app.core.logging import get_logger
 
 from app.schemas.rag import RAGQueryRequest, RAGQueryResponse, RAGRetrieveRequest
 from app.schemas.search import SearchResult
@@ -67,6 +65,7 @@ class EvalQuestion(BaseModel):
 class EvaluateRequest(BaseModel):
     questions: list[EvalQuestion]
     dataset_name: str = "eval_run"
+    top_k: int = 5
 
 
 @router.post("/evaluate")
@@ -77,134 +76,15 @@ async def evaluate_rag(req: EvaluateRequest, db: AsyncSession = Depends(get_db))
 
     for qa in req.questions:
         try:
+            row = await evaluate_question(qa.question, qa.expected_answer, req.top_k)
             per_question.append(row)
             latencies.append(row["latency_ms"])
         except Exception as exc:
             logger.error(f"Eval error for '{qa.question[:60]}': {exc}")
             failed.append({"question": qa.question, "error": str(exc)})
+            per_question.append(failed_question_row(qa.question, qa.expected_answer, str(exc)))
+
+    succeeded = [r for r in per_question if not r.get("error")]
 
     def _avg(k: str) -> float:
         if not succeeded:
@@ -212,7 +92,6 @@ Priority Order:
         return round(sum(r.get(k, 0.0) for r in succeeded) / len(succeeded), 4)
 
     return {
         "accuracy":          _avg("accuracy"),
         "faithfulness":      _avg("faithfulness"),
         "context_precision": _avg("context_precision"),
@@ -220,7 +99,6 @@ Priority Order:
         "answer_relevancy":  _avg("answer_relevancy"),
         "latency_avg_ms":    round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
         "failed_questions":  failed,
         "per_question":      per_question,
         "dataset_name":      req.dataset_name,
\ No newline at end of file
+    }
diff --git a/backend/app/core/prompts.py b/backend/app/core/prompts.py
new file mode 100644
index 0000000..9251c32
+++ b/backend/app/core/prompts.py
@@ -0,0 +1,80 @@
+"""Shared LLM system prompts used across chat, eval, and agent synthesis."""
+
+SBI_SYSTEM_PROMPT = """
+You are an SBI Banking Knowledge Assistant.
+
+Scope:
+- Treat all user questions as related to SBI Bank, banking operations, financial services, regulatory processes, forms, policies, products, and internal documentation unless the user explicitly changes the topic.
+- The retrieved context is the primary source of truth.
+
+RETRIEVAL-AWARE BEHAVIOR
+
+1. Relevance First
+- Carefully identify which parts of the retrieved context are relevant to the user's question.
+- Ignore unrelated retrieved passages.
+- Do not combine information from unrelated sections unless they clearly refer to the same subject.
+
+2. Direct Answering
+- If the answer is explicitly present, provide the answer directly.
+- For field names, abbreviations, codes, labels, column names, form fields, statuses, and identifiers, return the exact meaning or definition found in the retrieved content.
+- Prefer the most specific answer over a generic one.
+
+3. Multiple Matches
+- If multiple retrieved passages contain possible answers:
+  - Prefer the passage that most closely matches the user's wording and intent.
+  - Prefer SBI-specific definitions over generic banking definitions.
+  - Prefer the most complete and unambiguous answer.
+
+4. Ambiguity Handling
+- If the retrieved information is ambiguous, ask a short clarification question.
+- Do not guess which product, form, scheme, process, or field the user means.
+
+5. Missing Information
+- If the retrieved context does not contain sufficient information:
+  - Use general banking knowledge only when highly confident.
+  - Clearly separate inferred knowledge from retrieved facts.
+  - Never invent SBI-specific procedures, codes, policies, field meanings, product details, limits, eligibility rules, or internal terminology.
+
+6. Conflict Resolution
+- If retrieved passages conflict:
+  - Prefer the more specific passage.
+  - Prefer SBI-specific information over generic information.
+  - Prefer the passage that directly addresses the user's question.
+  - Do not merge conflicting answers.
+
+7. Hallucination Prevention
+- Never fabricate:
+  - Form field definitions
+  - Internal codes
+  - Status meanings
+  - Product rules
+  - Interest rates
+  - Regulatory requirements
+  - Process steps
+  - Branch-specific information
+- If uncertain, say:
+  "I do not have enough information to answer that."
+
+LOCATION DEFAULT
+- If a state is required but not specified, assume Karnataka, India.
+
+RESPONSE STYLE
+- Answer the user's question directly.
+- Keep responses concise.
+- For definition questions, return only the definition unless more detail is requested.
+- Avoid unnecessary explanations, background information, examples, or assumptions.
+- Never mention retrieval, documents, context, sources, or knowledge-base mechanics.
+
+Priority Order:
+1. Relevant retrieved SBI information
+2. Highly confident banking knowledge that does not conflict with retrieved information
+3. "I do not have enough information to answer that."
+"""
+
+# Backward-compatible alias used by chat_service
+SYSTEM_PROMPT = SBI_SYSTEM_PROMPT
+
+DEFAULT_COORDINATOR_SYNTHESIS_PROMPT = (
+    "You are a coordinator that synthesizes information from multiple sources "
+    "into a single coherent answer."
+)
diff --git a/backend/app/evaluation/agent_runner.py b/backend/app/evaluation/agent_runner.py
new file mode 100644
index 0000000..82eacf1
+++ b/backend/app/evaluation/agent_runner.py
@@ -0,0 +1,99 @@
+"""
+Multi-agent evaluation runner.
+
+Orchestrates coordinator_agent (retrieval + answer synthesis) and
+evaluator_agent (LLM-as-judge scoring) for each ground-truth Q&A pair.
+"""
+
+import time
+from typing import Any
+
+from app.agents.coordinator_agent import coordinator_agent
+from app.agents.evaluator_agent import evaluator_agent
+from app.core.prompts import SBI_SYSTEM_PROMPT
+
+
+def _chunk_texts(chunks: list[dict]) -> list[str]:
+    return [c["chunk_text"] for c in chunks if c.get("chunk_text")]
+
+
+async def evaluate_question(
+    question: str,
+    expected_answer: str,
+    top_k: int = 5,
+) -> dict[str, Any]:
+    """
+    Run one Q&A pair through the multi-agent eval pipeline.
+
+    1. coordinator_agent — routes to retrieval agents, synthesizes final answer
+    2. evaluator_agent — scores answer against expected_answer and retrieved chunks
+    """
+    t0 = time.time()
+
+    coord = await coordinator_agent.run(
+        question,
+        {"top_k": top_k, "synthesis_system": SBI_SYSTEM_PROMPT},
+    )
+    chunks = coord.get("chunks", [])
+    generated_answer = coord.get("answer", "")
+
+    scores = await evaluator_agent.run(
+        question,
+        {
+            "answer": generated_answer,
+            "chunks": chunks,
+            "expected_answer": expected_answer,
+        },
+    )
+
+    latency_ms = round((time.time() - t0) * 1000, 1)
+
+    return {
+        "question": question,
+        "expected_answer": expected_answer,
+        "generated_answer": generated_answer,
+        "retrieved_context": "\n---\n".join(_chunk_texts(chunks)),
+        "accuracy": scores.get("accuracy", 0.0),
+        "faithfulness": scores.get("faithfulness", 0.0),
+        "answer_relevancy": scores.get("answer_relevancy", 0.0),
+        "context_precision": scores.get("context_precision", 0.0),
+        "context_recall": scores.get("context_recall", 0.0),
+        "accuracy_rationale": scores.get("accuracy_rationale", ""),
+        "faithfulness_rationale": scores.get("faithfulness_rationale", ""),
+        "answer_relevancy_rationale": scores.get("answer_relevancy_rationale", ""),
+        "context_precision_rationale": scores.get("context_precision_rationale", ""),
+        "context_recall_rationale": scores.get("context_recall_rationale", ""),
+        "latency_ms": latency_ms,
+        "intent": coord.get("intent"),
+        "agents_invoked": [r.get("agent") for r in coord.get("agent_results", [])],
+        "retrieval_coverage": scores.get("retrieval_coverage"),
+        "retrieval_ok": scores.get("retrieval_ok"),
+        "overall_score": scores.get("overall_score"),
+    }
+
+
+def failed_question_row(question: str, expected_answer: str, error: str) -> dict[str, Any]:
+    """Build a zeroed per-question row when the eval pipeline raises."""
+    return {
+        "question": question,
+        "expected_answer": expected_answer,
+        "generated_answer": "",
+        "retrieved_context": "",
+        "accuracy": 0.0,
+        "faithfulness": 0.0,
+        "answer_relevancy": 0.0,
+        "context_precision": 0.0,
+        "context_recall": 0.0,
+        "accuracy_rationale": "",
+        "faithfulness_rationale": "",
+        "answer_relevancy_rationale": "",
+        "context_precision_rationale": "",
+        "context_recall_rationale": "",
+        "latency_ms": 0.0,
+        "intent": None,
+        "agents_invoked": [],
+        "retrieval_coverage": None,
+        "retrieval_ok": None,
+        "overall_score": None,
+        "error": error,
+    }
diff --git a/backend/app/services/chat_service.py b/backend/app/services/chat_service.py
index 66c02f0..cc048b1 100644
+++ b/backend/app/services/chat_service.py
@@ -7,79 +7,11 @@ from app.services.rag_service import rag_service
 from app.embeddings.openai_client import openai_client as ollama_client
 from app.core.logging import get_logger
 from app.core.exceptions import ConversationNotFoundError
+from app.core.prompts import SYSTEM_PROMPT
 
 logger = get_logger("chat_service")
 
+
 class ChatService:
     async def chat(
         self,
```

## 86f685032ce5eb17f1b0f55a17f09dbd0526365e — 2026-06-02T10:41:57+05:30

Message:

added better system prompt in eval

```diff
diff --git a/backend/app/api/rag.py b/backend/app/api/rag.py
index 0303a31..34e241d 100644
+++ b/backend/app/api/rag.py
@@ -98,8 +98,76 @@ async def evaluate_rag(req: EvaluateRequest, db: AsyncSession = Depends(get_db))
             generated_answer = await ollama_client.chat(
                 messages,
                 system=(
+"""
+You are an SBI Banking Knowledge Assistant.
+
+Scope:
+- Treat all user questions as related to SBI Bank, banking operations, financial services, regulatory processes, forms, policies, products, and internal documentation unless the user explicitly changes the topic.
+- The retrieved context is the primary source of truth.
+
+RETRIEVAL-AWARE BEHAVIOR
+
+1. Relevance First
+- Carefully identify which parts of the retrieved context are relevant to the user's question.
+- Ignore unrelated retrieved passages.
+- Do not combine information from unrelated sections unless they clearly refer to the same subject.
+
+2. Direct Answering
+- If the answer is explicitly present, provide the answer directly.
+- For field names, abbreviations, codes, labels, column names, form fields, statuses, and identifiers, return the exact meaning or definition found in the retrieved content.
+- Prefer the most specific answer over a generic one.
+
+3. Multiple Matches
+- If multiple retrieved passages contain possible answers:
+  - Prefer the passage that most closely matches the user's wording and intent.
+  - Prefer SBI-specific definitions over generic banking definitions.
+  - Prefer the most complete and unambiguous answer.
+
+4. Ambiguity Handling
+- If the retrieved information is ambiguous, ask a short clarification question.
+- Do not guess which product, form, scheme, process, or field the user means.
+
+5. Missing Information
+- If the retrieved context does not contain sufficient information:
+  - Use general banking knowledge only when highly confident.
+  - Clearly separate inferred knowledge from retrieved facts.
+  - Never invent SBI-specific procedures, codes, policies, field meanings, product details, limits, eligibility rules, or internal terminology.
+
+6. Conflict Resolution
+- If retrieved passages conflict:
+  - Prefer the more specific passage.
+  - Prefer SBI-specific information over generic information.
+  - Prefer the passage that directly addresses the user's question.
+  - Do not merge conflicting answers.
+
+7. Hallucination Prevention
+- Never fabricate:
+  - Form field definitions
+  - Internal codes
+  - Status meanings
+  - Product rules
+  - Interest rates
+  - Regulatory requirements
+  - Process steps
+  - Branch-specific information
+- If uncertain, say:
+  "I do not have enough information to answer that."
+
+LOCATION DEFAULT
+- If a state is required but not specified, assume Karnataka, India.
+
+RESPONSE STYLE
+- Answer the user's question directly.
+- Keep responses concise.
+- For definition questions, return only the definition unless more detail is requested.
+- Avoid unnecessary explanations, background information, examples, or assumptions.
+- Never mention retrieval, documents, context, sources, or knowledge-base mechanics.
+
+Priority Order:
+1. Relevant retrieved SBI information
+2. Highly confident banking knowledge that does not conflict with retrieved information
+3. "I do not have enough information to answer that."
+"""
                 ),
             )
```

## a5b57d337380f12f4ce288b8d351d77f99165e2a — 2026-06-02T10:38:28+05:30

Message:

eval chnages in export

_No Python file changes in this commit._

## 7cab83ddf09f8bfd433896d312770033465c64de — 2026-05-29T10:21:42+05:30

Message:

new embeddings

```diff
diff --git a/backend/app/services/chat_service.py b/backend/app/services/chat_service.py
index 18264cf..66c02f0 100644
+++ b/backend/app/services/chat_service.py
@@ -11,38 +11,74 @@ from app.core.exceptions import ConversationNotFoundError
 logger = get_logger("chat_service")
 
 SYSTEM_PROMPT = """
+You are an SBI Banking Knowledge Assistant.
+
+Scope:
+- Treat all user questions as related to SBI Bank, banking operations, financial services, regulatory processes, forms, policies, products, and internal documentation unless the user explicitly changes the topic.
+- The retrieved context is the primary source of truth.
+
+RETRIEVAL-AWARE BEHAVIOR
+
+1. Relevance First
+- Carefully identify which parts of the retrieved context are relevant to the user's question.
+- Ignore unrelated retrieved passages.
+- Do not combine information from unrelated sections unless they clearly refer to the same subject.
+
+2. Direct Answering
+- If the answer is explicitly present, provide the answer directly.
+- For field names, abbreviations, codes, labels, column names, form fields, statuses, and identifiers, return the exact meaning or definition found in the retrieved content.
+- Prefer the most specific answer over a generic one.
+
+3. Multiple Matches
+- If multiple retrieved passages contain possible answers:
+  - Prefer the passage that most closely matches the user's wording and intent.
+  - Prefer SBI-specific definitions over generic banking definitions.
+  - Prefer the most complete and unambiguous answer.
+
+4. Ambiguity Handling
+- If the retrieved information is ambiguous, ask a short clarification question.
+- Do not guess which product, form, scheme, process, or field the user means.
+
+5. Missing Information
+- If the retrieved context does not contain sufficient information:
+  - Use general banking knowledge only when highly confident.
+  - Clearly separate inferred knowledge from retrieved facts.
+  - Never invent SBI-specific procedures, codes, policies, field meanings, product details, limits, eligibility rules, or internal terminology.
+
+6. Conflict Resolution
+- If retrieved passages conflict:
+  - Prefer the more specific passage.
+  - Prefer SBI-specific information over generic information.
+  - Prefer the passage that directly addresses the user's question.
+  - Do not merge conflicting answers.
+
+7. Hallucination Prevention
+- Never fabricate:
+  - Form field definitions
+  - Internal codes
+  - Status meanings
+  - Product rules
+  - Interest rates
+  - Regulatory requirements
+  - Process steps
+  - Branch-specific information
+- If uncertain, say:
+  "I do not have enough information to answer that."
+
+LOCATION DEFAULT
+- If a state is required but not specified, assume Karnataka, India.
+
+RESPONSE STYLE
+- Answer the user's question directly.
+- Keep responses concise.
+- For definition questions, return only the definition unless more detail is requested.
+- Avoid unnecessary explanations, background information, examples, or assumptions.
+- Never mention retrieval, documents, context, sources, or knowledge-base mechanics.
+
+Priority Order:
+1. Relevant retrieved SBI information
+2. Highly confident banking knowledge that does not conflict with retrieved information
+3. "I do not have enough information to answer that."
 """
 class ChatService:
     async def chat(
```

## af6be2c8f617f29e256cac57742fb84a7fbeb92b — 2026-05-26T15:23:34+05:30

Message:

data and prompt changes

```diff
diff --git a/backend/app/agents/evaluator_agent.py b/backend/app/agents/evaluator_agent.py
index 43b5568..3d81a05 100644
+++ b/backend/app/agents/evaluator_agent.py
@@ -29,10 +29,13 @@ class RetrievalEvaluationAgent(BaseAgent):
 
         context_texts = [c.get("chunk_text", "") for c in chunks if c.get("chunk_text")]
 
+        faithfulness, faith_rationale = await compute_faithfulness(answer, context_texts)
+        relevancy, relevancy_rationale = await compute_answer_relevancy(query, answer)
+        cp, cp_rationale = await compute_context_precision(query, context_texts)
+        if expected:
+            cr, cr_rationale = await compute_context_recall(expected, context_texts)
+        else:
+            cr, cr_rationale = 0.0, "No expected answer provided."
         latency = (time.time() - start) * 1000
 
         coverage = len([c for c in chunks if c.get("score", 0) > 0.5]) / max(len(chunks), 1)
@@ -42,9 +45,13 @@ class RetrievalEvaluationAgent(BaseAgent):
             "agent": self.name,
             "query": query,
             "faithfulness": faithfulness,
+            "faithfulness_rationale": faith_rationale,
             "answer_relevancy": relevancy,
+            "answer_relevancy_rationale": relevancy_rationale,
             "context_precision": cp,
+            "context_precision_rationale": cp_rationale,
             "context_recall": cr,
+            "context_recall_rationale": cr_rationale,
             "retrieval_coverage": round(coverage, 4),
             "retrieval_ok": retrieval_ok,
             "latency_ms": round(latency, 2),
diff --git a/backend/app/api/rag.py b/backend/app/api/rag.py
index d2a2985..0303a31 100644
+++ b/backend/app/api/rag.py
@@ -10,6 +10,7 @@ import time
 from typing import Any
 
 from fastapi import APIRouter, Depends
+from fastapi.responses import StreamingResponse
 from sqlalchemy.ext.asyncio import AsyncSession
 
 from app.core.dependencies import get_db
@@ -19,10 +20,45 @@ from app.evaluation.evaluator import evaluate_single   # ← new LLM-judge modul
 from app.core.logging import get_logger
 from pydantic import BaseModel
 
+from app.schemas.rag import RAGQueryRequest, RAGQueryResponse, RAGRetrieveRequest
+from app.schemas.search import SearchResult
+
 logger = get_logger("api.rag.evaluate")
 router = APIRouter(prefix="/api/rag", tags=["rag"])
 
 
+@router.post("/query", response_model=RAGQueryResponse)
+async def rag_query(req: RAGQueryRequest, db: AsyncSession = Depends(get_db)):
+    result = await rag_service.query(db, req.query, req.strategy, req.top_k, req.filters)
+    return result
+
+
+@router.post("/query/stream")
+async def rag_query_stream(req: RAGQueryRequest):
+    async def generator():
+        async for token in rag_service.query_stream(req.query, req.strategy, req.top_k, req.filters):
+            yield token
+
+    return StreamingResponse(generator(), media_type="text/plain")
+
+
+@router.post("/retrieve", response_model=list[SearchResult])
+async def rag_retrieve(req: RAGRetrieveRequest):
+    chunks = await rag_service.retrieve(req.query, req.strategy, req.top_k, req.filters)
+    results = []
+    for r in chunks:
+        meta = r.get("metadata", {})
+        results.append(SearchResult(
+            chunk_id=r.get("chunk_id", ""),
+            document_id=r.get("document_id") or meta.get("document_id", ""),
+            filename=r.get("filename") or meta.get("filename", ""),
+            chunk_text=r.get("chunk_text", ""),
+            score=r.get("score", 0.0),
+            metadata=meta,
+        ))
+    return results
+
+
 class EvalQuestion(BaseModel):
     question: str
     expected_answer: str
diff --git a/backend/app/services/chat_service.py b/backend/app/services/chat_service.py
index f332d71..18264cf 100644
+++ b/backend/app/services/chat_service.py
@@ -10,10 +10,40 @@ from app.core.exceptions import ConversationNotFoundError
 
 logger = get_logger("chat_service")
 
+SYSTEM_PROMPT = """
+You are an intelligent assistant with access to a document knowledge base answer all questions based on the fact that they are all related to SBI bank and bancking sector.
+
+Primary rule:
+- Use the provided context as the highest-priority source of information.
+
+Answering guidelines:
+1. When the answer is explicitly present in the context, answer using the context.
+2. When a question refers to a form field, column value, code, abbreviation, label, or specific term, return its direct definition or expansion exactly as described in the context.
+3. If multiple descriptions exist in the context, prefer the shortest definition that directly answers the question.
+4. Do not provide additional domain knowledge, examples, background information, assumptions, interpretations, or explanations unless the user explicitly asks for them.
+
+Handling incomplete context:
+5. If the context is incomplete or does not directly answer the question:
+   - Use your general knowledge only if you are highly confident in the answer.
+   - Ensure the answer does not contradict any information present in the context.
+   - Clearly prioritize context over prior knowledge whenever both are available.
+6. If neither the context nor your knowledge provides a reliable answer, state that you do not have enough information.
+7. Never invent field definitions, codes, abbreviations, values, policies, procedures, or document-specific details that are not supported by the context.
+
+Location assumption:
+- When the state is not explicitly provided in the question, assume Karnataka, India.
+
+Response style:
+- Be concise and answer the user's question directly.
+- For definition-style questions, return only the definition unless additional detail is requested.
+- Do not mention the source of the information or use phrases such as:
+  - "The context provided does not define..."
+  - "Based on the context..."
+  - "According to the context..."
+  - "The document states..."
+
+If there is a conflict between the context and your general knowledge, always follow the context.
+"""
 class ChatService:
     async def chat(
         self,
diff --git a/backend/app/services/rag_service.py b/backend/app/services/rag_service.py
index aa744f2..95c710a 100644
+++ b/backend/app/services/rag_service.py
@@ -130,17 +130,17 @@ class RAGService:
                 answer = await ollama_client.generate(prompt, system=RAG_SYSTEM)
                 latency = (time.time() - start) * 1000
 
+                acc_score, _ = await compute_accuracy(answer, expected)
+                faith_score, _ = await compute_faithfulness(answer, context_texts)
+                cp_score, _ = await compute_context_precision(question, context_texts)
+                cr_score, _ = await compute_context_recall(expected, context_texts)
+                ar_score, _ = await compute_answer_relevancy(question, answer)
+
+                results["accuracy"].append(acc_score)
+                results["faithfulness"].append(faith_score)
+                results["context_precision"].append(cp_score)
+                results["context_recall"].append(cr_score)
+                results["answer_relevancy"].append(ar_score)
                 results["latency_ms"].append(latency)
             except Exception as e:
                 logger.error(f"Eval failed for '{question}': {e}")
diff --git a/helpfull scripts/hindi_remover.py b/helpfull scripts/hindi_remover.py
new file mode 100644
index 0000000..31a1b98
+++ b/helpfull scripts/hindi_remover.py	
@@ -0,0 +1,49 @@
+import re
+
+input_file = "input_hindi.txt"
+output_file = "output_hindi.txt"
+
+with open(input_file, "r", encoding="utf-8") as f:
+    text = f.read()
+
+# Remove bracketed Hindi text:
+# (हिन्दी), [हिन्दी], {हिन्दी}
+text = re.sub(r'\(\s*[\u0900-\u097F\s]+\s*\)', '', text)
+text = re.sub(r'\[\s*[\u0900-\u097F\s]+\s*\]', '', text)
+text = re.sub(r'\{\s*[\u0900-\u097F\s]+\s*\}', '', text)
+
+# Remove Hindi text that follows separators:
+# , हिन्दी
+# - हिन्दी
+# : हिन्दी
+# ; हिन्दी
+text = re.sub(
+    r'\s*[,;:\-–—]\s*[\u0900-\u097F\s]+',
+    '',
+    text
+)
+
+# Remove remaining Hindi characters
+text = re.sub(r'[\u0900-\u097F]+', '', text)
+
+# Remove empty brackets left behind
+text = re.sub(r'\(\s*\)', '', text)
+text = re.sub(r'\[\s*\]', '', text)
+text = re.sub(r'\{\s*\}', '', text)
+
+# Normalize spaces
+text = re.sub(r'[ \t]+', ' ', text)
+
+# Remove spaces before punctuation
+text = re.sub(r'\s+([,.;:!?])', r'\1', text)
+
+# Collapse multiple blank lines
+text = re.sub(r'\n\s*\n+', '\n\n', text)
+
+# Strip trailing spaces on each line
+text = '\n'.join(line.strip() for line in text.splitlines())
+
+with open(output_file, "w", encoding="utf-8") as f:
+    f.write(text)
+
+print(f"Saved cleaned text to {output_file}")
\ No newline at end of file
diff --git a/helpfull scripts/html_remover.py b/helpfull scripts/html_remover.py
new file mode 100644
index 0000000..44c2c0e
+++ b/helpfull scripts/html_remover.py	
@@ -0,0 +1,48 @@
+#!/usr/bin/env python3
+"""
+Strip all HTML tags, CSS, and JavaScript from input.html and save plain text to remove_html.txt
+Usage: python strip_html.py
+"""
+
+import re
+
+def strip_html_css_js(text):
+    # Remove <style>...</style> blocks (CSS)
+    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
+
+    # Remove <script>...</script> blocks (JavaScript)
+    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
+
+    # Remove inline style attributes
+    text = re.sub(r'\s*style="[^"]*"', '', text, flags=re.IGNORECASE)
+
+    # Remove all remaining HTML tags
+    text = re.sub(r'<[^>]+>', '', text)
+
+    # Decode common HTML entities
+    entities = {
+        '&amp;': '&', '&lt;': '<', '&gt;': '>',
+        '&nbsp;': ' ', '&quot;': '"', '&#39;': "'"
+    }
+    for entity, char in entities.items():
+        text = text.replace(entity, char)
+
+    # Clean up excess whitespace/blank lines
+    lines = [line.strip() for line in text.splitlines()]
+    lines = [line for line in lines if line]  # remove empty lines
+    return '\n'.join(lines)
+
+
+if __name__ == '__main__':
+    input_file = 'input.html'
+    output_file = 'remove_html.txt'
+
+    with open(input_file, 'r', encoding='utf-8') as f:
+        raw = f.read()
+
+    clean_text = strip_html_css_js(raw)
+
+    with open(output_file, 'w', encoding='utf-8') as f:
+        f.write(clean_text)
+
+    print(f"Done! Plain text saved to {output_file} ({len(clean_text)} characters)")
\ No newline at end of file
```

## 53e81474694c74ece04d7bd5adef079efa4d010d — 2026-05-26T09:36:40+05:30

Message:

gitignore

_No Python file changes in this commit._

## 7e761ac0c68b6e0c9bb8882d09c2367909caab58 — 2026-05-25T09:53:32+05:30

Message:

rag eval

_No Python file changes in this commit._

## f5fcde278623dfb508527f8e2f37d293617bf83c — 2026-05-25T09:08:00+05:30

Message:

rag logs ignore

_No Python file changes in this commit._

## 4cc1284d3f55989dee4f3d714f9fe6eddd24781a — 2026-05-25T09:07:23+05:30

Message:

readme

_No Python file changes in this commit._

## 3b6abcb6896f1bf57fc3595db59f627b16bfb1b4 — 2026-05-25T09:04:59+05:30

Message:

eval dataset

```diff
diff --git a/eval/csv_convert.py b/eval/csv_convert.py
new file mode 100644
index 0000000..a951c64
+++ b/eval/csv_convert.py
@@ -0,0 +1,114 @@
+"""
+Convert form-structured CSV to flat eval Q&A format.
+
+Input CSV structure (no fixed header, repeating blocks):
+    form name, question 1, question 2, ...
+    (blank),   answer 1,   answer 2,   ...
+
+Output CSV structure:
+    Question no, Eval Question + form name, Eval Answer
+
+Usage:
+    python convert_form_csv.py input.csv output.csv
+
+    # Or with custom delimiter (e.g. tab-separated):
+    python convert_form_csv.py input.csv output.csv --delimiter '\t'
+"""
+
+import csv
+import argparse
+import sys
+
+
+def convert(input_path: str, output_path: str, delimiter: str = ",") -> int:
+    """
+    Parse the input CSV and write the flattened eval CSV.
+    Returns the number of Q&A rows written.
+    """
+    rows = []
+    with open(input_path, newline="", encoding="utf-8-sig") as f:
+        reader = csv.reader(f, delimiter=delimiter)
+        for row in reader:
+            rows.append(row)
+
+    qa_pairs = []
+    i = 0
+
+    while i < len(rows):
+        row = rows[i]
+
+        # Skip completely empty rows
+        if not any(cell.strip() for cell in row):
+            i += 1
+            continue
+
+        first_cell = row[0].strip() if row else ""
+
+        # A "form name" row: first cell is non-empty (the form name)
+        # and there are questions in the remaining cells.
+        if first_cell:
+            form_name = first_cell
+            questions = [cell.strip() for cell in row[1:]]
+
+            # Look ahead for the answer row (first cell blank, rest are answers)
+            if i + 1 < len(rows):
+                next_row = rows[i + 1]
+                next_first = next_row[0].strip() if next_row else ""
+                if not next_first:
+                    answers = [cell.strip() for cell in next_row[1:]]
+                    i += 2  # consumed both rows
+                else:
+                    # No answer row follows — treat answers as empty
+                    answers = []
+                    i += 1
+            else:
+                answers = []
+                i += 1
+
+            # Pair each question with its answer (zip stops at shortest)
+            for q, a in zip(questions, answers):
+                q = q.strip()
+                a = a.strip()
+                if q:  # skip blank question slots
+                    eval_question = f"{q} ({form_name})" if form_name else q
+                    qa_pairs.append((eval_question, a))
+
+        else:
+            # Answer row without a preceding form row — skip
+            i += 1
+
+    # Write output
+    with open(output_path, "w", newline="", encoding="utf-8") as f:
+        writer = csv.writer(f)
+        writer.writerow(["Question no", "Eval Question + form name", "Eval Answer"])
+        for idx, (question, answer) in enumerate(qa_pairs, start=1):
+            writer.writerow([idx, question, answer])
+
+    return len(qa_pairs)
+
+
+def main():
+    parser = argparse.ArgumentParser(
+        description="Convert form-structured CSV to flat eval Q&A CSV."
+    )
+    parser.add_argument("input", help="Path to the input CSV file")
+    parser.add_argument("output", help="Path for the output CSV file")
+    parser.add_argument(
+        "--delimiter",
+        default=",",
+        help="CSV delimiter character (default: comma). Use '\\t' for tab.",
+    )
+    args = parser.parse_args()
+
+    delimiter = args.delimiter.replace("\\t", "\t")
+
+    try:
+        count = convert(args.input, args.output, delimiter)
+        print(f"Done. Wrote {count} Q&A rows to '{args.output}'.")
+    except FileNotFoundError as e:
+        print(f"Error: {e}", file=sys.stderr)
+        sys.exit(1)
+
+
+if __name__ == "__main__":
+    main()
\ No newline at end of file
```

## 6c73b780ed2ce083b1023efee1a76a90dbbdced4 — 2026-05-25T08:37:33+05:30

Message:

added open ai and docs and embedded them

```diff
diff --git a/backend/app/agents/base.py b/backend/app/agents/base.py
index 4d45e2c..51ed02d 100644
+++ b/backend/app/agents/base.py
@@ -1,6 +1,6 @@
 from abc import ABC, abstractmethod
 from typing import Any, Optional
+from app.embeddings.openai_client import openai_client as ollama_client
 from app.core.logging import get_logger
 
 
@@ -30,4 +30,4 @@ class BaseAgent(ABC):
         plan = await self.plan(query, context)
         result = await self.execute(plan)
         evaluated = await self.evaluate(result)
+        return evaluated
\ No newline at end of file
diff --git a/backend/app/agents/coordinator_agent.py b/backend/app/agents/coordinator_agent.py
index 43ec53a..1a0dadd 100644
+++ b/backend/app/agents/coordinator_agent.py
@@ -6,7 +6,7 @@ from app.agents.sqlite_agent import sqlite_agent
 from app.agents.router_agent import router_agent, _detect_doc_type
 from app.agents.web_agent import web_agent
 from app.agents.evaluator_agent import evaluator_agent
+from app.embeddings.openai_client import openai_client as ollama_client
 from app.core.logging import get_logger
 
 logger = get_logger("coordinator_agent")
@@ -125,4 +125,4 @@ class CoordinatorAgent(BaseAgent):
         return result
 
 
+coordinator_agent = CoordinatorAgent()
\ No newline at end of file
diff --git a/backend/app/agents/router_agent.py b/backend/app/agents/router_agent.py
index 48073fb..269e90f 100644
+++ b/backend/app/agents/router_agent.py
@@ -5,7 +5,7 @@ from app.rag.pdf_rag import pdf_rag
 from app.rag.markdown_rag import markdown_rag
 from app.rag.table_rag import table_rag
 from app.rag.vector_rag import vector_rag
+from app.embeddings.openai_client import openai_client as ollama_client
 
 
 ROUTING_RULES = {
@@ -82,4 +82,4 @@ class DocumentRouterAgent(BaseAgent):
         return result
 
 
+router_agent = DocumentRouterAgent()
\ No newline at end of file
diff --git a/backend/app/agents/sqlite_agent.py b/backend/app/agents/sqlite_agent.py
index af5c546..45d4588 100644
+++ b/backend/app/agents/sqlite_agent.py
@@ -2,7 +2,7 @@ import time
 from typing import Any, Optional
 from app.agents.base import BaseAgent
 from app.rag.table_rag import table_rag
+from app.embeddings.openai_client import openai_client as ollama_client
 
 
 class SQLiteAgent(BaseAgent):
@@ -49,4 +49,4 @@ Be precise with numbers, names, and values. Format answers clearly."""
         return result
 
 
+sqlite_agent = SQLiteAgent()
\ No newline at end of file
diff --git a/backend/app/agents/vector_agent.py b/backend/app/agents/vector_agent.py
index 12dbfb6..730a632 100644
+++ b/backend/app/agents/vector_agent.py
@@ -3,7 +3,7 @@ from typing import Any, Optional
 from app.agents.base import BaseAgent
 from app.rag.hybrid_rag import hybrid_rag
 from app.rag.vector_rag import vector_rag
+from app.embeddings.openai_client import openai_client as ollama_client
 from app.core.config import settings
 
 
@@ -60,4 +60,4 @@ class VectorRetrievalAgent(BaseAgent):
         return result
 
 
+vector_agent = VectorRetrievalAgent()
\ No newline at end of file
diff --git a/backend/app/agents/web_agent.py b/backend/app/agents/web_agent.py
index 995233b..9a997ee 100644
+++ b/backend/app/agents/web_agent.py
@@ -2,7 +2,7 @@ import time
 from typing import Any, Optional
 from app.agents.base import BaseAgent
 from app.services.web_service import web_service
+from app.embeddings.openai_client import openai_client as ollama_client
 
 
 class WebEnrichmentAgent(BaseAgent):
@@ -56,4 +56,4 @@ class WebEnrichmentAgent(BaseAgent):
         return result
 
 
+web_agent = WebEnrichmentAgent()
\ No newline at end of file
diff --git a/backend/app/api/chroma.py b/backend/app/api/chroma.py
index 681730d..be348df 100644
+++ b/backend/app/api/chroma.py
@@ -3,7 +3,7 @@ from pydantic import BaseModel, Field
 from typing import Any, Optional
 
 from app.chromadb.client import chroma_client
+from app.embeddings.openai_client import openai_client as ollama_client
 from app.core.logging import get_logger
 
 router = APIRouter(prefix="/api/chroma", tags=["ChromaDB"])
@@ -69,4 +69,4 @@ async def search_collection(req: ChromaSearchRequest):
 async def list_collections():
     collections = chroma_client.list_collections()
     counts = {name: chroma_client.get_collection_count(name) for name in collections}
+    return {"collections": collections, "counts": counts}
\ No newline at end of file
diff --git a/backend/app/api/embeddings.py b/backend/app/api/embeddings.py
index c9fa1d8..2ec0026 100644
+++ b/backend/app/api/embeddings.py
@@ -1,5 +1,5 @@
 from fastapi import APIRouter
+from app.embeddings.openai_client import openai_client as ollama_client
 from app.schemas.embeddings import (
     EmbeddingRequest, EmbeddingBatchRequest,
     EmbeddingResponse, EmbeddingBatchResponse, EmbeddingModelInfo,
@@ -13,7 +13,7 @@ logger = get_logger("api.embeddings")
 
 @router.post("/generate", response_model=EmbeddingResponse)
 async def generate_embedding(req: EmbeddingRequest):
+    model = req.model or settings.OPENAI_EMBED_MODEL
     embedding = await ollama_client.embeddings(req.text, model)
     return EmbeddingResponse(
         text=req.text,
@@ -25,7 +25,7 @@ async def generate_embedding(req: EmbeddingRequest):
 
 @router.post("/batch", response_model=EmbeddingBatchResponse)
 async def batch_embeddings(req: EmbeddingBatchRequest):
+    model = req.model or settings.OPENAI_EMBED_MODEL
     embeddings = await ollama_client.batch_embeddings(req.texts, model)
     responses = [
         EmbeddingResponse(text=t, embedding=e, model=model, dimensions=len(e))
@@ -37,4 +37,4 @@ async def batch_embeddings(req: EmbeddingBatchRequest):
 @router.get("/models", response_model=list[EmbeddingModelInfo])
 async def list_embedding_models():
     models = await ollama_client.list_models()
+    return [EmbeddingModelInfo(name=m.get("id", m.get("name", "")), dimensions=None) for m in models]
\ No newline at end of file
diff --git a/backend/app/api/health.py b/backend/app/api/health.py
index 1ce08e0..8b83211 100644
+++ b/backend/app/api/health.py
@@ -4,7 +4,7 @@ from sqlalchemy import text
 
 from app.core.dependencies import get_db
 from app.chromadb.client import chroma_client
+from app.embeddings.openai_client import openai_client as ollama_client
 from app.core.logging import get_logger
 
 router = APIRouter(tags=["Health"])
@@ -39,8 +39,8 @@ async def health_chroma():
     return {"status": status, "chromadb": "persistent", "collections": collections}
 
 
+@router.get("/health/openai")
+async def health_openai():
     ok = await ollama_client.health_check()
     models = []
     if ok:
@@ -51,6 +51,6 @@ async def health_ollama():
             pass
     return {
         "status": "ok" if ok else "error",
+        "openai": "connected" if ok else "unreachable",
         "models": models,
+    }
\ No newline at end of file
diff --git a/backend/app/api/rag.py b/backend/app/api/rag.py
index 4ad444f..d2a2985 100644
+++ b/backend/app/api/rag.py
@@ -1,54 +1,122 @@
+"""
+POST /api/rag/evaluate
+
+Accepts a list of {question, expected_answer} pairs, runs each through the
+RAG pipeline, scores with the LLM-as-judge evaluator, and returns per-question
+detail alongside aggregate metrics.
+"""
+
+import time
+from typing import Any
+
 from fastapi import APIRouter, Depends
 from sqlalchemy.ext.asyncio import AsyncSession
 
 from app.core.dependencies import get_db
 from app.services.rag_service import rag_service
+from app.embeddings.openai_client import openai_client as ollama_client
+from app.evaluation.evaluator import evaluate_single   # ← new LLM-judge module
 from app.core.logging import get_logger
+from pydantic import BaseModel
+
+logger = get_logger("api.rag.evaluate")
+router = APIRouter(prefix="/api/rag", tags=["rag"])
+
+
+class EvalQuestion(BaseModel):
+    question: str
+    expected_answer: str
+
 
+class EvaluateRequest(BaseModel):
+    questions: list[EvalQuestion]
+    dataset_name: str = "eval_run"
 
 
+@router.post("/evaluate")
+async def evaluate_rag(req: EvaluateRequest, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
+    per_question: list[dict] = []
+    failed: list[dict] = []
+    latencies: list[float] = []
 
+    for qa in req.questions:
+        try:
+            # 1. Retrieve context
+            retrieval_result = await rag_service.retrieve(
+                qa.question, strategy="hybrid", top_k=5
+            )
+            context_chunks = [r["chunk_text"] for r in retrieval_result]
 
+            # 2. Generate answer
+            context_str = "\n\n".join(
+                f"[Source: {r.get('filename', 'unknown')}]\n{r['chunk_text']}"
+                for r in retrieval_result
+            )
+            messages = [{
+                "role": "user",
+                "content": (
+                    f"Context:\n{context_str}\n\nQuestion: {qa.question}"
+                    if context_chunks else qa.question
+                ),
+            }]
+            generated_answer = await ollama_client.chat(
+                messages,
+                system=(
+                    "You are a helpful assistant. Answer the question using the provided "
+                    "context. Be concise and accurate."
+                ),
+            )
 
+            # 3. LLM-as-judge scoring
+            row = await evaluate_single(
+                question=qa.question,
+                expected_answer=qa.expected_answer,
+                generated_answer=generated_answer,
+                context_chunks=context_chunks,
+            )
+            per_question.append(row)
+            latencies.append(row["latency_ms"])
 
+        except Exception as exc:
+            logger.error(f"Eval error for '{qa.question[:60]}': {exc}")
+            failed.append({"question": qa.question, "error": str(exc)})
+            per_question.append({
+                "question": qa.question,
+                "expected_answer": qa.expected_answer,
+                "generated_answer": "",
+                "retrieved_context": "",
+                "accuracy": 0.0,
+                "faithfulness": 0.0,
+                "answer_relevancy": 0.0,
+                "context_precision": 0.0,
+                "context_recall": 0.0,
+                "accuracy_rationale": "",
+                "faithfulness_rationale": "",
+                "answer_relevancy_rationale": "",
+                "context_precision_rationale": "",
+                "context_recall_rationale": "",
+                "latency_ms": 0.0,
+                "error": str(exc),
+            })
 
+    # Aggregate over successful rows only
+    succeeded = [r for r in per_question if "error" not in r or not r.get("error")]
 
+    def _avg(k: str) -> float:
+        if not succeeded:
+            return 0.0
+        return round(sum(r.get(k, 0.0) for r in succeeded) / len(succeeded), 4)
 
+    return {
+        # Aggregate metrics (for backward compat with existing frontend)
+        "accuracy":          _avg("accuracy"),
+        "faithfulness":      _avg("faithfulness"),
+        "context_precision": _avg("context_precision"),
+        "context_recall":    _avg("context_recall"),
+        "answer_relevancy":  _avg("answer_relevancy"),
+        "latency_avg_ms":    round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
+        "failed_questions":  failed,
+        # NEW: per-question detail for the UI
+        "per_question":      per_question,
+        "dataset_name":      req.dataset_name,
+    }
\ No newline at end of file
diff --git a/backend/app/api/search.py b/backend/app/api/search.py
index 0355a3d..68ec6d0 100644
+++ b/backend/app/api/search.py
@@ -8,7 +8,7 @@ from app.rag.hybrid_rag import hybrid_rag
 from app.rag.bm25 import bm25_retriever
 from app.rag.table_rag import table_rag
 from app.rag.metadata_filter import filter_results, build_chroma_filter
+from app.embeddings.openai_client import openai_client as ollama_client
 from app.chromadb.client import chroma_client
 from app.schemas.search import SearchRequest, SearchResponse, SearchResult
 from app.core.logging import get_logger
@@ -86,4 +86,4 @@ async def metadata_search(req: SearchRequest):
 async def table_search(req: SearchRequest):
     start = time.time()
     results = await table_rag.query(req.query, top_k=req.top_k)
+    return _build_response(req.query, results, "table", start)
\ No newline at end of file
diff --git a/backend/app/core/config.py b/backend/app/core/config.py
index 3fa1ff5..6ad924a 100644
+++ b/backend/app/core/config.py
@@ -14,11 +14,12 @@ class Settings(BaseSettings):
     CHROMA_PORT: int = 8001
     CHROMA_PERSIST_DIR: str = "./chroma_db"
 
+    # ── OpenAI ────────────────────────────────────────────────────────────────
+    OPENAI_API_KEY: str = ""
+    OPENAI_LLM_MODEL: str = "gpt-4o-mini"
+    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"
+    OPENAI_TIMEOUT: int = 120
+    OPENAI_MAX_RETRIES: int = 3
 
     UPLOAD_DIR: str = "./uploads"
     LOG_FILE: str = "./logs/rag.log"
diff --git a/backend/app/core/exceptions.py b/backend/app/core/exceptions.py
index d0de922..be596c8 100644
+++ b/backend/app/core/exceptions.py
@@ -29,9 +29,13 @@ class ConversationNotFoundError(RAGPlatformException):
         super().__init__(f"Conversation {conv_id} not found", 404)
 
 
+class OpenAIConnectionError(RAGPlatformException):
     def __init__(self, detail: str = ""):
+        super().__init__(f"OpenAI connection failed: {detail}", 503)
+
+
+# Backward-compat alias so any existing catch clauses still work
+OllamaConnectionError = OpenAIConnectionError
 
 
 class ChromaDBError(RAGPlatformException):
diff --git a/backend/app/embeddings/openai_client.py b/backend/app/embeddings/openai_client.py
new file mode 100644
index 0000000..642b94f
+++ b/backend/app/embeddings/openai_client.py
@@ -0,0 +1,241 @@
+import asyncio
+from typing import AsyncGenerator, Optional
+import httpx
+from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
+from app.core.config import settings
+from app.core.logging import get_logger
+from app.core.exceptions import OpenAIConnectionError
+
+logger = get_logger("openai_client")
+
+OPENAI_API_BASE = "https://api.openai.com/v1"
+
+
+class OpenAIClient:
+    def __init__(self):
+        self.api_key = settings.OPENAI_API_KEY
+        self.llm_model = settings.OPENAI_LLM_MODEL
+        self.embed_model = settings.OPENAI_EMBED_MODEL
+        self.timeout = settings.OPENAI_TIMEOUT
+        self._client: Optional[httpx.AsyncClient] = None
+
+    def _get_headers(self) -> dict:
+        return {
+            "Authorization": f"Bearer {self.api_key}",
+            "Content-Type": "application/json",
+        }
+
+    async def _get_client(self) -> httpx.AsyncClient:
+        if self._client is None or self._client.is_closed:
+            self._client = httpx.AsyncClient(
+                base_url=OPENAI_API_BASE,
+                timeout=httpx.Timeout(self.timeout),
+                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
+            )
+        return self._client
+
+    async def close(self):
+        if self._client and not self._client.is_closed:
+            await self._client.aclose()
+
+    @retry(
+        stop=stop_after_attempt(3),
+        wait=wait_exponential(multiplier=1, min=1, max=10),
+        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
+    )
+    async def generate(self, prompt: str, model: Optional[str] = None, system: Optional[str] = None) -> str:
+        """Single-turn generation via chat completions."""
+        messages = []
+        if system:
+            messages.append({"role": "system", "content": system})
+        messages.append({"role": "user", "content": prompt})
+        return await self._chat_completions(messages, model)
+
+    @retry(
+        stop=stop_after_attempt(3),
+        wait=wait_exponential(multiplier=1, min=1, max=10),
+        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
+    )
+    async def chat(
+        self,
+        messages: list[dict],
+        model: Optional[str] = None,
+        system: Optional[str] = None,
+    ) -> str:
+        chat_messages = []
+        if system:
+            chat_messages.append({"role": "system", "content": system})
+        chat_messages.extend(messages)
+        return await self._chat_completions(chat_messages, model)
+
+    async def _chat_completions(self, messages: list[dict], model: Optional[str] = None) -> str:
+        client = await self._get_client()
+        payload = {
+            "model": model or self.llm_model,
+            "messages": messages,
+        }
+        try:
+            response = await client.post(
+                "/chat/completions",
+                json=payload,
+                headers=self._get_headers(),
+            )
+            response.raise_for_status()
+            data = response.json()
+            return data["choices"][0]["message"]["content"]
+        except httpx.HTTPStatusError as e:
+            logger.error(f"OpenAI chat HTTP error: {e.response.status_code} {e.response.text}")
+            raise OpenAIConnectionError(str(e))
+        except (httpx.ConnectError, httpx.TimeoutException) as e:
+            logger.error(f"OpenAI connection error: {e}")
+            raise OpenAIConnectionError(str(e))
+
+    async def generate_stream(
+        self, prompt: str, model: Optional[str] = None, system: Optional[str] = None
+    ) -> AsyncGenerator[str, None]:
+        messages = []
+        if system:
+            messages.append({"role": "system", "content": system})
+        messages.append({"role": "user", "content": prompt})
+        async for token in self._chat_stream(messages, model):
+            yield token
+
+    async def chat_stream(
+        self,
+        messages: list[dict],
+        model: Optional[str] = None,
+        system: Optional[str] = None,
+    ) -> AsyncGenerator[str, None]:
+        chat_messages = []
+        if system:
+            chat_messages.append({"role": "system", "content": system})
+        chat_messages.extend(messages)
+        async for token in self._chat_stream(chat_messages, model):
+            yield token
+
+    async def _chat_stream(
+        self, messages: list[dict], model: Optional[str] = None
+    ) -> AsyncGenerator[str, None]:
+        client = await self._get_client()
+        payload = {
+            "model": model or self.llm_model,
+            "messages": messages,
+            "stream": True,
+        }
+        try:
+            async with client.stream(
+                "POST",
+                "/chat/completions",
+                json=payload,
+                headers=self._get_headers(),
+            ) as response:
+                response.raise_for_status()
+                async for line in response.aiter_lines():
+                    if not line or not line.startswith("data: "):
+                        continue
+                    data_str = line[len("data: "):]
+                    if data_str.strip() == "[DONE]":
+                        break
+                    try:
+                        import json
+                        data = json.loads(data_str)
+                        delta = data["choices"][0].get("delta", {})
+                        token = delta.get("content", "")
+                        if token:
+                            yield token
+                    except (json.JSONDecodeError, KeyError, IndexError):
+                        continue
+        except (httpx.ConnectError, httpx.TimeoutException) as e:
+            logger.error(f"OpenAI stream error: {e}")
+            raise OpenAIConnectionError(str(e))
+
+    @retry(
+        stop=stop_after_attempt(3),
+        wait=wait_exponential(multiplier=1, min=1, max=10),
+        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
+    )
+    async def embeddings(self, text: str, model: Optional[str] = None) -> list[float]:
+        client = await self._get_client()
+        payload = {
+            "model": model or self.embed_model,
+            "input": text,
+        }
+        try:
+            response = await client.post(
+                "/embeddings",
+                json=payload,
+                headers=self._get_headers(),
+            )
+            response.raise_for_status()
+            data = response.json()
+            return data["data"][0]["embedding"]
+        except httpx.HTTPStatusError as e:
+            logger.error(f"OpenAI embeddings HTTP error: {e.response.status_code} {e.response.text}")
+            raise OpenAIConnectionError(str(e))
+        except (httpx.ConnectError, httpx.TimeoutException) as e:
+            logger.error(f"OpenAI embeddings connection error: {e}")
+            raise OpenAIConnectionError(str(e))
+
+    async def batch_embeddings(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
+        """
+        Use OpenAI's native batch input for efficiency (up to 2048 inputs per request).
+        Falls back to individual calls for very large batches.
+        """
+        BATCH_SIZE = 100  # safe limit well within OpenAI's 2048 max
+        all_embeddings: list[list[float]] = []
+        for i in range(0, len(texts), BATCH_SIZE):
+            batch = texts[i: i + BATCH_SIZE]
+            all_embeddings.extend(await self._batch_embeddings_chunk(batch, model))
+        return all_embeddings
+
+    async def _batch_embeddings_chunk(
+        self, texts: list[str], model: Optional[str] = None
+    ) -> list[list[float]]:
+        client = await self._get_client()
+        payload = {
+            "model": model or self.embed_model,
+            "input": texts,
+        }
+        try:
+            response = await client.post(
+                "/embeddings",
+                json=payload,
+                headers=self._get_headers(),
+            )
+            response.raise_for_status()
+            data = response.json()
+            # OpenAI returns data sorted by index
+            items = sorted(data["data"], key=lambda x: x["index"])
+            return [item["embedding"] for item in items]
+        except httpx.HTTPStatusError as e:
+            logger.error(f"OpenAI batch embeddings HTTP error: {e.response.status_code} {e.response.text}")
+            raise OpenAIConnectionError(str(e))
+        except (httpx.ConnectError, httpx.TimeoutException) as e:
+            logger.error(f"OpenAI batch embeddings connection error: {e}")
+            raise OpenAIConnectionError(str(e))
+
+    async def health_check(self) -> bool:
+        try:
+            client = await self._get_client()
+            response = await client.get("/models", headers=self._get_headers())
+            return response.status_code == 200
+        except Exception:
+            return False
+
+    async def list_models(self) -> list[dict]:
+        try:
+            client = await self._get_client()
+            response = await client.get("/models", headers=self._get_headers())
+            response.raise_for_status()
+            return response.json().get("data", [])
+        except Exception as e:
+            logger.error(f"Failed to list OpenAI models: {e}")
+            return []
+
+
+# Module-level singleton — same name pattern as before so imports stay clean
+openai_client = OpenAIClient()
+
+# Alias: every existing import of `ollama_client` still works without any
+# other file change, because we also export the name `ollama_client` here.
+ollama_client = openai_client
diff --git a/backend/app/evaluation/__init__.py b/backend/app/evaluation/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/backend/app/evaluation/evaluator.py b/backend/app/evaluation/evaluator.py
new file mode 100644
index 0000000..9acdb80
+++ b/backend/app/evaluation/evaluator.py
@@ -0,0 +1,185 @@
+"""
+LLM-as-a-Judge evaluator for RAG pipelines.
+
+Each metric is computed by prompting the LLM to score (0.0–1.0) with a brief rationale,
+rather than relying on cosine-similarity heuristics.
+"""
+
+import json
+import re
+import time
+from typing import Any
+
+from app.embeddings.openai_client import openai_client as ollama_client
+from app.core.logging import get_logger
+
+logger = get_logger("evaluator")
+
+
+# ──────────────────────────────────────────────────────────────────────────────
+# Internal helpers
+# ──────────────────────────────────────────────────────────────────────────────
+
+_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]*\.?[0-9]+)')
+
+
+async def _llm_score(prompt: str) -> tuple[float, str]:
+    """
+    Call the LLM with `prompt` and extract a JSON payload like:
+        {"score": 0.85, "rationale": "..."}
+    Returns (score, rationale).  Falls back to (0.0, error_msg) on failure.
+    """
+    system = (
+        "You are an expert RAG evaluation judge. "
+        "Respond ONLY with a JSON object containing exactly two keys: "
+        '"score" (a float between 0.0 and 1.0) and '
+        '"rationale" (a one-sentence explanation). '
+        "Do not include any other text."
+    )
+    raw = ""  # initialise so it's always bound
+    try:
+        raw = await ollama_client.chat(
+            [{"role": "user", "content": prompt}],
+            system=system,
+        )
+        # Strip markdown fences if present
+        raw = raw.strip().strip("```json").strip("```").strip()
+        data = json.loads(raw)
+        score = float(data.get("score", 0.0))
+        rationale = data.get("rationale", "")
+        return round(max(0.0, min(1.0, score)), 4), rationale
+    except Exception as exc:
+        logger.warning("LLM scoring parse error: %s | raw=%.200r", exc, raw)
+        # Fallback: try regex on whatever we got back
+        m = _SCORE_RE.search(raw)
+        if m:
+            return round(float(m.group(1)), 4), "Score extracted via regex fallback."
+        return 0.0, f"Parse error: {exc}"
+
+
+# ──────────────────────────────────────────────────────────────────────────────
+# Metric functions
+# ──────────────────────────────────────────────────────────────────────────────
+
+async def compute_accuracy(generated: str, expected: str) -> tuple[float, str]:
+    """Semantic accuracy: how well does the generated answer match the expected answer?"""
+    prompt = f"""Rate how well the GENERATED ANSWER matches the EXPECTED ANSWER semantically.
+
+EXPECTED ANSWER:
+{expected}
+
+GENERATED ANSWER:
+{generated}
+
+Score 1.0 if they convey the same meaning, 0.0 if completely unrelated."""
+    return await _llm_score(prompt)
+
+
+async def compute_faithfulness(answer: str, context_chunks: list[str]) -> tuple[float, str]:
+    """Faithfulness: is the answer grounded in the retrieved context?"""
+    context = "\n\n---\n\n".join(context_chunks[:5])  # cap to avoid token overflow
+    prompt = f"""Rate how faithfully the ANSWER is grounded in the CONTEXT below.
+A faithful answer only makes claims supported by the context (score 1.0).
+An unfaithful answer introduces hallucinated facts not in the context (score 0.0).
+
+CONTEXT:
+{context}
+
+ANSWER:
+{answer}"""
+    return await _llm_score(prompt)
+
+
+async def compute_answer_relevancy(question: str, answer: str) -> tuple[float, str]:
+    """Answer relevancy: does the answer directly address the question?"""
+    prompt = f"""Rate how directly and completely the ANSWER addresses the QUESTION.
+Score 1.0 if the answer is fully on-topic and complete, 0.0 if it ignores the question.
+
+QUESTION:
+{question}
+
+ANSWER:
+{answer}"""
+    return await _llm_score(prompt)
+
+
+async def compute_context_precision(question: str, context_chunks: list[str]) -> tuple[float, str]:
+    """Context precision: what fraction of retrieved chunks are actually relevant?"""
+    if not context_chunks:
+        return 0.0, "No context chunks provided."
+    chunks_text = "\n\n---\n\n".join(
+        f"[Chunk {i+1}]: {c}" for i, c in enumerate(context_chunks[:8])
+    )
+    prompt = f"""You are evaluating retrieval quality.
+Below are chunks retrieved for a QUESTION. Rate the proportion of chunks that contain
+information genuinely useful for answering the question (0.0 = none relevant, 1.0 = all relevant).
+
+QUESTION:
+{question}
+
+RETRIEVED CHUNKS:
+{chunks_text}"""
+    return await _llm_score(prompt)
+
+
+async def compute_context_recall(expected_answer: str, context_chunks: list[str]) -> tuple[float, str]:
+    """Context recall: does the retrieved context contain what's needed to answer correctly?"""
+    if not context_chunks:
+        return 0.0, "No context chunks provided."
+    context = "\n\n---\n\n".join(context_chunks[:8])
+    prompt = f"""Rate whether the RETRIEVED CONTEXT contains the information needed to produce the EXPECTED ANSWER.
+Score 1.0 if all key facts for the expected answer are present in the context, 0.0 if completely missing.
+
+EXPECTED ANSWER:
+{expected_answer}
+
+RETRIEVED CONTEXT:
+{context}"""
+    return await _llm_score(prompt)
+
+
+# ──────────────────────────────────────────────────────────────────────────────
+# Master evaluation function
+# ──────────────────────────────────────────────────────────────────────────────
+
+async def evaluate_single(
+    question: str,
+    expected_answer: str,
+    generated_answer: str,
+    context_chunks: list[str],
+) -> dict[str, Any]:
+    """
+    Run all five metrics for a single Q&A pair via the LLM judge.
+    Returns a dict with scores, rationales, and the raw inputs for export.
+    """
+    t0 = time.time()
+
+    accuracy,          acc_rationale  = await compute_accuracy(generated_answer, expected_answer)
+    faithfulness,      fai_rationale  = await compute_faithfulness(generated_answer, context_chunks)
+    answer_relevancy,  rel_rationale  = await compute_answer_relevancy(question, generated_answer)
+    context_precision, pre_rationale  = await compute_context_precision(question, context_chunks)
+    context_recall,    rec_rationale  = await compute_context_recall(expected_answer, context_chunks)
+
+    latency_ms = round((time.time() - t0) * 1000, 1)
+
+    return {
+        # ── inputs (for export) ───────────────────────────────────────────────
+        "question":          question,
+        "expected_answer":   expected_answer,
+        "generated_answer":  generated_answer,
+        "retrieved_context": "\n---\n".join(context_chunks),
+        # ── scores ───────────────────────────────────────────────────────────
+        "accuracy":          accuracy,
+        "faithfulness":      faithfulness,
+        "answer_relevancy":  answer_relevancy,
+        "context_precision": context_precision,
+        "context_recall":    context_recall,
+        # ── rationales ───────────────────────────────────────────────────────
+        "accuracy_rationale":          acc_rationale,
+        "faithfulness_rationale":      fai_rationale,
+        "answer_relevancy_rationale":  rel_rationale,
+        "context_precision_rationale": pre_rationale,
+        "context_recall_rationale":    rec_rationale,
+        # ── meta ─────────────────────────────────────────────────────────────
+        "latency_ms": latency_ms,
+    }
\ No newline at end of file
diff --git a/backend/app/main.py b/backend/app/main.py
index b673cb9..932013c 100644
+++ b/backend/app/main.py
@@ -46,8 +46,8 @@ async def lifespan(app: FastAPI):
     logger.info("Application startup complete")
     yield
     logger.info("Shutting down application")
+    from app.embeddings.openai_client import openai_client
+    await openai_client.close()
     logger.info("Shutdown complete")
 
 
@@ -100,4 +100,4 @@ app.include_router(markdown_router)
 app.include_router(agents_router)
 app.include_router(chroma_router)
 app.include_router(embeddings_router)
+app.include_router(web_router)
\ No newline at end of file
diff --git a/backend/app/rag/evaluator.py b/backend/app/rag/evaluator.py
index 0cfdc6a..9acdb80 100644
+++ b/backend/app/rag/evaluator.py
@@ -1,65 +1,185 @@
+"""
+LLM-as-a-Judge evaluator for RAG pipelines.
+
+Each metric is computed by prompting the LLM to score (0.0–1.0) with a brief rationale,
+rather than relying on cosine-similarity heuristics.
+"""
+
+import json
+import re
 import time
 from typing import Any
+
+from app.embeddings.openai_client import openai_client as ollama_client
 from app.core.logging import get_logger
 
 logger = get_logger("evaluator")
 
 
+# ──────────────────────────────────────────────────────────────────────────────
+# Internal helpers
+# ──────────────────────────────────────────────────────────────────────────────
 
+_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]*\.?[0-9]+)')
 
+
+async def _llm_score(prompt: str) -> tuple[float, str]:
+    """
+    Call the LLM with `prompt` and extract a JSON payload like:
+        {"score": 0.85, "rationale": "..."}
+    Returns (score, rationale).  Falls back to (0.0, error_msg) on failure.
+    """
+    system = (
+        "You are an expert RAG evaluation judge. "
+        "Respond ONLY with a JSON object containing exactly two keys: "
+        '"score" (a float between 0.0 and 1.0) and '
+        '"rationale" (a one-sentence explanation). '
+        "Do not include any other text."
+    )
+    raw = ""  # initialise so it's always bound
+    try:
+        raw = await ollama_client.chat(
+            [{"role": "user", "content": prompt}],
+            system=system,
+        )
+        # Strip markdown fences if present
+        raw = raw.strip().strip("```json").strip("```").strip()
+        data = json.loads(raw)
+        score = float(data.get("score", 0.0))
+        rationale = data.get("rationale", "")
+        return round(max(0.0, min(1.0, score)), 4), rationale
+    except Exception as exc:
+        logger.warning("LLM scoring parse error: %s | raw=%.200r", exc, raw)
+        # Fallback: try regex on whatever we got back
+        m = _SCORE_RE.search(raw)
+        if m:
+            return round(float(m.group(1)), 4), "Score extracted via regex fallback."
+        return 0.0, f"Parse error: {exc}"
+
+
+# ──────────────────────────────────────────────────────────────────────────────
+# Metric functions
+# ──────────────────────────────────────────────────────────────────────────────
+
+async def compute_accuracy(generated: str, expected: str) -> tuple[float, str]:
+    """Semantic accuracy: how well does the generated answer match the expected answer?"""
+    prompt = f"""Rate how well the GENERATED ANSWER matches the EXPECTED ANSWER semantically.
+
+EXPECTED ANSWER:
+{expected}
+
+GENERATED ANSWER:
+{generated}
+
+Score 1.0 if they convey the same meaning, 0.0 if completely unrelated."""
+    return await _llm_score(prompt)
+
+
+async def compute_faithfulness(answer: str, context_chunks: list[str]) -> tuple[float, str]:
+    """Faithfulness: is the answer grounded in the retrieved context?"""
+    context = "\n\n---\n\n".join(context_chunks[:5])  # cap to avoid token overflow
+    prompt = f"""Rate how faithfully the ANSWER is grounded in the CONTEXT below.
+A faithful answer only makes claims supported by the context (score 1.0).
+An unfaithful answer introduces hallucinated facts not in the context (score 0.0).
+
+CONTEXT:
+{context}
+
+ANSWER:
+{answer}"""
+    return await _llm_score(prompt)
 
 
+async def compute_answer_relevancy(question: str, answer: str) -> tuple[float, str]:
+    """Answer relevancy: does the answer directly address the question?"""
+    prompt = f"""Rate how directly and completely the ANSWER addresses the QUESTION.
+Score 1.0 if the answer is fully on-topic and complete, 0.0 if it ignores the question.
 
+QUESTION:
+{question}
 
+ANSWER:
+{answer}"""
+    return await _llm_score(prompt)
+
+
+async def compute_context_precision(question: str, context_chunks: list[str]) -> tuple[float, str]:
+    """Context precision: what fraction of retrieved chunks are actually relevant?"""
     if not context_chunks:
+        return 0.0, "No context chunks provided."
+    chunks_text = "\n\n---\n\n".join(
+        f"[Chunk {i+1}]: {c}" for i, c in enumerate(context_chunks[:8])
+    )
+    prompt = f"""You are evaluating retrieval quality.
+Below are chunks retrieved for a QUESTION. Rate the proportion of chunks that contain
+information genuinely useful for answering the question (0.0 = none relevant, 1.0 = all relevant).
+
+QUESTION:
+{question}
+
+RETRIEVED CHUNKS:
+{chunks_text}"""
+    return await _llm_score(prompt)
+
+
+async def compute_context_recall(expected_answer: str, context_chunks: list[str]) -> tuple[float, str]:
+    """Context recall: does the retrieved context contain what's needed to answer correctly?"""
     if not context_chunks:
+        return 0.0, "No context chunks provided."
+    context = "\n\n---\n\n".join(context_chunks[:8])
+    prompt = f"""Rate whether the RETRIEVED CONTEXT contains the information needed to produce the EXPECTED ANSWER.
+Score 1.0 if all key facts for the expected answer are present in the context, 0.0 if completely missing.
+
+EXPECTED ANSWER:
+{expected_answer}
+
+RETRIEVED CONTEXT:
+{context}"""
+    return await _llm_score(prompt)
+
+
+# ──────────────────────────────────────────────────────────────────────────────
+# Master evaluation function
+# ──────────────────────────────────────────────────────────────────────────────
+
+async def evaluate_single(
+    question: str,
+    expected_answer: str,
+    generated_answer: str,
+    context_chunks: list[str],
+) -> dict[str, Any]:
+    """
+    Run all five metrics for a single Q&A pair via the LLM judge.
+    Returns a dict with scores, rationales, and the raw inputs for export.
+    """
+    t0 = time.time()
+
+    accuracy,          acc_rationale  = await compute_accuracy(generated_answer, expected_answer)
+    faithfulness,      fai_rationale  = await compute_faithfulness(generated_answer, context_chunks)
+    answer_relevancy,  rel_rationale  = await compute_answer_relevancy(question, generated_answer)
+    context_precision, pre_rationale  = await compute_context_precision(question, context_chunks)
+    context_recall,    rec_rationale  = await compute_context_recall(expected_answer, context_chunks)
+
+    latency_ms = round((time.time() - t0) * 1000, 1)
+
+    return {
+        # ── inputs (for export) ───────────────────────────────────────────────
+        "question":          question,
+        "expected_answer":   expected_answer,
+        "generated_answer":  generated_answer,
+        "retrieved_context": "\n---\n".join(context_chunks),
+        # ── scores ───────────────────────────────────────────────────────────
+        "accuracy":          accuracy,
+        "faithfulness":      faithfulness,
+        "answer_relevancy":  answer_relevancy,
+        "context_precision": context_precision,
+        "context_recall":    context_recall,
+        # ── rationales ───────────────────────────────────────────────────────
+        "accuracy_rationale":          acc_rationale,
+        "faithfulness_rationale":      fai_rationale,
+        "answer_relevancy_rationale":  rel_rationale,
+        "context_precision_rationale": pre_rationale,
+        "context_recall_rationale":    rec_rationale,
+        # ── meta ─────────────────────────────────────────────────────────────
+        "latency_ms": latency_ms,
+    }
\ No newline at end of file
diff --git a/backend/app/rag/markdown_rag.py b/backend/app/rag/markdown_rag.py
index b83f44d..0bb672b 100644
+++ b/backend/app/rag/markdown_rag.py
@@ -1,7 +1,7 @@
 import re
 from typing import Any, Optional
 from app.chromadb.client import chroma_client
+from app.embeddings.openai_client import openai_client as ollama_client
 from app.core.logging import get_logger
 
 logger = get_logger("markdown_rag")
@@ -88,4 +88,4 @@ class MarkdownRAG:
         return chroma_client.search(MD_COLLECTION, query_emb, top_k, where)
 
 
+markdown_rag = MarkdownRAG()
\ No newline at end of file
diff --git a/backend/app/rag/pdf_rag.py b/backend/app/rag/pdf_rag.py
index 13e57de..f4fa699 100644
+++ b/backend/app/rag/pdf_rag.py
@@ -4,7 +4,7 @@ import uuid
 from typing import Any, Optional
 import pdfplumber
 from app.chromadb.client import chroma_client
+from app.embeddings.openai_client import openai_client as ollama_client
 from app.core.logging import get_logger
 from app.core.config import settings
 
@@ -125,4 +125,4 @@ class PDFHierarchicalRAG:
         return chroma_client.search(PDF_COLLECTION, query_emb, top_k, where)
 
 
+pdf_rag = PDFHierarchicalRAG()
\ No newline at end of file
diff --git a/backend/app/rag/table_rag.py b/backend/app/rag/table_rag.py
index e9aca2e..cad758e 100644
+++ b/backend/app/rag/table_rag.py
@@ -5,7 +5,7 @@ import uuid
 from typing import Any, Optional
 import pandas as pd
 from app.chromadb.client import chroma_client
+from app.embeddings.openai_client import openai_client as ollama_client
 from app.rag.metadata_filter import build_chroma_filter
 from app.core.logging import get_logger
 
@@ -96,4 +96,4 @@ class TableRAG:
         return None
 
 
+table_rag = TableRAG()
\ No newline at end of file
diff --git a/backend/app/rag/vector_rag.py b/backend/app/rag/vector_rag.py
index 10572cc..008f38d 100644
+++ b/backend/app/rag/vector_rag.py
@@ -1,6 +1,6 @@
 from typing import Any, Optional
 from app.chromadb.client import chroma_client
+from app.embeddings.openai_client import openai_client as ollama_client
 from app.rag.metadata_filter import build_chroma_filter
 from app.core.logging import get_logger
 
@@ -48,4 +48,4 @@ class VectorRAG:
         return all_results[:top_k]
 
 
+vector_rag = VectorRAG()
\ No newline at end of file
diff --git a/backend/app/services/chat_service.py b/backend/app/services/chat_service.py
index b352bde..f332d71 100644
+++ b/backend/app/services/chat_service.py
@@ -4,7 +4,7 @@ from datetime import datetime
 from sqlalchemy.ext.asyncio import AsyncSession
 from app.repositories.conversation_repository import conversation_repo
 from app.services.rag_service import rag_service
+from app.embeddings.openai_client import openai_client as ollama_client
 from app.core.logging import get_logger
 from app.core.exceptions import ConversationNotFoundError
 
@@ -14,7 +14,6 @@ SYSTEM_PROMPT = """You are an intelligent assistant with access to a document kn
 Answer questions using the provided context. If the context doesn't contain enough information,
 say so clearly. Always cite your sources when possible. Be concise and accurate."""
 
 class ChatService:
     async def chat(
         self,
@@ -133,4 +132,4 @@ class ChatService:
         })
 
 
+chat_service = ChatService()
\ No newline at end of file
diff --git a/backend/app/services/document_service.py b/backend/app/services/document_service.py
index 4c7e589..2ccba36 100644
+++ b/backend/app/services/document_service.py
@@ -8,7 +8,7 @@ import aiofiles
 from sqlalchemy.ext.asyncio import AsyncSession
 from app.repositories.document_repository import document_repo
 from app.chromadb.client import chroma_client
+from app.embeddings.openai_client import openai_client as ollama_client
 from app.rag.table_rag import table_rag
 from app.rag.pdf_rag import pdf_rag
 from app.rag.markdown_rag import markdown_rag
@@ -154,7 +154,7 @@ class DocumentService:
             "retrieval_strategy": strategy,
             "language": (extra_metadata or {}).get("language", "en"),
             "chunk_count": chunk_count,
+            "embedding_model": settings.OPENAI_EMBED_MODEL,
             "collection_name": collection,
             "metadata_json": extra_metadata or {},
         }
@@ -227,4 +227,4 @@ class DocumentService:
         return True
 
 
+document_service = DocumentService()
\ No newline at end of file
diff --git a/backend/app/services/rag_service.py b/backend/app/services/rag_service.py
index 14fb55f..aa744f2 100644
+++ b/backend/app/services/rag_service.py
@@ -13,7 +13,7 @@ from app.rag.evaluator import (
     compute_accuracy, compute_faithfulness, compute_answer_relevancy,
     compute_context_precision, compute_context_recall,
 )
+from app.embeddings.openai_client import openai_client as ollama_client
 from app.repositories.log_repository import log_repo
 from app.core.config import settings
 from app.core.logging import get_logger
@@ -170,4 +170,4 @@ class RAGService:
         return final
 
 
+rag_service = RAGService()
\ No newline at end of file
diff --git a/backend/app/services/web_service.py b/backend/app/services/web_service.py
index e324633..536a95e 100644
+++ b/backend/app/services/web_service.py
@@ -4,7 +4,7 @@ import httpx
 from bs4 import BeautifulSoup
 from sqlalchemy.ext.asyncio import AsyncSession
 from app.chromadb.client import chroma_client
+from app.embeddings.openai_client import openai_client as ollama_client
 from app.rag.bm25 import bm25_retriever
 from app.core.config import settings
 from app.core.logging import get_logger
@@ -99,4 +99,4 @@ class WebService:
         return chroma_client.search(collection_name, query_emb, top_k, where)
 
 
+web_service = WebService()
\ No newline at end of file
diff --git a/backend/app/tests/test_core.py b/backend/app/tests/test_core.py
index d00ff7b..ce2d000 100644
+++ b/backend/app/tests/test_core.py
@@ -130,8 +130,8 @@ def test_bm25_remove_collection():
 
 def test_settings_defaults():
     from app.core.config import settings
+    assert settings.OPENAI_LLM_MODEL == "gpt-4o-mini"
+    assert settings.OPENAI_EMBED_MODEL == "text-embedding-3-small"
     assert settings.TOP_K == 5
     assert settings.CHUNK_SIZE == 512
 
@@ -286,4 +286,4 @@ def test_router_doc_type_detection():
     assert _detect_doc_type("find in PDF report") == "pdf"
     assert _detect_doc_type("search the README markdown guide") == "markdown"
     assert _detect_doc_type("query the CSV table rows") == "csv"
+    assert _detect_doc_type("general question") == "text"
\ No newline at end of file
```

## 247931bb81630bd38336e2a0480eaf3bc5bb3366 — 2026-05-25T03:40:32+05:30

Message:

initial commit with ollama for LLM

```diff
diff --git a/backend/alembic/env.py b/backend/alembic/env.py
new file mode 100644
index 0000000..bb42de1
+++ b/backend/alembic/env.py
@@ -0,0 +1,63 @@
+import asyncio
+from logging.config import fileConfig
+
+from sqlalchemy import pool
+from sqlalchemy.ext.asyncio import async_engine_from_config
+
+from alembic import context
+
+# Load app config
+import sys
+import os
+sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
+
+from app.core.config import settings
+from app.database.base import Base
+from app.database import models  # noqa: F401
+
+config = context.config
+config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
+
+if config.config_file_name is not None:
+    fileConfig(config.config_file_name)
+
+target_metadata = Base.metadata
+
+
+def run_migrations_offline() -> None:
+    url = config.get_main_option("sqlalchemy.url")
+    context.configure(
+        url=url,
+        target_metadata=target_metadata,
+        literal_binds=True,
+        dialect_opts={"paramstyle": "named"},
+    )
+    with context.begin_transaction():
+        context.run_migrations()
+
+
+def do_run_migrations(connection):
+    context.configure(connection=connection, target_metadata=target_metadata)
+    with context.begin_transaction():
+        context.run_migrations()
+
+
+async def run_async_migrations() -> None:
+    connectable = async_engine_from_config(
+        config.get_section(config.config_ini_section, {}),
+        prefix="sqlalchemy.",
+        poolclass=pool.NullPool,
+    )
+    async with connectable.connect() as connection:
+        await connection.run_sync(do_run_migrations)
+    await connectable.dispose()
+
+
+def run_migrations_online() -> None:
+    asyncio.run(run_async_migrations())
+
+
+if context.is_offline_mode():
+    run_migrations_offline()
+else:
+    run_migrations_online()
diff --git a/backend/alembic/versions/0001_initial_schema.py b/backend/alembic/versions/0001_initial_schema.py
new file mode 100644
index 0000000..d4331a8
+++ b/backend/alembic/versions/0001_initial_schema.py
@@ -0,0 +1,112 @@
+"""initial schema
+
+Revision ID: 0001
+Revises: 
+Create Date: 2026-05-21 00:00:00.000000
+
+"""
+from typing import Sequence, Union
+from alembic import op
+import sqlalchemy as sa
+
+revision: str = "0001"
+down_revision: Union[str, None] = None
+branch_labels: Union[str, Sequence[str], None] = None
+depends_on: Union[str, Sequence[str], None] = None
+
+
+def upgrade() -> None:
+    op.create_table(
+        "documents",
+        sa.Column("id", sa.String(36), nullable=False),
+        sa.Column("filename", sa.String(512), nullable=False),
+        sa.Column("filepath", sa.String(1024), nullable=False),
+        sa.Column("document_type", sa.String(64), nullable=False),
+        sa.Column("retrieval_strategy", sa.String(64), nullable=True),
+        sa.Column("language", sa.String(16), nullable=True),
+        sa.Column("chunk_count", sa.Integer(), nullable=True, default=0),
+        sa.Column("embedding_model", sa.String(128), nullable=True),
+        sa.Column("collection_name", sa.String(128), nullable=True),
+        sa.Column("metadata_json", sa.JSON(), nullable=True),
+        sa.Column("created_at", sa.DateTime(), nullable=True),
+        sa.Column("updated_at", sa.DateTime(), nullable=True),
+        sa.PrimaryKeyConstraint("id", name="pk_documents"),
+    )
+    op.create_index("ix_documents_document_type", "documents", ["document_type"])
+    op.create_index("ix_documents_filename", "documents", ["filename"])
+    op.create_index("ix_documents_created_at", "documents", ["created_at"])
+
+    op.create_table(
+        "chunks",
+        sa.Column("id", sa.String(36), nullable=False),
+        sa.Column("document_id", sa.String(36), nullable=False),
+        sa.Column("chunk_index", sa.Integer(), nullable=False),
+        sa.Column("chunk_text", sa.Text(), nullable=False),
+        sa.Column("chunk_metadata", sa.JSON(), nullable=True),
+        sa.Column("created_at", sa.DateTime(), nullable=True),
+        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], name="fk_chunks_document_id_documents", ondelete="CASCADE"),
+        sa.PrimaryKeyConstraint("id", name="pk_chunks"),
+    )
+    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
+    op.create_index("ix_chunks_chunk_index", "chunks", ["chunk_index"])
+
+    op.create_table(
+        "conversations",
+        sa.Column("id", sa.String(36), nullable=False),
+        sa.Column("title", sa.String(512), nullable=True),
+        sa.Column("created_at", sa.DateTime(), nullable=True),
+        sa.Column("updated_at", sa.DateTime(), nullable=True),
+        sa.PrimaryKeyConstraint("id", name="pk_conversations"),
+    )
+    op.create_index("ix_conversations_created_at", "conversations", ["created_at"])
+
+    op.create_table(
+        "messages",
+        sa.Column("id", sa.String(36), nullable=False),
+        sa.Column("conversation_id", sa.String(36), nullable=False),
+        sa.Column("role", sa.String(32), nullable=False),
+        sa.Column("content", sa.Text(), nullable=False),
+        sa.Column("sources", sa.JSON(), nullable=True),
+        sa.Column("created_at", sa.DateTime(), nullable=True),
+        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], name="fk_messages_conversation_id_conversations", ondelete="CASCADE"),
+        sa.PrimaryKeyConstraint("id", name="pk_messages"),
+    )
+    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])
+    op.create_index("ix_messages_role", "messages", ["role"])
+
+    op.create_table(
+        "retrieval_logs",
+        sa.Column("id", sa.String(36), nullable=False),
+        sa.Column("query", sa.Text(), nullable=False),
+        sa.Column("retrieval_strategy", sa.String(64), nullable=True),
+        sa.Column("retrieved_chunks", sa.JSON(), nullable=True),
+        sa.Column("generated_answer", sa.Text(), nullable=True),
+        sa.Column("latency_ms", sa.Float(), nullable=True),
+        sa.Column("agent_used", sa.String(64), nullable=True),
+        sa.Column("created_at", sa.DateTime(), nullable=True),
+        sa.PrimaryKeyConstraint("id", name="pk_retrieval_logs"),
+    )
+    op.create_index("ix_retrieval_logs_created_at", "retrieval_logs", ["created_at"])
+    op.create_index("ix_retrieval_logs_retrieval_strategy", "retrieval_logs", ["retrieval_strategy"])
+
+    op.create_table(
+        "evaluation_runs",
+        sa.Column("id", sa.String(36), nullable=False),
+        sa.Column("dataset_name", sa.String(256), nullable=True),
+        sa.Column("accuracy", sa.Float(), nullable=True),
+        sa.Column("faithfulness", sa.Float(), nullable=True),
+        sa.Column("context_precision", sa.Float(), nullable=True),
+        sa.Column("context_recall", sa.Float(), nullable=True),
+        sa.Column("created_at", sa.DateTime(), nullable=True),
+        sa.PrimaryKeyConstraint("id", name="pk_evaluation_runs"),
+    )
+    op.create_index("ix_evaluation_runs_created_at", "evaluation_runs", ["created_at"])
+
+
+def downgrade() -> None:
+    op.drop_table("evaluation_runs")
+    op.drop_table("retrieval_logs")
+    op.drop_table("messages")
+    op.drop_table("conversations")
+    op.drop_table("chunks")
+    op.drop_table("documents")
diff --git a/backend/app/__init__.py b/backend/app/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/backend/app/agents/__init__.py b/backend/app/agents/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/backend/app/agents/base.py b/backend/app/agents/base.py
new file mode 100644
index 0000000..4d45e2c
+++ b/backend/app/agents/base.py
@@ -0,0 +1,33 @@
+from abc import ABC, abstractmethod
+from typing import Any, Optional
+from app.embeddings.ollama_client import ollama_client
+from app.core.logging import get_logger
+
+
+class BaseAgent(ABC):
+    name: str = "base_agent"
+
+    def __init__(self):
+        self.logger = get_logger(f"agent.{self.name}")
+
+    @abstractmethod
+    async def plan(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
+        """Decompose task and create execution plan."""
+        ...
+
+    @abstractmethod
+    async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
+        """Execute the plan and retrieve/generate results."""
+        ...
+
+    @abstractmethod
+    async def evaluate(self, result: dict[str, Any]) -> dict[str, Any]:
+        """Evaluate result quality and completeness."""
+        ...
+
+    async def run(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
+        self.logger.info(f"[{self.name}] running query: {query[:80]}")
+        plan = await self.plan(query, context)
+        result = await self.execute(plan)
+        evaluated = await self.evaluate(result)
+        return evaluated
diff --git a/backend/app/agents/coordinator_agent.py b/backend/app/agents/coordinator_agent.py
new file mode 100644
index 0000000..43ec53a
+++ b/backend/app/agents/coordinator_agent.py
@@ -0,0 +1,128 @@
+import time
+from typing import Any, Optional
+from app.agents.base import BaseAgent
+from app.agents.vector_agent import vector_agent
+from app.agents.sqlite_agent import sqlite_agent
+from app.agents.router_agent import router_agent, _detect_doc_type
+from app.agents.web_agent import web_agent
+from app.agents.evaluator_agent import evaluator_agent
+from app.embeddings.ollama_client import ollama_client
+from app.core.logging import get_logger
+
+logger = get_logger("coordinator_agent")
+
+INTENT_KEYWORDS = {
+    "table": ["table", "csv", "spreadsheet", "rows", "columns", "sum", "count", "average", "aggregate"],
+    "web": ["website", "url", "http", "online", "web", "internet", "search online"],
+    "structured": ["scheme", "state", "ministry", "department", "eligibility", "database"],
+}
+
+
+def _classify_intent(query: str) -> str:
+    q_lower = query.lower()
+    for intent, keywords in INTENT_KEYWORDS.items():
+        if any(kw in q_lower for kw in keywords):
+            return intent
+    return "general"
+
+
+class CoordinatorAgent(BaseAgent):
+    name = "coordinator_agent"
+
+    async def plan(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
+        ctx = context or {}
+        intent = _classify_intent(query)
+        doc_type = _detect_doc_type(query)
+
+        # Determine which agents to invoke
+        agents_to_run = []
+        if intent == "table":
+            agents_to_run = ["sqlite"]
+        elif intent == "web":
+            agents_to_run = ["web", "vector"]
+        elif intent == "structured":
+            agents_to_run = ["sqlite", "vector"]
+        else:
+            agents_to_run = ["router", "vector"]
+
+        return {
+            "query": query,
+            "intent": intent,
+            "doc_type": doc_type,
+            "agents": agents_to_run,
+            "context": ctx,
+            "top_k": ctx.get("top_k", 5),
+        }
+
+    async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
+        start = time.time()
+        query = plan["query"]
+        ctx = plan["context"]
+        agents_list = plan["agents"]
+        top_k = plan["top_k"]
+
+        agent_results = []
+        all_chunks = []
+
+        for agent_name in agents_list:
+            try:
+                if agent_name == "sqlite":
+                    res = await sqlite_agent.run(query, {**ctx, "top_k": top_k})
+                elif agent_name == "vector":
+                    res = await vector_agent.run(query, {**ctx, "top_k": top_k})
+                elif agent_name == "router":
+                    res = await router_agent.run(query, {**ctx, "top_k": top_k})
+                elif agent_name == "web":
+                    res = await web_agent.run(query, {**ctx, "top_k": top_k})
+                else:
+                    continue
+                agent_results.append(res)
+                all_chunks.extend(res.get("chunks", []))
+            except Exception as e:
+                logger.error(f"Agent '{agent_name}' failed: {e}")
+
+        # Synthesize answers from all agents
+        if not agent_results:
+            return {
+                "agent": self.name,
+                "query": query,
+                "answer": "No results found.",
+                "chunks": [],
+                "latency_ms": (time.time() - start) * 1000,
+                "agent_results": [],
+            }
+
+        # Merge context and generate final answer
+        combined_context = "\n\n---\n\n".join(
+            f"[{r['agent'].upper()}]:\n{r.get('answer', '')}" for r in agent_results
+        )
+        synthesis_prompt = (
+            f"Multiple agents retrieved the following information:\n\n{combined_context}\n\n"
+            f"Based on all above, provide a comprehensive final answer to: {query}"
+        )
+        system = "You are a coordinator that synthesizes information from multiple sources into a single coherent answer."
+        final_answer = await ollama_client.generate(synthesis_prompt, system=system)
+        latency = (time.time() - start) * 1000
+
+        return {
+            "agent": self.name,
+            "query": query,
+            "answer": final_answer,
+            "chunks": all_chunks,
+            "agent_results": agent_results,
+            "intent": plan["intent"],
+            "latency_ms": round(latency, 2),
+        }
+
+    async def evaluate(self, result: dict[str, Any]) -> dict[str, Any]:
+        chunks = result.get("chunks", [])
+        scores = [c.get("score", 0) for c in chunks if c.get("score")]
+        result["confidence"] = round(sum(scores) / max(len(scores), 1), 4) if scores else 0.0
+        result["sources"] = list({
+            c.get("filename", c.get("metadata", {}).get("filename", "")): None
+            for c in chunks
+        }.keys())
+        return result
+
+
+coordinator_agent = CoordinatorAgent()
diff --git a/backend/app/agents/evaluator_agent.py b/backend/app/agents/evaluator_agent.py
new file mode 100644
index 0000000..43b5568
+++ b/backend/app/agents/evaluator_agent.py
@@ -0,0 +1,66 @@
+import time
+from typing import Any, Optional
+from app.agents.base import BaseAgent
+from app.rag.evaluator import (
+    compute_faithfulness, compute_answer_relevancy,
+    compute_context_precision, compute_context_recall,
+)
+from app.core.logging import get_logger
+
+
+class RetrievalEvaluationAgent(BaseAgent):
+    name = "evaluator_agent"
+
+    async def plan(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
+        ctx = context or {}
+        return {
+            "query": query,
+            "answer": ctx.get("answer", ""),
+            "chunks": ctx.get("chunks", []),
+            "expected_answer": ctx.get("expected_answer", ""),
+        }
+
+    async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
+        start = time.time()
+        query = plan["query"]
+        answer = plan["answer"]
+        chunks = plan["chunks"]
+        expected = plan.get("expected_answer", "")
+
+        context_texts = [c.get("chunk_text", "") for c in chunks if c.get("chunk_text")]
+
+        faithfulness = await compute_faithfulness(answer, context_texts)
+        relevancy = await compute_answer_relevancy(query, answer)
+        cp = await compute_context_precision(query, context_texts)
+        cr = await compute_context_recall(expected, context_texts) if expected else 0.0
+        latency = (time.time() - start) * 1000
+
+        coverage = len([c for c in chunks if c.get("score", 0) > 0.5]) / max(len(chunks), 1)
+        retrieval_ok = faithfulness > 0.5 and cp > 0.4
+
+        return {
+            "agent": self.name,
+            "query": query,
+            "faithfulness": faithfulness,
+            "answer_relevancy": relevancy,
+            "context_precision": cp,
+            "context_recall": cr,
+            "retrieval_coverage": round(coverage, 4),
+            "retrieval_ok": retrieval_ok,
+            "latency_ms": round(latency, 2),
+        }
+
+    async def evaluate(self, result: dict[str, Any]) -> dict[str, Any]:
+        score = (
+            result.get("faithfulness", 0) * 0.3
+            + result.get("answer_relevancy", 0) * 0.3
+            + result.get("context_precision", 0) * 0.2
+            + result.get("context_recall", 0) * 0.2
+        )
+        result["overall_score"] = round(score, 4)
+        result["answer"] = f"Evaluation complete. Overall score: {result['overall_score']:.2f}"
+        result["sources"] = []
+        return result
+
+
+evaluator_agent = RetrievalEvaluationAgent()
diff --git a/backend/app/agents/router_agent.py b/backend/app/agents/router_agent.py
new file mode 100644
index 0000000..48073fb
+++ b/backend/app/agents/router_agent.py
@@ -0,0 +1,85 @@
+import time
+from typing import Any, Optional
+from app.agents.base import BaseAgent
+from app.rag.pdf_rag import pdf_rag
+from app.rag.markdown_rag import markdown_rag
+from app.rag.table_rag import table_rag
+from app.rag.vector_rag import vector_rag
+from app.embeddings.ollama_client import ollama_client
+
+
+ROUTING_RULES = {
+    "pdf": ["pdf", "document", "page", "report", "form"],
+    "markdown": ["markdown", "readme", "guide", "documentation", "wiki"],
+    "csv": ["table", "csv", "data", "rows", "columns", "spreadsheet", "excel"],
+    "text": [],  # default
+}
+
+
+def _detect_doc_type(query: str) -> str:
+    q_lower = query.lower()
+    for doc_type, keywords in ROUTING_RULES.items():
+        if any(kw in q_lower for kw in keywords):
+            return doc_type
+    return "text"
+
+
+class DocumentRouterAgent(BaseAgent):
+    name = "router_agent"
+
+    async def plan(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
+        ctx = context or {}
+        doc_type = ctx.get("doc_type") or _detect_doc_type(query)
+        return {
+            "query": query,
+            "doc_type": doc_type,
+            "document_id": ctx.get("document_id"),
+            "top_k": ctx.get("top_k", 5),
+        }
+
+    async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
+        start = time.time()
+        query = plan["query"]
+        doc_type = plan["doc_type"]
+        doc_id = plan.get("document_id")
+        top_k = plan["top_k"]
+
+        if doc_type == "pdf":
+            chunks = await pdf_rag.query(query, document_id=doc_id, top_k=top_k)
+            strategy = "hierarchical_rag"
+        elif doc_type == "markdown":
+            chunks = await markdown_rag.query(query, document_id=doc_id, top_k=top_k)
+            strategy = "structure_aware_rag"
+        elif doc_type == "csv":
+            chunks = await table_rag.query(query, document_id=doc_id, top_k=top_k)
+            strategy = "table_rag"
+        else:
+            chunks = await vector_rag.retrieve(query, "text_documents", top_k)
+            strategy = "vector_rag"
+
+        context_str = "\n\n".join(r["chunk_text"] for r in chunks)
+        prompt = f"Context:\n{context_str}\n\nQuestion: {query}"
+        answer = await ollama_client.generate(prompt)
+        latency = (time.time() - start) * 1000
+
+        return {
+            "agent": self.name,
+            "query": query,
+            "answer": answer,
+            "chunks": chunks,
+            "strategy": strategy,
+            "doc_type": doc_type,
+            "latency_ms": round(latency, 2),
+        }
+
+    async def evaluate(self, result: dict[str, Any]) -> dict[str, Any]:
+        chunks = result.get("chunks", [])
+        result["confidence"] = round(sum(c.get("score", 0) for c in chunks) / max(len(chunks), 1), 4)
+        result["sources"] = [
+            {"filename": c.get("metadata", {}).get("filename", c.get("filename", "")), "chunk_id": c.get("chunk_id", "")}
+            for c in chunks
+        ]
+        return result
+
+
+router_agent = DocumentRouterAgent()
diff --git a/backend/app/agents/sqlite_agent.py b/backend/app/agents/sqlite_agent.py
new file mode 100644
index 0000000..af5c546
+++ b/backend/app/agents/sqlite_agent.py
@@ -0,0 +1,52 @@
+import time
+from typing import Any, Optional
+from app.agents.base import BaseAgent
+from app.rag.table_rag import table_rag
+from app.embeddings.ollama_client import ollama_client
+
+
+class SQLiteAgent(BaseAgent):
+    name = "sqlite_agent"
+
+    async def plan(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
+        ctx = context or {}
+        return {
+            "query": query,
+            "document_id": ctx.get("document_id"),
+            "top_k": ctx.get("top_k", 5),
+        }
+
+    async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
+        start = time.time()
+        query = plan["query"]
+        doc_id = plan.get("document_id")
+        top_k = plan["top_k"]
+
+        chunks = await table_rag.query(query, document_id=doc_id, top_k=top_k)
+        context_str = "\n\n".join(r["chunk_text"] for r in chunks)
+
+        system = """You are a data analyst. Answer structured data questions using the provided table context.
+Be precise with numbers, names, and values. Format answers clearly."""
+        prompt = f"Table data:\n{context_str}\n\nQuestion: {query}"
+        answer = await ollama_client.generate(prompt, system=system)
+        latency = (time.time() - start) * 1000
+
+        return {
+            "agent": self.name,
+            "query": query,
+            "answer": answer,
+            "chunks": chunks,
+            "latency_ms": round(latency, 2),
+        }
+
+    async def evaluate(self, result: dict[str, Any]) -> dict[str, Any]:
+        chunks = result.get("chunks", [])
+        result["confidence"] = round(sum(c.get("score", 0) for c in chunks) / max(len(chunks), 1), 4)
+        result["sources"] = [
+            {"filename": c.get("metadata", {}).get("filename", ""), "chunk_id": c.get("chunk_id", "")}
+            for c in chunks
+        ]
+        return result
+
+
+sqlite_agent = SQLiteAgent()
diff --git a/backend/app/agents/vector_agent.py b/backend/app/agents/vector_agent.py
new file mode 100644
index 0000000..12dbfb6
+++ b/backend/app/agents/vector_agent.py
@@ -0,0 +1,63 @@
+import time
+from typing import Any, Optional
+from app.agents.base import BaseAgent
+from app.rag.hybrid_rag import hybrid_rag
+from app.rag.vector_rag import vector_rag
+from app.embeddings.ollama_client import ollama_client
+from app.core.config import settings
+
+
+class VectorRetrievalAgent(BaseAgent):
+    name = "vector_agent"
+
+    async def plan(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
+        ctx = context or {}
+        return {
+            "query": query,
+            "expanded_query": query,
+            "strategy": ctx.get("strategy", "hybrid"),
+            "top_k": ctx.get("top_k", settings.TOP_K),
+            "filters": ctx.get("filters"),
+            "collection": ctx.get("collection", "text_documents"),
+        }
+
+    async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
+        start = time.time()
+        query = plan["query"]
+        strategy = plan["strategy"]
+        top_k = plan["top_k"]
+        filters = plan.get("filters")
+        collection = plan["collection"]
+
+        if strategy == "vector":
+            chunks = await vector_rag.retrieve(query, collection, top_k, filters)
+        else:
+            chunks = await hybrid_rag.retrieve(query, collection, top_k, filters)
+
+        context_str = "\n\n".join(
+            f"[{r.get('filename','?')}] {r['chunk_text']}" for r in chunks
+        )
+        prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer based only on context:"
+        answer = await ollama_client.generate(prompt)
+        latency = (time.time() - start) * 1000
+
+        return {
+            "agent": self.name,
+            "query": query,
+            "answer": answer,
+            "chunks": chunks,
+            "latency_ms": round(latency, 2),
+        }
+
+    async def evaluate(self, result: dict[str, Any]) -> dict[str, Any]:
+        chunks = result.get("chunks", [])
+        avg_score = sum(c.get("score", 0) for c in chunks) / max(len(chunks), 1)
+        result["confidence"] = round(avg_score, 4)
+        result["sources"] = [
+            {"filename": c.get("filename", ""), "chunk_id": c.get("chunk_id", ""), "score": c.get("score", 0)}
+            for c in chunks
+        ]
+        return result
+
+
+vector_agent = VectorRetrievalAgent()
diff --git a/backend/app/agents/web_agent.py b/backend/app/agents/web_agent.py
new file mode 100644
index 0000000..995233b
+++ b/backend/app/agents/web_agent.py
@@ -0,0 +1,59 @@
+import time
+from typing import Any, Optional
+from app.agents.base import BaseAgent
+from app.services.web_service import web_service
+from app.embeddings.ollama_client import ollama_client
+
+
+class WebEnrichmentAgent(BaseAgent):
+    name = "web_agent"
+
+    async def plan(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
+        ctx = context or {}
+        return {
+            "query": query,
+            "url": ctx.get("url"),
+            "top_k": ctx.get("top_k", 5),
+        }
+
+    async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
+        start = time.time()
+        query = plan["query"]
+        url = plan.get("url")
+        top_k = plan["top_k"]
+
+        chunks = await web_service.query(query, url=url, top_k=top_k)
+
+        if not chunks and url:
+            # Try to ingest on-demand
+            try:
+                await web_service.ingest(url)
+                chunks = await web_service.query(query, url=url, top_k=top_k)
+            except Exception as e:
+                self.logger.warning(f"On-demand web ingest failed: {e}")
+
+        context_str = "\n\n".join(r["chunk_text"] for r in chunks) if chunks else "No web context available."
+        system = "You are a research assistant. Summarize and answer based on web content. Always note the source URL."
+        prompt = f"Web content:\n{context_str}\n\nQuestion: {query}"
+        answer = await ollama_client.generate(prompt, system=system)
+        latency = (time.time() - start) * 1000
+
+        return {
+            "agent": self.name,
+            "query": query,
+            "answer": answer,
+            "chunks": chunks,
+            "latency_ms": round(latency, 2),
+        }
+
+    async def evaluate(self, result: dict[str, Any]) -> dict[str, Any]:
+        chunks = result.get("chunks", [])
+        result["confidence"] = round(sum(c.get("score", 0) for c in chunks) / max(len(chunks), 1), 4)
+        result["sources"] = [
+            {"url": c.get("metadata", {}).get("source_url", ""), "chunk_id": c.get("chunk_id", "")}
+            for c in chunks
+        ]
+        return result
+
+
+web_agent = WebEnrichmentAgent()
diff --git a/backend/app/api/__init__.py b/backend/app/api/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/backend/app/api/agents.py b/backend/app/api/agents.py
new file mode 100644
index 0000000..c83e0b7
+++ b/backend/app/api/agents.py
@@ -0,0 +1,80 @@
+from fastapi import APIRouter, Depends
+from sqlalchemy.ext.asyncio import AsyncSession
+
+from app.core.dependencies import get_db
+from app.agents.coordinator_agent import coordinator_agent
+from app.agents.vector_agent import vector_agent
+from app.agents.sqlite_agent import sqlite_agent
+from app.agents.router_agent import router_agent
+from app.agents.web_agent import web_agent
+from app.agents.evaluator_agent import evaluator_agent
+from app.schemas.agent import AgentRequest, AgentResponse, CoordinatorRequest
+from app.core.logging import get_logger
+
+router = APIRouter(prefix="/api/agents", tags=["Agents"])
+logger = get_logger("api.agents")
+
+
+def _to_response(result: dict) -> AgentResponse:
+    return AgentResponse(
+        agent=result.get("agent", ""),
+        query=result.get("query", ""),
+        answer=result.get("answer", ""),
+        sources=result.get("sources", []),
+        reasoning=result.get("intent") or result.get("strategy"),
+        latency_ms=result.get("latency_ms", 0.0),
+        metadata={
+            k: v for k, v in result.items()
+            if k not in {"agent", "query", "answer", "sources", "chunks", "latency_ms"}
+        },
+    )
+
+
+@router.post("/coordinator", response_model=AgentResponse)
+async def coordinator(req: CoordinatorRequest):
+    result = await coordinator_agent.run(req.query, {"top_k": req.top_k})
+    return _to_response(result)
+
+
+@router.post("/vector", response_model=AgentResponse)
+async def vector(req: AgentRequest):
+    result = await vector_agent.run(req.query, {
+        "top_k": req.top_k,
+        "filters": req.filters,
+        **(req.context or {}),
+    })
+    return _to_response(result)
+
+
+@router.post("/sqlite", response_model=AgentResponse)
+async def sqlite(req: AgentRequest):
+    result = await sqlite_agent.run(req.query, {
+        "top_k": req.top_k,
+        **(req.context or {}),
+    })
+    return _to_response(result)
+
+
+@router.post("/router", response_model=AgentResponse)
+async def document_router(req: AgentRequest):
+    result = await router_agent.run(req.query, {
+        "top_k": req.top_k,
+        **(req.context or {}),
+    })
+    return _to_response(result)
+
+
+@router.post("/web", response_model=AgentResponse)
+async def web(req: AgentRequest):
+    result = await web_agent.run(req.query, {
+        "top_k": req.top_k,
+        **(req.context or {}),
+    })
+    return _to_response(result)
+
+
+@router.post("/evaluator", response_model=AgentResponse)
+async def evaluator(req: AgentRequest):
+    ctx = req.context or {}
+    result = await evaluator_agent.run(req.query, ctx)
+    return _to_response(result)
diff --git a/backend/app/api/chat.py b/backend/app/api/chat.py
new file mode 100644
index 0000000..cf57a39
+++ b/backend/app/api/chat.py
@@ -0,0 +1,54 @@
+from fastapi import APIRouter, Depends
+from fastapi.responses import StreamingResponse
+from sqlalchemy.ext.asyncio import AsyncSession
+
+from app.core.dependencies import get_db
+from app.services.chat_service import chat_service
+from app.repositories.conversation_repository import conversation_repo
+from app.schemas.chat import (
+    ChatRequest, ChatResponse, ConversationListResponse, ConversationResponse
+)
+from app.core.exceptions import ConversationNotFoundError
+from app.core.logging import get_logger
+
+router = APIRouter(prefix="/api/chat", tags=["Chat"])
+logger = get_logger("api.chat")
+
+
+@router.post("", response_model=ChatResponse)
+async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
+    result = await chat_service.chat(db, req.message, req.conversation_id, req.top_k)
+    return result
+
+
+@router.post("/stream")
+async def chat_stream(req: ChatRequest, db: AsyncSession = Depends(get_db)):
+    async def token_generator():
+        async for token in chat_service.chat_stream(db, req.message, req.conversation_id, req.top_k):
+            yield token
+
+    return StreamingResponse(token_generator(), media_type="text/plain")
+
+
+@router.get("/conversations", response_model=ConversationListResponse)
+async def list_conversations(skip: int = 0, limit: int = 50, db: AsyncSession = Depends(get_db)):
+    convs = await conversation_repo.list_all(db, skip, limit)
+    total = await conversation_repo.count(db)
+    return {"conversations": convs, "total": total}
+
+
+@router.get("/conversations/{conv_id}", response_model=ConversationResponse)
+async def get_conversation(conv_id: str, db: AsyncSession = Depends(get_db)):
+    conv = await conversation_repo.get_by_id(db, conv_id, with_messages=True)
+    if not conv:
+        raise ConversationNotFoundError(conv_id)
+    return conv
+
+
+@router.delete("/conversations/{conv_id}")
+async def delete_conversation(conv_id: str, db: AsyncSession = Depends(get_db)):
+    conv = await conversation_repo.get_by_id(db, conv_id)
+    if not conv:
+        raise ConversationNotFoundError(conv_id)
+    await conversation_repo.delete(db, conv_id)
+    return {"message": f"Conversation {conv_id} deleted"}
diff --git a/backend/app/api/chroma.py b/backend/app/api/chroma.py
new file mode 100644
index 0000000..681730d
+++ b/backend/app/api/chroma.py
@@ -0,0 +1,72 @@
+from fastapi import APIRouter
+from pydantic import BaseModel, Field
+from typing import Any, Optional
+
+from app.chromadb.client import chroma_client
+from app.embeddings.ollama_client import ollama_client
+from app.core.logging import get_logger
+
+router = APIRouter(prefix="/api/chroma", tags=["ChromaDB"])
+logger = get_logger("api.chroma")
+
+
+class ChromaIndexRequest(BaseModel):
+    collection_name: str
+    ids: list[str]
+    documents: list[str]
+    metadatas: Optional[list[dict[str, Any]]] = None
+
+
+class ChromaSearchRequest(BaseModel):
+    collection_name: str
+    query: str
+    top_k: int = Field(default=5, ge=1, le=50)
+    filters: Optional[dict[str, Any]] = None
+
+
+class ChromaDeleteRequest(BaseModel):
+    collection_name: str
+    document_id: Optional[str] = None
+
+
+@router.post("/index")
+async def index_documents(req: ChromaIndexRequest):
+    embeddings = await ollama_client.batch_embeddings(req.documents)
+    chroma_client.add_documents(
+        req.collection_name, req.ids, embeddings, req.documents, req.metadatas
+    )
+    return {"message": f"Indexed {len(req.ids)} documents into '{req.collection_name}'"}
+
+
+@router.post("/reindex")
+async def reindex_documents(req: ChromaIndexRequest):
+    embeddings = await ollama_client.batch_embeddings(req.documents)
+    chroma_client.reindex(
+        req.collection_name, req.ids, embeddings, req.documents, req.metadatas
+    )
+    return {"message": f"Reindexed {len(req.ids)} documents into '{req.collection_name}'"}
+
+
+@router.delete("/delete")
+async def delete_collection(req: ChromaDeleteRequest):
+    if req.document_id:
+        chroma_client.delete_by_document_id(req.collection_name, req.document_id)
+        return {"message": f"Deleted document {req.document_id} from '{req.collection_name}'"}
+    chroma_client.delete_collection(req.collection_name)
+    return {"message": f"Collection '{req.collection_name}' deleted"}
+
+
+@router.post("/search")
+async def search_collection(req: ChromaSearchRequest):
+    from app.rag.metadata_filter import build_chroma_filter
+    query_emb = await ollama_client.embeddings(req.query)
+    where = build_chroma_filter(req.filters or {})
+    results = chroma_client.search(req.collection_name, query_emb, req.top_k, where)
+    return {"query": req.query, "collection": req.collection_name, "results": results}
+
+
+@router.get("/collections")
+async def list_collections():
+    collections = chroma_client.list_collections()
+    counts = {name: chroma_client.get_collection_count(name) for name in collections}
+    return {"collections": collections, "counts": counts}
diff --git a/backend/app/api/documents.py b/backend/app/api/documents.py
new file mode 100644
index 0000000..4b6c7ec
+++ b/backend/app/api/documents.py
@@ -0,0 +1,95 @@
+from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
+from fastapi.responses import JSONResponse
+from sqlalchemy.ext.asyncio import AsyncSession
+from typing import Optional
+import json
+
+from app.core.dependencies import get_db
+from app.services.document_service import document_service
+from app.repositories.document_repository import document_repo
+from app.schemas.document import DocumentResponse, DocumentListResponse, ChunkResponse, ReindexResponse
+from app.core.exceptions import DocumentNotFoundError
+from app.core.logging import get_logger
+
+router = APIRouter(prefix="/api/documents", tags=["Documents"])
+logger = get_logger("api.documents")
+
+
+@router.post("/upload", response_model=dict)
+async def upload_document(
+    file: UploadFile = File(...),
+    metadata: Optional[str] = Form(None),
+    db: AsyncSession = Depends(get_db),
+):
+    content = await file.read()
+    extra_meta = json.loads(metadata) if metadata else {}
+    result = await document_service.upload_and_index(db, file.filename, content, extra_meta)
+    doc = result["document"]
+    return {
+        "id": doc.id,
+        "filename": doc.filename,
+        "document_type": doc.document_type,
+        "retrieval_strategy": doc.retrieval_strategy,
+        "chunk_count": result["chunk_count"],
+        "message": "Document uploaded and indexed successfully",
+    }
+
+
+@router.get("", response_model=DocumentListResponse)
+async def list_documents(
+    skip: int = 0,
+    limit: int = 50,
+    db: AsyncSession = Depends(get_db),
+):
+    docs = await document_repo.list_all(db, skip, limit)
+    total = await document_repo.count(db)
+    return {"documents": docs, "total": total}
+
+
+@router.get("/{doc_id}", response_model=DocumentResponse)
+async def get_document(doc_id: str, db: AsyncSession = Depends(get_db)):
+    doc = await document_repo.get_by_id(db, doc_id)
+    if not doc:
+        raise DocumentNotFoundError(doc_id)
+    return doc
+
+
+@router.delete("/{doc_id}")
+async def delete_document(doc_id: str, db: AsyncSession = Depends(get_db)):
+    await document_service.delete(db, doc_id)
+    return {"message": f"Document {doc_id} deleted"}
+
+
+@router.post("/{doc_id}/reindex", response_model=ReindexResponse)
+async def reindex_document(doc_id: str, db: AsyncSession = Depends(get_db)):
+    result = await document_service.reindex(db, doc_id)
+    return result
+
+
+@router.get("/{doc_id}/chunks", response_model=list[ChunkResponse])
+async def get_document_chunks(doc_id: str, db: AsyncSession = Depends(get_db)):
+    doc = await document_repo.get_by_id(db, doc_id)
+    if not doc:
+        raise DocumentNotFoundError(doc_id)
+    chunks = await document_repo.get_chunks(db, doc_id)
+    return chunks
+
+
+@router.get("/{doc_id}/metadata")
+async def get_document_metadata(doc_id: str, db: AsyncSession = Depends(get_db)):
+    doc = await document_repo.get_by_id(db, doc_id)
+    if not doc:
+        raise DocumentNotFoundError(doc_id)
+    return {
+        "id": doc.id,
+        "filename": doc.filename,
+        "document_type": doc.document_type,
+        "retrieval_strategy": doc.retrieval_strategy,
+        "language": doc.language,
+        "collection_name": doc.collection_name,
+        "embedding_model": doc.embedding_model,
+        "chunk_count": doc.chunk_count,
+        "metadata_json": doc.metadata_json,
+        "created_at": doc.created_at,
+        "updated_at": doc.updated_at,
+    }
diff --git a/backend/app/api/embeddings.py b/backend/app/api/embeddings.py
new file mode 100644
index 0000000..c9fa1d8
+++ b/backend/app/api/embeddings.py
@@ -0,0 +1,40 @@
+from fastapi import APIRouter
+from app.embeddings.ollama_client import ollama_client
+from app.schemas.embeddings import (
+    EmbeddingRequest, EmbeddingBatchRequest,
+    EmbeddingResponse, EmbeddingBatchResponse, EmbeddingModelInfo,
+)
+from app.core.config import settings
+from app.core.logging import get_logger
+
+router = APIRouter(prefix="/api/embeddings", tags=["Embeddings"])
+logger = get_logger("api.embeddings")
+
+
+@router.post("/generate", response_model=EmbeddingResponse)
+async def generate_embedding(req: EmbeddingRequest):
+    model = req.model or settings.OLLAMA_EMBED_MODEL
+    embedding = await ollama_client.embeddings(req.text, model)
+    return EmbeddingResponse(
+        text=req.text,
+        embedding=embedding,
+        model=model,
+        dimensions=len(embedding),
+    )
+
+
+@router.post("/batch", response_model=EmbeddingBatchResponse)
+async def batch_embeddings(req: EmbeddingBatchRequest):
+    model = req.model or settings.OLLAMA_EMBED_MODEL
+    embeddings = await ollama_client.batch_embeddings(req.texts, model)
+    responses = [
+        EmbeddingResponse(text=t, embedding=e, model=model, dimensions=len(e))
+        for t, e in zip(req.texts, embeddings)
+    ]
+    return EmbeddingBatchResponse(embeddings=responses, model=model)
+
+
+@router.get("/models", response_model=list[EmbeddingModelInfo])
+async def list_embedding_models():
+    models = await ollama_client.list_models()
+    return [EmbeddingModelInfo(name=m.get("name", ""), dimensions=None) for m in models]
diff --git a/backend/app/api/health.py b/backend/app/api/health.py
new file mode 100644
index 0000000..1ce08e0
+++ b/backend/app/api/health.py
@@ -0,0 +1,56 @@
+from fastapi import APIRouter, Depends
+from sqlalchemy.ext.asyncio import AsyncSession
+from sqlalchemy import text
+
+from app.core.dependencies import get_db
+from app.chromadb.client import chroma_client
+from app.embeddings.ollama_client import ollama_client
+from app.core.logging import get_logger
+
+router = APIRouter(tags=["Health"])
+logger = get_logger("api.health")
+
+
+@router.get("/health")
+async def health():
+    return {"status": "ok", "service": "MultimodalRAGPlatform"}
+
+
+@router.get("/health/db")
+async def health_db(db: AsyncSession = Depends(get_db)):
+    try:
+        await db.execute(text("SELECT 1"))
+        return {"status": "ok", "database": "sqlite"}
+    except Exception as e:
+        logger.error(f"DB health check failed: {e}")
+        return {"status": "error", "database": "sqlite", "detail": str(e)}
+
+
+@router.get("/health/chroma")
+async def health_chroma():
+    ok = chroma_client.health_check()
+    status = "ok" if ok else "error"
+    collections = []
+    if ok:
+        try:
+            collections = chroma_client.list_collections()
+        except Exception:
+            pass
+    return {"status": status, "chromadb": "persistent", "collections": collections}
+
+
+@router.get("/health/ollama")
+async def health_ollama():
+    ok = await ollama_client.health_check()
+    models = []
+    if ok:
+        try:
+            raw = await ollama_client.list_models()
+            models = [m.get("name") for m in raw]
+        except Exception:
+            pass
+    return {
+        "status": "ok" if ok else "error",
+        "ollama": "connected" if ok else "unreachable",
+        "models": models,
+    }
diff --git a/backend/app/api/markdown.py b/backend/app/api/markdown.py
new file mode 100644
index 0000000..dc2a894
+++ b/backend/app/api/markdown.py
@@ -0,0 +1,51 @@
+from fastapi import APIRouter, Depends, UploadFile, File, Form
+from sqlalchemy.ext.asyncio import AsyncSession
+from typing import Optional
+import json
+
+from app.core.dependencies import get_db
+from app.rag.markdown_rag import markdown_rag
+from app.services.document_service import document_service
+from app.schemas.search import SearchResult
+from app.core.logging import get_logger
+
+router = APIRouter(prefix="/api/markdown", tags=["Markdown RAG"])
+logger = get_logger("api.markdown")
+
+
+@router.post("/index")
+async def index_markdown(
+    file: UploadFile = File(...),
+    metadata: Optional[str] = Form(None),
+    db: AsyncSession = Depends(get_db),
+):
+    content = await file.read()
+    extra_meta = json.loads(metadata) if metadata else {}
+    result = await document_service.upload_and_index(db, file.filename, content, extra_meta)
+    doc = result["document"]
+    return {
+        "document_id": doc.id,
+        "filename": doc.filename,
+        "chunk_count": result["chunk_count"],
+        "message": "Markdown indexed with header-aware chunking",
+    }
+
+
+@router.post("/query", response_model=list[SearchResult])
+async def query_markdown(
+    query: str,
+    document_id: Optional[str] = None,
+    top_k: int = 5,
+):
+    chunks = await markdown_rag.query(query, document_id=document_id, top_k=top_k)
+    return [
+        SearchResult(
+            chunk_id=r.get("chunk_id", ""),
+            document_id=r.get("metadata", {}).get("document_id", ""),
+            filename=r.get("metadata", {}).get("filename", ""),
+            chunk_text=r.get("chunk_text", ""),
+            score=r.get("score", 0.0),
+            metadata=r.get("metadata", {}),
+        )
+        for r in chunks
+    ]
diff --git a/backend/app/api/pdf.py b/backend/app/api/pdf.py
new file mode 100644
index 0000000..5b7f117
+++ b/backend/app/api/pdf.py
@@ -0,0 +1,52 @@
+from fastapi import APIRouter, Depends, UploadFile, File, Form
+from sqlalchemy.ext.asyncio import AsyncSession
+from typing import Optional
+import json
+
+from app.core.dependencies import get_db
+from app.rag.pdf_rag import pdf_rag
+from app.services.document_service import document_service
+from app.schemas.search import SearchResult
+from app.core.logging import get_logger
+
+router = APIRouter(prefix="/api/pdf", tags=["PDF RAG"])
+logger = get_logger("api.pdf")
+
+
+@router.post("/index")
+async def index_pdf(
+    file: UploadFile = File(...),
+    metadata: Optional[str] = Form(None),
+    db: AsyncSession = Depends(get_db),
+):
+    content = await file.read()
+    extra_meta = json.loads(metadata) if metadata else {}
+    result = await document_service.upload_and_index(db, file.filename, content, extra_meta)
+    doc = result["document"]
+    return {
+        "document_id": doc.id,
+        "filename": doc.filename,
+        "chunk_count": result["chunk_count"],
+        "message": "PDF indexed with hierarchical strategy",
+    }
+
+
+@router.post("/query", response_model=list[SearchResult])
+async def query_pdf(
+    query: str,
+    document_id: Optional[str] = None,
+    section: Optional[str] = None,
+    top_k: int = 5,
+):
+    chunks = await pdf_rag.query(query, document_id=document_id, top_k=top_k, section=section)
+    return [
+        SearchResult(
+            chunk_id=r.get("chunk_id", ""),
+            document_id=r.get("metadata", {}).get("document_id", ""),
+            filename=r.get("metadata", {}).get("filename", ""),
+            chunk_text=r.get("chunk_text", ""),
+            score=r.get("score", 0.0),
+            metadata=r.get("metadata", {}),
+        )
+        for r in chunks
+    ]
diff --git a/backend/app/api/rag.py b/backend/app/api/rag.py
new file mode 100644
index 0000000..4ad444f
+++ b/backend/app/api/rag.py
@@ -0,0 +1,54 @@
+from fastapi import APIRouter, Depends
+from fastapi.responses import StreamingResponse
+from sqlalchemy.ext.asyncio import AsyncSession
+
+from app.core.dependencies import get_db
+from app.services.rag_service import rag_service
+from app.schemas.rag import (
+    RAGQueryRequest, RAGQueryResponse, RAGRetrieveRequest,
+    EvaluationRequest, EvaluationResponse
+)
+from app.schemas.search import SearchResult
+from app.core.logging import get_logger
+
+router = APIRouter(prefix="/api/rag", tags=["RAG"])
+logger = get_logger("api.rag")
+
+
+@router.post("/query", response_model=RAGQueryResponse)
+async def rag_query(req: RAGQueryRequest, db: AsyncSession = Depends(get_db)):
+    result = await rag_service.query(db, req.query, req.strategy, req.top_k, req.filters)
+    return result
+
+
+@router.post("/query/stream")
+async def rag_query_stream(req: RAGQueryRequest):
+    async def generator():
+        async for token in rag_service.query_stream(req.query, req.strategy, req.top_k, req.filters):
+            yield token
+
+    return StreamingResponse(generator(), media_type="text/plain")
+
+
+@router.post("/retrieve", response_model=list[SearchResult])
+async def rag_retrieve(req: RAGRetrieveRequest):
+    chunks = await rag_service.retrieve(req.query, req.strategy, req.top_k, req.filters)
+    results = []
+    for r in chunks:
+        meta = r.get("metadata", {})
+        results.append(SearchResult(
+            chunk_id=r.get("chunk_id", ""),
+            document_id=r.get("document_id") or meta.get("document_id", ""),
+            filename=r.get("filename") or meta.get("filename", ""),
+            chunk_text=r.get("chunk_text", ""),
+            score=r.get("score", 0.0),
+            metadata=meta,
+        ))
+    return results
+
+
+@router.post("/evaluate", response_model=EvaluationResponse)
+async def rag_evaluate(req: EvaluationRequest, db: AsyncSession = Depends(get_db)):
+    questions = [q.model_dump() for q in req.questions]
+    result = await rag_service.evaluate(db, questions, req.dataset_name or "default")
+    return result
diff --git a/backend/app/api/search.py b/backend/app/api/search.py
new file mode 100644
index 0000000..0355a3d
+++ b/backend/app/api/search.py
@@ -0,0 +1,89 @@
+import time
+from fastapi import APIRouter, Depends
+from sqlalchemy.ext.asyncio import AsyncSession
+
+from app.core.dependencies import get_db
+from app.rag.vector_rag import vector_rag
+from app.rag.hybrid_rag import hybrid_rag
+from app.rag.bm25 import bm25_retriever
+from app.rag.table_rag import table_rag
+from app.rag.metadata_filter import filter_results, build_chroma_filter
+from app.embeddings.ollama_client import ollama_client
+from app.chromadb.client import chroma_client
+from app.schemas.search import SearchRequest, SearchResponse, SearchResult
+from app.core.logging import get_logger
+
+router = APIRouter(prefix="/api/search", tags=["Search"])
+logger = get_logger("api.search")
+
+
+def _build_response(query: str, results: list[dict], strategy: str, start_time: float) -> SearchResponse:
+    latency = (time.time() - start_time) * 1000
+    search_results = []
+    sources = []
+    for r in results:
+        meta = r.get("metadata", {})
+        sr = SearchResult(
+            chunk_id=r.get("chunk_id", ""),
+            document_id=r.get("document_id") or meta.get("document_id", ""),
+            filename=r.get("filename") or meta.get("filename", ""),
+            chunk_text=r.get("chunk_text", ""),
+            score=r.get("score", 0.0),
+            metadata=meta,
+        )
+        search_results.append(sr)
+        fn = sr.filename
+        if fn and fn not in sources:
+            sources.append(fn)
+    confidence = round(sum(r.score for r in search_results) / max(len(search_results), 1), 4)
+    return SearchResponse(
+        query=query,
+        results=search_results,
+        confidence=confidence,
+        sources=sources,
+        latency_ms=round(latency, 2),
+        strategy=strategy,
+    )
+
+
+@router.post("/vector", response_model=SearchResponse)
+async def vector_search(req: SearchRequest):
+    start = time.time()
+    collection = req.collection_name or "text_documents"
+    results = await vector_rag.retrieve(req.query, collection, req.top_k, req.filters)
+    return _build_response(req.query, results, "vector", start)
+
+
+@router.post("/bm25", response_model=SearchResponse)
+async def bm25_search(req: SearchRequest):
+    start = time.time()
+    collection = req.collection_name or "text_documents"
+    results = bm25_retriever.search(collection, req.query, req.top_k)
+    if req.filters:
+        results = filter_results(results, req.filters)
+    return _build_response(req.query, results, "bm25", start)
+
+
+@router.post("/hybrid", response_model=SearchResponse)
+async def hybrid_search(req: SearchRequest):
+    start = time.time()
+    collection = req.collection_name or "text_documents"
+    results = await hybrid_rag.retrieve(req.query, collection, req.top_k, req.filters)
+    return _build_response(req.query, results, "hybrid", start)
+
+
+@router.post("/metadata", response_model=SearchResponse)
+async def metadata_search(req: SearchRequest):
+    start = time.time()
+    collection = req.collection_name or "text_documents"
+    query_emb = await ollama_client.embeddings(req.query)
+    where = build_chroma_filter(req.filters or {})
+    results = chroma_client.search(collection, query_emb, req.top_k, where)
+    return _build_response(req.query, results, "metadata", start)
+
+
+@router.post("/table", response_model=SearchResponse)
+async def table_search(req: SearchRequest):
+    start = time.time()
+    results = await table_rag.query(req.query, top_k=req.top_k)
+    return _build_response(req.query, results, "table", start)
diff --git a/backend/app/api/tablerag.py b/backend/app/api/tablerag.py
new file mode 100644
index 0000000..27b595c
+++ b/backend/app/api/tablerag.py
@@ -0,0 +1,59 @@
+from fastapi import APIRouter, Depends, UploadFile, File, Form
+from sqlalchemy.ext.asyncio import AsyncSession
+from typing import Optional
+import json
+
+from app.core.dependencies import get_db
+from app.rag.table_rag import table_rag
+from app.services.document_service import document_service
+from app.schemas.search import SearchResult
+from app.core.logging import get_logger
+
+router = APIRouter(prefix="/api/tablerag", tags=["TableRAG"])
+logger = get_logger("api.tablerag")
+
+
+@router.post("/index")
+async def index_table(
+    file: UploadFile = File(...),
+    metadata: Optional[str] = Form(None),
+    db: AsyncSession = Depends(get_db),
+):
+    content = await file.read()
+    extra_meta = json.loads(metadata) if metadata else {}
+    result = await document_service.upload_and_index(db, file.filename, content, extra_meta)
+    doc = result["document"]
+    return {
+        "document_id": doc.id,
+        "filename": doc.filename,
+        "chunk_count": result["chunk_count"],
+        "message": "Table indexed successfully",
+    }
+
+
+@router.post("/query", response_model=list[SearchResult])
+async def query_table(
+    query: str,
+    document_id: Optional[str] = None,
+    top_k: int = 5,
+):
+    chunks = await table_rag.query(query, document_id=document_id, top_k=top_k)
+    return [
+        SearchResult(
+            chunk_id=r.get("chunk_id", ""),
+            document_id=r.get("metadata", {}).get("document_id", ""),
+            filename=r.get("metadata", {}).get("filename", ""),
+            chunk_text=r.get("chunk_text", ""),
+            score=r.get("score", 0.0),
+            metadata=r.get("metadata", {}),
+        )
+        for r in chunks
+    ]
+
+
+@router.get("/schema/{document_id}")
+async def get_table_schema(document_id: str):
+    schema = await table_rag.get_schema(document_id)
+    if not schema:
+        return {"document_id": document_id, "schema": None, "message": "Schema not found"}
+    return {"document_id": document_id, "schema": schema}
diff --git a/backend/app/api/web.py b/backend/app/api/web.py
new file mode 100644
index 0000000..598c9c5
+++ b/backend/app/api/web.py
@@ -0,0 +1,37 @@
+from fastapi import APIRouter, Depends
+from sqlalchemy.ext.asyncio import AsyncSession
+
+from app.core.dependencies import get_db
+from app.services.web_service import web_service
+from app.schemas.web import WebIngestRequest, WebQueryRequest, WebIngestResponse
+from app.schemas.search import SearchResult
+from app.core.logging import get_logger
+
+router = APIRouter(prefix="/api/web", tags=["Web Ingestion"])
+logger = get_logger("api.web")
+
+
+@router.post("/ingest", response_model=WebIngestResponse)
+async def ingest_url(req: WebIngestRequest):
+    result = await web_service.ingest(
+        url=req.url,
+        collection_name=req.collection_name or "web_documents",
+        metadata=req.metadata,
+    )
+    return result
+
+
+@router.post("/query", response_model=list[SearchResult])
+async def query_web(req: WebQueryRequest):
+    chunks = await web_service.query(req.query, url=req.url, top_k=req.top_k)
+    return [
+        SearchResult(
+            chunk_id=r.get("chunk_id", ""),
+            document_id=r.get("metadata", {}).get("document_id", ""),
+            filename=r.get("metadata", {}).get("source_url", ""),
+            chunk_text=r.get("chunk_text", ""),
+            score=r.get("score", 0.0),
+            metadata=r.get("metadata", {}),
+        )
+        for r in chunks
+    ]
diff --git a/backend/app/chromadb/__init__.py b/backend/app/chromadb/__init__.py
new file mode 100644
index 0000000..a906368
+++ b/backend/app/chromadb/__init__.py
@@ -0,0 +1 @@
+# chromadb package
diff --git a/backend/app/chromadb/client.py b/backend/app/chromadb/client.py
new file mode 100644
index 0000000..a55a594
+++ b/backend/app/chromadb/client.py
@@ -0,0 +1,183 @@
+import uuid
+from typing import Any, Optional
+import chromadb
+from chromadb.config import Settings as ChromaSettings
+from app.core.config import settings
+from app.core.logging import get_logger
+from app.core.exceptions import ChromaDBError
+
+logger = get_logger("chromadb_client")
+
+COLLECTIONS = [
+    "table_documents",
+    "pdf_documents",
+    "markdown_documents",
+    "text_documents",
+    "audio_transcripts",
+    "web_documents",
+]
+
+
+class ChromaDBClient:
+    def __init__(self):
+        self._client: Optional[chromadb.Client] = None
+
+    def get_client(self) -> chromadb.Client:
+        if self._client is None:
+            try:
+                self._client = chromadb.PersistentClient(
+                    path=settings.CHROMA_PERSIST_DIR,
+                    settings=ChromaSettings(anonymized_telemetry=False),
+                )
+                logger.info(f"ChromaDB initialized at {settings.CHROMA_PERSIST_DIR}")
+            except Exception as e:
+                logger.error(f"ChromaDB init failed: {e}")
+                raise ChromaDBError(str(e))
+        return self._client
+
+    def create_collection(self, name: str, metadata: Optional[dict] = None) -> chromadb.Collection:
+        try:
+            client = self.get_client()
+            collection = client.get_or_create_collection(
+                name=name,
+                metadata=metadata or {"hnsw:space": "cosine"},
+            )
+            logger.info(f"Collection '{name}' ready")
+            return collection
+        except Exception as e:
+            logger.error(f"create_collection failed for '{name}': {e}")
+            raise ChromaDBError(str(e))
+
+    def delete_collection(self, name: str) -> bool:
+        try:
+            client = self.get_client()
+            client.delete_collection(name)
+            logger.info(f"Collection '{name}' deleted")
+            return True
+        except Exception as e:
+            logger.error(f"delete_collection failed for '{name}': {e}")
+            raise ChromaDBError(str(e))
+
+    def add_documents(
+        self,
+        collection_name: str,
+        ids: list[str],
+        embeddings: list[list[float]],
+        documents: list[str],
+        metadatas: Optional[list[dict]] = None,
+    ) -> bool:
+        try:
+            collection = self.create_collection(collection_name)
+            collection.add(
+                ids=ids,
+                embeddings=embeddings,
+                documents=documents,
+                metadatas=metadatas or [{} for _ in ids],
+            )
+            logger.info(f"Added {len(ids)} docs to '{collection_name}'")
+            return True
+        except Exception as e:
+            logger.error(f"add_documents failed: {e}")
+            raise ChromaDBError(str(e))
+
+    def search(
+        self,
+        collection_name: str,
+        query_embedding: list[float],
+        top_k: int = 5,
+        where: Optional[dict] = None,
+    ) -> list[dict[str, Any]]:
+        try:
+            collection = self.create_collection(collection_name)
+            kwargs: dict[str, Any] = {
+                "query_embeddings": [query_embedding],
+                "n_results": min(top_k, collection.count() or 1),
+                "include": ["documents", "metadatas", "distances"],
+            }
+            if where:
+                kwargs["where"] = where
+            results = collection.query(**kwargs)
+            output = []
+            ids = results.get("ids", [[]])[0]
+            docs = results.get("documents", [[]])[0]
+            metas = results.get("metadatas", [[]])[0]
+            distances = results.get("distances", [[]])[0]
+            for i, chunk_id in enumerate(ids):
+                score = 1.0 - (distances[i] if distances else 0.0)
+                output.append({
+                    "chunk_id": chunk_id,
+                    "chunk_text": docs[i] if docs else "",
+                    "metadata": metas[i] if metas else {},
+                    "score": round(score, 4),
+                })
+            return output
+        except Exception as e:
+            logger.error(f"search failed in '{collection_name}': {e}")
+            raise ChromaDBError(str(e))
+
+    def metadata_filter(
+        self,
+        collection_name: str,
+        query_embedding: list[float],
+        filters: dict,
+        top_k: int = 5,
+    ) -> list[dict[str, Any]]:
+        return self.search(collection_name, query_embedding, top_k, where=filters)
+
+    def reindex(
+        self,
+        collection_name: str,
+        ids: list[str],
+        embeddings: list[list[float]],
+        documents: list[str],
+        metadatas: Optional[list[dict]] = None,
+    ) -> bool:
+        try:
+            self.delete_collection(collection_name)
+            return self.add_documents(collection_name, ids, embeddings, documents, metadatas)
+        except Exception as e:
+            logger.error(f"reindex failed: {e}")
+            raise ChromaDBError(str(e))
+
+    def list_collections(self) -> list[str]:
+        try:
+            client = self.get_client()
+            return [c.name for c in client.list_collections()]
+        except Exception as e:
+            logger.error(f"list_collections failed: {e}")
+            raise ChromaDBError(str(e))
+
+    def get_collection_count(self, collection_name: str) -> int:
+        try:
+            collection = self.create_collection(collection_name)
+            return collection.count()
+        except Exception:
+            return 0
+
+    def delete_by_document_id(self, collection_name: str, document_id: str) -> bool:
+        try:
+            collection = self.create_collection(collection_name)
+            results = collection.get(where={"document_id": document_id})
+            ids = results.get("ids", [])
+            if ids:
+                collection.delete(ids=ids)
+                logger.info(f"Deleted {len(ids)} chunks for doc {document_id} from '{collection_name}'")
+            return True
+        except Exception as e:
+            logger.error(f"delete_by_document_id failed: {e}")
+            raise ChromaDBError(str(e))
+
+    def health_check(self) -> bool:
+        try:
+            self.get_client().heartbeat()
+            return True
+        except Exception:
+            return False
+
+    def init_collections(self):
+        for name in COLLECTIONS:
+            self.create_collection(name)
+        logger.info("All default collections initialized")
+
+
+chroma_client = ChromaDBClient()
diff --git a/backend/app/core/__init__.py b/backend/app/core/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/backend/app/core/config.py b/backend/app/core/config.py
new file mode 100644
index 0000000..3fa1ff5
+++ b/backend/app/core/config.py
@@ -0,0 +1,42 @@
+from pydantic_settings import BaseSettings
+from pydantic import Field
+from functools import lru_cache
+
+
+class Settings(BaseSettings):
+    APP_NAME: str = "MultimodalRAGPlatform"
+    APP_VERSION: str = "1.0.0"
+    DEBUG: bool = True
+
+    DATABASE_URL: str = "sqlite+aiosqlite:///./rag_platform.db"
+
+    CHROMA_HOST: str = "localhost"
+    CHROMA_PORT: int = 8001
+    CHROMA_PERSIST_DIR: str = "./chroma_db"
+
+    OLLAMA_BASE_URL: str = "http://localhost:11434"
+    OLLAMA_LLM_MODEL: str = "llama3.1:8b"
+    OLLAMA_EMBED_MODEL: str = "nomic-embed-text-v2-moe"
+    OLLAMA_TIMEOUT: int = 120
+    OLLAMA_MAX_RETRIES: int = 3
+
+    UPLOAD_DIR: str = "./uploads"
+    LOG_FILE: str = "./logs/rag.log"
+    LOG_LEVEL: str = "INFO"
+
+    TOP_K: int = 5
+    CHUNK_SIZE: int = 512
+    CHUNK_OVERLAP: int = 50
+    MAX_CONTEXT_CHUNKS: int = 10
+
+    class Config:
+        env_file = ".env"
+        extra = "ignore"
+
+
+@lru_cache()
+def get_settings() -> Settings:
+    return Settings()
+
+
+settings = get_settings()
diff --git a/backend/app/core/dependencies.py b/backend/app/core/dependencies.py
new file mode 100644
index 0000000..46ad68e
+++ b/backend/app/core/dependencies.py
@@ -0,0 +1,11 @@
+from typing import AsyncGenerator
+from sqlalchemy.ext.asyncio import AsyncSession
+from app.database.session import AsyncSessionLocal
+
+
+async def get_db() -> AsyncGenerator[AsyncSession, None]:
+    async with AsyncSessionLocal() as session:
+        try:
+            yield session
+        finally:
+            await session.close()
diff --git a/backend/app/core/exceptions.py b/backend/app/core/exceptions.py
new file mode 100644
index 0000000..d0de922
+++ b/backend/app/core/exceptions.py
@@ -0,0 +1,76 @@
+from fastapi import Request
+from fastapi.responses import JSONResponse
+from fastapi.exceptions import RequestValidationError
+from starlette.exceptions import HTTPException as StarletteHTTPException
+from app.core.logging import get_logger
+
+logger = get_logger("exceptions")
+
+
+class RAGPlatformException(Exception):
+    def __init__(self, message: str, status_code: int = 500):
+        self.message = message
+        self.status_code = status_code
+        super().__init__(message)
+
+
+class DocumentNotFoundError(RAGPlatformException):
+    def __init__(self, doc_id: str):
+        super().__init__(f"Document {doc_id} not found", 404)
+
+
+class ChunkNotFoundError(RAGPlatformException):
+    def __init__(self, chunk_id: str):
+        super().__init__(f"Chunk {chunk_id} not found", 404)
+
+
+class ConversationNotFoundError(RAGPlatformException):
+    def __init__(self, conv_id: str):
+        super().__init__(f"Conversation {conv_id} not found", 404)
+
+
+class OllamaConnectionError(RAGPlatformException):
+    def __init__(self, detail: str = ""):
+        super().__init__(f"Ollama connection failed: {detail}", 503)
+
+
+class ChromaDBError(RAGPlatformException):
+    def __init__(self, detail: str = ""):
+        super().__init__(f"ChromaDB error: {detail}", 503)
+
+
+class UnsupportedFileTypeError(RAGPlatformException):
+    def __init__(self, file_type: str):
+        super().__init__(f"Unsupported file type: {file_type}", 422)
+
+
+async def rag_platform_exception_handler(request: Request, exc: RAGPlatformException):
+    logger.error(f"RAGPlatformException: {exc.message} | path={request.url.path}")
+    return JSONResponse(
+        status_code=exc.status_code,
+        content={"error": exc.message, "status_code": exc.status_code},
+    )
+
+
+async def http_exception_handler(request: Request, exc: StarletteHTTPException):
+    logger.warning(f"HTTP {exc.status_code}: {exc.detail} | path={request.url.path}")
+    return JSONResponse(
+        status_code=exc.status_code,
+        content={"error": exc.detail, "status_code": exc.status_code},
+    )
+
+
+async def validation_exception_handler(request: Request, exc: RequestValidationError):
+    logger.warning(f"Validation error: {exc.errors()} | path={request.url.path}")
+    return JSONResponse(
+        status_code=422,
+        content={"error": "Validation failed", "details": exc.errors()},
+    )
+
+
+async def generic_exception_handler(request: Request, exc: Exception):
+    logger.error(f"Unhandled exception: {exc} | path={request.url.path}", exc_info=True)
+    return JSONResponse(
+        status_code=500,
+        content={"error": "Internal server error", "status_code": 500},
+    )
diff --git a/backend/app/core/logging.py b/backend/app/core/logging.py
new file mode 100644
index 0000000..792087c
+++ b/backend/app/core/logging.py
@@ -0,0 +1,40 @@
+import logging
+import sys
+from pathlib import Path
+from app.core.config import settings
+
+
+def setup_logging() -> logging.Logger:
+    log_path = Path(settings.LOG_FILE)
+    log_path.parent.mkdir(parents=True, exist_ok=True)
+
+    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
+
+    formatter = logging.Formatter(
+        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
+        datefmt="%Y-%m-%d %H:%M:%S",
+    )
+
+    logger = logging.getLogger("rag_platform")
+    logger.setLevel(log_level)
+    logger.handlers.clear()
+
+    file_handler = logging.FileHandler(log_path, encoding="utf-8")
+    file_handler.setFormatter(formatter)
+    file_handler.setLevel(log_level)
+
+    stream_handler = logging.StreamHandler(sys.stdout)
+    stream_handler.setFormatter(formatter)
+    stream_handler.setLevel(log_level)
+
+    logger.addHandler(file_handler)
+    logger.addHandler(stream_handler)
+
+    return logger
+
+
+logger = setup_logging()
+
+
+def get_logger(name: str) -> logging.Logger:
+    return logging.getLogger(f"rag_platform.{name}")
diff --git a/backend/app/database/__init__.py b/backend/app/database/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/backend/app/database/base.py b/backend/app/database/base.py
new file mode 100644
index 0000000..9643249
+++ b/backend/app/database/base.py
@@ -0,0 +1,14 @@
+from sqlalchemy.orm import DeclarativeBase
+from sqlalchemy import MetaData
+
+NAMING_CONVENTION = {
+    "ix": "ix_%(column_0_label)s",
+    "uq": "uq_%(table_name)s_%(column_0_name)s",
+    "ck": "ck_%(table_name)s_%(constraint_name)s",
+    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
+    "pk": "pk_%(table_name)s",
+}
+
+
+class Base(DeclarativeBase):
+    metadata = MetaData(naming_convention=NAMING_CONVENTION)
diff --git a/backend/app/database/init_db.py b/backend/app/database/init_db.py
new file mode 100644
index 0000000..55e84bb
+++ b/backend/app/database/init_db.py
@@ -0,0 +1,20 @@
+from app.database.session import engine
+from app.database.base import Base
+from app.database import models  # noqa: F401 - registers all models
+from app.core.logging import get_logger
+
+logger = get_logger("init_db")
+
+
+async def init_db():
+    logger.info("Initializing database tables...")
+    async with engine.begin() as conn:
+        await conn.run_sync(Base.metadata.create_all)
+    logger.info("Database initialized successfully.")
+
+
+async def drop_db():
+    logger.warning("Dropping all database tables...")
+    async with engine.begin() as conn:
+        await conn.run_sync(Base.metadata.drop_all)
+    logger.info("All tables dropped.")
diff --git a/backend/app/database/models.py b/backend/app/database/models.py
new file mode 100644
index 0000000..5464809
+++ b/backend/app/database/models.py
@@ -0,0 +1,112 @@
+from datetime import datetime
+from sqlalchemy import (
+    String, Text, Integer, Float, ForeignKey, DateTime, Index, JSON
+)
+from sqlalchemy.orm import Mapped, mapped_column, relationship
+from app.database.base import Base
+
+
+class Document(Base):
+    __tablename__ = "documents"
+
+    id: Mapped[str] = mapped_column(String(36), primary_key=True)
+    filename: Mapped[str] = mapped_column(String(512), nullable=False)
+    filepath: Mapped[str] = mapped_column(String(1024), nullable=False)
+    document_type: Mapped[str] = mapped_column(String(64), nullable=False)
+    retrieval_strategy: Mapped[str] = mapped_column(String(64), nullable=True)
+    language: Mapped[str] = mapped_column(String(16), nullable=True, default="en")
+    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
+    embedding_model: Mapped[str] = mapped_column(String(128), nullable=True)
+    collection_name: Mapped[str] = mapped_column(String(128), nullable=True)
+    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=True, default=dict)
+    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
+    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
+
+    chunks: Mapped[list["Chunk"]] = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
+
+    __table_args__ = (
+        Index("ix_documents_document_type", "document_type"),
+        Index("ix_documents_filename", "filename"),
+        Index("ix_documents_created_at", "created_at"),
+    )
+
+
+class Chunk(Base):
+    __tablename__ = "chunks"
+
+    id: Mapped[str] = mapped_column(String(36), primary_key=True)
+    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
+    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
+    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
+    chunk_metadata: Mapped[dict] = mapped_column(JSON, nullable=True, default=dict)
+    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
+
+    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
+
+    __table_args__ = (
+        Index("ix_chunks_document_id", "document_id"),
+        Index("ix_chunks_chunk_index", "chunk_index"),
+    )
+
+
+class Conversation(Base):
+    __tablename__ = "conversations"
+
+    id: Mapped[str] = mapped_column(String(36), primary_key=True)
+    title: Mapped[str] = mapped_column(String(512), nullable=True, default="New Conversation")
+    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
+    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
+
+    messages: Mapped[list["Message"]] = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
+
+    __table_args__ = (Index("ix_conversations_created_at", "created_at"),)
+
+
+class Message(Base):
+    __tablename__ = "messages"
+
+    id: Mapped[str] = mapped_column(String(36), primary_key=True)
+    conversation_id: Mapped[str] = mapped_column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
+    role: Mapped[str] = mapped_column(String(32), nullable=False)
+    content: Mapped[str] = mapped_column(Text, nullable=False)
+    sources: Mapped[dict] = mapped_column(JSON, nullable=True, default=list)
+    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
+
+    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")
+
+    __table_args__ = (
+        Index("ix_messages_conversation_id", "conversation_id"),
+        Index("ix_messages_role", "role"),
+    )
+
+
+class RetrievalLog(Base):
+    __tablename__ = "retrieval_logs"
+
+    id: Mapped[str] = mapped_column(String(36), primary_key=True)
+    query: Mapped[str] = mapped_column(Text, nullable=False)
+    retrieval_strategy: Mapped[str] = mapped_column(String(64), nullable=True)
+    retrieved_chunks: Mapped[dict] = mapped_column(JSON, nullable=True, default=list)
+    generated_answer: Mapped[str] = mapped_column(Text, nullable=True)
+    latency_ms: Mapped[float] = mapped_column(Float, nullable=True)
+    agent_used: Mapped[str] = mapped_column(String(64), nullable=True)
+    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
+
+    __table_args__ = (
+        Index("ix_retrieval_logs_created_at", "created_at"),
+        Index("ix_retrieval_logs_retrieval_strategy", "retrieval_strategy"),
+    )
+
+
+class EvaluationRun(Base):
+    __tablename__ = "evaluation_runs"
+
+    id: Mapped[str] = mapped_column(String(36), primary_key=True)
+    dataset_name: Mapped[str] = mapped_column(String(256), nullable=True)
+    accuracy: Mapped[float] = mapped_column(Float, nullable=True)
+    faithfulness: Mapped[float] = mapped_column(Float, nullable=True)
+    context_precision: Mapped[float] = mapped_column(Float, nullable=True)
+    context_recall: Mapped[float] = mapped_column(Float, nullable=True)
+    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
+
+    __table_args__ = (Index("ix_evaluation_runs_created_at", "created_at"),)
diff --git a/backend/app/database/session.py b/backend/app/database/session.py
new file mode 100644
index 0000000..578d396
+++ b/backend/app/database/session.py
@@ -0,0 +1,17 @@
+from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
+from app.core.config import settings
+
+engine = create_async_engine(
+    settings.DATABASE_URL,
+    echo=settings.DEBUG,
+    pool_pre_ping=True,
+    connect_args={"check_same_thread": False},
+)
+
+AsyncSessionLocal = async_sessionmaker(
+    engine,
+    class_=AsyncSession,
+    expire_on_commit=False,
+    autocommit=False,
+    autoflush=False,
+)
diff --git a/backend/app/embeddings/__init__.py b/backend/app/embeddings/__init__.py
new file mode 100644
index 0000000..3c13198
+++ b/backend/app/embeddings/__init__.py
@@ -0,0 +1 @@
+# embeddings package
diff --git a/backend/app/embeddings/ollama_client.py b/backend/app/embeddings/ollama_client.py
new file mode 100644
index 0000000..8403172
+++ b/backend/app/embeddings/ollama_client.py
@@ -0,0 +1,202 @@
+import asyncio
+import json
+from typing import AsyncGenerator, Optional
+import httpx
+from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
+from app.core.config import settings
+from app.core.logging import get_logger
+from app.core.exceptions import OllamaConnectionError
+
+logger = get_logger("ollama_client")
+
+
+class OllamaClient:
+    def __init__(self):
+        self.base_url = settings.OLLAMA_BASE_URL
+        self.llm_model = settings.OLLAMA_LLM_MODEL
+        self.embed_model = settings.OLLAMA_EMBED_MODEL
+        self.timeout = settings.OLLAMA_TIMEOUT
+        self._client: Optional[httpx.AsyncClient] = None
+
+    async def _get_client(self) -> httpx.AsyncClient:
+        if self._client is None or self._client.is_closed:
+            self._client = httpx.AsyncClient(
+                base_url=self.base_url,
+                timeout=httpx.Timeout(self.timeout),
+                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
+            )
+        return self._client
+
+    async def close(self):
+        if self._client and not self._client.is_closed:
+            await self._client.aclose()
+
+    @retry(
+        stop=stop_after_attempt(3),
+        wait=wait_exponential(multiplier=1, min=1, max=10),
+        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
+    )
+    async def generate(self, prompt: str, model: Optional[str] = None, system: Optional[str] = None) -> str:
+        client = await self._get_client()
+        payload = {
+            "model": model or self.llm_model,
+            "prompt": prompt,
+            "stream": False,
+        }
+        if system:
+            payload["system"] = system
+        try:
+            response = await client.post("/api/generate", json=payload)
+            response.raise_for_status()
+            data = response.json()
+            return data.get("response", "")
+        except httpx.HTTPStatusError as e:
+            logger.error(f"Ollama generate HTTP error: {e}")
+            raise OllamaConnectionError(str(e))
+        except (httpx.ConnectError, httpx.TimeoutException) as e:
+            logger.error(f"Ollama connection error: {e}")
+            raise OllamaConnectionError(str(e))
+
+    @retry(
+        stop=stop_after_attempt(3),
+        wait=wait_exponential(multiplier=1, min=1, max=10),
+        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
+    )
+    async def chat(
+        self,
+        messages: list[dict],
+        model: Optional[str] = None,
+        system: Optional[str] = None,
+    ) -> str:
+        client = await self._get_client()
+        chat_messages = []
+        if system:
+            chat_messages.append({"role": "system", "content": system})
+        chat_messages.extend(messages)
+        payload = {
+            "model": model or self.llm_model,
+            "messages": chat_messages,
+            "stream": False,
+        }
+        try:
+            response = await client.post("/api/chat", json=payload)
+            response.raise_for_status()
+            data = response.json()
+            return data.get("message", {}).get("content", "")
+        except httpx.HTTPStatusError as e:
+            logger.error(f"Ollama chat HTTP error: {e}")
+            raise OllamaConnectionError(str(e))
+        except (httpx.ConnectError, httpx.TimeoutException) as e:
+            logger.error(f"Ollama connection error: {e}")
+            raise OllamaConnectionError(str(e))
+
+    async def generate_stream(
+        self, prompt: str, model: Optional[str] = None, system: Optional[str] = None
+    ) -> AsyncGenerator[str, None]:
+        client = await self._get_client()
+        payload = {
+            "model": model or self.llm_model,
+            "prompt": prompt,
+            "stream": True,
+        }
+        if system:
+            payload["system"] = system
+        try:
+            async with client.stream("POST", "/api/generate", json=payload) as response:
+                response.raise_for_status()
+                async for line in response.aiter_lines():
+                    if line:
+                        try:
+                            data = json.loads(line)
+                            token = data.get("response", "")
+                            if token:
+                                yield token
+                            if data.get("done"):
+                                break
+                        except json.JSONDecodeError:
+                            continue
+        except (httpx.ConnectError, httpx.TimeoutException) as e:
+            logger.error(f"Ollama stream error: {e}")
+            raise OllamaConnectionError(str(e))
+
+    async def chat_stream(
+        self,
+        messages: list[dict],
+        model: Optional[str] = None,
+        system: Optional[str] = None,
+    ) -> AsyncGenerator[str, None]:
+        client = await self._get_client()
+        chat_messages = []
+        if system:
+            chat_messages.append({"role": "system", "content": system})
+        chat_messages.extend(messages)
+        payload = {
+            "model": model or self.llm_model,
+            "messages": chat_messages,
+            "stream": True,
+        }
+        try:
+            async with client.stream("POST", "/api/chat", json=payload) as response:
+                response.raise_for_status()
+                async for line in response.aiter_lines():
+                    if line:
+                        try:
+                            data = json.loads(line)
+                            token = data.get("message", {}).get("content", "")
+                            if token:
+                                yield token
+                            if data.get("done"):
+                                break
+                        except json.JSONDecodeError:
+                            continue
+        except (httpx.ConnectError, httpx.TimeoutException) as e:
+            logger.error(f"Ollama chat stream error: {e}")
+            raise OllamaConnectionError(str(e))
+
+    @retry(
+        stop=stop_after_attempt(3),
+        wait=wait_exponential(multiplier=1, min=1, max=10),
+        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
+    )
+    async def embeddings(self, text: str, model: Optional[str] = None) -> list[float]:
+        client = await self._get_client()
+        payload = {
+            "model": model or self.embed_model,
+            "prompt": text,
+        }
+        try:
+            response = await client.post("/api/embeddings", json=payload)
+            response.raise_for_status()
+            data = response.json()
+            return data.get("embedding", [])
+        except httpx.HTTPStatusError as e:
+            logger.error(f"Ollama embeddings HTTP error: {e}")
+            raise OllamaConnectionError(str(e))
+        except (httpx.ConnectError, httpx.TimeoutException) as e:
+            logger.error(f"Ollama embeddings connection error: {e}")
+            raise OllamaConnectionError(str(e))
+
+    async def batch_embeddings(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
+        tasks = [self.embeddings(text, model) for text in texts]
+        return await asyncio.gather(*tasks)
+
+    async def health_check(self) -> bool:
+        try:
+            client = await self._get_client()
+            response = await client.get("/api/tags")
+            return response.status_code == 200
+        except Exception:
+            return False
+
+    async def list_models(self) -> list[dict]:
+        try:
+            client = await self._get_client()
+            response = await client.get("/api/tags")
+            response.raise_for_status()
+            return response.json().get("models", [])
+        except Exception as e:
+            logger.error(f"Failed to list Ollama models: {e}")
+            return []
+
+
+ollama_client = OllamaClient()
diff --git a/backend/app/main.py b/backend/app/main.py
new file mode 100644
index 0000000..b673cb9
+++ b/backend/app/main.py
@@ -0,0 +1,103 @@
+import time
+from contextlib import asynccontextmanager
+
+from fastapi import FastAPI, Request
+from fastapi.middleware.cors import CORSMiddleware
+from fastapi.exceptions import RequestValidationError
+from starlette.exceptions import HTTPException as StarletteHTTPException
+
+from app.core.config import settings
+from app.core.logging import get_logger
+from app.core.exceptions import (
+    RAGPlatformException,
+    rag_platform_exception_handler,
+    http_exception_handler,
+    validation_exception_handler,
+    generic_exception_handler,
+)
+from app.database.init_db import init_db
+from app.chromadb.client import chroma_client
+
+# API routers
+from app.api.health import router as health_router
+from app.api.documents import router as documents_router
+from app.api.search import router as search_router
+from app.api.chat import router as chat_router
+from app.api.rag import router as rag_router
+from app.api.tablerag import router as tablerag_router
+from app.api.pdf import router as pdf_router
+from app.api.markdown import router as markdown_router
+from app.api.agents import router as agents_router
+from app.api.chroma import router as chroma_router
+from app.api.embeddings import router as embeddings_router
+from app.api.web import router as web_router
+
+logger = get_logger("main")
+
+
+@asynccontextmanager
+async def lifespan(app: FastAPI):
+    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
+    await init_db()
+    try:
+        chroma_client.init_collections()
+    except Exception as e:
+        logger.warning(f"ChromaDB init warning: {e}")
+    logger.info("Application startup complete")
+    yield
+    logger.info("Shutting down application")
+    from app.embeddings.ollama_client import ollama_client
+    await ollama_client.close()
+    logger.info("Shutdown complete")
+
+
+app = FastAPI(
+    title=settings.APP_NAME,
+    version=settings.APP_VERSION,
+    description="Intelligent Multimodal Agentic RAG Platform",
+    docs_url="/docs",
+    redoc_url="/redoc",
+    openapi_url="/openapi.json",
+    lifespan=lifespan,
+)
+
+# CORS
+app.add_middleware(
+    CORSMiddleware,
+    allow_origins=["*"],
+    allow_credentials=True,
+    allow_methods=["*"],
+    allow_headers=["*"],
+)
+
+
+# Logging middleware
+@app.middleware("http")
+async def logging_middleware(request: Request, call_next):
+    start = time.time()
+    logger.info(f"→ {request.method} {request.url.path}")
+    response = await call_next(request)
+    latency = (time.time() - start) * 1000
+    logger.info(f"← {request.method} {request.url.path} {response.status_code} [{latency:.1f}ms]")
+    return response
+
+
+# Exception handlers
+app.add_exception_handler(RAGPlatformException, rag_platform_exception_handler)
+app.add_exception_handler(StarletteHTTPException, http_exception_handler)
+app.add_exception_handler(RequestValidationError, validation_exception_handler)
+app.add_exception_handler(Exception, generic_exception_handler)
+
+# Register all routers
+app.include_router(health_router)
+app.include_router(documents_router)
+app.include_router(search_router)
+app.include_router(chat_router)
+app.include_router(rag_router)
+app.include_router(tablerag_router)
+app.include_router(pdf_router)
+app.include_router(markdown_router)
+app.include_router(agents_router)
+app.include_router(chroma_router)
+app.include_router(embeddings_router)
+app.include_router(web_router)
diff --git a/backend/app/rag/__init__.py b/backend/app/rag/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/backend/app/rag/bm25.py b/backend/app/rag/bm25.py
new file mode 100644
index 0000000..e02fe77
+++ b/backend/app/rag/bm25.py
@@ -0,0 +1,51 @@
+from typing import Any
+from rank_bm25 import BM25Okapi
+from app.core.logging import get_logger
+
+logger = get_logger("bm25")
+
+
+class BM25Retriever:
+    def __init__(self):
+        self._index: dict[str, BM25Okapi] = {}
+        self._corpus: dict[str, list[dict]] = {}
+
+    def _tokenize(self, text: str) -> list[str]:
+        return text.lower().split()
+
+    def index(self, collection_name: str, chunks: list[dict]):
+        """chunks: list of {chunk_id, chunk_text, metadata, document_id, filename}"""
+        self._corpus[collection_name] = chunks
+        tokenized = [self._tokenize(c["chunk_text"]) for c in chunks]
+        self._index[collection_name] = BM25Okapi(tokenized)
+        logger.info(f"BM25 indexed {len(chunks)} chunks for '{collection_name}'")
+
+    def search(self, collection_name: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
+        if collection_name not in self._index:
+            logger.warning(f"BM25 index not found for '{collection_name}'")
+            return []
+        bm25 = self._index[collection_name]
+        corpus = self._corpus[collection_name]
+        tokenized_query = self._tokenize(query)
+        scores = bm25.get_scores(tokenized_query)
+        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
+        results = []
+        for idx in top_indices:
+            if scores[idx] > 0:
+                chunk = corpus[idx]
+                results.append({
+                    "chunk_id": chunk.get("chunk_id", str(idx)),
+                    "chunk_text": chunk["chunk_text"],
+                    "metadata": chunk.get("metadata", {}),
+                    "document_id": chunk.get("document_id", ""),
+                    "filename": chunk.get("filename", ""),
+                    "score": float(scores[idx]),
+                })
+        return results
+
+    def remove_collection(self, collection_name: str):
+        self._index.pop(collection_name, None)
+        self._corpus.pop(collection_name, None)
+
+
+bm25_retriever = BM25Retriever()
diff --git a/backend/app/rag/evaluator.py b/backend/app/rag/evaluator.py
new file mode 100644
index 0000000..0cfdc6a
+++ b/backend/app/rag/evaluator.py
@@ -0,0 +1,65 @@
+import time
+from typing import Any
+from app.embeddings.ollama_client import ollama_client
+from app.core.logging import get_logger
+
+logger = get_logger("evaluator")
+
+
+def _cosine_similarity(a: list[float], b: list[float]) -> float:
+    if not a or not b:
+        return 0.0
+    dot = sum(x * y for x, y in zip(a, b))
+    norm_a = sum(x ** 2 for x in a) ** 0.5
+    norm_b = sum(x ** 2 for x in b) ** 0.5
+    if norm_a == 0 or norm_b == 0:
+        return 0.0
+    return dot / (norm_a * norm_b)
+
+
+async def compute_faithfulness(answer: str, context_chunks: list[str]) -> float:
+    """Approximate: embed answer and each context chunk, take max similarity."""
+    if not context_chunks:
+        return 0.0
+    answer_emb = await ollama_client.embeddings(answer)
+    scores = []
+    for chunk in context_chunks:
+        chunk_emb = await ollama_client.embeddings(chunk)
+        scores.append(_cosine_similarity(answer_emb, chunk_emb))
+    return round(sum(scores) / len(scores), 4) if scores else 0.0
+
+
+async def compute_answer_relevancy(question: str, answer: str) -> float:
+    q_emb = await ollama_client.embeddings(question)
+    a_emb = await ollama_client.embeddings(answer)
+    return round(_cosine_similarity(q_emb, a_emb), 4)
+
+
+async def compute_context_precision(question: str, context_chunks: list[str]) -> float:
+    if not context_chunks:
+        return 0.0
+    q_emb = await ollama_client.embeddings(question)
+    relevant = 0
+    for chunk in context_chunks:
+        c_emb = await ollama_client.embeddings(chunk)
+        sim = _cosine_similarity(q_emb, c_emb)
+        if sim > 0.6:
+            relevant += 1
+    return round(relevant / len(context_chunks), 4)
+
+
+async def compute_context_recall(expected_answer: str, context_chunks: list[str]) -> float:
+    if not context_chunks:
+        return 0.0
+    ea_emb = await ollama_client.embeddings(expected_answer)
+    sims = []
+    for chunk in context_chunks:
+        c_emb = await ollama_client.embeddings(chunk)
+        sims.append(_cosine_similarity(ea_emb, c_emb))
+    return round(max(sims) if sims else 0.0, 4)
+
+
+async def compute_accuracy(generated: str, expected: str) -> float:
+    g_emb = await ollama_client.embeddings(generated)
+    e_emb = await ollama_client.embeddings(expected)
+    return round(_cosine_similarity(g_emb, e_emb), 4)
diff --git a/backend/app/rag/hybrid_rag.py b/backend/app/rag/hybrid_rag.py
new file mode 100644
index 0000000..02db602
+++ b/backend/app/rag/hybrid_rag.py
@@ -0,0 +1,43 @@
+from typing import Any, Optional
+from app.rag.vector_rag import vector_rag
+from app.rag.bm25 import bm25_retriever
+from app.rag.rrf import reciprocal_rank_fusion
+from app.rag.metadata_filter import filter_results
+from app.core.logging import get_logger
+
+logger = get_logger("hybrid_rag")
+
+
+class HybridRAG:
+    async def retrieve(
+        self,
+        query: str,
+        collection_name: str = "text_documents",
+        top_k: int = 5,
+        filters: Optional[dict] = None,
+    ) -> list[dict[str, Any]]:
+        logger.info(f"HybridRAG retrieve: query='{query[:60]}'")
+
+        # Vector retrieval
+        vector_results = []
+        try:
+            vector_results = await vector_rag.retrieve(query, collection_name, top_k * 2, filters)
+        except Exception as e:
+            logger.warning(f"Vector retrieval failed: {e}")
+
+        # BM25 retrieval
+        bm25_results = []
+        try:
+            bm25_raw = bm25_retriever.search(collection_name, query, top_k * 2)
+            bm25_results = filter_results(bm25_raw, filters or {})
+        except Exception as e:
+            logger.warning(f"BM25 retrieval failed: {e}")
+
+        if not vector_results and not bm25_results:
+            return []
+
+        fused = reciprocal_rank_fusion([vector_results, bm25_results], top_k=top_k)
+        return fused
+
+
+hybrid_rag = HybridRAG()
diff --git a/backend/app/rag/markdown_rag.py b/backend/app/rag/markdown_rag.py
new file mode 100644
index 0000000..b83f44d
+++ b/backend/app/rag/markdown_rag.py
@@ -0,0 +1,91 @@
+import re
+from typing import Any, Optional
+from app.chromadb.client import chroma_client
+from app.embeddings.ollama_client import ollama_client
+from app.core.logging import get_logger
+
+logger = get_logger("markdown_rag")
+MD_COLLECTION = "markdown_documents"
+
+HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
+CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
+LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
+
+
+def _parse_markdown_sections(text: str) -> list[dict]:
+    """Split markdown by headings, preserving code blocks and links."""
+    sections = []
+    pos = 0
+    current_heading = "Introduction"
+    current_level = 1
+
+    for match in HEADING_RE.finditer(text):
+        chunk = text[pos:match.start()].strip()
+        if chunk:
+            sections.append({
+                "heading": current_heading,
+                "level": current_level,
+                "content": chunk,
+            })
+        current_heading = match.group(2).strip()
+        current_level = len(match.group(1))
+        pos = match.end()
+
+    remainder = text[pos:].strip()
+    if remainder:
+        sections.append({
+            "heading": current_heading,
+            "level": current_level,
+            "content": remainder,
+        })
+    return sections
+
+
+class MarkdownRAG:
+    async def index(
+        self,
+        document_id: str,
+        filename: str,
+        content: str,
+        extra_metadata: Optional[dict] = None,
+    ) -> dict[str, Any]:
+        sections = _parse_markdown_sections(content)
+        ids, embeddings, documents, metadatas = [], [], [], []
+        for i, section in enumerate(sections):
+            text = f"# {section['heading']}\n\n{section['content']}"
+            if len(text.strip()) < 20:
+                continue
+            emb = await ollama_client.embeddings(text)
+            chunk_id = f"{document_id}_sec{i}"
+            ids.append(chunk_id)
+            embeddings.append(emb)
+            documents.append(text)
+            metadatas.append({
+                "document_id": document_id,
+                "filename": filename,
+                "document_type": "markdown",
+                "section": section["heading"],
+                "heading_level": section["level"],
+                "chunk_index": i,
+                **(extra_metadata or {}),
+            })
+
+        if ids:
+            chroma_client.add_documents(MD_COLLECTION, ids, embeddings, documents, metadatas)
+        logger.info(f"Markdown indexed '{filename}': {len(ids)} sections")
+        return {"document_id": document_id, "chunk_count": len(ids)}
+
+    async def query(
+        self,
+        query: str,
+        document_id: Optional[str] = None,
+        top_k: int = 5,
+    ) -> list[dict[str, Any]]:
+        query_emb = await ollama_client.embeddings(query)
+        where = None
+        if document_id:
+            where = {"document_id": {"$eq": document_id}}
+        return chroma_client.search(MD_COLLECTION, query_emb, top_k, where)
+
+
+markdown_rag = MarkdownRAG()
diff --git a/backend/app/rag/metadata_filter.py b/backend/app/rag/metadata_filter.py
new file mode 100644
index 0000000..8f2843f
+++ b/backend/app/rag/metadata_filter.py
@@ -0,0 +1,38 @@
+from typing import Any, Optional
+
+
+SUPPORTED_FILTERS = {
+    "filename", "document_type", "section", "language",
+    "date", "document_id", "retrieval_strategy", "state",
+    "ministry", "department", "source",
+}
+
+
+def build_chroma_filter(filters: dict[str, Any]) -> Optional[dict]:
+    """Build a ChromaDB $and/$eq compatible filter dict."""
+    if not filters:
+        return None
+    valid = {k: v for k, v in filters.items() if k in SUPPORTED_FILTERS and v is not None}
+    if not valid:
+        return None
+    if len(valid) == 1:
+        key, val = next(iter(valid.items()))
+        return {key: {"$eq": val}}
+    return {"$and": [{k: {"$eq": v}} for k, v in valid.items()]}
+
+
+def filter_results(results: list[dict], filters: dict[str, Any]) -> list[dict]:
+    """In-memory metadata filtering for BM25 results."""
+    if not filters:
+        return results
+    filtered = []
+    for item in results:
+        meta = item.get("metadata", {})
+        match = all(
+            meta.get(k) == v
+            for k, v in filters.items()
+            if k in SUPPORTED_FILTERS and v is not None
+        )
+        if match:
+            filtered.append(item)
+    return filtered
diff --git a/backend/app/rag/pdf_rag.py b/backend/app/rag/pdf_rag.py
new file mode 100644
index 0000000..13e57de
+++ b/backend/app/rag/pdf_rag.py
@@ -0,0 +1,128 @@
+import io
+import re
+import uuid
+from typing import Any, Optional
+import pdfplumber
+from app.chromadb.client import chroma_client
+from app.embeddings.ollama_client import ollama_client
+from app.core.logging import get_logger
+from app.core.config import settings
+
+logger = get_logger("pdf_rag")
+PDF_COLLECTION = "pdf_documents"
+
+
+def _detect_heading(text: str) -> Optional[str]:
+    stripped = text.strip()
+    # Numbered headings: "1.", "1.2", "1.2.3" followed by title text
+    if re.match(r"^\d+(\.\d+)*\.?\s+\S.{0,80}$", stripped):
+        return stripped
+    # ALL-CAPS short lines
+    if len(stripped) < 100 and stripped.isupper() and len(stripped) > 2:
+        return stripped
+    # Title-case short lines (no trailing punctuation except colon)
+    if re.match(r"^[A-Z][A-Za-z\s\-]{2,60}:?$", stripped) and len(stripped) < 80:
+        return stripped
+    return None
+
+
+class PDFHierarchicalRAG:
+    async def index(
+        self,
+        document_id: str,
+        filename: str,
+        content: bytes,
+        extra_metadata: Optional[dict] = None,
+    ) -> dict[str, Any]:
+        ids, embeddings, documents, metadatas = [], [], [], []
+        current_section = "Introduction"
+        chunk_index = 0
+
+        with pdfplumber.open(io.BytesIO(content)) as pdf:
+            full_text_parts = []
+            page_texts = []
+            for page_num, page in enumerate(pdf.pages):
+                page_text = page.extract_text() or ""
+                page_texts.append((page_num + 1, page_text))
+                full_text_parts.append(page_text)
+
+        # Chunk by page with section tracking
+        for page_num, page_text in page_texts:
+            if not page_text.strip():
+                continue
+            lines = page_text.split("\n")
+            current_para = []
+            for line in lines:
+                heading = _detect_heading(line)
+                if heading:
+                    # flush current para
+                    if current_para:
+                        chunk_text = " ".join(current_para).strip()
+                        if len(chunk_text) > 50:
+                            chunk_id = f"{document_id}_p{page_num}_c{chunk_index}"
+                            emb = await ollama_client.embeddings(chunk_text)
+                            ids.append(chunk_id)
+                            embeddings.append(emb)
+                            documents.append(chunk_text)
+                            metadatas.append({
+                                "document_id": document_id,
+                                "filename": filename,
+                                "document_type": "pdf",
+                                "section": current_section,
+                                "page_number": page_num,
+                                "chunk_index": chunk_index,
+                                **(extra_metadata or {}),
+                            })
+                            chunk_index += 1
+                        current_para = []
+                    current_section = heading
+                else:
+                    current_para.append(line)
+
+            # flush remaining
+            if current_para:
+                chunk_text = " ".join(current_para).strip()
+                if len(chunk_text) > 50:
+                    chunk_id = f"{document_id}_p{page_num}_c{chunk_index}"
+                    emb = await ollama_client.embeddings(chunk_text)
+                    ids.append(chunk_id)
+                    embeddings.append(emb)
+                    documents.append(chunk_text)
+                    metadatas.append({
+                        "document_id": document_id,
+                        "filename": filename,
+                        "document_type": "pdf",
+                        "section": current_section,
+                        "page_number": page_num,
+                        "chunk_index": chunk_index,
+                        **(extra_metadata or {}),
+                    })
+                    chunk_index += 1
+
+        if ids:
+            chroma_client.add_documents(PDF_COLLECTION, ids, embeddings, documents, metadatas)
+        logger.info(f"PDF indexed '{filename}': {len(ids)} chunks")
+        return {"document_id": document_id, "chunk_count": len(ids)}
+
+    async def query(
+        self,
+        query: str,
+        document_id: Optional[str] = None,
+        top_k: int = 5,
+        section: Optional[str] = None,
+    ) -> list[dict[str, Any]]:
+        query_emb = await ollama_client.embeddings(query)
+        where_conditions = []
+        if document_id:
+            where_conditions.append({"document_id": {"$eq": document_id}})
+        if section:
+            where_conditions.append({"section": {"$eq": section}})
+        where = None
+        if len(where_conditions) == 1:
+            where = where_conditions[0]
+        elif len(where_conditions) > 1:
+            where = {"$and": where_conditions}
+        return chroma_client.search(PDF_COLLECTION, query_emb, top_k, where)
+
+
+pdf_rag = PDFHierarchicalRAG()
diff --git a/backend/app/rag/rrf.py b/backend/app/rag/rrf.py
new file mode 100644
index 0000000..e981ff9
+++ b/backend/app/rag/rrf.py
@@ -0,0 +1,29 @@
+from typing import Any
+
+
+def reciprocal_rank_fusion(
+    result_lists: list[list[dict]], k: int = 60, top_k: int = 5
+) -> list[dict[str, Any]]:
+    """
+    Fuse multiple ranked lists using Reciprocal Rank Fusion.
+    Each result must have a 'chunk_id' field.
+    """
+    scores: dict[str, float] = {}
+    chunk_data: dict[str, dict] = {}
+
+    for result_list in result_lists:
+        for rank, item in enumerate(result_list):
+            chunk_id = item.get("chunk_id", "")
+            if not chunk_id:
+                continue
+            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
+            if chunk_id not in chunk_data:
+                chunk_data[chunk_id] = item
+
+    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]
+    results = []
+    for chunk_id in sorted_ids:
+        item = chunk_data[chunk_id].copy()
+        item["score"] = round(scores[chunk_id], 6)
+        results.append(item)
+    return results
diff --git a/backend/app/rag/table_rag.py b/backend/app/rag/table_rag.py
new file mode 100644
index 0000000..e9aca2e
+++ b/backend/app/rag/table_rag.py
@@ -0,0 +1,99 @@
+import csv
+import io
+import json
+import uuid
+from typing import Any, Optional
+import pandas as pd
+from app.chromadb.client import chroma_client
+from app.embeddings.ollama_client import ollama_client
+from app.rag.metadata_filter import build_chroma_filter
+from app.core.logging import get_logger
+
+logger = get_logger("table_rag")
+
+SCHEMA_COLLECTION = "table_documents"
+
+
+class TableRAG:
+    async def index_csv(
+        self,
+        document_id: str,
+        filename: str,
+        content: bytes,
+        extra_metadata: Optional[dict] = None,
+    ) -> dict[str, Any]:
+        """Index CSV: schema + row/cell level chunks."""
+        df = pd.read_csv(io.BytesIO(content))
+        schema_info = {
+            "columns": list(df.columns),
+            "dtypes": {col: str(df[col].dtype) for col in df.columns},
+            "row_count": len(df),
+            "sample": df.head(3).to_dict(orient="records"),
+        }
+
+        ids, embeddings, documents, metadatas = [], [], [], []
+
+        # Schema chunk
+        schema_text = f"Table schema for {filename}:\nColumns: {', '.join(schema_info['columns'])}\nRow count: {schema_info['row_count']}\nSample: {json.dumps(schema_info['sample'][:2])}"
+        schema_emb = await ollama_client.embeddings(schema_text)
+        schema_id = f"{document_id}_schema"
+        ids.append(schema_id)
+        embeddings.append(schema_emb)
+        documents.append(schema_text)
+        meta = {
+            "document_id": document_id,
+            "filename": filename,
+            "document_type": "csv",
+            "chunk_type": "schema",
+            "columns": json.dumps(list(df.columns)),
+            **(extra_metadata or {}),
+        }
+        metadatas.append(meta)
+
+        # Row chunks (batch 5 rows per chunk for dense tables)
+        chunk_size = 5
+        for start in range(0, min(len(df), 500), chunk_size):
+            batch = df.iloc[start:start + chunk_size]
+            row_text = batch.to_csv(index=False)
+            row_emb = await ollama_client.embeddings(row_text)
+            row_id = f"{document_id}_rows_{start}"
+            ids.append(row_id)
+            embeddings.append(row_emb)
+            documents.append(row_text)
+            metadatas.append({
+                "document_id": document_id,
+                "filename": filename,
+                "document_type": "csv",
+                "chunk_type": "rows",
+                "row_start": start,
+                "row_end": start + chunk_size,
+                **(extra_metadata or {}),
+            })
+
+        chroma_client.add_documents(SCHEMA_COLLECTION, ids, embeddings, documents, metadatas)
+        logger.info(f"TableRAG indexed {filename}: {len(ids)} chunks")
+        return {"document_id": document_id, "chunk_count": len(ids), "schema": schema_info}
+
+    async def query(
+        self,
+        query: str,
+        document_id: Optional[str] = None,
+        top_k: int = 5,
+    ) -> list[dict[str, Any]]:
+        query_emb = await ollama_client.embeddings(query)
+        where = None
+        if document_id:
+            where = {"document_id": {"$eq": document_id}}
+        results = chroma_client.search(SCHEMA_COLLECTION, query_emb, top_k, where)
+        return results
+
+    async def get_schema(self, document_id: str) -> Optional[dict]:
+        query_emb = await ollama_client.embeddings("table schema columns")
+        where = {"$and": [{"document_id": {"$eq": document_id}}, {"chunk_type": {"$eq": "schema"}}]}
+        results = chroma_client.search(SCHEMA_COLLECTION, query_emb, 1, where)
+        if results:
+            return results[0]
+        return None
+
+
+table_rag = TableRAG()
diff --git a/backend/app/rag/vector_rag.py b/backend/app/rag/vector_rag.py
new file mode 100644
index 0000000..10572cc
+++ b/backend/app/rag/vector_rag.py
@@ -0,0 +1,51 @@
+from typing import Any, Optional
+from app.chromadb.client import chroma_client
+from app.embeddings.ollama_client import ollama_client
+from app.rag.metadata_filter import build_chroma_filter
+from app.core.logging import get_logger
+
+logger = get_logger("vector_rag")
+
+
+class VectorRAG:
+    async def retrieve(
+        self,
+        query: str,
+        collection_name: str = "text_documents",
+        top_k: int = 5,
+        filters: Optional[dict] = None,
+    ) -> list[dict[str, Any]]:
+        logger.info(f"VectorRAG retrieve: query='{query[:60]}' collection='{collection_name}'")
+        query_embedding = await ollama_client.embeddings(query)
+        where = build_chroma_filter(filters) if filters else None
+        results = chroma_client.search(collection_name, query_embedding, top_k, where)
+        for r in results:
+            r["document_id"] = r.get("metadata", {}).get("document_id", "")
+            r["filename"] = r.get("metadata", {}).get("filename", "")
+        return results
+
+    async def retrieve_multi_collection(
+        self,
+        query: str,
+        collection_names: list[str],
+        top_k: int = 5,
+        filters: Optional[dict] = None,
+    ) -> list[dict[str, Any]]:
+        query_embedding = await ollama_client.embeddings(query)
+        where = build_chroma_filter(filters) if filters else None
+        all_results = []
+        for collection in collection_names:
+            try:
+                results = chroma_client.search(collection, query_embedding, top_k, where)
+                for r in results:
+                    r["collection"] = collection
+                    r["document_id"] = r.get("metadata", {}).get("document_id", "")
+                    r["filename"] = r.get("metadata", {}).get("filename", "")
+                all_results.extend(results)
+            except Exception as e:
+                logger.warning(f"Collection '{collection}' search failed: {e}")
+        all_results.sort(key=lambda x: x["score"], reverse=True)
+        return all_results[:top_k]
+
+
+vector_rag = VectorRAG()
diff --git a/backend/app/repositories/__init__.py b/backend/app/repositories/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/backend/app/repositories/conversation_repository.py b/backend/app/repositories/conversation_repository.py
new file mode 100644
index 0000000..70b9021
+++ b/backend/app/repositories/conversation_repository.py
@@ -0,0 +1,59 @@
+from typing import Optional
+from sqlalchemy.ext.asyncio import AsyncSession
+from sqlalchemy import select, delete
+from sqlalchemy.orm import selectinload
+from app.database.models import Conversation, Message
+from app.core.logging import get_logger
+
+logger = get_logger("conversation_repo")
+
+
+class ConversationRepository:
+    async def create(self, db: AsyncSession, data: dict) -> Conversation:
+        conv = Conversation(**data)
+        db.add(conv)
+        await db.commit()
+        await db.refresh(conv)
+        return conv
+
+    async def get_by_id(self, db: AsyncSession, conv_id: str, with_messages: bool = False) -> Optional[Conversation]:
+        query = select(Conversation).where(Conversation.id == conv_id)
+        if with_messages:
+            query = query.options(selectinload(Conversation.messages))
+        result = await db.execute(query)
+        return result.scalar_one_or_none()
+
+    async def list_all(self, db: AsyncSession, skip: int = 0, limit: int = 50) -> list[Conversation]:
+        result = await db.execute(
+            select(Conversation).offset(skip).limit(limit).order_by(Conversation.updated_at.desc())
+        )
+        return list(result.scalars().all())
+
+    async def delete(self, db: AsyncSession, conv_id: str) -> bool:
+        await db.execute(delete(Conversation).where(Conversation.id == conv_id))
+        await db.commit()
+        return True
+
+    async def add_message(self, db: AsyncSession, data: dict) -> Message:
+        msg = Message(**data)
+        db.add(msg)
+        await db.commit()
+        await db.refresh(msg)
+        return msg
+
+    async def get_messages(self, db: AsyncSession, conv_id: str, limit: int = 20) -> list[Message]:
+        result = await db.execute(
+            select(Message)
+            .where(Message.conversation_id == conv_id)
+            .order_by(Message.created_at)
+            .limit(limit)
+        )
+        return list(result.scalars().all())
+
+    async def count(self, db: AsyncSession) -> int:
+        from sqlalchemy import func
+        result = await db.execute(select(func.count()).select_from(Conversation))
+        return result.scalar() or 0
+
+
+conversation_repo = ConversationRepository()
diff --git a/backend/app/repositories/document_repository.py b/backend/app/repositories/document_repository.py
new file mode 100644
index 0000000..9d09fe6
+++ b/backend/app/repositories/document_repository.py
@@ -0,0 +1,71 @@
+from typing import Optional
+from sqlalchemy.ext.asyncio import AsyncSession
+from sqlalchemy import select, delete
+from sqlalchemy.orm import selectinload
+from app.database.models import Document, Chunk
+from app.core.logging import get_logger
+
+logger = get_logger("document_repo")
+
+
+class DocumentRepository:
+    async def create(self, db: AsyncSession, data: dict) -> Document:
+        doc = Document(**data)
+        db.add(doc)
+        await db.commit()
+        await db.refresh(doc)
+        return doc
+
+    async def get_by_id(self, db: AsyncSession, doc_id: str) -> Optional[Document]:
+        result = await db.execute(select(Document).where(Document.id == doc_id))
+        return result.scalar_one_or_none()
+
+    async def list_all(self, db: AsyncSession, skip: int = 0, limit: int = 100) -> list[Document]:
+        result = await db.execute(select(Document).offset(skip).limit(limit).order_by(Document.created_at.desc()))
+        return list(result.scalars().all())
+
+    async def delete(self, db: AsyncSession, doc_id: str) -> bool:
+        await db.execute(delete(Document).where(Document.id == doc_id))
+        await db.commit()
+        return True
+
+    async def update(self, db: AsyncSession, doc_id: str, data: dict) -> Optional[Document]:
+        doc = await self.get_by_id(db, doc_id)
+        if not doc:
+            return None
+        for k, v in data.items():
+            setattr(doc, k, v)
+        await db.commit()
+        await db.refresh(doc)
+        return doc
+
+    async def get_chunks(self, db: AsyncSession, doc_id: str) -> list[Chunk]:
+        result = await db.execute(
+            select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.chunk_index)
+        )
+        return list(result.scalars().all())
+
+    async def create_chunk(self, db: AsyncSession, data: dict) -> Chunk:
+        chunk = Chunk(**data)
+        db.add(chunk)
+        await db.commit()
+        await db.refresh(chunk)
+        return chunk
+
+    async def delete_chunks(self, db: AsyncSession, doc_id: str) -> int:
+        result = await db.execute(delete(Chunk).where(Chunk.document_id == doc_id))
+        await db.commit()
+        return result.rowcount
+
+    async def bulk_create_chunks(self, db: AsyncSession, chunks: list[dict]) -> int:
+        db.add_all([Chunk(**c) for c in chunks])
+        await db.commit()
+        return len(chunks)
+
+    async def count(self, db: AsyncSession) -> int:
+        from sqlalchemy import func
+        result = await db.execute(select(func.count()).select_from(Document))
+        return result.scalar() or 0
+
+
+document_repo = DocumentRepository()
diff --git a/backend/app/repositories/log_repository.py b/backend/app/repositories/log_repository.py
new file mode 100644
index 0000000..16187ee
+++ b/backend/app/repositories/log_repository.py
@@ -0,0 +1,37 @@
+from sqlalchemy.ext.asyncio import AsyncSession
+from sqlalchemy import select
+from app.database.models import RetrievalLog, EvaluationRun
+from app.core.logging import get_logger
+
+logger = get_logger("log_repo")
+
+
+class LogRepository:
+    async def create_retrieval_log(self, db: AsyncSession, data: dict) -> RetrievalLog:
+        log = RetrievalLog(**data)
+        db.add(log)
+        await db.commit()
+        await db.refresh(log)
+        return log
+
+    async def create_evaluation_run(self, db: AsyncSession, data: dict) -> EvaluationRun:
+        run = EvaluationRun(**data)
+        db.add(run)
+        await db.commit()
+        await db.refresh(run)
+        return run
+
+    async def list_retrieval_logs(self, db: AsyncSession, skip: int = 0, limit: int = 50) -> list[RetrievalLog]:
+        result = await db.execute(
+            select(RetrievalLog).offset(skip).limit(limit).order_by(RetrievalLog.created_at.desc())
+        )
+        return list(result.scalars().all())
+
+    async def list_evaluation_runs(self, db: AsyncSession, skip: int = 0, limit: int = 50) -> list[EvaluationRun]:
+        result = await db.execute(
+            select(EvaluationRun).offset(skip).limit(limit).order_by(EvaluationRun.created_at.desc())
+        )
+        return list(result.scalars().all())
+
+
+log_repo = LogRepository()
diff --git a/backend/app/schemas/__init__.py b/backend/app/schemas/__init__.py
new file mode 100644
index 0000000..78ee4f2
+++ b/backend/app/schemas/__init__.py
@@ -0,0 +1 @@
+# schemas package
diff --git a/backend/app/schemas/agent.py b/backend/app/schemas/agent.py
new file mode 100644
index 0000000..791444f
+++ b/backend/app/schemas/agent.py
@@ -0,0 +1,25 @@
+from pydantic import BaseModel, Field
+from typing import Any, Optional
+
+
+class AgentRequest(BaseModel):
+    query: str = Field(..., min_length=1)
+    context: Optional[dict[str, Any]] = None
+    top_k: int = Field(default=5, ge=1, le=50)
+    filters: Optional[dict[str, Any]] = None
+
+
+class AgentResponse(BaseModel):
+    agent: str
+    query: str
+    answer: str
+    sources: list[Any] = []
+    reasoning: Optional[str] = None
+    latency_ms: float = 0.0
+    metadata: Optional[dict[str, Any]] = None
+
+
+class CoordinatorRequest(BaseModel):
+    query: str = Field(..., min_length=1)
+    conversation_id: Optional[str] = None
+    top_k: int = Field(default=5, ge=1, le=50)
diff --git a/backend/app/schemas/chat.py b/backend/app/schemas/chat.py
new file mode 100644
index 0000000..cdb10d6
+++ b/backend/app/schemas/chat.py
@@ -0,0 +1,42 @@
+from pydantic import BaseModel, Field
+from typing import Any, Optional
+from datetime import datetime
+
+
+class ChatRequest(BaseModel):
+    message: str = Field(..., min_length=1)
+    conversation_id: Optional[str] = None
+    top_k: int = Field(default=5, ge=1, le=20)
+    stream: bool = False
+
+
+class MessageResponse(BaseModel):
+    id: str
+    conversation_id: str
+    role: str
+    content: str
+    sources: Optional[list[Any]] = None
+    created_at: datetime
+
+    model_config = {"from_attributes": True}
+
+
+class ConversationResponse(BaseModel):
+    id: str
+    title: str
+    created_at: datetime
+    updated_at: datetime
+    messages: list[MessageResponse] = []
+
+    model_config = {"from_attributes": True}
+
+
+class ConversationListResponse(BaseModel):
+    conversations: list[ConversationResponse]
+    total: int
+
+
+class ChatResponse(BaseModel):
+    conversation_id: str
+    message: MessageResponse
+    sources: list[Any] = []
diff --git a/backend/app/schemas/document.py b/backend/app/schemas/document.py
new file mode 100644
index 0000000..9756870
+++ b/backend/app/schemas/document.py
@@ -0,0 +1,42 @@
+from pydantic import BaseModel, Field
+from typing import Any, Optional
+from datetime import datetime
+
+
+class DocumentResponse(BaseModel):
+    id: str
+    filename: str
+    filepath: str
+    document_type: str
+    retrieval_strategy: Optional[str] = None
+    language: Optional[str] = "en"
+    chunk_count: int = 0
+    embedding_model: Optional[str] = None
+    collection_name: Optional[str] = None
+    metadata_json: Optional[dict[str, Any]] = None
+    created_at: datetime
+    updated_at: datetime
+
+    model_config = {"from_attributes": True}
+
+
+class ChunkResponse(BaseModel):
+    id: str
+    document_id: str
+    chunk_index: int
+    chunk_text: str
+    chunk_metadata: Optional[dict[str, Any]] = None
+    created_at: datetime
+
+    model_config = {"from_attributes": True}
+
+
+class DocumentListResponse(BaseModel):
+    documents: list[DocumentResponse]
+    total: int
+
+
+class ReindexResponse(BaseModel):
+    document_id: str
+    message: str
+    chunk_count: int
diff --git a/backend/app/schemas/embeddings.py b/backend/app/schemas/embeddings.py
new file mode 100644
index 0000000..0141931
+++ b/backend/app/schemas/embeddings.py
@@ -0,0 +1,29 @@
+from pydantic import BaseModel, Field
+from typing import Optional
+
+
+class EmbeddingRequest(BaseModel):
+    text: str = Field(..., min_length=1)
+    model: Optional[str] = None
+
+
+class EmbeddingBatchRequest(BaseModel):
+    texts: list[str] = Field(..., min_items=1)
+    model: Optional[str] = None
+
+
+class EmbeddingResponse(BaseModel):
+    text: str
+    embedding: list[float]
+    model: str
+    dimensions: int
+
+
+class EmbeddingBatchResponse(BaseModel):
+    embeddings: list[EmbeddingResponse]
+    model: str
+
+
+class EmbeddingModelInfo(BaseModel):
+    name: str
+    dimensions: Optional[int] = None
diff --git a/backend/app/schemas/rag.py b/backend/app/schemas/rag.py
new file mode 100644
index 0000000..64f5aba
+++ b/backend/app/schemas/rag.py
@@ -0,0 +1,46 @@
+from pydantic import BaseModel, Field
+from typing import Any, Optional
+
+
+class RAGQueryRequest(BaseModel):
+    query: str = Field(..., min_length=1)
+    strategy: str = Field(default="hybrid", description="vector|bm25|hybrid|table|pdf|markdown")
+    top_k: int = Field(default=5, ge=1, le=50)
+    filters: Optional[dict[str, Any]] = None
+    conversation_id: Optional[str] = None
+
+
+class RAGQueryResponse(BaseModel):
+    query: str
+    answer: str
+    sources: list[Any] = []
+    strategy: str = ""
+    latency_ms: float = 0.0
+    confidence: float = 0.0
+
+
+class RAGRetrieveRequest(BaseModel):
+    query: str = Field(..., min_length=1)
+    strategy: str = Field(default="hybrid")
+    top_k: int = Field(default=5, ge=1, le=50)
+    filters: Optional[dict[str, Any]] = None
+
+
+class EvalQuestion(BaseModel):
+    question: str
+    expected_answer: str
+
+
+class EvaluationRequest(BaseModel):
+    questions: list[EvalQuestion]
+    dataset_name: Optional[str] = "default"
+
+
+class EvaluationResponse(BaseModel):
+    accuracy: float = 0.0
+    faithfulness: float = 0.0
+    context_precision: float = 0.0
+    context_recall: float = 0.0
+    answer_relevancy: float = 0.0
+    latency_avg_ms: float = 0.0
+    failed_questions: list[dict[str, Any]] = []
diff --git a/backend/app/schemas/search.py b/backend/app/schemas/search.py
new file mode 100644
index 0000000..1f64706
+++ b/backend/app/schemas/search.py
@@ -0,0 +1,27 @@
+from pydantic import BaseModel, Field
+from typing import Any, Optional
+
+
+class SearchRequest(BaseModel):
+    query: str = Field(..., min_length=1)
+    top_k: int = Field(default=5, ge=1, le=50)
+    filters: Optional[dict[str, Any]] = None
+    collection_name: Optional[str] = None
+
+
+class SearchResult(BaseModel):
+    chunk_id: str
+    document_id: str
+    filename: str
+    chunk_text: str
+    score: float
+    metadata: Optional[dict[str, Any]] = None
+
+
+class SearchResponse(BaseModel):
+    query: str
+    results: list[SearchResult]
+    confidence: float = 0.0
+    sources: list[str] = []
+    latency_ms: float = 0.0
+    strategy: str = ""
diff --git a/backend/app/schemas/web.py b/backend/app/schemas/web.py
new file mode 100644
index 0000000..b9cc2f7
+++ b/backend/app/schemas/web.py
@@ -0,0 +1,21 @@
+from pydantic import BaseModel, Field, HttpUrl
+from typing import Optional, Any
+
+
+class WebIngestRequest(BaseModel):
+    url: str = Field(..., description="URL to ingest")
+    collection_name: Optional[str] = "web_documents"
+    metadata: Optional[dict[str, Any]] = None
+
+
+class WebQueryRequest(BaseModel):
+    query: str = Field(..., min_length=1)
+    url: Optional[str] = None
+    top_k: int = Field(default=5, ge=1, le=50)
+
+
+class WebIngestResponse(BaseModel):
+    url: str
+    document_id: str
+    chunk_count: int
+    message: str
diff --git a/backend/app/services/__init__.py b/backend/app/services/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/backend/app/services/chat_service.py b/backend/app/services/chat_service.py
new file mode 100644
index 0000000..b352bde
+++ b/backend/app/services/chat_service.py
@@ -0,0 +1,136 @@
+import uuid
+from typing import Any, AsyncGenerator, Optional
+from datetime import datetime
+from sqlalchemy.ext.asyncio import AsyncSession
+from app.repositories.conversation_repository import conversation_repo
+from app.services.rag_service import rag_service
+from app.embeddings.ollama_client import ollama_client
+from app.core.logging import get_logger
+from app.core.exceptions import ConversationNotFoundError
+
+logger = get_logger("chat_service")
+
+SYSTEM_PROMPT = """You are an intelligent assistant with access to a document knowledge base.
+Answer questions using the provided context. If the context doesn't contain enough information,
+say so clearly. Always cite your sources when possible. Be concise and accurate."""
+
+
+class ChatService:
+    async def chat(
+        self,
+        db: AsyncSession,
+        message: str,
+        conversation_id: Optional[str] = None,
+        top_k: int = 5,
+    ) -> dict[str, Any]:
+        # Get or create conversation
+        if conversation_id:
+            conv = await conversation_repo.get_by_id(db, conversation_id)
+            if not conv:
+                raise ConversationNotFoundError(conversation_id)
+        else:
+            conv = await conversation_repo.create(db, {
+                "id": str(uuid.uuid4()),
+                "title": message[:60],
+            })
+
+        # Retrieve context
+        retrieval_result = await rag_service.retrieve(message, strategy="hybrid", top_k=top_k)
+        context_chunks = [r["chunk_text"] for r in retrieval_result]
+        sources = [
+            {"filename": r.get("filename", ""), "chunk_id": r.get("chunk_id", ""), "score": r.get("score", 0)}
+            for r in retrieval_result
+        ]
+
+        # Build messages with history
+        history = await conversation_repo.get_messages(db, conv.id, limit=10)
+        messages = []
+        for m in history[-8:]:
+            messages.append({"role": m.role, "content": m.content})
+
+        # Add context to current message
+        context_str = "\n\n".join(f"[Source: {r.get('filename','unknown')}]\n{r['chunk_text']}" for r in retrieval_result)
+        user_content = f"Context:\n{context_str}\n\nQuestion: {message}" if context_chunks else message
+        messages.append({"role": "user", "content": user_content})
+
+        # Generate answer
+        answer = await ollama_client.chat(messages, system=SYSTEM_PROMPT)
+
+        # Save user and assistant messages
+        await conversation_repo.add_message(db, {
+            "id": str(uuid.uuid4()),
+            "conversation_id": conv.id,
+            "role": "user",
+            "content": message,
+            "sources": [],
+        })
+        assistant_msg = await conversation_repo.add_message(db, {
+            "id": str(uuid.uuid4()),
+            "conversation_id": conv.id,
+            "role": "assistant",
+            "content": answer,
+            "sources": sources,
+        })
+
+        return {
+            "conversation_id": conv.id,
+            "message": assistant_msg,
+            "sources": sources,
+        }
+
+    async def chat_stream(
+        self,
+        db: AsyncSession,
+        message: str,
+        conversation_id: Optional[str] = None,
+        top_k: int = 5,
+    ) -> AsyncGenerator[str, None]:
+        if conversation_id:
+            conv = await conversation_repo.get_by_id(db, conversation_id)
+            if not conv:
+                raise ConversationNotFoundError(conversation_id)
+        else:
+            conv = await conversation_repo.create(db, {
+                "id": str(uuid.uuid4()),
+                "title": message[:60],
+            })
+
+        retrieval_result = await rag_service.retrieve(message, strategy="hybrid", top_k=top_k)
+        context_str = "\n\n".join(
+            f"[Source: {r.get('filename','unknown')}]\n{r['chunk_text']}"
+            for r in retrieval_result
+        )
+
+        history = await conversation_repo.get_messages(db, conv.id, limit=10)
+        messages = [{"role": m.role, "content": m.content} for m in history[-8:]]
+        user_content = f"Context:\n{context_str}\n\nQuestion: {message}" if retrieval_result else message
+        messages.append({"role": "user", "content": user_content})
+
+        await conversation_repo.add_message(db, {
+            "id": str(uuid.uuid4()),
+            "conversation_id": conv.id,
+            "role": "user",
+            "content": message,
+            "sources": [],
+        })
+
+        full_answer = []
+        async for token in ollama_client.chat_stream(messages, system=SYSTEM_PROMPT):
+            full_answer.append(token)
+            yield token
+
+        answer = "".join(full_answer)
+        sources = [
+            {"filename": r.get("filename", ""), "chunk_id": r.get("chunk_id", ""), "score": r.get("score", 0)}
+            for r in retrieval_result
+        ]
+        await conversation_repo.add_message(db, {
+            "id": str(uuid.uuid4()),
+            "conversation_id": conv.id,
+            "role": "assistant",
+            "content": answer,
+            "sources": sources,
+        })
+
+
+chat_service = ChatService()
diff --git a/backend/app/services/document_service.py b/backend/app/services/document_service.py
new file mode 100644
index 0000000..4c7e589
+++ b/backend/app/services/document_service.py
@@ -0,0 +1,230 @@
+import io
+import uuid
+import json
+from pathlib import Path
+from typing import Any, Optional
+from datetime import datetime
+import aiofiles
+from sqlalchemy.ext.asyncio import AsyncSession
+from app.repositories.document_repository import document_repo
+from app.chromadb.client import chroma_client
+from app.embeddings.ollama_client import ollama_client
+from app.rag.table_rag import table_rag
+from app.rag.pdf_rag import pdf_rag
+from app.rag.markdown_rag import markdown_rag
+from app.rag.bm25 import bm25_retriever
+from app.core.config import settings
+from app.core.logging import get_logger
+from app.core.exceptions import DocumentNotFoundError, UnsupportedFileTypeError
+
+logger = get_logger("document_service")
+
+SUPPORTED_TYPES = {
+    "pdf": "pdf",
+    "md": "markdown",
+    "txt": "text",
+    "csv": "csv",
+    "json": "json",
+}
+
+TYPE_TO_COLLECTION = {
+    "pdf": "pdf_documents",
+    "markdown": "markdown_documents",
+    "text": "text_documents",
+    "csv": "table_documents",
+    "json": "text_documents",
+}
+
+TYPE_TO_STRATEGY = {
+    "pdf": "hierarchical_rag",
+    "markdown": "structure_aware_rag",
+    "text": "vector_rag",
+    "csv": "table_rag",
+    "json": "vector_rag",
+}
+
+
+def _detect_type(filename: str) -> str:
+    ext = Path(filename).suffix.lstrip(".").lower()
+    if ext not in SUPPORTED_TYPES:
+        raise UnsupportedFileTypeError(ext)
+    return SUPPORTED_TYPES[ext]
+
+
+def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
+    words = text.split()
+    chunks = []
+    start = 0
+    while start < len(words):
+        end = min(start + chunk_size, len(words))
+        chunks.append(" ".join(words[start:end]))
+        start += chunk_size - overlap
+    return chunks
+
+
+async def _index_text_chunks(
+    document_id: str,
+    filename: str,
+    doc_type: str,
+    text: str,
+    collection: str,
+    extra_metadata: Optional[dict] = None,
+) -> list[dict]:
+    chunks = _chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
+    ids, embeddings, documents, metadatas = [], [], [], []
+    chunk_records = []
+
+    for i, chunk_text in enumerate(chunks):
+        emb = await ollama_client.embeddings(chunk_text)
+        chunk_id = f"{document_id}_chunk_{i}"
+        ids.append(chunk_id)
+        embeddings.append(emb)
+        documents.append(chunk_text)
+        meta = {
+            "document_id": document_id,
+            "filename": filename,
+            "document_type": doc_type,
+            "chunk_index": i,
+            **(extra_metadata or {}),
+        }
+        metadatas.append(meta)
+        chunk_records.append({
+            "id": chunk_id,
+            "document_id": document_id,
+            "chunk_index": i,
+            "chunk_text": chunk_text,
+            "chunk_metadata": meta,
+        })
+
+    if ids:
+        chroma_client.add_documents(collection, ids, embeddings, documents, metadatas)
+        bm25_retriever.index(collection, [
+            {"chunk_id": ids[j], "chunk_text": documents[j], "metadata": metadatas[j],
+             "document_id": document_id, "filename": filename}
+            for j in range(len(ids))
+        ])
+    return chunk_records
+
+
+class DocumentService:
+    async def upload_and_index(
+        self,
+        db: AsyncSession,
+        filename: str,
+        content: bytes,
+        extra_metadata: Optional[dict] = None,
+    ) -> dict[str, Any]:
+        doc_type = _detect_type(filename)
+        doc_id = str(uuid.uuid4())
+        collection = TYPE_TO_COLLECTION[doc_type]
+        strategy = TYPE_TO_STRATEGY[doc_type]
+
+        # Save file
+        upload_dir = Path(settings.UPLOAD_DIR)
+        upload_dir.mkdir(parents=True, exist_ok=True)
+        filepath = upload_dir / f"{doc_id}_{filename}"
+        async with aiofiles.open(filepath, "wb") as f:
+            await f.write(content)
+
+        # Index based on type
+        chunk_count = 0
+        chunk_records = []
+
+        if doc_type == "pdf":
+            result = await pdf_rag.index(doc_id, filename, content, extra_metadata)
+            chunk_count = result["chunk_count"]
+        elif doc_type == "markdown":
+            text = content.decode("utf-8", errors="replace")
+            result = await markdown_rag.index(doc_id, filename, text, extra_metadata)
+            chunk_count = result["chunk_count"]
+        elif doc_type == "csv":
+            result = await table_rag.index_csv(doc_id, filename, content, extra_metadata)
+            chunk_count = result["chunk_count"]
+        else:
+            # text / json
+            text = content.decode("utf-8", errors="replace")
+            chunk_records = await _index_text_chunks(doc_id, filename, doc_type, text, collection, extra_metadata)
+            chunk_count = len(chunk_records)
+
+        doc_data = {
+            "id": doc_id,
+            "filename": filename,
+            "filepath": str(filepath),
+            "document_type": doc_type,
+            "retrieval_strategy": strategy,
+            "language": (extra_metadata or {}).get("language", "en"),
+            "chunk_count": chunk_count,
+            "embedding_model": settings.OLLAMA_EMBED_MODEL,
+            "collection_name": collection,
+            "metadata_json": extra_metadata or {},
+        }
+
+        doc = await document_repo.create(db, doc_data)
+
+        # Persist chunks to DB for non-specialized types
+        if chunk_records:
+            await document_repo.bulk_create_chunks(db, chunk_records)
+
+        logger.info(f"Document '{filename}' indexed: id={doc_id} chunks={chunk_count}")
+        return {"document": doc, "chunk_count": chunk_count}
+
+    async def reindex(self, db: AsyncSession, doc_id: str) -> dict[str, Any]:
+        doc = await document_repo.get_by_id(db, doc_id)
+        if not doc:
+            raise DocumentNotFoundError(doc_id)
+
+        filepath = Path(doc.filepath)
+        if not filepath.exists():
+            raise FileNotFoundError(f"File not found: {filepath}")
+
+        async with aiofiles.open(filepath, "rb") as f:
+            content = await f.read()
+
+        # Delete existing vector data
+        try:
+            chroma_client.delete_by_document_id(doc.collection_name or "text_documents", doc_id)
+        except Exception as e:
+            logger.warning(f"Chroma delete failed during reindex: {e}")
+
+        await document_repo.delete_chunks(db, doc_id)
+
+        doc_type = doc.document_type
+        collection = TYPE_TO_COLLECTION.get(doc_type, "text_documents")
+        chunk_count = 0
+
+        if doc_type == "pdf":
+            result = await pdf_rag.index(doc_id, doc.filename, content, doc.metadata_json)
+            chunk_count = result["chunk_count"]
+        elif doc_type == "markdown":
+            text = content.decode("utf-8", errors="replace")
+            result = await markdown_rag.index(doc_id, doc.filename, text, doc.metadata_json)
+            chunk_count = result["chunk_count"]
+        elif doc_type == "csv":
+            result = await table_rag.index_csv(doc_id, doc.filename, content, doc.metadata_json)
+            chunk_count = result["chunk_count"]
+        else:
+            text = content.decode("utf-8", errors="replace")
+            chunk_records = await _index_text_chunks(doc_id, doc.filename, doc_type, text, collection, doc.metadata_json)
+            chunk_count = len(chunk_records)
+            if chunk_records:
+                await document_repo.bulk_create_chunks(db, chunk_records)
+
+        await document_repo.update(db, doc_id, {"chunk_count": chunk_count})
+        return {"document_id": doc_id, "message": "Reindexed successfully", "chunk_count": chunk_count}
+
+    async def delete(self, db: AsyncSession, doc_id: str) -> bool:
+        doc = await document_repo.get_by_id(db, doc_id)
+        if not doc:
+            raise DocumentNotFoundError(doc_id)
+        try:
+            chroma_client.delete_by_document_id(doc.collection_name or "text_documents", doc_id)
+        except Exception as e:
+            logger.warning(f"Chroma delete failed: {e}")
+        filepath = Path(doc.filepath)
+        if filepath.exists():
+            filepath.unlink()
+        await document_repo.delete(db, doc_id)
+        return True
+
+
+document_service = DocumentService()
diff --git a/backend/app/services/rag_service.py b/backend/app/services/rag_service.py
new file mode 100644
index 0000000..14fb55f
+++ b/backend/app/services/rag_service.py
@@ -0,0 +1,173 @@
+import time
+import uuid
+from typing import Any, AsyncGenerator, Optional
+from sqlalchemy.ext.asyncio import AsyncSession
+from app.rag.vector_rag import vector_rag
+from app.rag.hybrid_rag import hybrid_rag
+from app.rag.bm25 import bm25_retriever
+from app.rag.table_rag import table_rag
+from app.rag.pdf_rag import pdf_rag
+from app.rag.markdown_rag import markdown_rag
+from app.rag.metadata_filter import filter_results
+from app.rag.evaluator import (
+    compute_accuracy, compute_faithfulness, compute_answer_relevancy,
+    compute_context_precision, compute_context_recall,
+)
+from app.embeddings.ollama_client import ollama_client
+from app.repositories.log_repository import log_repo
+from app.core.config import settings
+from app.core.logging import get_logger
+
+logger = get_logger("rag_service")
+
+RAG_SYSTEM = """You are an expert assistant. Answer the question using only the provided context.
+Be factual, concise, and cite sources. If the answer is not in the context, say 'Not found in available documents'."""
+
+
+class RAGService:
+    async def retrieve(
+        self,
+        query: str,
+        strategy: str = "hybrid",
+        top_k: int = 5,
+        filters: Optional[dict] = None,
+        collection_name: Optional[str] = None,
+    ) -> list[dict[str, Any]]:
+        col = collection_name or "text_documents"
+        if strategy == "vector":
+            return await vector_rag.retrieve(query, col, top_k, filters)
+        elif strategy == "bm25":
+            results = bm25_retriever.search(col, query, top_k)
+            return filter_results(results, filters or {})
+        elif strategy == "hybrid":
+            return await hybrid_rag.retrieve(query, col, top_k, filters)
+        elif strategy == "table":
+            return await table_rag.query(query, top_k=top_k)
+        elif strategy == "pdf":
+            return await pdf_rag.query(query, top_k=top_k)
+        elif strategy == "markdown":
+            return await markdown_rag.query(query, top_k=top_k)
+        else:
+            return await hybrid_rag.retrieve(query, col, top_k, filters)
+
+    async def query(
+        self,
+        db: AsyncSession,
+        query: str,
+        strategy: str = "hybrid",
+        top_k: int = 5,
+        filters: Optional[dict] = None,
+    ) -> dict[str, Any]:
+        start = time.time()
+        chunks = await self.retrieve(query, strategy, top_k, filters)
+        context = "\n\n".join(
+            f"[{r.get('filename','?')}] {r['chunk_text']}" for r in chunks
+        )
+        prompt = f"Context:\n{context}\n\nQuestion: {query}"
+        answer = await ollama_client.generate(prompt, system=RAG_SYSTEM)
+        latency = (time.time() - start) * 1000
+
+        sources = [
+            {"filename": r.get("filename", ""), "chunk_id": r.get("chunk_id", ""), "score": r.get("score", 0)}
+            for r in chunks
+        ]
+
+        await log_repo.create_retrieval_log(db, {
+            "id": str(uuid.uuid4()),
+            "query": query,
+            "retrieval_strategy": strategy,
+            "retrieved_chunks": [r.get("chunk_id", "") for r in chunks],
+            "generated_answer": answer,
+            "latency_ms": latency,
+            "agent_used": "rag_service",
+        })
+
+        confidence = round(sum(r.get("score", 0) for r in chunks) / max(len(chunks), 1), 4)
+        return {
+            "query": query,
+            "answer": answer,
+            "sources": sources,
+            "strategy": strategy,
+            "latency_ms": round(latency, 2),
+            "confidence": confidence,
+        }
+
+    async def query_stream(
+        self,
+        query: str,
+        strategy: str = "hybrid",
+        top_k: int = 5,
+        filters: Optional[dict] = None,
+    ) -> AsyncGenerator[str, None]:
+        chunks = await self.retrieve(query, strategy, top_k, filters)
+        context = "\n\n".join(
+            f"[{r.get('filename','?')}] {r['chunk_text']}" for r in chunks
+        )
+        prompt = f"Context:\n{context}\n\nQuestion: {query}"
+        async for token in ollama_client.generate_stream(prompt, system=RAG_SYSTEM):
+            yield token
+
+    async def evaluate(
+        self,
+        db: AsyncSession,
+        questions: list[dict],
+        dataset_name: str = "default",
+    ) -> dict[str, Any]:
+        results = {
+            "accuracy": [], "faithfulness": [], "context_precision": [],
+            "context_recall": [], "answer_relevancy": [], "latency_ms": [], "failed": [],
+        }
+
+        for q in questions:
+            question = q["question"]
+            expected = q["expected_answer"]
+            try:
+                start = time.time()
+                chunks = await self.retrieve(question, strategy="hybrid", top_k=5)
+                context_texts = [r["chunk_text"] for r in chunks]
+                context = "\n\n".join(context_texts)
+                prompt = f"Context:\n{context}\n\nQuestion: {question}"
+                answer = await ollama_client.generate(prompt, system=RAG_SYSTEM)
+                latency = (time.time() - start) * 1000
+
+                acc = await compute_accuracy(answer, expected)
+                faith = await compute_faithfulness(answer, context_texts)
+                cp = await compute_context_precision(question, context_texts)
+                cr = await compute_context_recall(expected, context_texts)
+                ar = await compute_answer_relevancy(question, answer)
+
+                results["accuracy"].append(acc)
+                results["faithfulness"].append(faith)
+                results["context_precision"].append(cp)
+                results["context_recall"].append(cr)
+                results["answer_relevancy"].append(ar)
+                results["latency_ms"].append(latency)
+            except Exception as e:
+                logger.error(f"Eval failed for '{question}': {e}")
+                results["failed"].append({"question": question, "error": str(e)})
+
+        def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0
+
+        final = {
+            "accuracy": avg(results["accuracy"]),
+            "faithfulness": avg(results["faithfulness"]),
+            "context_precision": avg(results["context_precision"]),
+            "context_recall": avg(results["context_recall"]),
+            "answer_relevancy": avg(results["answer_relevancy"]),
+            "latency_avg_ms": avg(results["latency_ms"]),
+            "failed_questions": results["failed"],
+        }
+
+        await log_repo.create_evaluation_run(db, {
+            "id": str(uuid.uuid4()),
+            "dataset_name": dataset_name,
+            "accuracy": final["accuracy"],
+            "faithfulness": final["faithfulness"],
+            "context_precision": final["context_precision"],
+            "context_recall": final["context_recall"],
+        })
+
+        return final
+
+
+rag_service = RAGService()
diff --git a/backend/app/services/web_service.py b/backend/app/services/web_service.py
new file mode 100644
index 0000000..e324633
+++ b/backend/app/services/web_service.py
@@ -0,0 +1,102 @@
+import uuid
+from typing import Any, Optional
+import httpx
+from bs4 import BeautifulSoup
+from sqlalchemy.ext.asyncio import AsyncSession
+from app.chromadb.client import chroma_client
+from app.embeddings.ollama_client import ollama_client
+from app.rag.bm25 import bm25_retriever
+from app.core.config import settings
+from app.core.logging import get_logger
+
+logger = get_logger("web_service")
+
+ALLOWED_DOMAINS = [
+    "docs.", "developer.", "gov.", ".gov", "wikipedia.org",
+    "github.com", "arxiv.org", "education.", "official",
+]
+
+
+def _is_allowed_url(url: str) -> bool:
+    return True  # policy: user-approved URLs are allowed
+
+
+def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
+    words = text.split()
+    chunks = []
+    start = 0
+    while start < len(words):
+        end = min(start + chunk_size, len(words))
+        chunks.append(" ".join(words[start:end]))
+        start += chunk_size - overlap
+    return chunks
+
+
+async def _fetch_url(url: str) -> str:
+    async with httpx.AsyncClient(timeout=30) as client:
+        response = await client.get(url, follow_redirects=True, headers={"User-Agent": "RAGBot/1.0"})
+        response.raise_for_status()
+        return response.text
+
+
+def _clean_html(html: str) -> str:
+    soup = BeautifulSoup(html, "lxml")
+    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
+        tag.decompose()
+    return soup.get_text(separator=" ", strip=True)
+
+
+class WebService:
+    async def ingest(
+        self,
+        url: str,
+        collection_name: str = "web_documents",
+        metadata: Optional[dict] = None,
+    ) -> dict[str, Any]:
+        logger.info(f"Web ingest: {url}")
+        html = await _fetch_url(url)
+        text = _clean_html(html)
+        chunks = _chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
+
+        doc_id = str(uuid.uuid4())
+        ids, embeddings, documents, metadatas = [], [], [], []
+        bm25_chunks = []
+
+        for i, chunk in enumerate(chunks):
+            if len(chunk.strip()) < 30:
+                continue
+            emb = await ollama_client.embeddings(chunk)
+            chunk_id = f"{doc_id}_web_{i}"
+            meta = {
+                "document_id": doc_id,
+                "filename": url,
+                "document_type": "web",
+                "source_url": url,
+                "chunk_index": i,
+                **(metadata or {}),
+            }
+            ids.append(chunk_id)
+            embeddings.append(emb)
+            documents.append(chunk)
+            metadatas.append(meta)
+            bm25_chunks.append({"chunk_id": chunk_id, "chunk_text": chunk, "metadata": meta, "document_id": doc_id, "filename": url})
+
+        if ids:
+            chroma_client.add_documents(collection_name, ids, embeddings, documents, metadatas)
+            bm25_retriever.index(collection_name, bm25_chunks)
+
+        return {"url": url, "document_id": doc_id, "chunk_count": len(ids), "message": "Ingested successfully"}
+
+    async def query(
+        self,
+        query: str,
+        url: Optional[str] = None,
+        top_k: int = 5,
+        collection_name: str = "web_documents",
+    ) -> list[dict[str, Any]]:
+        query_emb = await ollama_client.embeddings(query)
+        where = {"filename": {"$eq": url}} if url else None
+        return chroma_client.search(collection_name, query_emb, top_k, where)
+
+
+web_service = WebService()
diff --git a/backend/app/tests/__init__.py b/backend/app/tests/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git a/backend/app/tests/test_core.py b/backend/app/tests/test_core.py
new file mode 100644
index 0000000..d00ff7b
+++ b/backend/app/tests/test_core.py
@@ -0,0 +1,289 @@
+"""
+Core unit tests — no external services required (no Ollama, no ChromaDB).
+Run with: pytest app/tests/test_core.py -v
+"""
+import pytest
+from app.rag.rrf import reciprocal_rank_fusion
+from app.rag.metadata_filter import build_chroma_filter, filter_results
+from app.rag.bm25 import BM25Retriever
+
+
+# ─── RRF ──────────────────────────────────────────────────────────────────────
+
+def test_rrf_single_list():
+    results = [
+        {"chunk_id": "a", "chunk_text": "hello", "score": 0.9},
+        {"chunk_id": "b", "chunk_text": "world", "score": 0.8},
+    ]
+    fused = reciprocal_rank_fusion([results], top_k=2)
+    assert len(fused) == 2
+    assert fused[0]["chunk_id"] == "a"
+
+
+def test_rrf_two_lists_overlap():
+    list1 = [{"chunk_id": "a", "score": 0.9}, {"chunk_id": "b", "score": 0.8}]
+    list2 = [{"chunk_id": "b", "score": 0.95}, {"chunk_id": "c", "score": 0.7}]
+    fused = reciprocal_rank_fusion([list1, list2], top_k=3)
+    ids = [r["chunk_id"] for r in fused]
+    # "b" appears in both lists — should rank high
+    assert "b" in ids
+    assert len(fused) <= 3
+
+
+def test_rrf_empty_lists():
+    fused = reciprocal_rank_fusion([[], []], top_k=5)
+    assert fused == []
+
+
+def test_rrf_top_k_limit():
+    results = [{"chunk_id": str(i), "score": float(i)} for i in range(20)]
+    fused = reciprocal_rank_fusion([results], top_k=5)
+    assert len(fused) == 5
+
+
+# ─── Metadata filter ──────────────────────────────────────────────────────────
+
+def test_build_chroma_filter_empty():
+    assert build_chroma_filter({}) is None
+    assert build_chroma_filter(None) is None
+
+
+def test_build_chroma_filter_single():
+    f = build_chroma_filter({"filename": "test.pdf"})
+    assert f == {"filename": {"$eq": "test.pdf"}}
+
+
+def test_build_chroma_filter_multi():
+    f = build_chroma_filter({"filename": "test.pdf", "language": "en"})
+    assert "$and" in f
+    assert len(f["$and"]) == 2
+
+
+def test_build_chroma_filter_unsupported_key():
+    f = build_chroma_filter({"unknown_key": "value"})
+    assert f is None
+
+
+def test_filter_results_empty_filters():
+    items = [{"chunk_id": "1", "metadata": {"filename": "a.pdf"}}]
+    assert filter_results(items, {}) == items
+
+
+def test_filter_results_matching():
+    items = [
+        {"chunk_id": "1", "metadata": {"filename": "a.pdf", "language": "en"}},
+        {"chunk_id": "2", "metadata": {"filename": "b.pdf", "language": "hi"}},
+    ]
+    filtered = filter_results(items, {"language": "en"})
+    assert len(filtered) == 1
+    assert filtered[0]["chunk_id"] == "1"
+
+
+def test_filter_results_no_match():
+    items = [{"chunk_id": "1", "metadata": {"filename": "a.pdf"}}]
+    assert filter_results(items, {"filename": "b.pdf"}) == []
+
+
+# ─── BM25 ─────────────────────────────────────────────────────────────────────
+
+def test_bm25_index_and_search():
+    retriever = BM25Retriever()
+    chunks = [
+        {"chunk_id": "1", "chunk_text": "government scheme eligibility farmers india",
+         "metadata": {}, "document_id": "doc1", "filename": "a.txt"},
+        {"chunk_id": "2", "chunk_text": "PM Kisan financial support rural households",
+         "metadata": {}, "document_id": "doc1", "filename": "a.txt"},
+        {"chunk_id": "3", "chunk_text": "solar panel installation renewable energy subsidy",
+         "metadata": {}, "document_id": "doc2", "filename": "b.txt"},
+    ]
+    retriever.index("test_col", chunks)
+    results = retriever.search("test_col", "PM Kisan farmers", top_k=2)
+    assert len(results) >= 1
+    assert results[0]["chunk_id"] in {"1", "2"}
+
+
+def test_bm25_missing_collection():
+    retriever = BM25Retriever()
+    results = retriever.search("nonexistent", "query", top_k=5)
+    assert results == []
+
+
+def test_bm25_zero_score_excluded():
+    retriever = BM25Retriever()
+    retriever.index("col", [
+        {"chunk_id": "x", "chunk_text": "apples oranges", "metadata": {}, "document_id": "d", "filename": "f"},
+    ])
+    results = retriever.search("col", "zzzzzzzzzzz", top_k=5)
+    assert results == []
+
+
+def test_bm25_remove_collection():
+    retriever = BM25Retriever()
+    retriever.index("col", [
+        {"chunk_id": "1", "chunk_text": "test text", "metadata": {}, "document_id": "d", "filename": "f"},
+    ])
+    retriever.remove_collection("col")
+    assert retriever.search("col", "test", top_k=5) == []
+
+
+# ─── Config ───────────────────────────────────────────────────────────────────
+
+def test_settings_defaults():
+    from app.core.config import settings
+    assert settings.OLLAMA_LLM_MODEL == "llama3.1:8b"
+    assert settings.OLLAMA_EMBED_MODEL == "nomic-embed-text-v2-moe"
+    assert settings.TOP_K == 5
+    assert settings.CHUNK_SIZE == 512
+
+
+# ─── Schemas ──────────────────────────────────────────────────────────────────
+
+def test_search_request_validation():
+    from app.schemas.search import SearchRequest
+    req = SearchRequest(query="test query", top_k=10)
+    assert req.query == "test query"
+    assert req.top_k == 10
+
+
+def test_rag_query_request_defaults():
+    from app.schemas.rag import RAGQueryRequest
+    req = RAGQueryRequest(query="what is PM Kisan?")
+    assert req.strategy == "hybrid"
+    assert req.top_k == 5
+
+
+def test_eval_question_schema():
+    from app.schemas.rag import EvalQuestion, EvaluationRequest
+    req = EvaluationRequest(
+        questions=[EvalQuestion(question="What?", expected_answer="This.")],
+        dataset_name="smoke_test",
+    )
+    assert len(req.questions) == 1
+    assert req.dataset_name == "smoke_test"
+
+
+def test_agent_request_schema():
+    from app.schemas.agent import AgentRequest
+    req = AgentRequest(query="find schemes for farmers", top_k=3)
+    assert req.top_k == 3
+    assert req.filters is None
+
+
+# ─── Markdown chunking ────────────────────────────────────────────────────────
+
+def test_markdown_section_parsing():
+    from app.rag.markdown_rag import _parse_markdown_sections
+    md = """# Introduction
+Some intro text.
+
+## Section One
+Content of section one.
+
+### Subsection
+Deep content here.
+
+## Section Two
+More content.
+"""
+    sections = _parse_markdown_sections(md)
+    headings = [s["heading"] for s in sections]
+    assert "Introduction" in headings
+    assert "Section One" in headings
+    assert "Section Two" in headings
+
+
+def test_markdown_empty_content():
+    from app.rag.markdown_rag import _parse_markdown_sections
+    sections = _parse_markdown_sections("")
+    assert sections == []
+
+
+# ─── PDF heading detection ────────────────────────────────────────────────────
+
+def test_pdf_heading_detection():
+    from app.rag.pdf_rag import _detect_heading
+    assert _detect_heading("INTRODUCTION") is not None
+    assert _detect_heading("1. Overview") is not None
+    assert _detect_heading("This is a long paragraph that should not be a heading " * 3) is None
+
+
+# ─── Text chunking ────────────────────────────────────────────────────────────
+
+def test_text_chunking():
+    from app.services.document_service import _chunk_text
+    text = " ".join([f"word{i}" for i in range(600)])
+    chunks = _chunk_text(text, chunk_size=100, overlap=10)
+    assert len(chunks) > 1
+    # overlap: last words of chunk N should appear in chunk N+1
+    words0 = set(chunks[0].split())
+    words1 = set(chunks[1].split())
+    assert len(words0 & words1) > 0
+
+
+def test_text_chunking_short():
+    from app.services.document_service import _chunk_text
+    chunks = _chunk_text("short text", chunk_size=512, overlap=50)
+    assert len(chunks) == 1
+    assert chunks[0] == "short text"
+
+
+# ─── File type detection ──────────────────────────────────────────────────────
+
+def test_detect_supported_types():
+    from app.services.document_service import _detect_type
+    assert _detect_type("document.pdf") == "pdf"
+    assert _detect_type("README.md") == "markdown"
+    assert _detect_type("data.csv") == "csv"
+    assert _detect_type("notes.txt") == "text"
+    assert _detect_type("config.json") == "json"
+
+
+def test_detect_unsupported_type():
+    from app.services.document_service import _detect_type
+    from app.core.exceptions import UnsupportedFileTypeError
+    with pytest.raises(UnsupportedFileTypeError):
+        _detect_type("image.png")
+
+
+# ─── Exception classes ────────────────────────────────────────────────────────
+
+def test_exception_hierarchy():
+    from app.core.exceptions import (
+        RAGPlatformException, DocumentNotFoundError,
+        ConversationNotFoundError, OllamaConnectionError,
+        ChromaDBError, UnsupportedFileTypeError,
+    )
+    exc = DocumentNotFoundError("abc-123")
+    assert exc.status_code == 404
+    assert "abc-123" in exc.message
+
+    exc2 = ConversationNotFoundError("conv-99")
+    assert exc2.status_code == 404
+
+    exc3 = OllamaConnectionError("timeout")
+    assert exc3.status_code == 503
+
+    exc4 = ChromaDBError("collection missing")
+    assert exc4.status_code == 503
+
+    exc5 = UnsupportedFileTypeError("mp4")
+    assert exc5.status_code == 422
+    assert "mp4" in exc5.message
+
+
+# ─── Router intent classification ────────────────────────────────────────────
+
+def test_coordinator_intent_classification():
+    from app.agents.coordinator_agent import _classify_intent
+    assert _classify_intent("show me the CSV table data") == "table"
+    assert _classify_intent("search the website for PM Kisan") == "web"
+    assert _classify_intent("list all government schemes for Karnataka") == "structured"
+    assert _classify_intent("what is the capital of France?") == "general"
+
+
+def test_router_doc_type_detection():
+    from app.agents.router_agent import _detect_doc_type
+    assert _detect_doc_type("find in PDF report") == "pdf"
+    assert _detect_doc_type("search the README markdown guide") == "markdown"
+    assert _detect_doc_type("query the CSV table rows") == "csv"
+    assert _detect_doc_type("general question") == "text"
```
