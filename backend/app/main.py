import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.models import EvaluationRequest, QueryRequest
from app.rag import query
from app.llm import llm


app = FastAPI(title="RAG Chat and Evaluation", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "openai_models": [settings.openai_model]}


def _format_response(query_text: str, response_text: str, status: int) -> JSONResponse:
    return JSONResponse({"query": query_text, "response": response_text, "status": status}, status_code=status)


def _build_context_prompt(form_entry: str, voice_query: str, scheme_name: str, context: str) -> str:
    scheme = scheme_name or ""
    context_block = f"\n\nContext:\n{context}" if context else ""
    return (
        "You are a helpful assistant for form filling and schemes. "
        "Use the provided context when available."
        f"{context_block}\n\nForm text: {form_entry}\nScheme: {scheme}\nUser query: {voice_query}"
    )


@app.post("/get_llm_response_schemes")
async def get_llm_response_schemes(req: QueryRequest):
    query_text = req.voice_query.strip() if req.voice_query else ""
    if not query_text:
        return _format_response(query_text, "Please enter a valid query.", 400)

    form_entry = req.form_entry or ""
    scheme_name = req.scheme_name or ""

    try:
        context, sources, _, _ = query(query_text, 5)
        prompt = _build_context_prompt(form_entry, query_text, scheme_name, context)
        answer = llm.generate("You are a helpful assistant.", prompt)
        return _format_response(query_text, answer, 200)
    except HTTPException as exc:
        return _format_response(query_text, str(exc.detail), exc.status_code)
    except Exception as exc:  # noqa: BLE001
        return _format_response(query_text, f"Processing Error: {exc}", 500)


@app.post("/api/rag/evaluate")
async def rag_evaluate(req: EvaluationRequest):
    if not req.questions:
        return _format_response("", "At least one question is required", 400)

    q = req.questions[0]
    query_text = q.question.strip() if q.question else ""
    if not query_text:
        return _format_response(query_text, "Question text is required", 400)

    try:
        context, sources, _, _ = query(query_text, 5)
        system_prompt = "You are a RAG evaluator. Answer the question based on the context."
        user_prompt = f"Context:\n{context}\n\nQuestion: {query_text}"
        answer = llm.generate(system_prompt, user_prompt)
        return _format_response(query_text, answer, 200)
    except HTTPException as exc:
        return _format_response(query_text, str(exc.detail), exc.status_code)
    except Exception as exc:  # noqa: BLE001
        return _format_response(query_text, f"Processing Error: {exc}", 500)
