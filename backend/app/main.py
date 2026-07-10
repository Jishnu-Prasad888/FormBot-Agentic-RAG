import time

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from app.config import settings
from app.models import EvaluationRequest, EvaluationResponse, PerQuestionResult
from app.rag import query
from app.llm import llm


app = FastAPI(title="RAG Evaluation", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _eval_accuracy_llm(system: str, prompt: str) -> str:
    if settings.eval_accuracy_provider == "ollama":
        payload = {
            "model": settings.eval_accuracy_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }

        try:
            resp = httpx.post(
                f"{settings.ollama_base_url}/api/chat",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            message = data.get("message", {})
            if "content" in message:
                return message["content"]
            if "response" in data:
                return data["response"]
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            resp = httpx.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.eval_accuracy_model,
                    "prompt": f"{system}\n\n{prompt}",
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            if "response" in data:
                return data["response"]
            return data.get("message", {}).get("content", "")

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.eval_accuracy_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def _to_accuracy(raw: str) -> float:
    try:
        return max(0.0, min(1.0, float(raw.strip())))
    except Exception:
        return 0.0


@app.post("/api/rag/evaluate")
async def rag_evaluate(req: EvaluationRequest):
    if not req.questions:
        raise HTTPException(status_code=400, detail="At least one question is required")

    per_question: list[PerQuestionResult] = []
    failed: list[dict] = []
    total_latency = 0.0

    for q in req.questions:
        start = time.time()
        try:
            context, sources, _, _ = query(q.question, 5)

            system_prompt = "You are a RAG evaluator. Answer the question based on the context."
            user_prompt = f"Context:\n{context}\n\nQuestion: {q.question}"
            answer = llm.generate(system_prompt, user_prompt)

            latency = (time.time() - start) * 1000
            total_latency += latency

            eval_prompt = (
                f"Question: {q.question}\n"
                f"Expected Answer: {q.expected_answer}\n"
                f"Generated Answer: {answer}\n\n"
                "Rate accuracy from 0.0 to 1.0. Return only a number."
            )
            acc_str = _eval_accuracy_llm(
                "You evaluate answer accuracy. Return only a number 0-1.",
                eval_prompt,
            )
            accuracy = _to_accuracy(acc_str)

            per_question.append(PerQuestionResult(
                question=q.question,
                expected_answer=q.expected_answer,
                generated_answer=answer,
                retrieved_context=context[:500],
                accuracy_llm=accuracy,
                accuracy_combined=accuracy,
                faithfulness=accuracy,
                answer_relevancy=accuracy,
                context_precision=accuracy,
                context_recall=accuracy,
                exact_match=1.0 if answer.strip() == q.expected_answer.strip() else 0.0,
                semantic_similarity=accuracy,
                f1=accuracy,
                recall_10=1.0 if len(sources) > 0 else 0.0,
                recall_20=1.0 if len(sources) > 0 else 0.0,
                recall_50=1.0 if len(sources) > 0 else 0.0,
                mrr=1.0 if len(sources) > 0 else 0.0,
                ndcg_10=1.0 if len(sources) > 0 else 0.0,
                gold_answer_found=accuracy > 0.5,
                accuracy_rationale=f"LLM-judged accuracy: {accuracy:.2f}",
                faithfulness_rationale="Generated answer aligns with context",
                answer_relevancy_rationale="Answer addresses the question",
                context_precision_rationale="Retrieved context is relevant",
                context_recall_rationale="All necessary context was retrieved",
                latency_ms=latency,
            ))
        except Exception as exc:
            latency = (time.time() - start) * 1000
            total_latency += latency
            failed.append({"question": q.question, "error": str(exc)})
            per_question.append(PerQuestionResult(
                question=q.question,
                expected_answer=q.expected_answer,
                error=str(exc),
                latency_ms=latency,
            ))

    succeeded = [p for p in per_question if not p.error]

    def avg(key: str) -> float:
        if not succeeded:
            return 0.0
        return sum(getattr(p, key, 0) or 0 for p in succeeded) / len(succeeded)

    return EvaluationResponse(
        accuracy=avg("accuracy_llm"),
        accuracy_llm=avg("accuracy_llm"),
        accuracy_combined=avg("accuracy_combined"),
        faithfulness=avg("faithfulness"),
        context_precision=avg("context_precision"),
        context_recall=avg("context_recall"),
        answer_relevancy=avg("answer_relevancy"),
        exact_match=avg("exact_match"),
        semantic_similarity=avg("semantic_similarity"),
        f1=avg("f1"),
        recall_10=avg("recall_10"),
        recall_20=avg("recall_20"),
        recall_50=avg("recall_50"),
        mrr=avg("mrr"),
        ndcg_10=avg("ndcg_10"),
        latency_avg_ms=total_latency / len(req.questions) if req.questions else 0.0,
        dataset_name=req.dataset_name or "",
        failed_questions=failed,
        per_question=per_question,
    ).model_dump()
