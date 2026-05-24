import time
from typing import Any, Optional
from app.agents.base import BaseAgent
from app.rag.evaluator import (
    compute_faithfulness, compute_answer_relevancy,
    compute_context_precision, compute_context_recall,
)
from app.core.logging import get_logger


class RetrievalEvaluationAgent(BaseAgent):
    name = "evaluator_agent"

    async def plan(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
        ctx = context or {}
        return {
            "query": query,
            "answer": ctx.get("answer", ""),
            "chunks": ctx.get("chunks", []),
            "expected_answer": ctx.get("expected_answer", ""),
        }

    async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
        start = time.time()
        query = plan["query"]
        answer = plan["answer"]
        chunks = plan["chunks"]
        expected = plan.get("expected_answer", "")

        context_texts = [c.get("chunk_text", "") for c in chunks if c.get("chunk_text")]

        faithfulness = await compute_faithfulness(answer, context_texts)
        relevancy = await compute_answer_relevancy(query, answer)
        cp = await compute_context_precision(query, context_texts)
        cr = await compute_context_recall(expected, context_texts) if expected else 0.0
        latency = (time.time() - start) * 1000

        coverage = len([c for c in chunks if c.get("score", 0) > 0.5]) / max(len(chunks), 1)
        retrieval_ok = faithfulness > 0.5 and cp > 0.4

        return {
            "agent": self.name,
            "query": query,
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": cp,
            "context_recall": cr,
            "retrieval_coverage": round(coverage, 4),
            "retrieval_ok": retrieval_ok,
            "latency_ms": round(latency, 2),
        }

    async def evaluate(self, result: dict[str, Any]) -> dict[str, Any]:
        score = (
            result.get("faithfulness", 0) * 0.3
            + result.get("answer_relevancy", 0) * 0.3
            + result.get("context_precision", 0) * 0.2
            + result.get("context_recall", 0) * 0.2
        )
        result["overall_score"] = round(score, 4)
        result["answer"] = f"Evaluation complete. Overall score: {result['overall_score']:.2f}"
        result["sources"] = []
        return result


evaluator_agent = RetrievalEvaluationAgent()
