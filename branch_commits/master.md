# Branch master

Generated on 2026-07-13T07:28:43Z
Total commits: 13

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
