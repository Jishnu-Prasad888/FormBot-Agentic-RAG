import time
from typing import Any, Optional
from app.agents.base import BaseAgent
from app.agents.vector_agent import vector_agent
from app.agents.sqlite_agent import sqlite_agent
from app.agents.router_agent import router_agent, _detect_doc_type
from app.agents.web_agent import web_agent
from app.agents.evaluator_agent import evaluator_agent
from app.embeddings.ollama_client import ollama_client
from app.core.logging import get_logger

logger = get_logger("coordinator_agent")

INTENT_KEYWORDS = {
    "table": ["table", "csv", "spreadsheet", "rows", "columns", "sum", "count", "average", "aggregate"],
    "web": ["website", "url", "http", "online", "web", "internet", "search online"],
    "structured": ["scheme", "state", "ministry", "department", "eligibility", "database"],
}


def _classify_intent(query: str) -> str:
    q_lower = query.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return intent
    return "general"


class CoordinatorAgent(BaseAgent):
    name = "coordinator_agent"

    async def plan(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
        ctx = context or {}
        intent = _classify_intent(query)
        doc_type = _detect_doc_type(query)

        # Determine which agents to invoke
        agents_to_run = []
        if intent == "table":
            agents_to_run = ["sqlite"]
        elif intent == "web":
            agents_to_run = ["web", "vector"]
        elif intent == "structured":
            agents_to_run = ["sqlite", "vector"]
        else:
            agents_to_run = ["router", "vector"]

        return {
            "query": query,
            "intent": intent,
            "doc_type": doc_type,
            "agents": agents_to_run,
            "context": ctx,
            "top_k": ctx.get("top_k", 5),
        }

    async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
        start = time.time()
        query = plan["query"]
        ctx = plan["context"]
        agents_list = plan["agents"]
        top_k = plan["top_k"]

        agent_results = []
        all_chunks = []

        for agent_name in agents_list:
            try:
                if agent_name == "sqlite":
                    res = await sqlite_agent.run(query, {**ctx, "top_k": top_k})
                elif agent_name == "vector":
                    res = await vector_agent.run(query, {**ctx, "top_k": top_k})
                elif agent_name == "router":
                    res = await router_agent.run(query, {**ctx, "top_k": top_k})
                elif agent_name == "web":
                    res = await web_agent.run(query, {**ctx, "top_k": top_k})
                else:
                    continue
                agent_results.append(res)
                all_chunks.extend(res.get("chunks", []))
            except Exception as e:
                logger.error(f"Agent '{agent_name}' failed: {e}")

        # Synthesize answers from all agents
        if not agent_results:
            return {
                "agent": self.name,
                "query": query,
                "answer": "No results found.",
                "chunks": [],
                "latency_ms": (time.time() - start) * 1000,
                "agent_results": [],
            }

        # Merge context and generate final answer
        combined_context = "\n\n---\n\n".join(
            f"[{r['agent'].upper()}]:\n{r.get('answer', '')}" for r in agent_results
        )
        synthesis_prompt = (
            f"Multiple agents retrieved the following information:\n\n{combined_context}\n\n"
            f"Based on all above, provide a comprehensive final answer to: {query}"
        )
        system = "You are a coordinator that synthesizes information from multiple sources into a single coherent answer."
        final_answer = await ollama_client.generate(synthesis_prompt, system=system)
        latency = (time.time() - start) * 1000

        return {
            "agent": self.name,
            "query": query,
            "answer": final_answer,
            "chunks": all_chunks,
            "agent_results": agent_results,
            "intent": plan["intent"],
            "latency_ms": round(latency, 2),
        }

    async def evaluate(self, result: dict[str, Any]) -> dict[str, Any]:
        chunks = result.get("chunks", [])
        scores = [c.get("score", 0) for c in chunks if c.get("score")]
        result["confidence"] = round(sum(scores) / max(len(scores), 1), 4) if scores else 0.0
        result["sources"] = list({
            c.get("filename", c.get("metadata", {}).get("filename", "")): None
            for c in chunks
        }.keys())
        return result


coordinator_agent = CoordinatorAgent()
