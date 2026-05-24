from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db
from app.agents.coordinator_agent import coordinator_agent
from app.agents.vector_agent import vector_agent
from app.agents.sqlite_agent import sqlite_agent
from app.agents.router_agent import router_agent
from app.agents.web_agent import web_agent
from app.agents.evaluator_agent import evaluator_agent
from app.schemas.agent import AgentRequest, AgentResponse, CoordinatorRequest
from app.core.logging import get_logger

router = APIRouter(prefix="/api/agents", tags=["Agents"])
logger = get_logger("api.agents")


def _to_response(result: dict) -> AgentResponse:
    return AgentResponse(
        agent=result.get("agent", ""),
        query=result.get("query", ""),
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        reasoning=result.get("intent") or result.get("strategy"),
        latency_ms=result.get("latency_ms", 0.0),
        metadata={
            k: v for k, v in result.items()
            if k not in {"agent", "query", "answer", "sources", "chunks", "latency_ms"}
        },
    )


@router.post("/coordinator", response_model=AgentResponse)
async def coordinator(req: CoordinatorRequest):
    result = await coordinator_agent.run(req.query, {"top_k": req.top_k})
    return _to_response(result)


@router.post("/vector", response_model=AgentResponse)
async def vector(req: AgentRequest):
    result = await vector_agent.run(req.query, {
        "top_k": req.top_k,
        "filters": req.filters,
        **(req.context or {}),
    })
    return _to_response(result)


@router.post("/sqlite", response_model=AgentResponse)
async def sqlite(req: AgentRequest):
    result = await sqlite_agent.run(req.query, {
        "top_k": req.top_k,
        **(req.context or {}),
    })
    return _to_response(result)


@router.post("/router", response_model=AgentResponse)
async def document_router(req: AgentRequest):
    result = await router_agent.run(req.query, {
        "top_k": req.top_k,
        **(req.context or {}),
    })
    return _to_response(result)


@router.post("/web", response_model=AgentResponse)
async def web(req: AgentRequest):
    result = await web_agent.run(req.query, {
        "top_k": req.top_k,
        **(req.context or {}),
    })
    return _to_response(result)


@router.post("/evaluator", response_model=AgentResponse)
async def evaluator(req: AgentRequest):
    ctx = req.context or {}
    result = await evaluator_agent.run(req.query, ctx)
    return _to_response(result)
