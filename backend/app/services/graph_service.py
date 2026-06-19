from typing import Optional

from app.graph.graph_retriever import GraphResult, retrieve_candidates


class GraphService:
    async def get_candidates(self, query: str, filters: Optional[dict] = None) -> GraphResult:
        return await retrieve_candidates(query, filters or {})


graph_service = GraphService()
