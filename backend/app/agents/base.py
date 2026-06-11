from abc import ABC, abstractmethod
from typing import Any, Optional
from app.embeddings.openai_client import openai_client as ollama_client


class BaseAgent(ABC):
    name: str = "base_agent"

    def __init__(self):
        pass

    @abstractmethod
    async def plan(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
        """Decompose task and create execution plan."""
        ...

    @abstractmethod
    async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Execute the plan and retrieve/generate results."""
        ...

    @abstractmethod
    async def evaluate(self, result: dict[str, Any]) -> dict[str, Any]:
        """Evaluate result quality and completeness."""
        ...

    async def run(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
        plan = await self.plan(query, context)
        result = await self.execute(plan)
        evaluated = await self.evaluate(result)
        return evaluated