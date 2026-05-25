from abc import ABC, abstractmethod
from typing import Any, Optional
from app.embeddings.openai_client import openai_client as ollama_client
from app.core.logging import get_logger


class BaseAgent(ABC):
    name: str = "base_agent"

    def __init__(self):
        self.logger = get_logger(f"agent.{self.name}")

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
        self.logger.info(f"[{self.name}] running query: {query[:80]}")
        plan = await self.plan(query, context)
        result = await self.execute(plan)
        evaluated = await self.evaluate(result)
        return evaluated