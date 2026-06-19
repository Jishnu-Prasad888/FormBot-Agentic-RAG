import logging
from typing import Any, Optional

from app.core.config import settings

try:
    from neo4j import AsyncGraphDatabase  # type: ignore
except ImportError:  # pragma: no cover - neo4j optional at runtime
    AsyncGraphDatabase = None


logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self) -> None:
        self._driver = None
        self.enabled = settings.USE_KG_RETRIEVAL and AsyncGraphDatabase is not None

    async def _get_driver(self):
        if not self.enabled:
            return None
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )
        return self._driver

    async def close(self):
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def run_read(self, cypher: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        driver = await self._get_driver()
        if not driver:
            return []
        async with driver.session() as session:
            result = await session.run(cypher, params or {})
            records = await result.to_list()
            return [r.data() for r in records]

    async def run_write(self, cypher: str, params: Optional[dict[str, Any]] = None) -> None:
        driver = await self._get_driver()
        if not driver:
            return
        async with driver.session() as session:
            await session.run(cypher, params or {})

    async def health_check(self) -> bool:
        try:
            rows = await self.run_read("RETURN 1 AS ok")
            return bool(rows)
        except Exception as exc:
            logger.warning("Neo4j health check failed: %s", exc)
            return False


neo4j_client = Neo4jClient()
