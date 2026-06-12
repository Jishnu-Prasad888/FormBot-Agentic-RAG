from typing import Optional
from elasticsearch import AsyncElasticsearch
from app.embeddings.openai_client import openai_client

ES_HOST = "http://localhost:9200"
INDEX_NAME = "rag_documents"

class ElasticsearchService:
    def __init__(self):
        self.client: Optional[AsyncElasticsearch] = None
        
    async def init(self):
        if not self.client:
            self.client = AsyncElasticsearch([ES_HOST])
            await self._create_index()
    
    async def close(self):
        if self.client:
            await self.client.close()
    
    async def _create_index(self):
        if not await self.client.indices.exists(index=INDEX_NAME):
            await self.client.indices.create(
                index=INDEX_NAME,
                body={
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "metadata": {"type": "object"}
                        }
                    }
                }
            )
    
    async def check_status(self) -> dict:
        try:
            if not self.client:
                await self.init()
            health = await self.client.cluster.health()
            count = await self.client.count(index=INDEX_NAME)
            return {"status": health["status"], "online": True, "doc_count": count["count"]}
        except Exception as e:
            return {"status": "offline", "online": False, "error": str(e)}
    
    async def index_document(self, doc_id: str, content: str, metadata: dict = None):
        await self.init()
        await self.client.index(index=INDEX_NAME, id=doc_id, body={
            "content": content,
            "metadata": metadata or {}
        })
    
    async def bulk_index(self, documents: list[dict]):
        await self.init()
        actions = []
        for doc in documents:
            actions.append({"index": {"_index": INDEX_NAME, "_id": doc["id"]}})
            actions.append({"content": doc["content"], "metadata": doc.get("metadata", {})})
        if actions:
            await self.client.bulk(operations=actions)
    
    async def search(self, query: str, size: int = 10) -> list[dict]:
        await self.init()
        result = await self.client.search(
            index=INDEX_NAME,
            body={"query": {"match": {"content": query}}, "size": size}
        )
        return [{"id": hit["_id"], "content": hit["_source"]["content"], 
                 "score": hit["_score"], "metadata": hit["_source"].get("metadata", {})}
                for hit in result["hits"]["hits"]]
    
    async def enhance_with_iterative_query(self, chunks: list[str], query: str, max_tries: int = 5, logger=None) -> list[str]:
        """Enhance nearest neighbor chunks by iteratively asking ES for missing info"""
        enhanced = chunks.copy()
        
        try:
            for attempt in range(max_tries):
                if logger:
                    logger.log(f"ES_ITERATION_{attempt+1}", f"Current chunks: {len(enhanced)}")
                
                prompt = f"Based on these chunks:\n{chr(10).join(enhanced)}\n\nWhat specific question should I ask to get more relevant info for: {query}?"
                llm_question = await openai_client.generate(prompt, system="Generate a concise search question.")
                
                if logger:
                    logger.log(f"ES_LLM_QUESTION_{attempt+1}", f"Generated question: {llm_question}")
                
                es_results = await self.search(llm_question, size=3)
                if logger:
                    logger.log(f"ES_SEARCH_RESULTS_{attempt+1}", f"Found {len(es_results)} results")
                
                if not es_results:
                    break
                
                new_content = [r["content"] for r in es_results if r["content"] not in enhanced]
                if not new_content:
                    break
                    
                enhanced.extend(new_content[:2])
        except Exception as e:
            if logger:
                logger.log("ES_ENHANCEMENT_ERROR", str(e))
        
        return enhanced[:len(chunks) + 5]

es_service = ElasticsearchService()
