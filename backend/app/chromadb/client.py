import uuid
from typing import Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import ChromaDBError

logger = get_logger("chromadb_client")

COLLECTIONS = [
    "table_documents",
    "pdf_documents",
    "markdown_documents",
    "text_documents",
    "audio_transcripts",
    "web_documents",
]


class ChromaDBClient:
    def __init__(self):
        self._client: Optional[chromadb.Client] = None

    def get_client(self) -> chromadb.Client:
        if self._client is None:
            try:
                self._client = chromadb.PersistentClient(
                    path=settings.CHROMA_PERSIST_DIR,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                logger.info(f"ChromaDB initialized at {settings.CHROMA_PERSIST_DIR}")
            except Exception as e:
                logger.error(f"ChromaDB init failed: {e}")
                raise ChromaDBError(str(e))
        return self._client

    def create_collection(self, name: str, metadata: Optional[dict] = None) -> chromadb.Collection:
        try:
            client = self.get_client()
            collection = client.get_or_create_collection(
                name=name,
                metadata=metadata or {"hnsw:space": "cosine"},
            )
            logger.info(f"Collection '{name}' ready")
            return collection
        except Exception as e:
            logger.error(f"create_collection failed for '{name}': {e}")
            raise ChromaDBError(str(e))

    def delete_collection(self, name: str) -> bool:
        try:
            client = self.get_client()
            client.delete_collection(name)
            logger.info(f"Collection '{name}' deleted")
            return True
        except Exception as e:
            logger.error(f"delete_collection failed for '{name}': {e}")
            raise ChromaDBError(str(e))

    def add_documents(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> bool:
        try:
            collection = self.create_collection(collection_name)
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas or [{} for _ in ids],
            )
            logger.info(f"Added {len(ids)} docs to '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"add_documents failed: {e}")
            raise ChromaDBError(str(e))

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        try:
            collection = self.create_collection(collection_name)
            kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": min(top_k, collection.count() or 1),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where
            results = collection.query(**kwargs)
            output = []
            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            for i, chunk_id in enumerate(ids):
                score = 1.0 - (distances[i] if distances else 0.0)
                output.append({
                    "chunk_id": chunk_id,
                    "chunk_text": docs[i] if docs else "",
                    "metadata": metas[i] if metas else {},
                    "score": round(score, 4),
                })
            return output
        except Exception as e:
            logger.error(f"search failed in '{collection_name}': {e}")
            raise ChromaDBError(str(e))

    def metadata_filter(
        self,
        collection_name: str,
        query_embedding: list[float],
        filters: dict,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return self.search(collection_name, query_embedding, top_k, where=filters)

    def reindex(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> bool:
        try:
            self.delete_collection(collection_name)
            return self.add_documents(collection_name, ids, embeddings, documents, metadatas)
        except Exception as e:
            logger.error(f"reindex failed: {e}")
            raise ChromaDBError(str(e))

    def list_collections(self) -> list[str]:
        try:
            client = self.get_client()
            return [c.name for c in client.list_collections()]
        except Exception as e:
            logger.error(f"list_collections failed: {e}")
            raise ChromaDBError(str(e))

    def get_collection_count(self, collection_name: str) -> int:
        try:
            collection = self.create_collection(collection_name)
            return collection.count()
        except Exception:
            return 0

    def delete_by_document_id(self, collection_name: str, document_id: str) -> bool:
        try:
            collection = self.create_collection(collection_name)
            results = collection.get(where={"document_id": document_id})
            ids = results.get("ids", [])
            if ids:
                collection.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} chunks for doc {document_id} from '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"delete_by_document_id failed: {e}")
            raise ChromaDBError(str(e))

    def health_check(self) -> bool:
        try:
            self.get_client().heartbeat()
            return True
        except Exception:
            return False

    def init_collections(self):
        for name in COLLECTIONS:
            self.create_collection(name)
        logger.info("All default collections initialized")


chroma_client = ChromaDBClient()
