import logging
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings
from app.core.exceptions import ChromaDBError

logger = logging.getLogger(__name__)

COLLECTIONS = [
    "table_documents",
    "pdf_documents",
    "markdown_documents",
    "text_documents",
    "audio_transcripts",
    "web_documents",
    "bank_forms_collection",
    "regulations_collection",
    "guidelines_collection",
]


# ── Qdrant Backend ------------------------------------------------------------
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchAny,
        MatchValue,
        PointStruct,
        VectorParams,
        HnswConfigDiff,
        ScalarQuantization,
        ScalarQuantizationConfig,
    )
except ImportError:
    QdrantClient = None  # type: ignore


class _QdrantBackend:
    def __init__(self):
        if QdrantClient is None:
            raise ImportError("qdrant-client not installed")
        self._client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY or None,
            prefer_grpc=False,
        )

    # Helpers -----------------------------------------------------------------
    def _distance(self):
        dist = (settings.QDRANT_DISTANCE or "Cosine").lower()
        return Distance.COSINE if dist == "cosine" else Distance.DOT

    def _ensure_collection(self, name: str, vector_size: Optional[int] = None) -> None:
        collections = {c.name for c in self._client.get_collections().collections}
        if name not in collections:
            size = vector_size or settings.QDRANT_VECTOR_SIZE
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=size,
                    distance=self._distance(),
                ),
                hnsw_config=HnswConfigDiff(
                    m=settings.QDRANT_HNSW_M,
                    ef_construct=settings.QDRANT_HNSW_EF_CONSTRUCT,
                ),
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(enabled=settings.QDRANT_USE_SCALAR_QUANTIZATION)
                ) if settings.QDRANT_USE_SCALAR_QUANTIZATION else None,
            )

    def _to_filter(self, where: Optional[dict]) -> Optional[Filter]:
        if not where:
            return None

        def _from_clause(clause: dict) -> Optional[Filter]:
            # clause may be {field: {"$eq": val}} or {"$and": [clauses]}
            if "$and" in clause:
                must = []
                for sub in clause.get("$and", []):
                    f = _from_clause(sub)
                    if f:
                        must.extend(f.must or [])
                return Filter(must=must) if must else None

            if len(clause) != 1:
                return None
            field, expr = next(iter(clause.items()))
            if isinstance(expr, dict):
                if "$eq" in expr:
                    return Filter(must=[FieldCondition(key=field, match=MatchValue(value=expr["$eq"]))])
                if "$in" in expr and isinstance(expr["$in"], list):
                    return Filter(must=[FieldCondition(key=field, match=MatchAny(any=expr["$in"]))])
            return None

        return _from_clause(where)

    # CRUD --------------------------------------------------------------------
    def add_documents(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> bool:
        vector_size = len(embeddings[0]) if embeddings else settings.QDRANT_VECTOR_SIZE
        self._ensure_collection(collection_name, vector_size)
        payloads = metadatas or [{} for _ in ids]
        points = [
            PointStruct(id=ids[i], vector=embeddings[i], payload={**payloads[i], "chunk_text": documents[i]})
            for i in range(len(ids))
        ]
        self._client.upsert(collection_name=collection_name, points=points, wait=True)
        return True

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        self._ensure_collection(collection_name)
        flt = self._to_filter(where)
        results = self._client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=flt,
        )
        output: list[dict[str, Any]] = []
        for hit in results:
            payload = hit.payload or {}
            output.append({
                "chunk_id": str(hit.id),
                "chunk_text": payload.get("chunk_text", ""),
                "metadata": {k: v for k, v in payload.items() if k != "chunk_text"},
                "score": round(float(hit.score), 4),
            })
        return output

    def get_client(self):
        return self._client

    def metadata_filter(self, collection_name: str, query_embedding: list[float], filters: dict, top_k: int = 5):
        return self.search(collection_name, query_embedding, top_k, where=filters)

    def delete_by_document_id(self, collection_name: str, document_id: str) -> bool:
        self._ensure_collection(collection_name)
        flt = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        self._client.delete(collection_name=collection_name, filter=flt, wait=True)
        return True

    def delete_collection(self, name: str) -> bool:
        self._client.delete_collection(name)
        return True

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.get_collections().collections]

    def get_collection_count(self, collection_name: str) -> int:
        try:
            self._ensure_collection(collection_name)
            info = self._client.get_collection(collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def reindex(self, collection_name: str, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: Optional[list[dict]] = None) -> bool:
        self.delete_collection(collection_name)
        return self.add_documents(collection_name, ids, embeddings, documents, metadatas)

    def init_collections(self):
        for name in COLLECTIONS:
            try:
                self._ensure_collection(name)
            except Exception as exc:
                logger.warning("Qdrant collection init failed for %s: %s", name, exc)

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False


# ── Chroma Backend (legacy / fallback) --------------------------------------
class _ChromaBackend:
    def __init__(self):
        self._client: Optional[chromadb.Client] = None

    def get_client(self) -> chromadb.Client:
        if self._client is None:
            try:
                self._client = chromadb.PersistentClient(
                    path=settings.CHROMA_PERSIST_DIR,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            except Exception as e:
                raise ChromaDBError(str(e))
        return self._client

    def create_collection(self, name: str, metadata: Optional[dict] = None):
        try:
            client = self.get_client()
            collection = client.get_or_create_collection(
                name=name,
                metadata=metadata or {"hnsw:space": "cosine"},
            )
            return collection
        except Exception as e:
            raise ChromaDBError(str(e))

    def delete_collection(self, name: str) -> bool:
        try:
            client = self.get_client()
            client.delete_collection(name)
            return True
        except Exception as e:
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
            return True
        except Exception as e:
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
            raise ChromaDBError(str(e))

    def list_collections(self) -> list[str]:
        try:
            client = self.get_client()
            return [c.name for c in client.list_collections()]
        except Exception as e:
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
            return True
        except Exception as e:
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


# ── Facade that preserves the old name chroma_client ------------------------
class VectorClient:
    def __init__(self):
        self._backend: Optional[Any] = None
        self.backend_name: str = ""

    def _ensure_backend(self):
        if self._backend is not None:
            return
        if settings.USE_QDRANT:
            try:
                self._backend = _QdrantBackend()
                self.backend_name = "qdrant"
                return
            except Exception as exc:
                logger.warning("Qdrant backend unavailable, falling back to Chroma: %s", exc)
        self._backend = _ChromaBackend()
        self.backend_name = "chroma"

    # Proxy all public methods ------------------------------------------------
    def add_documents(self, *args, **kwargs):
        self._ensure_backend()
        return self._backend.add_documents(*args, **kwargs)

    def search(self, *args, **kwargs):
        self._ensure_backend()
        return self._backend.search(*args, **kwargs)

    def metadata_filter(self, *args, **kwargs):
        self._ensure_backend()
        return self._backend.metadata_filter(*args, **kwargs)

    def reindex(self, *args, **kwargs):
        self._ensure_backend()
        return self._backend.reindex(*args, **kwargs)

    def list_collections(self, *args, **kwargs):
        self._ensure_backend()
        return self._backend.list_collections(*args, **kwargs)

    def get_collection_count(self, *args, **kwargs):
        self._ensure_backend()
        return self._backend.get_collection_count(*args, **kwargs)

    def delete_by_document_id(self, *args, **kwargs):
        self._ensure_backend()
        return self._backend.delete_by_document_id(*args, **kwargs)

    def delete_collection(self, *args, **kwargs):
        self._ensure_backend()
        return self._backend.delete_collection(*args, **kwargs)

    def health_check(self, *args, **kwargs):
        self._ensure_backend()
        return self._backend.health_check(*args, **kwargs)

    def init_collections(self, *args, **kwargs):
        self._ensure_backend()
        return self._backend.init_collections(*args, **kwargs)

    def get_client(self):
        self._ensure_backend()
        if hasattr(self._backend, "get_client"):
            return self._backend.get_client()
        return getattr(self._backend, "_client", None)


# Expose under the legacy name to avoid touching call sites
chroma_client = VectorClient()
