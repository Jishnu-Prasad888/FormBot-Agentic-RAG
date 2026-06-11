from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Any, Optional

from app.chromadb.client import chroma_client
from app.embeddings.openai_client import openai_client as ollama_client

router = APIRouter(prefix="/api/chroma", tags=["ChromaDB"])


class ChromaIndexRequest(BaseModel):
    collection_name: str
    ids: list[str]
    documents: list[str]
    metadatas: Optional[list[dict[str, Any]]] = None


class ChromaSearchRequest(BaseModel):
    collection_name: str
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[dict[str, Any]] = None


class ChromaDeleteRequest(BaseModel):
    collection_name: str
    document_id: Optional[str] = None


@router.post("/index")
async def index_documents(req: ChromaIndexRequest):
    embeddings = await ollama_client.batch_embeddings(req.documents)
    chroma_client.add_documents(
        req.collection_name, req.ids, embeddings, req.documents, req.metadatas
    )
    return {"message": f"Indexed {len(req.ids)} documents into '{req.collection_name}'"}


@router.post("/reindex")
async def reindex_documents(req: ChromaIndexRequest):
    embeddings = await ollama_client.batch_embeddings(req.documents)
    chroma_client.reindex(
        req.collection_name, req.ids, embeddings, req.documents, req.metadatas
    )
    return {"message": f"Reindexed {len(req.ids)} documents into '{req.collection_name}'"}


@router.delete("/delete")
async def delete_collection(req: ChromaDeleteRequest):
    if req.document_id:
        chroma_client.delete_by_document_id(req.collection_name, req.document_id)
        return {"message": f"Deleted document {req.document_id} from '{req.collection_name}'"}
    chroma_client.delete_collection(req.collection_name)
    return {"message": f"Collection '{req.collection_name}' deleted"}


@router.post("/search")
async def search_collection(req: ChromaSearchRequest):
    from app.rag.metadata_filter import build_chroma_filter
    query_emb = await ollama_client.embeddings(req.query)
    where = build_chroma_filter(req.filters or {})
    results = chroma_client.search(req.collection_name, query_emb, req.top_k, where)
    return {"query": req.query, "collection": req.collection_name, "results": results}


@router.get("/collections")
async def list_collections():
    collections = chroma_client.list_collections()
    counts = {name: chroma_client.get_collection_count(name) for name in collections}
    return {"collections": collections, "counts": counts}