from fastapi import APIRouter, UploadFile, File
from app.services.elasticsearch_service import es_service

router = APIRouter(prefix="/api/elasticsearch", tags=["elasticsearch"])

@router.get("/status")
async def get_status():
    return await es_service.check_status()

@router.post("/upload")
async def upload_documents(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    docs = [{"id": f"doc_{i}", "content": line} for i, line in enumerate(lines)]
    await es_service.bulk_index(docs)
    return {"message": f"Indexed {len(docs)} documents", "count": len(docs)}

@router.post("/search")
async def search_documents(query: str, size: int = 10):
    results = await es_service.search(query, size)
    return {"query": query, "results": results, "count": len(results)}
