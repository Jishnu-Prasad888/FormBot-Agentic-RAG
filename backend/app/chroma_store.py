import os
import uuid
import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError, InvalidCollectionException

os.environ.setdefault("CHROMADB_DISABLE_TELEMETRY", "1")

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma")
COLLECTION_NAME = "documents"

_client = None
_collection = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_collection():
    global _collection
    if _collection is None:
        client = _get_client()
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except (ValueError, NotFoundError, InvalidCollectionException):
            _collection = client.create_collection(COLLECTION_NAME)
    return _collection


def add_chunks(doc_id: str, filename: str, chunks: list[str],
               embeddings: list[list[float]], metadata_list: list[dict] = None):
    collection = _get_collection()
    ids = []
    metadatas = []
    for i, (text, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = str(uuid.uuid4())
        ids.append(chunk_id)
        meta = {
            "document_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "chunk_text": text,
        }
        if metadata_list and i < len(metadata_list):
            meta.update(metadata_list[i])
        metadatas.append(meta)
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)


def search(query_emb: list[float], top_k: int = 5) -> list[dict]:
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["metadatas", "distances"],
    )
    output = []
    if not results["ids"] or not results["ids"][0]:
        return output
    for i, chunk_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        score = 1.0 - distance
        output.append({
            "chunk_id": chunk_id,
            "document_id": meta.get("document_id", ""),
            "filename": meta.get("filename", "unknown"),
            "chunk_text": meta.get("chunk_text", ""),
            "score": score,
            "metadata": {k: v for k, v in meta.items() if k not in ("document_id", "filename", "chunk_text")},
        })
    return output


def get_chunks_by_doc(doc_id: str) -> list[dict]:
    collection = _get_collection()
    results = collection.get(
        where={"document_id": doc_id},
        include=["metadatas"],
    )
    output = []
    for i, chunk_id in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        output.append({
            "id": chunk_id,
            "document_id": meta.get("document_id", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "chunk_text": meta.get("chunk_text", ""),
            "chunk_metadata": {k: v for k, v in meta.items() if k not in ("document_id", "filename", "chunk_text", "chunk_index")},
            "created_at": "",
        })
    output.sort(key=lambda x: x["chunk_index"])
    return output


def delete_chunks_by_doc(doc_id: str):
    collection = _get_collection()
    results = collection.get(where={"document_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])


def count_chunks() -> int:
    collection = _get_collection()
    return collection.count()


def list_collections_info() -> list[dict]:
    client = _get_client()
    cols = client.list_collections()
    return [{"name": c.name, "count": c.count()} for c in cols]


def reset_collection():
    global _collection
    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except (ValueError, NotFoundError):
        pass
    _collection = None
