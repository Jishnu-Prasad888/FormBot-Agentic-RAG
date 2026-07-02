import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.models import Document, Conversation, Message, Source
from app.chroma_store import delete_chunks_by_doc

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DOCS_FILE = os.path.join(DATA_DIR, "documents.json")
CONVS_FILE = os.path.join(DATA_DIR, "conversations.json")

os.makedirs(DATA_DIR, exist_ok=True)


def _load_json(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _save_json(path: str, data: list):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


# ─── Documents ────────────────────────────────────────────────────────────────

def list_docs(skip: int = 0, limit: int = 50) -> tuple[list[Document], int]:
    docs = _load_json(DOCS_FILE)
    total = len(docs)
    docs = docs[skip: skip + limit]
    return [Document(**d) for d in docs], total


def get_doc(doc_id: str) -> Optional[Document]:
    docs = _load_json(DOCS_FILE)
    for d in docs:
        if d["id"] == doc_id:
            return Document(**d)
    return None


def create_doc(filename: str, filepath: str, doc_type: str,
               embedding_model: str, metadata: dict = None) -> Document:
    now = _utcnow()
    doc = Document(
        id=_new_id(),
        filename=filename,
        filepath=filepath,
        document_type=doc_type,
        embedding_model=embedding_model,
        metadata_json=metadata or {},
        created_at=now,
        updated_at=now,
    )
    docs = _load_json(DOCS_FILE)
    docs.append(doc.model_dump())
    _save_json(DOCS_FILE, docs)
    return doc


def update_doc(doc_id: str, **kwargs):
    docs = _load_json(DOCS_FILE)
    for i, d in enumerate(docs):
        if d["id"] == doc_id:
            d.update(kwargs)
            d["updated_at"] = _utcnow()
            docs[i] = d
            _save_json(DOCS_FILE, docs)
            return Document(**d)
    return None


def delete_doc(doc_id: str) -> bool:
    docs = _load_json(DOCS_FILE)
    new_docs = [d for d in docs if d["id"] != doc_id]
    if len(new_docs) == len(docs):
        return False
    _save_json(DOCS_FILE, new_docs)
    delete_chunks_by_doc(doc_id)
    return True


# ─── Conversations ────────────────────────────────────────────────────────────

def list_convs(skip: int = 0, limit: int = 50) -> tuple[list[Conversation], int]:
    convs = _load_json(CONVS_FILE)
    total = len(convs)
    convs = convs[skip: skip + limit]
    return [Conversation(**c) for c in convs], total


def get_conv(conv_id: str) -> Optional[Conversation]:
    convs = _load_json(CONVS_FILE)
    for c in convs:
        if c["id"] == conv_id:
            return Conversation(**c)
    return None


def create_conv(title: str = "") -> Conversation:
    now = _utcnow()
    conv = Conversation(
        id=_new_id(),
        title=title or f"Chat {now[:10]}",
        created_at=now,
        updated_at=now,
    )
    convs = _load_json(CONVS_FILE)
    convs.append(conv.model_dump())
    _save_json(CONVS_FILE, convs)
    return conv


def add_message(conv_id: str, role: str, content: str,
                sources: list[dict] = None) -> Optional[Message]:
    convs = _load_json(CONVS_FILE)
    for i, c in enumerate(convs):
        if c["id"] == conv_id:
            now = _utcnow()
            msg = Message(
                id=_new_id(),
                conversation_id=conv_id,
                role=role,
                content=content,
                sources=[Source(**s) for s in (sources or [])],
                created_at=now,
            )
            c["messages"].append(msg.model_dump())
            c["updated_at"] = now
            if role == "user" and not c.get("title") or c["title"].startswith("Chat "):
                c["title"] = content[:60]
            convs[i] = c
            _save_json(CONVS_FILE, convs)
            return msg
    return None


def delete_conv(conv_id: str) -> bool:
    convs = _load_json(CONVS_FILE)
    new_convs = [c for c in convs if c["id"] != conv_id]
    if len(new_convs) == len(convs):
        return False
    _save_json(CONVS_FILE, new_convs)
    return True
