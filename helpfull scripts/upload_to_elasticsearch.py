#!/usr/bin/env python3

import json
import hashlib
from pathlib import Path

import fitz
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

ES_HOST = "http://10.64.240.47:9200"
INDEX_NAME = "rag_documents"

DATA_DIR = "/home/wtc6/formbot-jishnu/data"

EMBEDDING_MODEL = "BAAI/bge-m3"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

BULK_SIZE = 500

# BGE-M3 produces 1024-dimensional embeddings
EMBEDDING_DIMS = 1024


# ============================================================
# INDEX SETUP
# ============================================================

def create_index_if_needed(es):

    if es.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists")
        return

    print(f"Creating index '{INDEX_NAME}'...")

    es.indices.create(
        index=INDEX_NAME,
        body={
            "mappings": {
                "properties": {

                    "content": {
                        "type": "text"
                    },

                    "embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIMS,
                        "index": True,
                        "similarity": "cosine"
                    },

                    # ========================
                    # FILE METADATA
                    # ========================

                    "source_id": {
                        "type": "keyword"
                    },

                    "file_name": {
                        "type": "keyword"
                    },

                    "file_stem": {
                        "type": "keyword"
                    },

                    "file_path": {
                        "type": "keyword"
                    },

                    "file_type": {
                        "type": "keyword"
                    },

                    # ========================
                    # DOCUMENT METADATA
                    # ========================

                    "document_id": {
                        "type": "keyword"
                    },

                    "chunk_number": {
                        "type": "integer"
                    },

                    "total_chunks": {
                        "type": "integer"
                    },

                    # ========================
                    # SOURCE INFO
                    # ========================

                    "url": {
                        "type": "keyword"
                    },

                    # ========================
                    # STATS
                    # ========================

                    "content_length": {
                        "type": "integer"
                    }
                }
            }
        }
    )

    print("Index created")


# ============================================================
# FILE READERS
# ============================================================

def read_json(path: Path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    content = data.get("content", "").strip()
    url = data.get("url", "")

    return content, url


def read_txt(path: Path):

    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip(), ""


def read_pdf(path: Path):

    doc = fitz.open(path)

    pages = []

    for page in doc:
        pages.append(page.get_text())

    doc.close()

    return "\n".join(pages).strip(), ""


def read_file(path: Path):

    suffix = path.suffix.lower()

    try:

        if suffix == ".json":
            return read_json(path)

        elif suffix == ".txt":
            return read_txt(path)

        elif suffix == ".pdf":
            return read_pdf(path)

        return None, None

    except Exception as e:
        print(f"ERROR reading {path}: {e}")
        return None, None


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(text):

    if not text:
        return []

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []
    current = ""

    for para in paragraphs:

        if len(current) + len(para) < CHUNK_SIZE:

            current += "\n\n" + para

        else:

            chunks.append(current.strip())

            overlap_text = (
                current[-CHUNK_OVERLAP:]
                if len(current) > CHUNK_OVERLAP
                else current
            )

            current = overlap_text + "\n\n" + para

    if current:
        chunks.append(current.strip())

    return chunks


# ============================================================
# MAIN
# ============================================================

def main():

    print("Connecting to Elasticsearch...")

    es = Elasticsearch(ES_HOST)

    if not es.ping():
        raise RuntimeError(
            f"Cannot connect to Elasticsearch at {ES_HOST}"
        )

    create_index_if_needed(es)

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    files = []

    for ext in ("*.json", "*.txt", "*.pdf"):
        files.extend(Path(DATA_DIR).rglob(ext))

    print(f"Found {len(files)} files")

    actions = []

    total_chunks = 0
    total_docs = 0

    for file_path in tqdm(files, desc="Processing"):

        content, url = read_file(file_path)

        if not content:
            continue

        chunks = chunk_text(content)

        if not chunks:
            continue

        embeddings = model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Stable document identifier
        source_id = hashlib.sha256(
            str(file_path.resolve()).encode("utf-8")
        ).hexdigest()

        total_file_chunks = len(chunks)

        for i, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            actions.append(
                {
                    "_index": INDEX_NAME,
                    "_id": f"{source_id}_{i}",
                    "_source": {

                        # ========================
                        # RAG DATA
                        # ========================

                        "content": chunk,
                        "embedding": embedding.tolist(),

                        # ========================
                        # FILE METADATA
                        # ========================

                        "source_id": source_id,
                        "file_name": file_path.name,
                        "file_stem": file_path.stem,
                        "file_path": str(
                            file_path.resolve()
                        ),
                        "file_type": file_path.suffix.lower().replace(
                            ".", ""
                        ),

                        # ========================
                        # DOCUMENT METADATA
                        # ========================

                        "document_id": file_path.stem,
                        "chunk_number": i,
                        "total_chunks": total_file_chunks,

                        # ========================
                        # SOURCE INFO
                        # ========================

                        "url": url,

                        # ========================
                        # STATS
                        # ========================

                        "content_length": len(chunk)
                    }
                }
            )

            total_chunks += 1

            if len(actions) >= BULK_SIZE:

                helpers.bulk(
                    es,
                    actions,
                    request_timeout=300
                )

                actions.clear()

        total_docs += 1

    if actions:

        helpers.bulk(
            es,
            actions,
            request_timeout=300
        )

    es.indices.refresh(index=INDEX_NAME)

    print()
    print("=" * 60)
    print(f"Documents indexed : {total_docs}")
    print(f"Chunks indexed    : {total_chunks}")
    print(f"Index             : {INDEX_NAME}")
    print("=" * 60)

    print("\nUseful filters:")
    print("  file_name")
    print("  file_stem")
    print("  file_type")
    print("  source_id")
    print("  document_id")


if __name__ == "__main__":
    main()