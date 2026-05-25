import csv
import io
import json
import uuid
from typing import Any, Optional
import pandas as pd
from app.chromadb.client import chroma_client
from app.embeddings.openai_client import openai_client as ollama_client
from app.rag.metadata_filter import build_chroma_filter
from app.core.logging import get_logger

logger = get_logger("table_rag")

SCHEMA_COLLECTION = "table_documents"


class TableRAG:
    async def index_csv(
        self,
        document_id: str,
        filename: str,
        content: bytes,
        extra_metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Index CSV: schema + row/cell level chunks."""
        df = pd.read_csv(io.BytesIO(content))
        schema_info = {
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "row_count": len(df),
            "sample": df.head(3).to_dict(orient="records"),
        }

        ids, embeddings, documents, metadatas = [], [], [], []

        # Schema chunk
        schema_text = f"Table schema for {filename}:\nColumns: {', '.join(schema_info['columns'])}\nRow count: {schema_info['row_count']}\nSample: {json.dumps(schema_info['sample'][:2])}"
        schema_emb = await ollama_client.embeddings(schema_text)
        schema_id = f"{document_id}_schema"
        ids.append(schema_id)
        embeddings.append(schema_emb)
        documents.append(schema_text)
        meta = {
            "document_id": document_id,
            "filename": filename,
            "document_type": "csv",
            "chunk_type": "schema",
            "columns": json.dumps(list(df.columns)),
            **(extra_metadata or {}),
        }
        metadatas.append(meta)

        # Row chunks (batch 5 rows per chunk for dense tables)
        chunk_size = 5
        for start in range(0, min(len(df), 500), chunk_size):
            batch = df.iloc[start:start + chunk_size]
            row_text = batch.to_csv(index=False)
            row_emb = await ollama_client.embeddings(row_text)
            row_id = f"{document_id}_rows_{start}"
            ids.append(row_id)
            embeddings.append(row_emb)
            documents.append(row_text)
            metadatas.append({
                "document_id": document_id,
                "filename": filename,
                "document_type": "csv",
                "chunk_type": "rows",
                "row_start": start,
                "row_end": start + chunk_size,
                **(extra_metadata or {}),
            })

        chroma_client.add_documents(SCHEMA_COLLECTION, ids, embeddings, documents, metadatas)
        logger.info(f"TableRAG indexed {filename}: {len(ids)} chunks")
        return {"document_id": document_id, "chunk_count": len(ids), "schema": schema_info}

    async def query(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        query_emb = await ollama_client.embeddings(query)
        where = None
        if document_id:
            where = {"document_id": {"$eq": document_id}}
        results = chroma_client.search(SCHEMA_COLLECTION, query_emb, top_k, where)
        return results

    async def get_schema(self, document_id: str) -> Optional[dict]:
        query_emb = await ollama_client.embeddings("table schema columns")
        where = {"$and": [{"document_id": {"$eq": document_id}}, {"chunk_type": {"$eq": "schema"}}]}
        results = chroma_client.search(SCHEMA_COLLECTION, query_emb, 1, where)
        if results:
            return results[0]
        return None


table_rag = TableRAG()