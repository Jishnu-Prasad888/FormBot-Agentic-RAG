import csv
import io
import json
import uuid
from typing import Any, Optional
import pandas as pd
from app.chromadb.client import chroma_client
from app.embeddings.openai_client import openai_client as ollama_client
from app.rag.metadata_filter import build_chroma_filter
from app.rag.bm25 import bm25_retriever


SCHEMA_COLLECTION = "table_documents"


class TableRAG:
    def _convert_row_to_text(self, row: pd.Series, headers: list[str]) -> str:
        """Convert a row to readable key-value text format."""
        parts = []
        for header in headers:
            val = row.get(header, "")
            parts.append(f"{header}: {val}")
        return " | ".join(parts)

    def _convert_table_section_to_text(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> str:
        """Convert table rows to readable text preserving headers and structure."""
        lines = []
        headers = list(df.columns)
        
        # Header row
        lines.append(" | ".join(headers))
        lines.append("-" * (sum(len(str(h)) for h in headers) + len(headers) * 3))
        
        # Data rows
        for idx in range(start_idx, min(end_idx, len(df))):
            row = df.iloc[idx]
            row_text = self._convert_row_to_text(row, headers)
            lines.append(row_text)
        
        return "\n".join(lines)

    async def index_csv(
        self,
        document_id: str,
        filename: str,
        content: bytes,
        extra_metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Index CSV: schema + structured text chunks."""
        df = pd.read_csv(io.BytesIO(content))
        schema_info = {
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "row_count": len(df),
        }

        ids, embeddings, documents, metadatas = [], [], [], []
        chunk_records = []

        # Schema chunk with column details
        col_types = ", ".join([f"{col} ({dtype})" for col, dtype in schema_info["dtypes"].items()])
        schema_text = f"Table: {filename}\nColumns: {col_types}\nTotal rows: {schema_info['row_count']}"
        schema_emb = await ollama_client.embeddings(schema_text)
        schema_id = str(uuid.uuid4())
        ids.append(schema_id)
        embeddings.append(schema_emb)
        documents.append(schema_text)
        schema_meta = {
            "document_id": document_id,
            "filename": filename,
            "document_type": "csv",
            "chunk_type": "schema",
            "columns": json.dumps(list(df.columns)),
            "chunk_id": schema_id,
            **(extra_metadata or {}),
        }
        metadatas.append(schema_meta)
        chunk_records.append({
            "id": schema_id,
            "document_id": document_id,
            "chunk_index": 0,
            "chunk_text": schema_text,
            "chunk_metadata": schema_meta,
            "metadata_json": schema_meta,
            "qdrant_point_id": schema_id,
        })

        # Row chunks: convert to readable text format (preserve headers in each chunk)
        chunk_size = 10  # Increased from 5 to capture more context
        for idx, start in enumerate(range(0, min(len(df), 500), chunk_size), start=1):
            end = start + chunk_size
            row_text = self._convert_table_section_to_text(df, start, end)
            row_emb = await ollama_client.embeddings(row_text)
            row_id = str(uuid.uuid4())
            ids.append(row_id)
            embeddings.append(row_emb)
            documents.append(row_text)
            row_meta = {
                "document_id": document_id,
                "filename": filename,
                "document_type": "csv",
                "chunk_type": "rows",
                "row_start": start,
                "row_end": end,
                "chunk_id": row_id,
                **(extra_metadata or {}),
            }
            metadatas.append(row_meta)
            chunk_records.append({
                "id": row_id,
                "document_id": document_id,
                "chunk_index": idx,
                "chunk_text": row_text,
                "chunk_metadata": row_meta,
                "metadata_json": row_meta,
                "qdrant_point_id": row_id,
            })

        chroma_client.add_documents(SCHEMA_COLLECTION, ids, embeddings, documents, metadatas)
        try:
            bm25_retriever.index(SCHEMA_COLLECTION, [
                {
                    "chunk_id": ids[i],
                    "chunk_text": documents[i],
                    "metadata": metadatas[i],
                    "document_id": metadatas[i].get("document_id", ""),
                    "filename": metadatas[i].get("filename", ""),
                }
                for i in range(len(ids))
            ])
        except Exception:
            pass
        return {"document_id": document_id, "chunk_count": len(ids), "schema": schema_info, "chunks": chunk_records}

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
