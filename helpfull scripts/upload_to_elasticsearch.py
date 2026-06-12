#!/usr/bin/env python3
import os
import sys
import asyncio
from pathlib import Path
from elasticsearch import AsyncElasticsearch

ES_HOST = "http://localhost:9200"
INDEX_NAME = "rag_documents"
DATA_DIR = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/data"

async def upload_files():
    client = AsyncElasticsearch([ES_HOST])
    
    try:
        # Create index
        if not await client.indices.exists(index=INDEX_NAME):
            await client.indices.create(index=INDEX_NAME, body={
                "mappings": {"properties": {"content": {"type": "text"}, "metadata": {"type": "object"}}}
            })
            print(f"Created index: {INDEX_NAME}")
        
        doc_id = 0
        total = 0
        
        # Process all files in data directory
        for file_path in sorted(Path(DATA_DIR).rglob("*")):
            if file_path.is_file() and file_path.suffix in [".txt", ".csv", ".md"]:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                        if not content:
                            continue
                        
                        # Split by lines and index
                        lines = [l.strip() for l in content.split("\n") if l.strip()]
                        for line in lines:
                            await client.index(
                                index=INDEX_NAME,
                                id=f"doc_{doc_id}",
                                body={"content": line, "metadata": {"source": file_path.name}}
                            )
                            doc_id += 1
                        
                        total += len(lines)
                        print(f"✓ {file_path.name}: {len(lines)} documents")
                except Exception as e:
                    print(f"✗ {file_path.name}: {e}")
        
        print(f"\nTotal indexed: {total} documents")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        await client.close()

if __name__ == "__main__":
    print(f"Uploading data from {DATA_DIR} to Elasticsearch...")
    asyncio.run(upload_files())
