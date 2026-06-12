#!/usr/bin/env python3

import asyncio
import json
import sys
from pathlib import Path

from elasticsearch import AsyncElasticsearch

ES_HOST = "http://localhost:9200"
INDEX_NAME = "rag_documents"
DATA_DIR = "/media/jishnu/Windows-SSD/Users/Jishnu/Desktop/SRAG/helpfull scripts/downloaded"


async def upload_files():
    client = AsyncElasticsearch([ES_HOST])

    try:
        if not await client.indices.exists(index=INDEX_NAME):
            await client.indices.create(
                index=INDEX_NAME,
                body={
                    "mappings": {
                        "properties": {
                            "content": {
                                "type": "text"
                            },
                            "metadata": {
                                "type": "object"
                            }
                        }
                    }
                }
            )

            print(f"Created index: {INDEX_NAME}")

        total_docs = 0

        for file_path in sorted(Path(DATA_DIR).rglob("*.json")):

            try:
                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as f:
                    data = json.load(f)

                content = data.get("content", "").strip()

                if not content:
                    continue

                source_url = data.get("url", "")

                document = {
                    "content": content,
                    "metadata": {
                        "source": file_path.name,
                        "filepath": str(file_path.resolve()),
                        "url": source_url,
                        "document_id": file_path.stem,
                        "file_type": file_path.suffix.lstrip("."),
                        "content_length": len(content),
                    }
                }

                await client.index(
                    index=INDEX_NAME,
                    id=file_path.stem,
                    body=document
                )

                total_docs += 1

                print(
                    f"✓ {file_path.name}"
                )

            except Exception as e:
                print(
                    f"✗ {file_path.name}: {e}"
                )

        await client.indices.refresh(
            index=INDEX_NAME
        )

        print()
        print("=" * 50)
        print(f"Indexed documents : {total_docs}")
        print(f"Index             : {INDEX_NAME}")
        print("=" * 50)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    finally:
        await client.close()


if __name__ == "__main__":
    print(
        f"Uploading crawler JSON files from:\n{DATA_DIR}\n"
    )

    asyncio.run(upload_files())