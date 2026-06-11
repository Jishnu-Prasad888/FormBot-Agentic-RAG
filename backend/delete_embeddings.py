from app.chromadb.client import chroma_client

# Delete all collections
for collection_name in ["text_documents", "pdf_documents", "table_documents", 
                        "markdown_documents", "audio_transcripts", "web_documents"]:
    try:
        chroma_client.delete_collection(collection_name)
        print(f"Deleted {collection_name}")
    except Exception as e:
        print(f"Failed to delete {collection_name}: {e}")

# Reinitialize empty collections
chroma_client.init_collections()