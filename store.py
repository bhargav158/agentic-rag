import chromadb
from chromadb.config import Settings

import config

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=config.CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
        )
    return _client


def collection():
    return _get_client().get_or_create_collection(
        config.COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def reset():
    try:
        _get_client().delete_collection(config.COLLECTION)
    except Exception:
        pass


def add(ids, docs, embeddings, metas):
    col = collection()
    for i in range(0, len(ids), 100):
        col.upsert(
            ids=ids[i:i + 100],
            documents=docs[i:i + 100],
            embeddings=embeddings[i:i + 100],
            metadatas=metas[i:i + 100],
        )


def delete_repo(name):
    """Remove every chunk of one repo (used when a repo is re-indexed)."""
    collection().delete(where={"repo": name})
