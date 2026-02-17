"""検索ロジック."""

import chromadb

from src.config import PipelineConfig
from src.embedder import get_embeddings
from src.store import query_collection


def retrieve(
    query: str,
    collection: chromadb.Collection,
    config: PipelineConfig,
) -> list[dict]:
    """質問に関連するチャンクを検索する."""
    query_embedding = get_embeddings([query], config.embedding)[0]
    results = query_collection(
        collection=collection,
        query_embedding=query_embedding,
        n_results=config.retriever.top_k,
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append(
            {
                "text": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            }
        )
    return chunks
