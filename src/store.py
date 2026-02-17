"""Chromaベクトルストアの管理."""

import chromadb

from src.config import PipelineConfig


def get_collection(config: PipelineConfig) -> chromadb.Collection:
    """Chromaコレクションを取得または作成する."""
    client = chromadb.PersistentClient(path=config.chroma_path)
    return client.get_or_create_collection(
        name=config.retriever.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    collection: chromadb.Collection,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict] | None = None,
) -> None:
    """チャンクをコレクションに保存する."""
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query_collection(
    collection: chromadb.Collection,
    query_embedding: list[float],
    n_results: int = 5,
) -> dict:
    """ベクトル類似検索を実行する."""
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
