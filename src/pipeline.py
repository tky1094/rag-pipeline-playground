"""RAGパイプラインの統合."""

import hashlib

from src.config import PipelineConfig
from src.embedder import get_embeddings
from src.generator import TokenUsage, generate, generate_stream
from src.loader import load_markdown_files, split_by_headers, split_chunks
from src.retriever import retrieve
from src.store import get_collection, upsert_chunks


def _make_chunk_id(source: str, index: int) -> str:
    """チャンクのユニークIDを生成する."""
    content = f"{source}:{index}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def ingest(config: PipelineConfig) -> int:
    """ナレッジベースのドキュメントを取り込む."""
    docs = load_markdown_files(config.knowledge_dir)
    collection = get_collection(config)

    total = 0
    for doc in docs:
        # ヘッダー単位で分割
        header_chunks = split_by_headers(doc["text"])
        texts = [chunk["text"] for chunk in header_chunks]

        # さらに固定長で分割
        chunks = split_chunks(texts, config.chunk)
        if not chunks:
            continue

        # Embedding生成
        embeddings = get_embeddings(chunks, config.embedding)

        # メタデータ付きでChromaに保存
        ids = [_make_chunk_id(doc["source"], i) for i in range(len(chunks))]
        metadatas = [{"source": doc["source"]} for _ in chunks]
        upsert_chunks(collection, ids, chunks, embeddings, metadatas)
        total += len(chunks)

    return total


def ask(query: str, config: PipelineConfig, *, stream: bool = False) -> dict:
    """質問に対してRAGで回答を生成する."""
    collection = get_collection(config)
    chunks = retrieve(query, collection, config)
    contexts = [chunk["text"] for chunk in chunks]

    if stream:
        usage = TokenUsage()
        return {
            "query": query,
            "stream": generate_stream(query, contexts, config.generator, usage),
            "usage": usage,
            "contexts": chunks,
        }

    result = generate(query, contexts, config.generator)
    return {
        "query": query,
        "answer": result.text,
        "usage": result.usage,
        "contexts": chunks,
    }
