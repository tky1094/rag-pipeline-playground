"""Embeddingの生成."""

from openai import OpenAI

from src.config import EmbeddingConfig


def get_embeddings(texts: list[str], config: EmbeddingConfig) -> list[list[float]]:
    """OpenAI互換のEmbedding APIでテキストをベクトル化する."""
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )
    response = client.embeddings.create(
        input=texts,
        model=config.model,
    )
    return [item.embedding for item in response.data]
