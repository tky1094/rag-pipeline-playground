"""環境変数で上書き可能なパイプライン設定.

環境変数のプレフィックス: RAG_
ネストはアンダースコアで区切る (例: RAG_CHUNK__SIZE=1024)
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkConfig(BaseSettings):
    """チャンク分割の設定."""

    model_config = SettingsConfigDict(env_prefix="RAG_CHUNK__")

    size: int = Field(default=512, description="チャンクサイズ（文字数）")
    overlap: int = Field(default=64, description="チャンク間のオーバーラップ（文字数）")
    split_by_header: bool = Field(default=True, description="Markdownヘッダーで分割するか")


class EmbeddingConfig(BaseSettings):
    """Embeddingモデルの設定."""

    model_config = SettingsConfigDict(env_prefix="RAG_EMBEDDING__")

    model: str = Field(default="text-embedding-3-small", description="Embeddingモデル名")
    dimensions: int = Field(default=1536, description="ベクトルの次元数")
    api_key: str | None = Field(default=None, description="APIキー（未指定時はOPENAI_API_KEY）")
    base_url: str | None = Field(default=None, description="OpenAI互換APIのベースURL")


class RetrieverConfig(BaseSettings):
    """検索の設定."""

    model_config = SettingsConfigDict(env_prefix="RAG_RETRIEVER__")

    top_k: int = Field(default=5, description="検索結果の上位件数")
    collection_name: str = Field(default="knowledge", description="Chromaコレクション名")


class GeneratorConfig(BaseSettings):
    """LLM生成の設定."""

    model_config = SettingsConfigDict(env_prefix="RAG_GENERATOR__")

    provider: Literal["anthropic", "openai"] = Field(default="openai", description="LLMプロバイダー")
    model: str = Field(default="gpt-4o", description="LLMモデル名")
    max_tokens: int = Field(default=1024, description="最大トークン数")
    temperature: float = Field(default=0.0, description="温度パラメータ")
    api_key: str | None = Field(default=None, description="APIキー（未指定時は各プロバイダーのデフォルト）")
    base_url: str | None = Field(default=None, description="OpenAI互換APIのベースURL")


class PipelineConfig(BaseSettings):
    """パイプライン全体の設定."""

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    chroma_path: str = Field(default=".chroma", description="Chroma永続化パス")
    knowledge_dir: str = Field(default="knowledge", description="ナレッジベースのディレクトリ")
