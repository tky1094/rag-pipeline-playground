"""Markdownドキュメントの読み込みとチャンク分割."""

from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.config import ChunkConfig


def load_markdown_files(knowledge_dir: str) -> list[dict]:
    """knowledge_dir 配下の .md ファイルを読み込む."""
    docs = []
    for md_path in sorted(Path(knowledge_dir).rglob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        docs.append({"text": text, "source": str(md_path)})
    return docs


def split_by_headers(text: str) -> list[dict]:
    """Markdownのヘッダー構造でチャンク分割する."""
    headers_to_split = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)
    return [{"text": doc.page_content, "metadata": doc.metadata} for doc in splitter.split_text(text)]


def split_chunks(texts: list[str], config: ChunkConfig) -> list[str]:
    """テキストを固定長でチャンク分割する."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.size,
        chunk_overlap=config.overlap,
    )
    return splitter.split_text("\n\n".join(texts))
