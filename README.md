# rag-pipeline-playground

Markdown ドキュメントをナレッジベースとした RAG パイプラインの実装検証・精度検証用リポジトリ。

## セットアップ

```bash
# 依存インストール
uv sync

# 環境変数の設定
cp .env.example .env
# .env を編集して API キーを設定
```

## 使い方

```bash
# ナレッジベースの取り込み
uv run python -m src.cli index

# 質問（ストリーミング出力）
uv run python -m src.cli query "リモートワークは週何日まで？"

# 質問（一括出力）
uv run python -m src.cli query "リモートワークは週何日まで？" --no-stream
```

## 設定

環境変数で各パラメータを変更できる。デフォルト値は `.env.example` を参照。

```bash
# 例: チャンクサイズを変更
RAG_CHUNK__SIZE=1024

# 例: Ollama を使う場合
RAG_GENERATOR__PROVIDER=openai
RAG_GENERATOR__MODEL=gemma3
RAG_GENERATOR__BASE_URL=http://localhost:11434/v1
RAG_GENERATOR__API_KEY=ollama
RAG_EMBEDDING__MODEL=nomic-embed-text
RAG_EMBEDDING__BASE_URL=http://localhost:11434/v1
RAG_EMBEDDING__API_KEY=ollama
```

## 構成

```
src/
  cli.py          CLI エントリーポイント (index / query)
  config.py       環境変数ベースの設定管理
  loader.py       Markdown 読み込み・チャンク分割
  embedder.py     Embedding 生成 (OpenAI 互換 API)
  store.py        Chroma ベクトルストア
  retriever.py    ベクトル類似検索
  generator.py    LLM 回答生成 (Anthropic / OpenAI 互換 API)
  pipeline.py     上記を統合した ingest / ask
eval/
  evaluate.py     Ragas による評価（未実装）
  dataset.json    評価用データセット
knowledge/
  sample/         サンプルナレッジベース（架空の社内規程）
```

## 技術スタック

- Python 3.12 / uv
- ChromaDB (ベクトルストア)
- OpenAI API (Embedding)
- Anthropic API / OpenAI 互換 API (LLM)
- LangChain Text Splitters (チャンク分割)
- Ragas (評価)
- ruff (lint / format)
