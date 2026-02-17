"""CLIエントリーポイント."""

import os

import typer
from dotenv import load_dotenv

from src.config import PipelineConfig
from src.pipeline import ask, ingest

load_dotenv()

app = typer.Typer(help="RAG Pipeline Playground")


@app.command()
def index(
    knowledge_dir: str = typer.Option("knowledge", help="ナレッジベースのディレクトリ"),
) -> None:
    """ナレッジベースのドキュメントを取り込む."""
    config = PipelineConfig(knowledge_dir=knowledge_dir)
    count = ingest(config)
    typer.echo(f"{count} チャンクを取り込みました")


@app.command()
def query(
    question: str = typer.Argument(help="質問"),
    no_stream: bool = typer.Option(False, "--no-stream", help="ストリーミング出力を無効化"),
) -> None:
    """質問に対してRAGで回答を生成する."""
    config = PipelineConfig()
    if no_stream:
        result = ask(question, config)
        typer.echo(f"\n回答:\n{result['answer']}")
    else:
        result = ask(question, config, stream=True)
        typer.echo("\n回答:")
        for chunk in result["stream"]:
            print(chunk, end="", flush=True)
        print()
    sources = dict.fromkeys(c["metadata"].get("source", "不明") for c in result["contexts"])
    typer.echo(f"\n参照ドキュメント ({len(result['contexts'])}チャンク):")
    for source in sources:
        typer.echo(f"  - {source}")


def main() -> None:
    try:
        app()
    finally:
        # Chromadbの非デーモンスレッドがプロセス終了をブロックするためOS レベルで終了
        os._exit(0)


if __name__ == "__main__":
    main()
