"""LLM呼び出しとプロンプト構築."""

from collections.abc import Iterator

from anthropic import Anthropic
from openai import OpenAI

from src.config import GeneratorConfig

SYSTEM_PROMPT = """\
あなたはナレッジベースに基づいて質問に回答するアシスタントです。
以下のルールに従ってください:
- 提供されたコンテキストの情報のみに基づいて回答してください
- コンテキストに情報がない場合は「提供された情報からは回答できません」と答えてください
- 回答は簡潔かつ正確にしてください
"""


def build_prompt(query: str, contexts: list[str]) -> str:
    """検索結果を含むプロンプトを構築する."""
    context_text = "\n\n---\n\n".join(contexts)
    return f"""\
以下のコンテキストを参考に、質問に回答してください。

## コンテキスト
{context_text}

## 質問
{query}
"""


def _generate_anthropic(prompt: str, config: GeneratorConfig) -> str:
    client = Anthropic(
        api_key=config.api_key,
        base_url=config.base_url,
    )
    message = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _generate_openai(prompt: str, config: GeneratorConfig) -> str:
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )
    response = client.chat.completions.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def _stream_anthropic(prompt: str, config: GeneratorConfig) -> Iterator[str]:
    client = Anthropic(
        api_key=config.api_key,
        base_url=config.base_url,
    )
    with client.messages.stream(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        yield from stream.text_stream


def _stream_openai(prompt: str, config: GeneratorConfig) -> Iterator[str]:
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )
    stream = client.chat.completions.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def generate(query: str, contexts: list[str], config: GeneratorConfig) -> str:
    """LLM APIで回答を生成する."""
    prompt = build_prompt(query, contexts)
    if config.provider == "anthropic":
        return _generate_anthropic(prompt, config)
    return _generate_openai(prompt, config)


def generate_stream(query: str, contexts: list[str], config: GeneratorConfig) -> Iterator[str]:
    """LLM APIでストリーミング回答を生成する."""
    prompt = build_prompt(query, contexts)
    if config.provider == "anthropic":
        return _stream_anthropic(prompt, config)
    return _stream_openai(prompt, config)
