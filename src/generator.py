"""LLM呼び出しとプロンプト構築."""

from collections.abc import Iterator
from dataclasses import dataclass, field

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


@dataclass
class TokenUsage:
    """トークン使用量."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class GenerateResult:
    """生成結果."""

    text: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)


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


def _generate_anthropic(prompt: str, config: GeneratorConfig) -> GenerateResult:
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
    return GenerateResult(
        text=message.content[0].text,
        usage=TokenUsage(input_tokens=message.usage.input_tokens, output_tokens=message.usage.output_tokens),
    )


def _generate_openai(prompt: str, config: GeneratorConfig) -> GenerateResult:
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
    usage = TokenUsage()
    if response.usage:
        usage = TokenUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)
    return GenerateResult(text=response.choices[0].message.content, usage=usage)


def _stream_anthropic(prompt: str, config: GeneratorConfig, usage: TokenUsage) -> Iterator[str]:
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
        final = stream.get_final_message()
        usage.input_tokens = final.usage.input_tokens
        usage.output_tokens = final.usage.output_tokens


def _stream_openai(prompt: str, config: GeneratorConfig, usage: TokenUsage) -> Iterator[str]:
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
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
        if chunk.usage:
            usage.input_tokens = chunk.usage.prompt_tokens
            usage.output_tokens = chunk.usage.completion_tokens


def generate(query: str, contexts: list[str], config: GeneratorConfig) -> GenerateResult:
    """LLM APIで回答を生成する."""
    prompt = build_prompt(query, contexts)
    if config.provider == "anthropic":
        return _generate_anthropic(prompt, config)
    return _generate_openai(prompt, config)


def generate_stream(
    query: str,
    contexts: list[str],
    config: GeneratorConfig,
    usage: TokenUsage,
) -> Iterator[str]:
    """LLM APIでストリーミング回答を生成する."""
    prompt = build_prompt(query, contexts)
    if config.provider == "anthropic":
        return _stream_anthropic(prompt, config, usage)
    return _stream_openai(prompt, config, usage)
