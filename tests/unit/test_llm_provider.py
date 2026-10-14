import json
from collections.abc import Callable

import httpx
import pytest

from packages.agent.llm import (
    LlmConfigurationError,
    LlmMessage,
    LlmProviderError,
    LlmRequest,
    extract_json_object,
    provider_from_settings,
)
from packages.agent.providers import NVIDIA_NIM_BASE_URL
from packages.agent.providers.openai_compatible import OpenAICompatibleProvider
from packages.common.settings import Settings

KEY = "nvapi-test-" + "x" * 30
REQUEST = LlmRequest(
    messages=[LlmMessage(role="system", content="rules"), LlmMessage(role="user", content="hi")],
    max_output_tokens=300,
)


def _completion(content: str | None = '{"ok": true}') -> dict[str, object]:
    return {
        "model": "meta/llama-test",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
    }


def _provider(
    handler: Callable[[httpx.Request], httpx.Response], attempts: int = 2
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="nvidia_nim",
        base_url="https://nim.example.test/v1/",
        model="meta/llama-test",
        api_key=KEY,
        attempts=attempts,
        backoff_seconds=0,
        transport=httpx.MockTransport(handler),
    )


async def test_request_follows_the_chat_completions_contract() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion())

    response = await _provider(handler).complete(REQUEST)
    request = seen[0]
    assert str(request.url) == "https://nim.example.test/v1/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert body["model"] == "meta/llama-test" and body["max_tokens"] == 300
    assert body["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
    ]
    assert body["stream"] is False
    assert (response.text, response.input_tokens, response.output_tokens) == ('{"ok": true}', 12, 7)


async def test_rate_limits_and_server_errors_are_retried() -> None:
    statuses = [429, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        return httpx.Response(status, json=_completion() if status == 200 else {"error": "slow"})

    response = await _provider(handler).complete(REQUEST)
    assert response.text == '{"ok": true}' and statuses == []


async def test_persistent_errors_raise_without_leaking_the_key_or_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": f"internal trace for {KEY}"})

    with pytest.raises(LlmProviderError) as raised:
        await _provider(handler).complete(REQUEST)
    assert raised.value.details == {"provider": "nvidia_nim", "status": 503}
    assert KEY not in str(raised.value) and "trace" not in str(raised.value)


async def test_client_errors_are_not_retried() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(401, json={"error": "bad key"})

    with pytest.raises(LlmProviderError):
        await _provider(handler).complete(REQUEST)
    assert len(calls) == 1


async def test_network_failures_become_provider_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    with pytest.raises(LlmProviderError, match="could not be reached"):
        await _provider(handler).complete(REQUEST)


@pytest.mark.parametrize("payload", [{"choices": []}, {"nope": 1}, _completion(content=None)])
async def test_unusable_payloads_raise(payload: dict[str, object]) -> None:
    with pytest.raises(LlmProviderError):
        await _provider(lambda request: httpx.Response(200, json=payload)).complete(REQUEST)


def test_extract_json_object_strips_reasoning_fences_and_chatter() -> None:
    reply = '<think>{"draft": 1}</think>Sure! ```json\n{"answer": "x"}\n```'
    assert json.loads(extract_json_object(reply)) == {"answer": "x"}
    with pytest.raises(ValueError):
        extract_json_object("no object here")


def _settings(**values: object) -> Settings:
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_nvidia_nim_is_built_from_settings() -> None:
    provider = provider_from_settings(
        _settings(llm_provider="NVIDIA_NIM", llm_model="meta/llama-test", llm_api_key=KEY)
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.name == "nvidia_nim"
    assert provider._url == f"{NVIDIA_NIM_BASE_URL}/chat/completions"


def test_base_url_can_point_at_another_nim() -> None:
    provider = provider_from_settings(
        _settings(
            llm_provider="nvidia_nim",
            llm_model="m",
            llm_api_key=KEY,
            llm_base_url="https://nim.internal.example/v1/",
        )
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._url == "https://nim.internal.example/v1/chat/completions"


def test_a_provider_without_model_or_key_is_refused() -> None:
    with pytest.raises(LlmConfigurationError) as raised:
        provider_from_settings(_settings(llm_provider="nvidia_nim"))
    assert raised.value.details == {"missing": ["LLM_MODEL", "LLM_API_KEY"]}


def test_plain_http_base_urls_are_rejected() -> None:
    with pytest.raises(ValueError):
        _settings(llm_base_url="http://nim.example/v1")
