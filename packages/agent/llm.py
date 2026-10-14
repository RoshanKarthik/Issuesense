"""Provider-neutral LLM interface. Business code depends on this, never on a vendor SDK.

Adapters live in `packages.agent.providers` and are chosen by `LLM_PROVIDER`.
`provider_from_settings` returns None when none is configured, and refuses a provider it has no
adapter for (or one missing its model or key), rather than silently answering without it.
"""

import re
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from packages.common.errors import IssueSenseError
from packages.common.settings import Settings


class LlmMessage(BaseModel):
    role: Literal["system", "user"]
    content: str


class LlmRequest(BaseModel):
    messages: list[LlmMessage] = Field(min_length=1)
    max_output_tokens: int = Field(default=1500, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    json_output: bool = True


class LlmResponse(BaseModel):
    text: str
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class LlmProvider(Protocol):
    name: str

    async def complete(self, request: LlmRequest) -> LlmResponse: ...


class LlmConfigurationError(IssueSenseError):
    """An LLM provider is configured that IssueSense cannot use."""


class LlmProviderError(IssueSenseError):
    """The provider did not return a usable completion (network, HTTP or payload error)."""


_THINKING = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def extract_json_object(text: str) -> str:
    """The JSON object in a model reply, without reasoning blocks, code fences or chatter around it.

    Raises ValueError when the reply holds no object at all.
    """
    body = _FENCE.sub("", _THINKING.sub("", text).strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end < start:
        raise ValueError("the reply contains no JSON object")
    return body[start : end + 1]


def provider_from_settings(settings: Settings) -> LlmProvider | None:
    name = (settings.llm_provider or "").strip().lower()
    if not name:
        return None
    from packages.agent.providers import ADAPTERS  # adapters import this module's types

    if name not in ADAPTERS:
        raise LlmConfigurationError(
            f"LLM_PROVIDER={name!r} is set, but no adapter exists for it.",
            {"available": sorted(ADAPTERS)},
        )
    missing = [
        variable
        for variable, value in (
            ("LLM_MODEL", settings.llm_model),
            ("LLM_API_KEY", settings.llm_api_key),
        )
        if not value
    ]
    if missing:
        raise LlmConfigurationError(
            f"LLM_PROVIDER={name!r} also needs {' and '.join(missing)}.", {"missing": missing}
        )
    return ADAPTERS[name](settings)
