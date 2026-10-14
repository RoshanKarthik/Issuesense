"""Provider-neutral LLM interface. Business code depends on this, never on a vendor SDK.

No provider adapter is implemented yet (the owner has not chosen one). `provider_from_settings`
returns None when none is configured, and refuses a configured provider it has no adapter for,
rather than silently answering without it.
"""

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


# Provider name -> adapter factory. Adapters are added once the owner picks a provider.
_ADAPTERS: dict[str, type[LlmProvider]] = {}


def provider_from_settings(settings: Settings) -> LlmProvider | None:
    name = (settings.llm_provider or "").strip().lower()
    if not name:
        return None
    if name not in _ADAPTERS:
        raise LlmConfigurationError(
            f"LLM_PROVIDER={name!r} is set, but no adapter exists for it yet.",
            {"available": sorted(_ADAPTERS)},
        )
    return _ADAPTERS[name]()
