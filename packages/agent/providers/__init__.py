"""LLM provider adapters, keyed by the `LLM_PROVIDER` value that selects them."""

from collections.abc import Callable

from packages.agent.llm import LlmProvider
from packages.agent.providers.openai_compatible import OpenAICompatibleProvider
from packages.common.settings import Settings

# NVIDIA's hosted API catalog; `LLM_BASE_URL` points elsewhere (for example a self-hosted NIM).
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


def nvidia_nim(settings: Settings) -> LlmProvider:
    assert settings.llm_model and settings.llm_api_key  # checked by provider_from_settings
    return OpenAICompatibleProvider(
        name="nvidia_nim",
        base_url=settings.llm_base_url or NVIDIA_NIM_BASE_URL,
        model=settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value(),
        timeout=settings.llm_timeout_seconds,
    )


ADAPTERS: dict[str, Callable[[Settings], LlmProvider]] = {"nvidia_nim": nvidia_nim}
