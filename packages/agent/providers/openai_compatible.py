"""An adapter for OpenAI-compatible Chat Completions APIs, such as NVIDIA NIM, over HTTPS (httpx).

The model, key and endpoint come from settings; nothing vendor-specific is hardcoded in business
code. Rate limits and server errors are retried with a short backoff; every other failure raises
`LlmProviderError` without echoing the key or the provider's raw error body.
"""

import asyncio
import logging
from typing import Any

import httpx

from packages.agent.llm import LlmProviderError, LlmRequest, LlmResponse

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 60.0,
        attempts: int = 2,
        backoff_seconds: float = 1.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        self._timeout = timeout
        self._attempts = max(1, attempts)
        self._backoff = backoff_seconds
        self._transport = transport  # tests inject httpx.MockTransport

    def _body(self, request: LlmRequest) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [message.model_dump() for message in request.messages],
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "stream": False,
        }

    async def complete(self, request: LlmRequest) -> LlmResponse:
        async with httpx.AsyncClient(
            headers=self._headers, timeout=self._timeout, transport=self._transport
        ) as client:
            for attempt in range(1, self._attempts + 1):
                retry = attempt < self._attempts
                try:
                    response = await client.post(self._url, json=self._body(request))
                except httpx.TransportError as exc:  # includes timeouts
                    logger.warning("%s request failed (attempt %d): %r", self.name, attempt, exc)
                    if retry:
                        await asyncio.sleep(self._backoff * attempt)
                        continue
                    raise LlmProviderError(
                        "The language model could not be reached.", {"provider": self.name}
                    ) from exc
                if response.status_code in RETRYABLE_STATUS and retry:
                    logger.warning(
                        "%s returned HTTP %d (attempt %d); retrying",
                        self.name,
                        response.status_code,
                        attempt,
                    )
                    await asyncio.sleep(self._backoff * attempt)
                    continue
                return self._parse(response)
        raise AssertionError("unreachable")  # the loop always returns or raises

    def _parse(self, response: httpx.Response) -> LlmResponse:
        if response.status_code != 200:
            logger.error("%s returned HTTP %d", self.name, response.status_code)
            raise LlmProviderError(
                "The language model returned an error.",
                {"provider": self.name, "status": response.status_code},
            )
        try:
            payload = response.json()
            message = payload["choices"][0]["message"]
            text = message.get("content") or ""
            usage = payload.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LlmProviderError(
                "The language model returned an unexpected payload.", {"provider": self.name}
            ) from exc
        if not isinstance(text, str) or not text.strip():
            raise LlmProviderError("The language model returned no text.", {"provider": self.name})
        return LlmResponse(
            text=text,
            model=payload.get("model") or self._model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )
