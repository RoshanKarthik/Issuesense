"""Optional LangSmith tracing of graph runs, on only when `LANGSMITH_API_KEY` is set.

The key is read from IssueSense's settings (`.env`), not from the process environment, and tracing
is explicitly disabled when it is absent, so a stray `LANGSMITH_TRACING=true` cannot send data
anywhere. When on, LangSmith records each node's inputs and outputs, which include the question,
the caller's user ID and scopes, and retrieved evidence text: enable it only for a LangSmith
workspace that may hold the demo workspace's data.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cached_property
from typing import Any

from langsmith import Client
from langsmith.run_helpers import tracing_context

from packages.common.settings import Settings

ClientFactory = Callable[..., Any]


class LangSmithTracing:
    def __init__(self, settings: Settings, client_factory: ClientFactory = Client) -> None:
        self._key = settings.langsmith_api_key
        self._endpoint = settings.langsmith_endpoint
        self.project = settings.langsmith_project
        self._client_factory = client_factory  # tests pass a stub: a real client calls LangSmith

    @property
    def enabled(self) -> bool:
        return self._key is not None

    @cached_property
    def _client(self) -> Any:
        assert self._key is not None
        return self._client_factory(api_key=self._key.get_secret_value(), api_url=self._endpoint)

    @contextmanager
    def run(self, query_id: str, user_id: str) -> Iterator[None]:
        """Trace the graph runs inside this block to LangSmith, or explicitly not at all."""
        if not self.enabled:
            with tracing_context(enabled=False):
                yield
            return
        metadata: dict[str, Any] = {"query_id": query_id, "user_id": user_id}
        with tracing_context(
            enabled=True,
            client=self._client,
            project_name=self.project,
            metadata=metadata,
            tags=["issuesense"],
        ):
            yield


def disabled() -> LangSmithTracing:
    """Tracing that never sends anything (the default outside the API, and in tests)."""
    return LangSmithTracing(Settings(_env_file=None, langsmith_api_key=None))
