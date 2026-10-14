"""In-memory fakes for testing the query graph without MCP or an LLM."""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from packages.agent.llm import LlmRequest, LlmResponse
from packages.connectors.base import Connector, RetrievalRequest, SourceRecord

SITE = "https://atlas-demo.atlassian.net"


def issue(key: str, summary: str, description: str = "", **fields: Any) -> dict[str, Any]:
    return {
        "key": key,
        "fields": {
            "summary": summary,
            "description": description,
            "status": {"name": fields.pop("status", "In Progress")},
            "updated": fields.pop("updated", "2026-10-12T09:00:00+00:00"),
            **fields,
        },
    }


class StaticJiraConnector(Connector):
    """Returns the given issues for every granted project; records each request."""

    source = "jira"

    def __init__(self, issues: list[dict[str, Any]]) -> None:
        self.issues = issues
        self.requests: list[RetrievalRequest] = []

    async def _fetch(self, request: RetrievalRequest, grants: dict[str, str]) -> list[SourceRecord]:
        self.requests.append(request)
        return [
            SourceRecord(
                source="jira",
                resource=project,
                tool="searchJiraIssuesUsingJql",
                payload={"issue": item, "comments": [], "changelog": [], "details_fetched": True},
                permission_context=scope,
            )
            for project, scope in grants.items()
            for item in self.issues
            if str(item["key"]).startswith(f"{project}-")
        ]


class Opener:
    """A connector opener that counts how often it was opened and closed."""

    def __init__(self, connector: Connector | None = None, error: Exception | None = None) -> None:
        self.connector = connector
        self.error = error
        self.opened = 0
        self.closed = 0

    def __call__(self) -> AbstractAsyncContextManager[Connector]:
        return self._open()

    @asynccontextmanager
    async def _open(self) -> AsyncIterator[Connector]:
        self.opened += 1
        if self.error is not None:
            raise self.error
        assert self.connector is not None
        try:
            yield self.connector
        finally:
            self.closed += 1


class ScriptedProvider:
    """An LLM provider that answers each request with the next scripted reply."""

    name = "scripted"

    def __init__(self, *replies: str | Callable[[LlmRequest], str]) -> None:
        self.replies = list(replies)
        self.requests: list[LlmRequest] = []

    async def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        reply = self.replies.pop(0)
        text = reply(request) if callable(reply) else reply
        return LlmResponse(text=text, model="scripted-1", input_tokens=100, output_tokens=50)
