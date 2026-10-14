"""Attempts to read data the caller was not granted, or to reach tools outside the allowlist."""

from typing import Any

import pytest
from pydantic import ValidationError

from packages.common.errors import AuthorizationError
from packages.common.models import AuthorizationContext
from packages.connectors.base import (
    AllowlistedToolClient,
    Connector,
    FetchResult,
    RetrievalRequest,
    ToolNotAllowedError,
    ToolResult,
)


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def list_tools(self) -> dict[str, dict[str, Any]]:
        return {name: {} for name in ["search", "createJiraIssue", "deleteComment"]}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append(name)
        return ToolResult(tool=name)


class _Jira(Connector):
    source = "jira"

    async def _fetch(self, request: RetrievalRequest, grants: dict[str, str]) -> FetchResult:
        return FetchResult()


@pytest.mark.parametrize(
    "resource", ["*", "ATLAS,PAY", "ATLAS PAY", "../ATLAS", "ATLAS:*", "", "-ATLAS", "A" * 65]
)
def test_malformed_resource_ids_are_rejected(resource: str) -> None:
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", resources=[resource])


@pytest.mark.parametrize(
    "scopes",
    [
        [],
        ["jira:read:ATLAS2"],  # a different project whose key starts the same
        ["jira:read:ATLA"],  # a prefix of the project key
        ["notion:read:ATLAS"],  # the right project, wrong source
        ["jira:write:ATLAS"],  # not a read scope
        ["jira:read:atlas"],  # resource IDs are case-sensitive
        ["jira:read:ATLAS "],  # trailing whitespace is not normalized into a match
        ["jira:*"],
        ["*"],
        ["notion:read:*"],
    ],
)
def test_near_miss_scopes_do_not_grant_access(scopes: list[str]) -> None:
    caller = AuthorizationContext(user_id="mallory", scopes=scopes)
    with pytest.raises(AuthorizationError):
        _Jira().authorize(caller, RetrievalRequest(query="q", resources=["ATLAS"]))


def test_one_denied_resource_denies_the_whole_request() -> None:
    caller = AuthorizationContext(user_id="mallory", scopes=["jira:read:ATLAS"])
    with pytest.raises(AuthorizationError):
        _Jira().authorize(caller, RetrievalRequest(query="q", resources=["ATLAS", "HR"]))


@pytest.mark.parametrize("tool", ["createJiraIssue", "deleteComment", "SEARCH", "search ", ""])
async def test_tools_outside_the_allowlist_never_reach_the_server(tool: str) -> None:
    recording = _RecordingClient()
    client = AllowlistedToolClient(recording, "atlassian", frozenset({"search"}))
    with pytest.raises(ToolNotAllowedError) as excinfo:
        await client.call_tool(tool, {})
    assert recording.calls == []
    assert excinfo.value.details == {"server": "atlassian", "tool": tool}
