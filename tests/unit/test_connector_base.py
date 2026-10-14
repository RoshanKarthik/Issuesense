from typing import Any

import pytest
from pydantic import ValidationError

from packages.common.errors import AuthorizationError
from packages.common.models import AuthorizationContext
from packages.connectors.base import (
    AllowlistedToolClient,
    Connector,
    RetrievalRequest,
    SourceRecord,
    ToolResult,
    read_scope,
)


class FakeMcpClient:
    def __init__(self, tools: list[str]) -> None:
        self.tools = tools
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> dict[str, dict[str, Any]]:
        return {name: {"type": "object"} for name in self.tools}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append((name, arguments))
        return ToolResult(tool=name, content={"ok": True})


class FakeConnector(Connector):
    source = "jira"

    def __init__(self, extra_resources: list[str] | None = None) -> None:
        self.fetched: list[dict[str, str]] = []
        self.extra_resources = extra_resources or []

    async def _fetch(self, request: RetrievalRequest, grants: dict[str, str]) -> list[SourceRecord]:
        self.fetched.append(grants)
        resources = [*grants, *self.extra_resources]
        return [
            SourceRecord(
                source=self.source,
                resource=resource,
                tool="search",
                payload={"key": f"{resource}-1"},
                permission_context=grants.get(resource, "none"),
            )
            for resource in resources
        ]


def _caller(*scopes: str) -> AuthorizationContext:
    return AuthorizationContext(user_id="alice", scopes=list(scopes))


def _request(*resources: str) -> RetrievalRequest:
    return RetrievalRequest(query="what changed?", resources=list(resources))


def test_read_scope_format() -> None:
    assert read_scope("jira", "ATLAS") == "jira:read:ATLAS"


def test_exact_scope_grants_its_resource() -> None:
    assert FakeConnector().authorize(_caller("jira:read:ATLAS"), _request("ATLAS")) == {
        "ATLAS": "jira:read:ATLAS"
    }


def test_wildcard_scope_grants_every_resource_of_its_source() -> None:
    grants = FakeConnector().authorize(_caller("jira:read:*"), _request("ATLAS", "PAY"))
    assert grants == {"ATLAS": "jira:read:*", "PAY": "jira:read:*"}


def test_exact_scope_is_recorded_in_preference_to_wildcard() -> None:
    grants = FakeConnector().authorize(_caller("jira:read:*", "jira:read:ATLAS"), _request("ATLAS"))
    assert grants == {"ATLAS": "jira:read:ATLAS"}


def test_missing_scope_names_every_denied_resource() -> None:
    with pytest.raises(AuthorizationError) as excinfo:
        FakeConnector().authorize(_caller("jira:read:ATLAS"), _request("ATLAS", "PAY", "HR"))
    assert excinfo.value.details == {"source": "jira", "resources": ["PAY", "HR"]}


async def test_retrieve_checks_access_before_fetching() -> None:
    connector = FakeConnector()
    with pytest.raises(AuthorizationError):
        await connector.retrieve(_caller(), _request("ATLAS"))
    assert connector.fetched == []


async def test_retrieve_passes_grants_and_tags_records() -> None:
    result = await FakeConnector().retrieve(_caller("jira:read:ATLAS"), _request("ATLAS"))
    assert result.source == "jira"
    assert [(r.resource, r.permission_context) for r in result.records] == [
        ("ATLAS", "jira:read:ATLAS")
    ]


async def test_records_for_ungranted_resources_are_dropped() -> None:
    connector = FakeConnector(extra_resources=["HR"])
    result = await connector.retrieve(_caller("jira:read:ATLAS"), _request("ATLAS"))
    assert [r.resource for r in result.records] == ["ATLAS"]


@pytest.mark.parametrize("resources", [[], ["ATLAS"] * 21])
def test_request_bounds_the_number_of_resources(resources: list[str]) -> None:
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", resources=resources)


async def test_allowlisted_tool_is_forwarded() -> None:
    fake = FakeMcpClient(["search", "get_issue"])
    client = AllowlistedToolClient(fake, "atlassian", frozenset({"search"}))
    result = await client.call_tool("search", {"jql": "project = ATLAS"})
    assert result.content == {"ok": True}
    assert fake.calls == [("search", {"jql": "project = ATLAS"})]


async def test_missing_tools_reports_allowlisted_tools_the_server_lacks() -> None:
    client = AllowlistedToolClient(
        FakeMcpClient(["search"]), "atlassian", frozenset({"search", "get_issue"})
    )
    assert await client.missing_tools() == {"get_issue"}


def test_an_empty_allowlist_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="at least one"):
        AllowlistedToolClient(FakeMcpClient([]), "atlassian", frozenset())
