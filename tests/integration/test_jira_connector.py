from collections.abc import Callable

import pytest

from packages.common.errors import AuthorizationError, ConnectorError
from packages.common.models import AuthorizationContext
from packages.common.settings import Settings
from packages.connectors.base import RetrievalRequest, ToolNotAllowedError
from packages.connectors.jira import (
    JIRA_READ_TOOLS,
    check_jira,
    open_jira_connector,
    project_jql,
)
from packages.connectors.mcp_session import basic_auth_header, connect
from packages.connectors.resilience import RetryPolicy
from packages.evaluation.atlas_dataset import ATLAS_ISSUES
from tests.integration.fake_atlassian import SITE, FakeJiraMcp

ATLAS_READER = AuthorizationContext(user_id="alice", scopes=["jira:read:ATLAS"])


def _request(limit: int = 50) -> RetrievalRequest:
    return RetrievalRequest(query="What changed in Atlas?", resources=["ATLAS"], limit=limit)


def test_basic_auth_header_encodes_email_and_token() -> None:
    assert basic_auth_header("owner@example.com", "tok") == {
        "Authorization": "Basic b3duZXJAZXhhbXBsZS5jb206dG9r"
    }


def test_jql_contains_only_the_project_key() -> None:
    assert project_jql("ATLAS") == 'project = "ATLAS" ORDER BY updated DESC'


def test_only_read_tools_are_allowlisted() -> None:
    assert JIRA_READ_TOOLS == {
        "searchJiraIssuesUsingJql",
        "listJiraIssueComments",
        "listJiraIssueChangelogs",
    }


async def test_retrieves_every_issue_with_comments_and_changelog(
    fake_jira: FakeJiraMcp, jira_settings: Callable[..., Settings]
) -> None:
    async with open_jira_connector(jira_settings(), fake_jira.server()) as jira:
        result = await jira.retrieve(ATLAS_READER, _request())

    assert len(result.records) == len(ATLAS_ISSUES)
    assert {r.permission_context for r in result.records} == {"jira:read:ATLAS"}
    card_auth = next(
        r.payload
        for r in result.records
        if r.payload["issue"]["key"] == fake_jira.keys["atlas-card-auth"]
    )
    assert card_auth["issue"]["fields"]["duedate"] == "2026-12-05"
    assert len(card_auth["comments"]) == 2
    assert [item["items"][0]["toString"] for item in card_auth["changelog"]] == [
        "2026-11-21",
        "2026-12-05",
    ]
    search = next(args for tool, args in fake_jira.calls if tool == "searchJiraIssuesUsingJql")
    assert search == {"cloudId": SITE, "jql": project_jql("ATLAS"), "maxResults": 50}


async def test_details_are_fetched_only_for_the_first_issues(
    fake_jira: FakeJiraMcp, jira_settings: Callable[..., Settings]
) -> None:
    async with open_jira_connector(jira_settings(), fake_jira.server()) as jira:
        jira._detail_limit = 3
        result = await jira.retrieve(ATLAS_READER, _request())
    fetched = [r.payload["details_fetched"] for r in result.records]
    assert fetched == [True] * 3 + [False] * (len(ATLAS_ISSUES) - 3)
    detail_calls = [t for t, _ in fake_jira.calls if t != "searchJiraIssuesUsingJql"]
    assert len(detail_calls) == 3 * 2


async def test_unauthorized_caller_triggers_no_tool_calls(
    fake_jira: FakeJiraMcp, jira_settings: Callable[..., Settings]
) -> None:
    outsider = AuthorizationContext(user_id="mallory", scopes=["jira:read:PAY"])
    async with open_jira_connector(jira_settings(), fake_jira.server()) as jira:
        with pytest.raises(AuthorizationError):
            await jira.retrieve(outsider, _request())
    assert fake_jira.calls == []


async def test_issues_from_other_projects_are_dropped(
    fake_jira: FakeJiraMcp, jira_settings: Callable[..., Settings]
) -> None:
    # Simulate a server bug: an HR issue comes back from a search scoped to ATLAS.
    fake_jira.issues["ATLAS-0"] = {**next(iter(fake_jira.issues.values())), "key": "HR-1"}
    async with open_jira_connector(jira_settings(), fake_jira.server()) as jira:
        result = await jira.retrieve(ATLAS_READER, _request())
    assert all(r.payload["issue"]["key"].startswith("ATLAS-") for r in result.records)
    assert "HR-1" not in {r.payload["issue"]["key"] for r in result.records}


async def test_write_tools_on_the_server_are_unreachable(
    fake_jira: FakeJiraMcp, jira_settings: Callable[..., Settings]
) -> None:
    async with open_jira_connector(jira_settings(), fake_jira.server()) as jira:
        await jira.retrieve(ATLAS_READER, _request())
        for tool in ("createJiraIssue", "deleteJiraIssue"):
            with pytest.raises(ToolNotAllowedError):
                await jira._tools.call_tool(tool, {"cloudId": SITE, "issueIdOrKey": "ATLAS-1"})
        assert set(await jira._tools.tool_schemas()) == JIRA_READ_TOOLS
    assert fake_jira.tools_called() <= JIRA_READ_TOOLS


async def test_a_failed_detail_call_keeps_everything_else(
    fake_jira: FakeJiraMcp, jira_settings: Callable[..., Settings]
) -> None:
    fake_jira.failing_tools.add("listJiraIssueChangelogs")
    async with open_jira_connector(jira_settings(), fake_jira.server()) as jira:
        result = await jira.retrieve(ATLAS_READER, _request())
    assert len(result.records) == len(ATLAS_ISSUES)
    assert all(not r.payload["details_fetched"] for r in result.records)
    assert sum(len(r.payload["comments"]) for r in result.records) > 0  # comments survived
    assert {f.tool for f in result.failures} == {"listJiraIssueChangelogs"}
    assert len(result.failures) == len(ATLAS_ISSUES)
    assert "jira-db-7" not in str([f.model_dump() for f in result.failures])


async def test_a_failed_search_is_a_connector_error_without_internals(
    fake_jira: FakeJiraMcp, jira_settings: Callable[..., Settings]
) -> None:
    fake_jira.failing_tools.add("searchJiraIssuesUsingJql")
    async with open_jira_connector(jira_settings(), fake_jira.server()) as jira:
        with pytest.raises(ConnectorError) as excinfo:
            await jira.retrieve(ATLAS_READER, _request())
    assert excinfo.value.details == {"server": "atlassian"}
    assert "jira-db-7" not in str(excinfo.value)
    # The failing search was tried twice (the default retry policy), then given up.
    assert len(fake_jira.searches()) == 2


async def test_transient_failures_are_retried(
    fake_jira: FakeJiraMcp, jira_settings: Callable[..., Settings]
) -> None:
    fake_jira.flaky = {"searchJiraIssuesUsingJql": 1, "listJiraIssueComments": 3}
    async with open_jira_connector(jira_settings(), fake_jira.server()) as jira:
        jira._retry = RetryPolicy(attempts=2, backoff_seconds=0)
        result = await jira.retrieve(ATLAS_READER, _request())
    assert len(result.records) == len(ATLAS_ISSUES)
    assert len(fake_jira.searches()) == 2
    # Three comment calls failed once each and were retried; nothing is missing.
    assert result.failures == []


async def test_a_slow_call_times_out_and_is_reported(
    fake_jira: FakeJiraMcp, jira_settings: Callable[..., Settings]
) -> None:
    fake_jira.delays = {"listJiraIssueComments": 5.0}
    async with open_jira_connector(jira_settings(), fake_jira.server()) as jira:
        jira._retry = RetryPolicy(attempts=1, timeout_seconds=0.05)
        jira._detail_limit = 2
        result = await jira.retrieve(ATLAS_READER, _request())
    assert len(result.records) == len(ATLAS_ISSUES)
    assert [f.tool for f in result.failures] == ["listJiraIssueComments"] * 2


async def test_schema_mismatch_is_reported_before_any_retrieval(
    fake_jira: FakeJiraMcp, jira_settings: Callable[..., Settings]
) -> None:
    with pytest.raises(ConnectorError) as excinfo:
        async with open_jira_connector(jira_settings(), fake_jira.server(rename_issue_arg=True)):
            pytest.fail("connector must not open")
    problems = excinfo.value.details["problems"]
    assert "listJiraIssueComments: server does not accept issueIdOrKey, maxResults" in problems
    assert "listJiraIssueComments: server requires issueKey" in problems
    assert fake_jira.calls == []


async def test_missing_configuration_is_explicit(fake_jira: FakeJiraMcp) -> None:
    unconfigured = Settings(
        _env_file=None, jira_base_url=None, jira_email=None, jira_api_token=None
    )
    with pytest.raises(ConnectorError, match="not configured"):
        async with open_jira_connector(unconfigured, fake_jira.server()):
            pytest.fail("connector must not open")


async def test_unreachable_server_is_a_connector_error() -> None:
    with pytest.raises(ConnectorError) as excinfo:
        async with connect("http://127.0.0.1:9/mcp", server="atlassian", timeout=3):
            pytest.fail("must not connect")
    assert excinfo.value.details == {"server": "atlassian"}


async def test_errors_inside_the_session_pass_through_unchanged(fake_jira: FakeJiraMcp) -> None:
    with pytest.raises(KeyError):
        async with connect(fake_jira.server(), server="atlassian"):
            raise KeyError("caller bug")


async def test_check_jira_reports_tools_and_recent_issues(
    fake_jira: FakeJiraMcp, jira_settings: Callable[..., Settings]
) -> None:
    lines = await check_jira(jira_settings(), "ATLAS", fake_jira.server())
    assert lines[1] == (
        "Read tools available: listJiraIssueChangelogs, listJiraIssueComments, "
        "searchJiraIssuesUsingJql."
    )
    assert lines[2] == "Read 5 recent issues from ATLAS:"
    assert len(lines) == 3 + 5
