import pytest
from pydantic import ValidationError

from packages.common.errors import AuthorizationError, IssueSenseError
from packages.common.models import (
    Answerability,
    AuthorizationContext,
    Evidence,
    Query,
    TraceEvent,
)
from packages.common.settings import Settings


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ISSUESENSE_ENV",
        "ISSUESENSE_LOG_LEVEL",
        "ISSUESENSE_API_HOST",
        "ISSUESENSE_API_PORT",
    ):
        monkeypatch.delenv(name, raising=False)
    s = Settings(_env_file=None)
    assert s.env == "development"
    assert s.log_level == "INFO"
    assert s.api_host == "127.0.0.1"
    assert s.api_port == 8000


def test_settings_read_prefixed_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISSUESENSE_ENV", "production")
    monkeypatch.setenv("ISSUESENSE_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("ISSUESENSE_API_HOST", "0.0.0.0")
    monkeypatch.setenv("ISSUESENSE_API_PORT", "9000")
    s = Settings(_env_file=None)
    assert s.env == "production"
    assert s.log_level == "WARNING"
    assert s.api_host == "0.0.0.0"
    assert s.api_port == 9000


def test_settings_reject_unknown_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISSUESENSE_ENV", "staging")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_ignore_unprefixed_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ISSUESENSE_ENV", raising=False)
    monkeypatch.setenv("ENV", "production")
    assert Settings(_env_file=None).env == "development"


def test_error_hierarchy() -> None:
    with pytest.raises(IssueSenseError) as excinfo:
        raise AuthorizationError("Unauthorized access", {"scope": "read:repo"})
    assert excinfo.value.message == "Unauthorized access"
    assert excinfo.value.details == {"scope": "read:repo"}
    assert AuthorizationError("no details").details == {}


def test_evidence_model_validation() -> None:
    # Valid evidence
    ev = Evidence(
        evidence_id="E1",
        source="github",
        authority="implementation",
        location="repo/main.py#L1",
        content="print('hello')",
    )
    assert ev.evidence_id == "E1"

    # Invalid evidence (missing required fields)
    with pytest.raises(ValidationError):
        Evidence(evidence_id="E2")  # type: ignore[call-arg]


def test_authorization_context() -> None:
    ctx = AuthorizationContext(user_id="user123", scopes=["read:repo"])
    assert ctx.user_id == "user123"
    assert "read:repo" in ctx.scopes


def test_answerability_enum() -> None:
    assert Answerability.SUFFICIENT == "SUFFICIENT"
    assert {state.value for state in Answerability} >= {
        "INSUFFICIENT_EVIDENCE",
        "PARTIALLY_ANSWERABLE",
        "CONFLICTING_EVIDENCE",
        "TOOL_FAILURE",
        "UNAUTHORIZED_EVIDENCE",
    }


def test_default_timestamps_are_timezone_aware() -> None:
    query = Query(query_id="Q1", text="Where is auth handled?")
    event = TraceEvent(event_id="T1", node_id="triage", event_type="started")
    assert query.timestamp.utcoffset() is not None
    assert event.timestamp.utcoffset() is not None


def test_jira_settings_use_the_documented_env_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIRA_BASE_URL", "https://atlas-demo.atlassian.net/")
    monkeypatch.setenv("JIRA_EMAIL", "owner@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "test-token-value")
    monkeypatch.setenv("ISSUESENSE_JIRA_PROJECTS", '["ATLAS"]')
    s = Settings(_env_file=None)
    assert s.jira_base_url == "https://atlas-demo.atlassian.net"
    assert s.jira_email == "owner@example.com"
    assert s.jira_api_token is not None
    assert s.jira_api_token.get_secret_value() == "test-token-value"
    assert s.jira_projects == ["ATLAS"]
    assert s.jira_mcp_url == "https://mcp.atlassian.com/v2/mcp"
    assert s.jira_configured
    assert "test-token-value" not in repr(s)


def test_jira_is_unconfigured_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    assert not Settings(_env_file=None).jira_configured


def test_blank_jira_values_mean_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN"):
        monkeypatch.setenv(name, "")
    s = Settings(_env_file=None)
    assert s.jira_base_url is None
    assert not s.jira_configured


@pytest.mark.parametrize("url", ["http://atlas-demo.atlassian.net", "atlas-demo.atlassian.net"])
def test_jira_base_url_must_be_https(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("JIRA_BASE_URL", url)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
