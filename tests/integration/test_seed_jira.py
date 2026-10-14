import json
import re
from typing import Any

import httpx
import pytest

from packages.evaluation.atlas_dataset import ATLAS_ISSUES, IssueType, issue_by_ref
from packages.evaluation.seed_jira import SEED_LABEL, JiraRestClient, SeedError, adf, seed

SITE = "https://atlas-demo.atlassian.net"


class FakeJira:
    """Just enough of the Jira Cloud REST API for the seeder, with its state in memory."""

    def __init__(self, transitions: tuple[str, ...] = ("To Do", "In Progress", "Done")) -> None:
        self.issues: dict[str, dict[str, Any]] = {}
        self.comments: dict[str, list[dict[str, Any]]] = {}
        self.due_history: dict[str, list[str]] = {}
        self.statuses: dict[str, str] = {}
        self.transitions = transitions
        self.writes = 0
        self.auth_headers: set[str] = set()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.auth_headers.add(request.headers.get("authorization", ""))
        path, method = request.url.path, request.method
        if method != "GET":
            self.writes += 1
        if path == "/rest/api/3/search/jql" and method == "GET":
            project = re.search(r'project = "(\w+)"', request.url.params["jql"])
            assert project is not None
            matching = [
                {"key": key, "fields": {"labels": fields["labels"]}}
                for key, fields in self.issues.items()
                if fields["project"]["key"] == project.group(1) and SEED_LABEL in fields["labels"]
            ]
            return httpx.Response(200, json={"issues": matching, "isLast": True})
        if path == "/rest/api/3/issue" and method == "POST":
            fields = json.loads(request.content)["fields"]
            if "parent" in fields:
                assert fields["parent"]["key"] in self.issues, "parent must exist first"
            key = f"{fields['project']['key']}-{len(self.issues) + 1}"
            self.issues[key] = fields
            self.statuses[key] = "To Do"
            self.due_history[key] = [fields["duedate"]] if "duedate" in fields else []
            return httpx.Response(201, json={"key": key})
        match = re.fullmatch(r"/rest/api/3/issue/([A-Z]+-\d+)(/\w+)?", path)
        assert match, f"unexpected {method} {path}"
        key, sub = match.group(1), match.group(2)
        if key not in self.issues:
            return httpx.Response(404, json={"errorMessages": ["Issue does not exist"]})
        if sub == "/comment" and method == "POST":
            self.comments.setdefault(key, []).append(json.loads(request.content)["body"])
            return httpx.Response(201, json={"id": "1"})
        if sub is None and method == "PUT":
            due = json.loads(request.content)["fields"]["duedate"]
            self.issues[key]["duedate"] = due
            self.due_history[key].append(due)
            return httpx.Response(204)
        if sub == "/transitions" and method == "GET":
            options = [{"id": str(i), "to": {"name": n}} for i, n in enumerate(self.transitions)]
            return httpx.Response(200, json={"transitions": options})
        if sub == "/transitions" and method == "POST":
            chosen = int(json.loads(request.content)["transition"]["id"])
            self.statuses[key] = self.transitions[chosen]
            return httpx.Response(204)
        raise AssertionError(f"unexpected {method} {path}")

    def client(self) -> JiraRestClient:
        return JiraRestClient.connect(
            SITE, "owner@example.com", "test-token", transport=httpx.MockTransport(self.handler)
        )

    def key_for(self, ref: str) -> str:
        return next(k for k, f in self.issues.items() if f"seed-{ref}" in f["labels"])


def test_adf_makes_one_paragraph_per_non_empty_line() -> None:
    doc = adf("First line\n\nSecond line")
    assert doc["type"] == "doc" and doc["version"] == 1
    assert [p["content"][0]["text"] for p in doc["content"]] == ["First line", "Second line"]


async def test_preview_writes_nothing() -> None:
    jira = FakeJira()
    report = await seed(jira.client(), "ATLAS", ATLAS_ISSUES, apply=False)
    assert jira.writes == 0
    assert not report.applied
    assert set(report.created) == {issue.ref for issue in ATLAS_ISSUES}
    assert any("move atlas-card-auth due date to 2026-12-05" in a for a in report.actions)


async def test_apply_creates_the_whole_story() -> None:
    jira = FakeJira()
    report = await seed(jira.client(), "ATLAS", ATLAS_ISSUES, apply=True)
    assert len(jira.issues) == len(ATLAS_ISSUES) == len(report.created)

    card_auth = jira.key_for("atlas-card-auth")
    assert jira.due_history[card_auth] == ["2026-11-07", "2026-11-21", "2026-12-05"]
    assert jira.statuses[card_auth] == "In Progress"
    assert len(jira.comments[card_auth]) == 2
    assert jira.issues[card_auth]["parent"]["key"] == jira.key_for("atlas-epic-checkout")
    assert {"blocker", SEED_LABEL, "seed-atlas-card-auth"} <= set(jira.issues[card_auth]["labels"])
    assert jira.statuses[jira.key_for("atlas-refunds")] == "Done"
    assert jira.statuses[jira.key_for("atlas-runbook")] == "To Do"
    epics = [k for k, f in jira.issues.items() if f["issuetype"]["name"] == IssueType.EPIC.value]
    assert len(epics) == 2
    assert jira.auth_headers == {httpx.BasicAuth("owner@example.com", "test-token")._auth_header}


async def test_reseeding_is_idempotent() -> None:
    jira = FakeJira()
    await seed(jira.client(), "ATLAS", ATLAS_ISSUES, apply=True)
    writes = jira.writes
    report = await seed(jira.client(), "ATLAS", ATLAS_ISSUES, apply=True)
    assert jira.writes == writes
    assert report.created == {}
    assert len(report.skipped) == len(ATLAS_ISSUES)


async def test_partial_seed_resumes_with_existing_parents() -> None:
    jira = FakeJira()
    epic = issue_by_ref("atlas-epic-checkout")
    await seed(jira.client(), "ATLAS", [epic], apply=True)
    child = issue_by_ref("atlas-runbook")
    report = await seed(jira.client(), "ATLAS", [epic, child], apply=True)
    assert list(report.created) == ["atlas-runbook"]
    assert jira.issues[report.created["atlas-runbook"]]["parent"]["key"] == jira.key_for(epic.ref)


async def test_missing_transition_is_an_explicit_error() -> None:
    jira = FakeJira(transitions=("To Do", "Backlog"))
    with pytest.raises(SeedError, match="no transition to 'In Progress'"):
        await seed(jira.client(), "ATLAS", [issue_by_ref("atlas-epic-checkout")], apply=True)


async def test_http_errors_surface_with_jira_message() -> None:
    def deny(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"errorMessages": ["You do not have permission"]})

    client = JiraRestClient.connect(SITE, "o@example.com", "t", transport=httpx.MockTransport(deny))
    with pytest.raises(SeedError, match="HTTP 403.*permission"):
        await seed(client, "ATLAS", ATLAS_ISSUES, apply=False)


def test_dataset_refs_are_unique_and_parents_exist() -> None:
    refs = [issue.ref for issue in ATLAS_ISSUES]
    assert len(refs) == len(set(refs))
    assert {i.parent_ref for i in ATLAS_ISSUES if i.parent_ref} <= set(refs)
    assert sum(issue.adversarial for issue in ATLAS_ISSUES) == 1


def test_dataset_slips_are_later_than_the_original_due_date() -> None:
    for issue in ATLAS_ISSUES:
        dates = [d for d in [issue.due, *issue.due_changes] if d]
        assert dates == sorted(dates), issue.ref
