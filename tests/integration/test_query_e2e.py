"""End-to-end: HTTP, auth, guardrails, Jira over MCP (fake server), answer, verification."""

import json
from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

from packages.agent.llm import LlmRequest, LlmResponse
from packages.agent.synthesis import DETERMINISTIC_INTRO, FALLBACK_INTRO
from tests.integration.fake_atlassian import SITE, FakeJiraMcp

HEADLINE = "What changed in Project Atlas this quarter, and what are the major risks?"
InstallFake = Callable[..., FakeJiraMcp]


class ScriptedProvider:
    """A fake LLM that answers with a reply built from the prompt it receives."""

    name = "scripted"

    def __init__(self, build: Callable[[str], dict[str, Any]]) -> None:
        self.build = build
        self.prompts: list[str] = []

    async def complete(self, request: LlmRequest) -> LlmResponse:
        user = request.messages[-1].content
        self.prompts.append(user)
        return LlmResponse(text=json.dumps(self.build(user)), input_tokens=900, output_tokens=120)


def _evidence_id(prompt: str, location: str) -> str:
    marker = f'location="{location}"'
    head = prompt[: prompt.index(marker)]
    return head[head.rindex('id="') + 4 :].split('"')[0]


def _ask(client: TestClient, question: str = HEADLINE) -> dict[str, Any]:
    response = client.post("/v1/query", json={"question": question})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def test_headline_question_returns_a_verified_cited_answer(
    client: TestClient, use_fake_jira: InstallFake
) -> None:
    fake = use_fake_jira()
    body = _ask(client)
    answer = body["answer"]

    assert body["triage"] == "RETRIEVAL"
    assert answer["answerability"] == "PARTIALLY_ANSWERABLE"
    assert answer["text"].startswith(DETERMINISTIC_INTRO)
    assert body["verification"]["issues"] == []
    assert body["verification"]["supported_claims"] == len(answer["claims"]) > 0

    evidence = {item["evidence_id"]: item for item in body["evidence"]}
    assert {c["evidence_id"] for c in answer["citations"]} == set(evidence)
    for item in evidence.values():
        assert item["source"] == "jira" and item["permission_context"] == "jira:read:ATLAS"
        assert item["url"].startswith(f"{SITE}/browse/ATLAS-")
    cited_issues = {item["location"].split("#")[0] for item in evidence.values()}
    assert fake.keys["atlas-card-auth"] in cited_issues
    assert fake.keys["atlas-double-charge"] in cited_issues
    # The planted ticket is never cited, and the answer says it was withheld.
    assert fake.keys["atlas-feedback-import"] not in cited_issues
    assert "on track with no risks" not in answer["text"]
    assert fake.keys["atlas-feedback-import"] in answer["reasoning_summary"]
    assert fake.tools_called() <= {
        "searchJiraIssuesUsingJql",
        "listJiraIssueComments",
        "listJiraIssueChangelogs",
    }


def test_a_valid_llm_answer_is_returned_as_sufficient(
    client: TestClient, use_fake_jira: InstallFake
) -> None:
    fake = use_fake_jira()

    def reply(prompt: str) -> dict[str, Any]:
        slip = _evidence_id(prompt, f"{fake.keys['atlas-card-auth']}#change-30031")
        return {
            "answer": f"Card authorization slipped to Dec 5 [{slip}].",
            "answerability": "SUFFICIENT",
            "claims": [
                {
                    "text": "The due date changed from 2026-11-21 to 2026-12-05.",
                    "evidence_ids": [slip],
                }
            ],
            "reasoning_summary": "The changelog records the slip.",
        }

    use_fake_jira(provider=ScriptedProvider(reply))
    body = _ask(client, "Which deadlines slipped?")
    assert body["answer"]["answerability"] == "SUFFICIENT"
    assert body["answer"]["text"].startswith("Card authorization slipped to Dec 5")
    assert [e["location"] for e in body["evidence"]] == [
        f"{fake.keys['atlas-card-auth']}#change-30031"
    ]
    assert body["verification"]["issues"] == []


def test_a_hallucinated_date_is_replaced_by_quoted_evidence(
    client: TestClient, use_fake_jira: InstallFake
) -> None:
    fake = use_fake_jira()

    def reply(prompt: str) -> dict[str, Any]:
        slip = _evidence_id(prompt, f"{fake.keys['atlas-card-auth']}#change-30031")
        return {
            "answer": f"It slipped to Jan 15 [{slip}].",
            "answerability": "SUFFICIENT",
            "claims": [{"text": "The due date moved to Jan 15.", "evidence_ids": [slip]}],
        }

    use_fake_jira(provider=ScriptedProvider(reply))
    body = _ask(client, "Which deadlines slipped?")
    assert body["answer"]["text"].startswith(FALLBACK_INTRO)
    assert "Jan 15" not in body["answer"]["text"]
    assert body["answer"]["answerability"] == "PARTIALLY_ANSWERABLE"
    assert body["verification"]["issues"] == []


def test_a_question_nothing_matches_is_insufficient(
    client: TestClient, use_fake_jira: InstallFake
) -> None:
    use_fake_jira()
    body = _ask(client, "What is the quarterly marketing budget for Kubernetes?")
    assert body["answer"]["answerability"] == "INSUFFICIENT_EVIDENCE"
    assert body["answer"]["citations"] == [] and body["evidence"] == []


def test_an_empty_project_is_insufficient(client: TestClient, use_fake_jira: InstallFake) -> None:
    fake = use_fake_jira()
    fake.issues.clear()
    body = _ask(client)
    assert body["answer"]["answerability"] == "INSUFFICIENT_EVIDENCE"


def test_a_jira_failure_is_an_explicit_tool_failure(
    client: TestClient, use_fake_jira: InstallFake
) -> None:
    fake = use_fake_jira()
    fake.failing_tools.add("searchJiraIssuesUsingJql")
    response = client.post("/v1/query", json={"question": HEADLINE})
    body = response.json()
    assert response.status_code == 200
    assert body["answer"]["answerability"] == "TOOL_FAILURE"
    assert body["evidence"] == []
    assert "jira-db-7" not in response.text


def test_a_schema_mismatch_on_the_server_is_a_tool_failure(
    client: TestClient, use_fake_jira: InstallFake
) -> None:
    fake = use_fake_jira(rename_issue_arg=True)
    assert _ask(client)["answer"]["answerability"] == "TOOL_FAILURE"
    assert fake.calls == []


def test_a_named_project_the_caller_cannot_read_is_skipped(
    client: TestClient, use_fake_jira: InstallFake
) -> None:
    fake = use_fake_jira(projects=["ATLAS", "HR"])
    body = _ask(client, "What changed in ATLAS and in HR this quarter?")
    assert "Skipped Jira project(s) you cannot read: HR." in body["answer"]["reasoning_summary"]
    assert body["answer"]["answerability"] == "PARTIALLY_ANSWERABLE"
    searched = fake.searches()
    assert searched and all(jql.startswith('project = "ATLAS"') for jql in searched)
    assert not any("HR" in jql or '"hr"' in jql for jql in searched)


def test_issue_keys_of_unconfigured_projects_are_not_treated_as_projects(
    client: TestClient, use_fake_jira: InstallFake
) -> None:
    use_fake_jira()
    body = _ask(client, "Is vendor ticket PV2-4471 still blocking card authorization?")
    assert body["answer"]["answerability"] == "PARTIALLY_ANSWERABLE"
    assert any("PV2-4471" in e["content"] for e in body["evidence"])
