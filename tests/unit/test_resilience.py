import asyncio
from datetime import date

import pytest

from packages.agent.pipeline import PipelineResult, QueryPipeline
from packages.agent.planner import QueryPlan, SubQuestion
from packages.agent.synthesis import GroundedSynthesizer
from packages.common.errors import AuthorizationError, ConnectorError
from packages.common.models import Answerability, AuthorizationContext
from packages.connectors.base import ToolFailure
from packages.connectors.resilience import RetryPolicy, call_with_retry
from tests.unit.agent_fakes import SITE, Opener, StaticJiraConnector, issue

FAST = RetryPolicy(attempts=3, timeout_seconds=0.05, backoff_seconds=0)

# --- call_with_retry --------------------------------------------------------------------------


async def test_a_call_that_recovers_is_retried_until_it_succeeds() -> None:
    attempts = []

    async def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectorError("The source did not answer.")
        return "ok"

    assert await call_with_retry(flaky, FAST, "test") == "ok"
    assert len(attempts) == 3


async def test_timeouts_are_retried_then_reported_safely() -> None:
    attempts = []

    async def hangs() -> str:
        attempts.append(1)
        await asyncio.sleep(5)
        return "never"

    with pytest.raises(ConnectorError, match="did not answer in time") as raised:
        await call_with_retry(hangs, FAST, "Jira search")
    assert len(attempts) == 3
    assert raised.value.details == {"call": "Jira search"}


async def test_authorization_errors_are_never_retried() -> None:
    attempts = []

    async def denied() -> str:
        attempts.append(1)
        raise AuthorizationError("Not authorized.")

    with pytest.raises(AuthorizationError):
        await call_with_retry(denied, FAST, "test")
    assert len(attempts) == 1


def test_policies_are_bounded() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(attempts=50)


# --- partial results in the graph -------------------------------------------------------------

READER = AuthorizationContext(user_id="reader", scopes=["jira:read:ATLAS"])
ISSUES = [
    issue(
        "ATLAS-3", "Integrate card authorization", "The vendor outage blocks card authorization."
    ),
    issue("ATLAS-7", "Duplicate charge bug", "A risk: two clicks on Pay create two charges."),
]
TWO_PARTS = [
    SubQuestion(id="Q1", text="What blocks card authorization?", terms=["authorization"]),
    SubQuestion(id="Q2", text="Which charge risk is open?", terms=["charge"]),
]


class FixedPlanner:
    async def plan(self, question: str, today: date, projects: list[str]) -> QueryPlan:
        return QueryPlan(
            question=question, rewritten=question, sub_questions=TWO_PARTS, planner="rules"
        )


async def _ask(connector: StaticJiraConnector, question: str) -> PipelineResult:
    pipeline = QueryPipeline(
        open_jira=Opener(connector),
        site_url=SITE,
        projects=["ATLAS"],
        synthesizer=GroundedSynthesizer(None),
        planner=FixedPlanner(),
        retrieval_timeout=0.2,
    )
    return await pipeline.answer(question, READER)


QUESTION = "What blocks card authorization, and which charge risk is open?"


async def test_one_failed_search_keeps_the_other_searches_evidence() -> None:
    result = await _ask(StaticJiraConnector(ISSUES, fail_calls={2}), QUESTION)
    failed = [r for r in result.retrievals if r.error]
    assert [(r.task.sub_question, r.error) for r in failed] == [
        ("Q2", "Jira could not be reached.")
    ]
    assert result.answer.answerability is Answerability.PARTIALLY_ANSWERABLE
    assert any(item.location == "ATLAS-3" for item in result.evidence)
    summary = result.answer.reasoning_summary or ""
    assert "Jira searches failed" in summary and "may be incomplete" in summary


async def test_one_slow_search_times_out_without_losing_the_others() -> None:
    result = await _ask(StaticJiraConnector(ISSUES, slow_calls={1}), QUESTION)
    assert [r.error for r in result.retrievals if r.round == 1] == [
        "Jira did not answer in time.",
        None,
    ]
    assert any(item.location == "ATLAS-7" for item in result.evidence)
    assert result.answer.answerability is Answerability.PARTIALLY_ANSWERABLE


async def test_missing_details_are_noted_in_the_answer() -> None:
    failure = ToolFailure(
        tool="listJiraIssueChangelogs",
        resource="ATLAS",
        item="ATLAS-3",
        reason="Jira did not return this issue's history.",
    )
    result = await _ask(StaticJiraConnector(ISSUES, failures=[failure]), QUESTION)
    assert all(r.error is None for r in result.retrievals)
    assert "ATLAS-3: Jira did not return this issue's history." in (
        result.answer.reasoning_summary or ""
    )


async def test_nothing_found_while_searches_failed_is_a_tool_failure() -> None:
    # The working search finds nothing relevant; the answer may be in the failed one.
    unrelated = [issue("ATLAS-9", "Update the runbook", "Dashboards and alerts.")]
    result = await _ask(StaticJiraConnector(unrelated, fail_calls={1}), QUESTION)
    assert result.answer.answerability is Answerability.TOOL_FAILURE
    assert "Jira searches failed" in (result.answer.reasoning_summary or "")


async def test_every_search_failing_is_a_tool_failure_with_no_evidence() -> None:
    result = await _ask(StaticJiraConnector(ISSUES, fail_calls={1, 2, 3, 4}), QUESTION)
    assert result.answer.answerability is Answerability.TOOL_FAILURE
    assert result.evidence == []
