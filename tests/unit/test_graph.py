import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date

import pytest

from packages.agent.context import RunContext
from packages.agent.graph import build_graph
from packages.agent.pipeline import QueryPipeline
from packages.agent.planner import LlmPlanner
from packages.agent.state import GraphState
from packages.agent.synthesis import GroundedSynthesizer
from packages.common.errors import ConnectorError
from packages.common.models import Answerability, AuthorizationContext
from packages.connectors.base import Connector
from packages.connectors.lazy import LazyConnector
from tests.unit.agent_fakes import SITE, Opener, ScriptedProvider, StaticJiraConnector, issue

READER = AuthorizationContext(user_id="reader", scopes=["jira:read:ATLAS"])
ISSUES = [
    issue("ATLAS-3", "Integrate the card authorization API", "Blocked by the vendor outage."),
    issue("ATLAS-4", "Load test checkout", "Waiting for card authorization."),
]


def _pipeline(opener: Opener | None, provider: ScriptedProvider | None = None) -> QueryPipeline:
    return QueryPipeline(
        open_jira=opener,
        site_url=SITE,
        projects=["ATLAS"],
        synthesizer=GroundedSynthesizer(provider),
    )


async def _visited(opener: Opener, provider: ScriptedProvider | None, question: str) -> list[str]:
    jira = LazyConnector(opener, "jira")
    context = RunContext(
        jira=jira, site_url=SITE, projects=["ATLAS"], synthesizer=GroundedSynthesizer(provider)
    )
    nodes: list[str] = []
    try:
        async for update in build_graph().astream(
            GraphState(question=question, caller=READER), context=context, stream_mode="updates"
        ):
            nodes.extend(update)
    finally:
        await jira.aclose()
    return nodes


def test_graph_has_the_planned_nodes() -> None:
    nodes = set(build_graph().get_graph().nodes)
    assert {"authorize", "plan", "retrieve", "assess", "synthesize", "verify", "finish"} <= nodes


async def test_answer_runs_every_node_in_order() -> None:
    opener = Opener(StaticJiraConnector(ISSUES))
    visited = await _visited(opener, None, "What blocks the card authorization work?")
    assert visited == ["authorize", "plan", "retrieve", "assess", "synthesize", "verify", "finish"]


async def test_unverifiable_llm_answer_loops_back_to_quoting() -> None:
    invented = json.dumps(
        {
            "answer": "Card authorization slipped 9 weeks [E1].",
            "answerability": "SUFFICIENT",
            "claims": [{"text": "Card authorization slipped 9 weeks", "evidence_ids": ["E1"]}],
        }
    )
    opener = Opener(StaticJiraConnector(ISSUES))
    visited = await _visited(
        opener, ScriptedProvider(invented), "What blocks the card authorization work?"
    )
    assert visited == [
        "authorize",
        "plan",
        "retrieve",
        "assess",
        "synthesize",
        "verify",
        "synthesize",
        "verify",
        "finish",
    ]


async def test_quoted_answer_is_verified_and_cites_returned_evidence() -> None:
    result = await _pipeline(Opener(StaticJiraConnector(ISSUES))).answer(
        "What blocks the card authorization work?", READER
    )
    assert result.answer.answerability is Answerability.PARTIALLY_ANSWERABLE
    assert result.verification is not None and result.verification.verified
    cited = {c.evidence_id for c in result.answer.citations}
    assert cited and cited <= {item.evidence_id for item in result.evidence}
    assert result.projects == ["ATLAS"]


async def test_unauthorized_question_never_opens_the_connector() -> None:
    opener = Opener(StaticJiraConnector(ISSUES))
    outsider = AuthorizationContext(user_id="outsider", scopes=["jira:read:OTHER"])
    result = await _pipeline(opener).answer("What changed in ATLAS?", outsider)
    assert result.answer.answerability is Answerability.UNAUTHORIZED_EVIDENCE
    assert opener.opened == 0


async def test_unconfigured_jira_is_a_tool_failure() -> None:
    result = await _pipeline(None).answer("What changed?", READER)
    assert result.answer.answerability is Answerability.TOOL_FAILURE


async def test_connection_failure_is_a_tool_failure() -> None:
    opener = Opener(error=ConnectorError("Could not connect to the source.", {}))
    result = await _pipeline(opener).answer("What changed?", READER)
    assert result.answer.answerability is Answerability.TOOL_FAILURE
    assert result.evidence == []


async def test_connector_is_opened_once_and_closed_after_the_run() -> None:
    opener = Opener(StaticJiraConnector(ISSUES))
    await _pipeline(opener).answer("What blocks the card authorization work?", READER)
    assert (opener.opened, opener.closed) == (1, 1)


async def test_lazy_connector_closes_in_the_task_that_opened_it() -> None:
    tasks: list[asyncio.Task[object] | None] = []

    @asynccontextmanager
    async def open_tracked() -> AsyncIterator[Connector]:
        tasks.append(asyncio.current_task())
        yield StaticJiraConnector([])
        tasks.append(asyncio.current_task())

    lazy = LazyConnector(open_tracked, "jira")
    await asyncio.create_task(lazy.get())  # first use from another task, as a graph node would
    await lazy.get()
    await lazy.aclose()
    assert len(tasks) == 2 and tasks[0] is tasks[1]


async def test_lazy_connector_wraps_unexpected_open_errors() -> None:
    lazy = LazyConnector(Opener(error=RuntimeError("socket details")), "jira")
    with pytest.raises(ConnectorError) as raised:
        await lazy.get()
    assert "socket" not in raised.value.message
    with pytest.raises(ConnectorError):
        await lazy.get()  # the failure is cached for the run, not retried silently
    await lazy.aclose()


async def test_the_plan_is_part_of_the_result() -> None:
    pipeline = QueryPipeline(
        open_jira=Opener(StaticJiraConnector(ISSUES)),
        site_url=SITE,
        projects=["ATLAS"],
        synthesizer=GroundedSynthesizer(None),
        today=lambda: date(2026, 10, 15),
    )
    result = await pipeline.answer("What changed this quarter, and what are the risks?", READER)
    assert result.plan is not None and result.plan.planner == "rules"
    assert [q.id for q in result.plan.sub_questions] == ["Q1", "Q2"]
    assert result.plan.time_range is not None


async def test_unauthorized_question_is_never_planned() -> None:
    provider = ScriptedProvider()  # any LLM call would fail: no replies are scripted
    pipeline = QueryPipeline(
        open_jira=Opener(StaticJiraConnector(ISSUES)),
        site_url=SITE,
        projects=["ATLAS"],
        synthesizer=GroundedSynthesizer(None),
        planner=LlmPlanner(provider),
    )
    outsider = AuthorizationContext(user_id="outsider", scopes=[])
    result = await pipeline.answer("What changed?", outsider)
    assert result.answer.answerability is Answerability.UNAUTHORIZED_EVIDENCE
    assert result.plan is None and provider.requests == []
