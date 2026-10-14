import asyncio
import json
from datetime import date

import pytest
from langsmith.run_helpers import get_tracing_context

from packages.agent.pipeline import QueryPipeline
from packages.agent.planner import QueryPlan
from packages.agent.synthesis import GroundedSynthesizer
from packages.common.models import AuthorizationContext
from packages.common.settings import Settings
from packages.observability.langsmith import LangSmithTracing, disabled
from packages.observability.trace import TraceRecorder
from tests.unit.agent_fakes import SITE, Opener, StaticJiraConnector, issue

READER = AuthorizationContext(user_id="reader", scopes=["jira:read:ATLAS"])
SECRET_TEXT = "The vendor sandbox returns HTTP 503 on most authorize calls"
ISSUES = [
    issue("ATLAS-3", "Integrate card authorization", f"{SECRET_TEXT}. Parent: ATLAS-1"),
    issue("ATLAS-1", "Checkout migration epic", "Target launch Nov 30."),
]
NODES = ["authorize", "plan", "retrieve", "assess", "synthesize", "verify", "finish"]


def _pipeline(**overrides: object) -> QueryPipeline:
    values: dict[str, object] = {
        "open_jira": Opener(StaticJiraConnector(ISSUES)),
        "site_url": SITE,
        "projects": ["ATLAS"],
        "synthesizer": GroundedSynthesizer(None),
        "today": lambda: date(2026, 10, 15),
    }
    values.update(overrides)
    return QueryPipeline(**values)  # type: ignore[arg-type]


# --- the recorder -------------------------------------------------------------------------------


def test_events_are_numbered_in_order() -> None:
    trace = TraceRecorder("q1", "alice")
    first = trace.emit("plan", "node.started")
    second = trace.emit("plan", "node.completed", {"planner": "rules"}, 12.3456)
    assert (first.event_id, second.event_id) == ("1", "2")
    assert second.duration_ms == 12.35 and second.payload == {"planner": "rules"}


async def test_follow_replays_then_streams_live_until_closed() -> None:
    trace = TraceRecorder("q1", "alice")
    trace.emit("query", "query.started")
    seen: list[str] = []

    async def watch() -> None:
        async for event in trace.follow():
            seen.append(event.event_type)

    watcher = asyncio.create_task(watch())
    await asyncio.sleep(0)
    trace.emit("plan", "node.started")
    trace.emit("plan", "node.completed")
    trace.close("completed")
    await asyncio.wait_for(watcher, 1)
    assert seen == ["query.started", "node.started", "node.completed"]


async def test_follow_after_the_end_returns_the_rest_and_stops() -> None:
    trace = TraceRecorder("q1", "alice")
    for kind in ("a", "b", "c"):
        trace.emit("n", kind)
    trace.close("completed")
    assert [e.event_type async for e in trace.follow(after=1)] == ["b", "c"]
    trace.close("failed")  # a closed trace keeps its first status
    assert trace.status == "completed"


# --- traces of query runs -----------------------------------------------------------------------


async def test_every_node_emits_started_then_completed_in_order() -> None:
    trace = TraceRecorder("q1", "reader")
    await _pipeline().answer("What blocks card authorization?", READER, trace)
    node_events = [(e.node_id, e.event_type) for e in trace.events if e.node_id in NODES]
    expected = []
    for node in ["authorize", "plan", "retrieve", "assess", "retrieve", "assess"]:
        expected += [(node, "node.started"), (node, "node.completed")]
    for node in ["synthesize", "verify", "finish"]:
        expected += [(node, "node.started"), (node, "node.completed")]
    assert [e for e in node_events if e[1] != "retrieval.completed"] == expected
    assert trace.events[0].event_type == "query.started"
    assert trace.events[-1].event_type == "query.completed"
    assert trace.status == "completed"
    assert all(
        e.duration_ms is not None and e.duration_ms >= 0
        for e in trace.events
        if e.event_type == "node.completed"
    )


async def test_trace_shows_the_plan_searches_and_evidence_but_no_evidence_text() -> None:
    trace = TraceRecorder("q1", "reader")
    result = await _pipeline().answer("What blocks card authorization?", READER, trace)
    by_type: dict[str, list[dict[str, object]]] = {}
    for event in trace.events:
        by_type.setdefault(f"{event.node_id}:{event.event_type}", []).append(event.payload)

    plan = by_type["plan:node.completed"][0]
    assert plan["planner"] == "rules" and plan["sub_questions"]
    searches = by_type["retrieve:retrieval.completed"]
    assert searches[0]["kind"] == "search" and searches[0]["items"] == 1
    assert any(s["kind"] == "follow" for s in searches)  # ATLAS-1 is referenced
    retrieved = by_type["retrieve:node.completed"][0]["evidence"]
    assert {"evidence_id", "location", "title", "url"} <= set(retrieved[0])  # type: ignore[index]
    verify = by_type["verify:node.completed"][0]
    assert verify["verified"] is True
    finish = by_type["finish:node.completed"][0]
    assert finish["answerability"] == result.answer.answerability.value

    dumped = json.dumps([e.model_dump(mode="json") for e in trace.events])
    assert SECRET_TEXT not in dumped


async def test_a_failing_node_is_traced_and_the_trace_closed() -> None:
    class BrokenPlanner:
        async def plan(self, question: str, today: date, projects: list[str]) -> QueryPlan:
            raise RuntimeError("planner bug")

    trace = TraceRecorder("q1", "reader")
    with pytest.raises(RuntimeError):
        await _pipeline(planner=BrokenPlanner()).answer("What changed?", READER, trace)
    failed = [e for e in trace.events if e.event_type == "node.failed"]
    assert [(e.node_id, e.payload) for e in failed] == [("plan", {"error": "RuntimeError"})]
    assert trace.events[-1].event_type == "query.failed"
    assert trace.status == "failed"


async def test_a_cancelled_run_closes_its_trace() -> None:
    trace = TraceRecorder("q1", "reader")
    connector = StaticJiraConnector(ISSUES, slow_calls={1})
    run = asyncio.create_task(
        _pipeline(open_jira=Opener(connector)).answer("What blocks it?", READER, trace)
    )
    await asyncio.sleep(0.05)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert trace.status == "cancelled"
    assert trace.events[-1].event_type == "query.cancelled"


def test_the_step_limit_grows_with_the_follow_up_budget() -> None:
    from packages.agent.state import Budget

    assert _pipeline().step_limit < _pipeline(budget=Budget(max_follow_ups=5)).step_limit


# --- LangSmith ----------------------------------------------------------------------------------


def test_langsmith_is_off_without_a_key_even_if_the_environment_says_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    tracing = LangSmithTracing(Settings(_env_file=None))
    assert not tracing.enabled
    with tracing.run("q1", "alice"):
        assert get_tracing_context()["enabled"] is False
    with disabled().run("q1", "alice"):
        assert get_tracing_context()["enabled"] is False


def test_langsmith_is_on_with_a_key_and_names_the_project_and_query() -> None:
    settings = Settings(
        _env_file=None,
        langsmith_api_key="lsv2_test_" + "k" * 30,
        langsmith_project="issuesense-demo",
        langsmith_endpoint="https://langsmith.example.test",
    )
    clients: list[StubClient] = []

    def make_client(**kwargs: object) -> StubClient:
        clients.append(StubClient(**kwargs))
        return clients[-1]

    tracing = LangSmithTracing(settings, client_factory=make_client)
    assert tracing.enabled
    with tracing.run("q1", "alice"):
        context = get_tracing_context()
        assert context["enabled"] is True
        assert context["project_name"] == "issuesense-demo"
        assert context["metadata"] == {"query_id": "q1", "user_id": "alice"}
        assert context["client"] is clients[0]
    assert clients[0].kwargs == {
        "api_key": "lsv2_test_" + "k" * 30,
        "api_url": "https://langsmith.example.test",
    }
    assert get_tracing_context()["enabled"] is not True


class StubClient:
    """Stands in for `langsmith.Client`, which contacts LangSmith as soon as it is created."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
