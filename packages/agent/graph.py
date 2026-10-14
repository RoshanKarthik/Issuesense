"""The query workflow as a typed LangGraph graph.

    authorize -> plan -> retrieve -> assess -> synthesize -> verify -> finish
                                                    ^           |
                                                    +-----------+  (quote instead of an LLM answer
                                                                    that failed verification)

A node that settles the outcome early (not configured, unauthorized, nothing retrievable) sets
`answer` and the graph goes straight to `finish`. The guarantees of the Phase 1 pipeline hold:
access is checked before any retrieval, retrieved text is normalized into evidence before
synthesis, and no answer leaves without passing citation verification.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from packages.agent.planner import Planner, RulePlanner
from packages.agent.state import GraphState
from packages.agent.synthesis import GroundedSynthesizer
from packages.common.errors import ConnectorError
from packages.common.models import Answer, Answerability, utc_now
from packages.connectors.base import RetrievalRequest, grant_for
from packages.connectors.lazy import LazyConnector
from packages.evidence.citations import verify_answer
from packages.evidence.jira import normalize_jira_records
from packages.evidence.ranking import rank_evidence

logger = logging.getLogger(__name__)

SOURCE = "jira"
_ISSUE_KEY = re.compile(r"\b([A-Z][A-Z0-9_]+)-\d+\b")
_WORD = re.compile(r"\b[A-Z][A-Z0-9_]+\b")

NOT_CONFIGURED_TEXT = "Jira is not configured on this server, so there is no answer."
UNAVAILABLE_TEXT = "Jira could not be reached, so there is no answer. Please try again later."
UNVERIFIED_TEXT = "The answer could not be verified against the evidence, so it was withheld."

Update = dict[str, Any]


@dataclass
class RunContext:
    """Per-run dependencies, kept out of the state so the state stays plain data."""

    jira: LazyConnector | None
    site_url: str
    projects: list[str]  # configured project keys
    synthesizer: GroundedSynthesizer
    planner: Planner = field(default_factory=RulePlanner)
    today: Callable[[], date] = field(default=lambda: utc_now().date())
    retrieval_limit: int = 20


def mentioned_projects(question: str, configured: list[str]) -> list[str]:
    """Configured project keys the question names directly ("ATLAS") or via an issue key."""
    named = [m.group(1) for m in _ISSUE_KEY.finditer(question)] + _WORD.findall(question)
    return [key for key in dict.fromkeys(named) if key in configured]


def plain_answer(text: str, answerability: Answerability, summary: str | None = None) -> Answer:
    return Answer(text=text, answerability=answerability, citations=[], reasoning_summary=summary)


async def authorize(state: GraphState, runtime: Runtime[RunContext]) -> Update:
    ctx = runtime.context
    if ctx.jira is None or not ctx.projects:
        return {"answer": plain_answer(NOT_CONFIGURED_TEXT, Answerability.TOOL_FAILURE)}

    named = mentioned_projects(state.question, ctx.projects)
    requested = named or ctx.projects
    granted = [p for p in requested if grant_for(state.caller, SOURCE, p)]
    skipped = [p for p in named if p not in granted]
    if granted:
        return {"projects": granted, "skipped": skipped}

    logger.info("User %r cannot read any project the question needs", state.caller.user_id)
    # Name projects only when the caller named them; never reveal the configured list.
    text = (
        f"You are not authorized to read the Jira project(s) {', '.join(skipped)}."
        if skipped
        else "You are not authorized to read any of the Jira projects IssueSense searches."
    )
    return {
        "skipped": skipped,
        "answer": plain_answer(text, Answerability.UNAUTHORIZED_EVIDENCE),
    }


async def plan(state: GraphState, runtime: Runtime[RunContext]) -> Update:
    ctx = runtime.context
    query_plan = await ctx.planner.plan(state.question, ctx.today(), state.projects)
    logger.info(
        "Planned %d sub-question(s) with %s%s",
        len(query_plan.sub_questions),
        query_plan.planner,
        f" ({query_plan.note})" if query_plan.note else "",
    )
    return {"plan": query_plan}


async def retrieve(state: GraphState, runtime: Runtime[RunContext]) -> Update:
    ctx = runtime.context
    assert ctx.jira is not None and state.plan is not None
    request = RetrievalRequest(
        query=state.question, resources=state.projects, limit=ctx.retrieval_limit
    )
    try:
        jira = await ctx.jira.get()
        retrieved = await jira.retrieve(state.caller, request)
    except ConnectorError:
        logger.exception("Jira retrieval failed for projects %s", ", ".join(state.projects))
        return {"failures": [*state.failures, "Jira could not be reached."]}
    evidence = normalize_jira_records(retrieved.records, ctx.site_url)
    return {"evidence": [*state.evidence, *evidence]}


async def assess(state: GraphState, runtime: Runtime[RunContext]) -> Update:
    if state.failures and not state.evidence:
        return {"answer": plain_answer(UNAVAILABLE_TEXT, Answerability.TOOL_FAILURE)}
    return {"ranked": rank_evidence(state.question, state.evidence)}


async def synthesize(state: GraphState, runtime: Runtime[RunContext]) -> Update:
    synthesis = await runtime.context.synthesizer.synthesize(
        state.question, state.ranked, allow_llm=not state.quote_only
    )
    return {"synthesis": synthesis}


async def verify(state: GraphState, runtime: Runtime[RunContext]) -> Update:
    assert state.synthesis is not None
    report = verify_answer(state.synthesis.answer, state.synthesis.evidence)
    if report.verified:
        return {"verification": report}
    codes = ", ".join(sorted({issue.code.value for issue in report.issues}))
    if state.synthesis.mode == "llm" and not state.quote_only:
        logger.warning("LLM answer failed verification (%s); quoting evidence instead", codes)
        return {"verification": report, "quote_only": True}
    logger.error("Answer withheld: %s", codes)
    return {
        "verification": report,
        "answer": plain_answer(UNVERIFIED_TEXT, Answerability.INSUFFICIENT_EVIDENCE),
    }


async def finish(state: GraphState, runtime: Runtime[RunContext]) -> Update:
    if state.answer is not None:
        return {}
    assert state.synthesis is not None
    answer = state.synthesis.answer.model_copy(deep=True)
    if state.skipped:
        note = f"Skipped Jira project(s) you cannot read: {', '.join(state.skipped)}."
        answer.reasoning_summary = f"{answer.reasoning_summary or ''} {note}".strip()
        if answer.answerability is Answerability.SUFFICIENT:
            answer.answerability = Answerability.PARTIALLY_ANSWERABLE
    return {"answer": answer}


def _settled_or(next_node: str) -> Any:
    def route(state: GraphState) -> str:
        return "finish" if state.answer is not None else next_node

    return route


def _after_verify(state: GraphState) -> Literal["synthesize", "finish"]:
    if state.answer is None and state.verification is not None and not state.verification.verified:
        return "synthesize"
    return "finish"


def build_graph() -> CompiledStateGraph[GraphState, RunContext, GraphState, GraphState]:
    builder = StateGraph(GraphState, context_schema=RunContext)
    builder.add_node("authorize", authorize)
    builder.add_node("plan", plan)
    builder.add_node("retrieve", retrieve)
    builder.add_node("assess", assess)
    builder.add_node("synthesize", synthesize)
    builder.add_node("verify", verify)
    builder.add_node("finish", finish)

    builder.add_edge(START, "authorize")
    builder.add_conditional_edges("authorize", _settled_or("plan"), ["plan", "finish"])
    builder.add_edge("plan", "retrieve")
    builder.add_edge("retrieve", "assess")
    builder.add_conditional_edges("assess", _settled_or("synthesize"), ["synthesize", "finish"])
    builder.add_edge("synthesize", "verify")
    builder.add_conditional_edges("verify", _after_verify, ["synthesize", "finish"])
    builder.add_edge("finish", END)
    return builder.compile(name="issuesense-query")
