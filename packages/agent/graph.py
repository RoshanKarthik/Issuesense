"""The query workflow as a typed LangGraph graph.

    authorize -> plan -> retrieve -> assess -> synthesize -> verify -> finish
                           ^           |           ^           |
                           +-----------+           +-----------+
                      (follow-up retrieval      (quote instead of an LLM answer
                       when evidence has gaps)   that failed verification)

`plan` turns the question into sub-questions, and `retrieve` runs one search per sub-question,
concurrently. `assess` looks for gaps: a sub-question its evidence does not cover gets one broader
search, and issues the evidence refers to but the pool lacks (a parent epic, "blocked by ATLAS-7")
are fetched by key. A `Budget` bounds the rounds, the searches and the LLM tokens; reaching it is
reported in the answer, never hidden.

A node that settles the outcome early (not configured, unauthorized, nothing retrievable) sets
`answer` and the graph goes straight to `finish`. The guarantees of Phase 1 hold: access is checked
before any retrieval, retrieved text is normalized into evidence before synthesis, and no answer
leaves without passing citation verification.
"""

import asyncio
import logging
import re
import time
from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from packages.agent import tracing
from packages.agent.context import RunContext
from packages.agent.state import GraphState, RetrievalRecord, RetrievalTask
from packages.agent.synthesis import MIN_QUESTION_COVERAGE
from packages.common.errors import ConnectorError
from packages.common.models import Answer, Answerability, Evidence
from packages.connectors.base import RetrievalRequest, grant_for
from packages.evidence.citations import verify_answer
from packages.evidence.jira import normalize_jira_records
from packages.evidence.pool import merge_evidence, within_period
from packages.evidence.ranking import query_coverage, rank_evidence, rank_for_questions

logger = logging.getLogger(__name__)

SOURCE = "jira"
_ISSUE_KEY = re.compile(r"\b([A-Z][A-Z0-9_]+)-\d+\b")
_WORD = re.compile(r"\b[A-Z][A-Z0-9_]+\b")

NOT_CONFIGURED_TEXT = "Jira is not configured on this server, so there is no answer."
UNAVAILABLE_TEXT = "Jira could not be reached, so there is no answer. Please try again later."
UNVERIFIED_TEXT = "The answer could not be verified against the evidence, so it was withheld."
RETRIEVAL_ERROR = "Jira could not be reached."
RETRIEVAL_TIMEOUT = "Jira did not answer in time."

FOLLOW_UP = "follow-up"
GAP_DEPTH = 5  # how many of a sub-question's best items decide whether it is covered
MAX_FOLLOW_KEYS = 10

Update = dict[str, Any]


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
    # Every configured key, so that a project the caller may not read is not searched as a word.
    query_plan = await ctx.planner.plan(state.question, ctx.today(), ctx.projects)
    logger.info(
        "Planned %d sub-question(s) with %s%s",
        len(query_plan.sub_questions),
        query_plan.planner,
        f" ({query_plan.note})" if query_plan.note else "",
    )
    period = query_plan.time_range
    tasks = [
        RetrievalTask(
            sub_question=sub_question.id,
            kind="search",
            query=sub_question.text,
            terms=sub_question.terms,
            updated_since=period.start if period and sub_question.time_bound else None,
            updated_until=period.end if period and sub_question.time_bound else None,
        )
        for sub_question in query_plan.sub_questions
    ]
    return {
        "plan": query_plan,
        "pending": tasks,
        "llm_tokens": state.llm_tokens + query_plan.input_tokens + query_plan.output_tokens,
    }


class _Outcome:
    def __init__(
        self, evidence: list[Evidence], error: str | None = None, warnings: list[str] | None = None
    ) -> None:
        self.evidence = evidence
        self.error = error
        self.warnings = warnings or []


async def _run_task(task: RetrievalTask, state: GraphState, ctx: RunContext) -> _Outcome:
    """One retrieval, bounded by the retrieval timeout. It never raises a source failure: a failed
    or slow retrieval comes back as an error, so the other retrievals' evidence is kept."""
    assert ctx.jira is not None
    request = RetrievalRequest(
        query=task.query,
        resources=state.projects,
        limit=ctx.retrieval_limit,
        terms=task.terms,
        updated_since=task.updated_since,
        keys=task.keys,
    )
    start = time.perf_counter()
    outcome = await _retrieve_one(task, request, state, ctx)
    ctx.trace.emit(
        "retrieve",
        "retrieval.failed" if outcome.error else "retrieval.completed",
        {
            **tracing.task_summary(task),
            "items": len(outcome.evidence),
            "error": outcome.error,
            "warnings": outcome.warnings,
        },
        (time.perf_counter() - start) * 1000,
    )
    return outcome


async def _retrieve_one(
    task: RetrievalTask, request: RetrievalRequest, state: GraphState, ctx: RunContext
) -> _Outcome:
    assert ctx.jira is not None
    try:
        async with asyncio.timeout(ctx.retrieval_timeout):
            jira = await ctx.jira.get()
            retrieved = await jira.retrieve(state.caller, request)
    except TimeoutError:
        logger.error("Jira retrieval %s/%s timed out", task.sub_question, task.kind)
        return _Outcome([], RETRIEVAL_TIMEOUT)
    except ConnectorError:
        logger.exception("Jira retrieval %s/%s failed", task.sub_question, task.kind)
        return _Outcome([], RETRIEVAL_ERROR)
    evidence = normalize_jira_records(retrieved.records, ctx.site_url)
    if task.updated_since and task.updated_until:
        evidence = within_period(evidence, task.updated_since, task.updated_until)
    warnings = [
        f"{failure.item}: {failure.reason}" if failure.item else failure.reason
        for failure in retrieved.failures
    ]
    return _Outcome(evidence, warnings=warnings)


async def retrieve(state: GraphState, runtime: Runtime[RunContext]) -> Update:
    ctx = runtime.context
    room = ctx.budget.max_retrievals - len(state.retrievals)
    tasks, dropped = state.pending[:room], state.pending[room:]
    notes = list(state.notes)
    if dropped:
        notes.append(
            f"Skipped {len(dropped)} search(es): the limit of {ctx.budget.max_retrievals} "
            "searches per question was reached."
        )
    outcomes = await asyncio.gather(*(_run_task(task, state, ctx) for task in tasks))

    pool = state.evidence
    records = list(state.retrievals)
    current_round = state.rounds + 1
    for task, outcome in zip(tasks, outcomes, strict=True):
        pool, added = merge_evidence(pool, outcome.evidence, task.sub_question)
        records.append(
            RetrievalRecord(
                task=task,
                round=current_round,
                items=len(outcome.evidence),
                new_evidence=[item.evidence_id for item in added],
                error=outcome.error,
                warnings=outcome.warnings,
            )
        )
    return {
        "evidence": pool,
        "retrievals": records,
        "rounds": current_round,
        "pending": [],
        "notes": notes,
    }


def _referenced_keys(items: list[Evidence], projects: list[str]) -> list[str]:
    keys = []
    for item in items:
        for match in _ISSUE_KEY.finditer(f"{item.title or ''}\n{item.content}"):
            if match.group(1) in projects:
                keys.append(match.group(0))
    return list(dict.fromkeys(keys))


def find_gaps(state: GraphState) -> list[RetrievalTask]:
    """Follow-up searches the evidence calls for: broader searches for uncovered sub-questions,
    and issues the relevant evidence refers to that are not in the pool yet."""
    assert state.plan is not None
    broadened = {r.task.sub_question for r in state.retrievals if r.task.kind == "broaden"}
    first = {r.task.sub_question: r.task for r in state.retrievals if r.task.kind == "search"}
    requested = {key for r in state.retrievals for key in r.task.keys}
    pooled = {str(item.metadata.get("issue_key")) for item in state.evidence}

    tasks: list[RetrievalTask] = []
    follow: list[str] = []
    for sub_question in state.plan.sub_questions:
        ranked = rank_evidence(sub_question.text, state.evidence)
        relevant = [
            item
            for item in ranked
            if item.metadata.get("relevance", 0) > 0 and not item.metadata.get("injection_findings")
        ][:GAP_DEPTH]
        covered = bool(relevant) and (
            query_coverage(sub_question.text, relevant) >= MIN_QUESTION_COVERAGE
        )
        search = first.get(sub_question.id)
        if not covered and sub_question.id not in broadened and search and search.terms:
            tasks.append(
                RetrievalTask(
                    sub_question=sub_question.id,
                    kind="broaden",
                    query=sub_question.text,
                    updated_since=search.updated_since,
                    updated_until=search.updated_until,
                )
            )
        follow += _referenced_keys(relevant, state.projects)
    missing = [key for key in dict.fromkeys(follow) if key not in pooled | requested]
    if missing:
        keys = missing[:MAX_FOLLOW_KEYS]
        tasks.append(
            RetrievalTask(
                sub_question=FOLLOW_UP,
                kind="follow",
                query=f"Issues the evidence refers to: {', '.join(keys)}",
                keys=keys,
            )
        )
    return tasks


async def assess(state: GraphState, runtime: Runtime[RunContext]) -> Update:
    assert state.plan is not None
    budget = runtime.context.budget
    if state.retrievals and len(state.failed_retrievals) == len(state.retrievals):
        return {"answer": plain_answer(UNAVAILABLE_TEXT, Answerability.TOOL_FAILURE)}

    gaps = find_gaps(state)
    follow_ups_done = state.rounds - 1
    if gaps and follow_ups_done < budget.max_follow_ups:
        if len(state.retrievals) < budget.max_retrievals:
            logger.info("Following up %d gap(s) in round %d", len(gaps), state.rounds + 1)
            return {"pending": gaps}
    notes = list(state.notes)
    if gaps:
        notes.append(
            f"Stopped before {len(gaps)} follow-up search(es): the limit of "
            f"{budget.max_follow_ups} follow-up round(s) or {budget.max_retrievals} searches "
            "was reached."
        )
    questions = [state.question, *(q.text for q in state.plan.sub_questions)]
    return {"ranked": rank_for_questions(questions, state.evidence), "notes": notes}


async def synthesize(state: GraphState, runtime: Runtime[RunContext]) -> Update:
    ctx = runtime.context
    assert state.plan is not None
    notes = list(state.notes)
    allow_llm = not state.quote_only
    if allow_llm and ctx.synthesizer.uses_llm and state.llm_tokens >= ctx.budget.max_llm_tokens:
        allow_llm = False
        notes.append(
            f"Quoted the evidence: the {ctx.budget.max_llm_tokens}-token budget was spent."
        )
    sub_questions = [q.text for q in state.plan.sub_questions]
    if sub_questions == [state.question]:
        sub_questions = []
    synthesis = await ctx.synthesizer.synthesize(
        state.question, state.ranked, allow_llm=allow_llm, sub_questions=sub_questions
    )
    spent = (synthesis.input_tokens or 0) + (synthesis.output_tokens or 0)
    return {"synthesis": synthesis, "llm_tokens": state.llm_tokens + spent, "notes": notes}


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


def _failure_note(state: GraphState) -> str | None:
    failed = state.failed_retrievals
    incomplete = [r for r in state.retrievals if r.warnings and not r.error]
    if not failed and not incomplete:
        return None
    parts = []
    if failed:
        parts.append(f"{len(failed)} of {len(state.retrievals)} Jira searches failed")
    if incomplete:
        missing = sorted({w for r in incomplete for w in r.warnings})
        shown = "; ".join(missing[:3]) + ("; ..." if len(missing) > 3 else "")
        parts.append(f"some details are missing ({shown})")
    return f"The evidence may be incomplete: {' and '.join(parts)}."


async def finish(state: GraphState, runtime: Runtime[RunContext]) -> Update:
    if state.answer is not None:
        return {}
    assert state.synthesis is not None
    answer = state.synthesis.answer.model_copy(deep=True)
    notes = list(state.notes)
    if state.skipped:
        notes.append(f"Skipped Jira project(s) you cannot read: {', '.join(state.skipped)}.")
        if answer.answerability is Answerability.SUFFICIENT:
            answer.answerability = Answerability.PARTIALLY_ANSWERABLE
    if failure_note := _failure_note(state):
        notes.append(failure_note)
        if answer.answerability is Answerability.SUFFICIENT:
            answer.answerability = Answerability.PARTIALLY_ANSWERABLE
        elif answer.answerability is Answerability.INSUFFICIENT_EVIDENCE and (
            state.failed_retrievals
        ):
            # The evidence that would answer it may be in what failed: say so, not "none exists".
            answer.answerability = Answerability.TOOL_FAILURE
    if notes:
        answer.reasoning_summary = " ".join([answer.reasoning_summary or "", *notes]).strip()
    return {"answer": answer}


def _settled_or(next_node: str) -> Callable[[GraphState], str]:
    def route(state: GraphState) -> str:
        return "finish" if state.answer is not None else next_node

    return route


def _after_assess(state: GraphState) -> Literal["retrieve", "synthesize", "finish"]:
    if state.answer is not None:
        return "finish"
    return "retrieve" if state.pending else "synthesize"


def _after_verify(state: GraphState) -> Literal["synthesize", "finish"]:
    if state.answer is None and state.verification is not None and not state.verification.verified:
        return "synthesize"
    return "finish"


def build_graph() -> CompiledStateGraph[GraphState, RunContext, GraphState, GraphState]:
    builder = StateGraph(GraphState, context_schema=RunContext)
    for name, node, summary in (
        ("authorize", authorize, tracing.authorize_summary),
        ("plan", plan, tracing.plan_summary),
        ("retrieve", retrieve, tracing.retrieve_summary),
        ("assess", assess, tracing.assess_summary),
        ("synthesize", synthesize, tracing.synthesize_summary),
        ("verify", verify, tracing.verify_summary),
        ("finish", finish, tracing.finish_summary),
    ):
        builder.add_node(name, tracing.traced(name, node, summary))

    builder.add_edge(START, "authorize")
    builder.add_conditional_edges("authorize", _settled_or("plan"), ["plan", "finish"])
    builder.add_edge("plan", "retrieve")
    builder.add_edge("retrieve", "assess")
    builder.add_conditional_edges("assess", _after_assess, ["retrieve", "synthesize", "finish"])
    builder.add_edge("synthesize", "verify")
    builder.add_conditional_edges("verify", _after_verify, ["synthesize", "finish"])
    builder.add_edge("finish", END)
    return builder.compile(name="issuesense-query")
