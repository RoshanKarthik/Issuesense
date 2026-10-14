"""A trace event for every graph node: `node.started`, then `node.completed` with a summary of
what the node decided (or `node.failed`), each with its duration.

Summaries hold IDs, locations, titles, plans, counts and verdicts. Evidence text, prompts and model
output stay out of the trace.
"""

import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from langgraph.runtime import Runtime

from packages.agent.context import RunContext
from packages.agent.state import GraphState, RetrievalRecord, RetrievalTask
from packages.common.models import Evidence

Update = dict[str, Any]
Summary = Callable[[GraphState, GraphState], dict[str, Any]]  # (before, after) -> payload

MAX_LISTED = 20


class Node(Protocol):
    """A graph node that receives the run context (LangGraph passes `runtime` by name)."""

    def __call__(self, state: GraphState, *, runtime: Runtime[RunContext]) -> Awaitable[Update]: ...


def task_summary(task: RetrievalTask) -> dict[str, Any]:
    return {
        "sub_question": task.sub_question,
        "kind": task.kind,
        "terms": task.terms,
        "keys": task.keys,
        "period": (
            [task.updated_since.isoformat(), task.updated_until.isoformat()]
            if task.updated_since and task.updated_until
            else None
        ),
    }


def record_summary(record: RetrievalRecord) -> dict[str, Any]:
    return {
        **task_summary(record.task),
        "round": record.round,
        "items": record.items,
        "new_evidence": record.new_evidence,
        "error": record.error,
        "warnings": record.warnings,
    }


def evidence_ref(item: Evidence) -> dict[str, Any]:
    return {
        "evidence_id": item.evidence_id,
        "source": item.source,
        "location": item.location,
        "title": item.title,
        "url": item.url,
        "sub_questions": item.metadata.get("sub_questions", []),
        "flagged": bool(item.metadata.get("injection_findings")),
    }


def _outcome(after: GraphState) -> dict[str, Any]:
    if after.answer is None:
        return {}
    return {"settled": after.answer.answerability.value}


def authorize_summary(before: GraphState, after: GraphState) -> dict[str, Any]:
    return {"projects": after.projects, "skipped": after.skipped, **_outcome(after)}


def plan_summary(before: GraphState, after: GraphState) -> dict[str, Any]:
    plan = after.plan
    if plan is None:
        return {}
    period = plan.time_range
    return {
        "planner": plan.planner,
        "rewritten": plan.rewritten,
        "period": (
            {
                "start": period.start.isoformat(),
                "end": period.end.isoformat(),
                "phrase": period.phrase,
            }
            if period
            else None
        ),
        "sub_questions": [q.model_dump() for q in plan.sub_questions],
        "note": plan.note,
        "llm_tokens": plan.input_tokens + plan.output_tokens,
    }


def retrieve_summary(before: GraphState, after: GraphState) -> dict[str, Any]:
    records = after.retrievals[len(before.retrievals) :]
    new_ids = {i for record in records for i in record.new_evidence}
    added = [item for item in after.evidence if item.evidence_id in new_ids]
    return {
        "round": after.rounds,
        "retrievals": [record_summary(record) for record in records],
        "evidence": [evidence_ref(item) for item in added],
        "pool_size": len(after.evidence),
        "notes": after.notes[len(before.notes) :],
    }


def assess_summary(before: GraphState, after: GraphState) -> dict[str, Any]:
    if after.pending:
        return {"follow_ups": [task_summary(task) for task in after.pending]}
    ranked = [
        {"evidence_id": item.evidence_id, "relevance": item.metadata.get("relevance", 0.0)}
        for item in after.ranked[:MAX_LISTED]
        if item.metadata.get("relevance", 0.0) > 0
    ]
    return {"ranked": ranked, "notes": after.notes[len(before.notes) :], **_outcome(after)}


def synthesize_summary(before: GraphState, after: GraphState) -> dict[str, Any]:
    synthesis = after.synthesis
    if synthesis is None:
        return {}
    return {
        "mode": synthesis.mode,
        "answerability": synthesis.answer.answerability.value,
        "citations": [c.evidence_id for c in synthesis.answer.citations],
        "withheld": synthesis.withheld,
        "llm_tokens": after.llm_tokens - before.llm_tokens,
    }


def verify_summary(before: GraphState, after: GraphState) -> dict[str, Any]:
    report = after.verification
    if report is None:
        return {}
    return {
        "verified": report.verified,
        "supported_claims": report.supported_claims,
        "total_claims": report.total_claims,
        "issues": [
            {"code": i.code.value, "claim_index": i.claim_index, "evidence_id": i.evidence_id}
            for i in report.issues
        ],
        "quote_instead": after.quote_only and not before.quote_only,
        **_outcome(after),
    }


def finish_summary(before: GraphState, after: GraphState) -> dict[str, Any]:
    if after.answer is None:
        return {}
    return {
        "answerability": after.answer.answerability.value,
        "citations": [c.evidence_id for c in after.answer.citations],
        "reasoning_summary": after.answer.reasoning_summary,
    }


def traced(name: str, node: Node, summarize: Summary) -> Node:
    """`node`, emitting its trace events around every run."""

    async def run(state: GraphState, *, runtime: Runtime[RunContext]) -> Update:
        context = runtime.context
        context.trace.emit(name, "node.started")
        start = time.perf_counter()
        try:
            update = await node(state, runtime=runtime)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            context.trace.emit(name, "node.failed", {"error": type(exc).__name__}, elapsed)
            raise
        elapsed = (time.perf_counter() - start) * 1000
        payload = summarize(state, state.model_copy(update=update))
        context.trace.emit(name, "node.completed", payload, elapsed)
        return update

    run.__name__ = name
    return run
