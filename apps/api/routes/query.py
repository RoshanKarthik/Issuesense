import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import suppress
from functools import lru_cache, partial
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent

from apps.api.auth import Caller
from apps.api.schemas import ErrorBody, ErrorResponse, QueryRequest, QueryResponse
from packages.agent.llm import provider_from_settings
from packages.agent.pipeline import PipelineResult, QueryPipeline
from packages.agent.planner import planner_for
from packages.agent.synthesis import GroundedSynthesizer
from packages.common.errors import GuardrailError
from packages.common.models import Answer, Answerability, AuthorizationContext
from packages.common.settings import settings
from packages.connectors.jira import open_jira_connector
from packages.guardrails.input import FindingCategory, InputAssessment, Triage, assess_question
from packages.observability.langsmith import LangSmithTracing
from packages.observability.trace import TraceRecorder, TraceStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Query"])

USAGE_HINT = (
    "I answer questions about your project sources and cite the evidence for every claim. "
    'Try: "What changed in Project Atlas this quarter, and what are the risks?"'
)
READ_ONLY_MESSAGE = (
    "IssueSense is read-only: it can't create, change, send or run anything. "
    "Ask a question about your sources instead."
)
BLOCKED_MESSAGE = "The question was blocked by the input guardrails."


def _reject(assessment: InputAssessment) -> GuardrailError:
    blocking = assessment.blocking_findings
    rule_ids = [finding.rule_id for finding in blocking]
    logger.info(
        "Question blocked by input rules: %s", ", ".join(rule_ids), extra={"rule_ids": rule_ids}
    )
    read_only = any(finding.category is FindingCategory.TOOL_INTENT for finding in blocking)
    # Rule IDs explain the block; the matched text is never echoed back.
    return GuardrailError(
        READ_ONLY_MESSAGE if read_only else BLOCKED_MESSAGE,
        {"findings": [finding.model_dump(mode="json") for finding in blocking]},
    )


@lru_cache(maxsize=1)
def get_pipeline() -> QueryPipeline:
    """The configured pipeline (a FastAPI dependency, so tests can swap in fakes)."""
    provider = provider_from_settings(settings)
    return QueryPipeline(
        open_jira=partial(open_jira_connector, settings) if settings.jira_configured else None,
        site_url=settings.jira_base_url,
        projects=settings.jira_projects,
        synthesizer=GroundedSynthesizer(provider),
        planner=planner_for(provider),
        retrieval_timeout=settings.retrieval_timeout_seconds,
        tracing=LangSmithTracing(settings),
    )


@lru_cache(maxsize=1)
def get_trace_store() -> TraceStore:
    """The traces of recent queries (a FastAPI dependency, so tests get a fresh store)."""
    return TraceStore()


def assess_request(request: QueryRequest, caller: Caller) -> InputAssessment:
    """Check the question before anything runs. A dependency, so a blocked question is refused
    with the usual error response even on the streaming endpoint, before its stream starts."""
    assessment = assess_question(request.question)
    if assessment.blocked:
        raise _reject(assessment)
    return assessment


Assessed = Annotated[InputAssessment, Depends(assess_request)]
Pipeline = Annotated[QueryPipeline, Depends(get_pipeline)]
Traces = Annotated[TraceStore, Depends(get_trace_store)]

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    HTTPStatus.BAD_REQUEST: {"model": ErrorResponse},
    HTTPStatus.UNAUTHORIZED: {"model": ErrorResponse},
    HTTPStatus.UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
}


def _start(caller: AuthorizationContext, traces: TraceStore) -> TraceRecorder:
    trace = TraceRecorder(uuid.uuid4().hex, caller.user_id)
    traces.add(trace)
    return trace


def _small_talk(trace: TraceRecorder) -> QueryResponse:
    trace.emit("triage", "query.small_talk", {"answerability": "NOT_APPLICABLE"})
    trace.close("completed")
    return QueryResponse(
        query_id=trace.query_id,
        triage=Triage.SMALL_TALK,
        answer=Answer(text=USAGE_HINT, answerability=Answerability.NOT_APPLICABLE, citations=[]),
    )


def _response(query_id: str, result: PipelineResult) -> QueryResponse:
    logger.info(
        "Answered query %s: %s, %d citation(s), projects %s",
        query_id,
        result.answer.answerability.value,
        len(result.answer.citations),
        ", ".join(result.projects) or "none",
    )
    return QueryResponse(
        query_id=query_id,
        answer=result.answer,
        evidence=result.evidence,
        verification=result.verification,
    )


@router.post("/query", responses=ERROR_RESPONSES)
async def query(
    caller: Caller, assessment: Assessed, pipeline: Pipeline, traces: Traces
) -> QueryResponse:
    """Answer a question from authorized source evidence, citing every claim.

    The steps taken are recorded under the returned `query_id`: see `GET /v1/query/{id}/trace`.
    """
    trace = _start(caller, traces)
    if assessment.triage is Triage.SMALL_TALK:
        return _small_talk(trace)
    result = await pipeline.answer(assessment.normalized_text, caller, trace)
    return _response(trace.query_id, result)


@router.post("/query/stream", response_class=EventSourceResponse, responses=ERROR_RESPONSES)
async def query_stream(
    caller: Caller, assessment: Assessed, pipeline: Pipeline, traces: Traces
) -> AsyncGenerator[ServerSentEvent]:
    """Answer a question like `POST /v1/query`, streaming each step as it happens.

    Server-sent events, in order: `query` (the query ID), one `trace` per trace event (its `id` is
    the event's sequence number), then `answer` (the body `POST /v1/query` returns) or `error`.
    Closing the stream cancels the query.
    """
    trace = _start(caller, traces)
    yield ServerSentEvent(event="query", data={"query_id": trace.query_id})
    if assessment.triage is Triage.SMALL_TALK:
        yield ServerSentEvent(event="answer", data=_small_talk(trace).model_dump(mode="json"))
        return

    run = asyncio.create_task(pipeline.answer(assessment.normalized_text, caller, trace))
    try:
        async for event in trace.follow():
            yield ServerSentEvent(
                event="trace", data=event.model_dump(mode="json"), id=event.event_id
            )
        try:
            result = await run
        except Exception:
            logger.exception("Streamed query %s failed", trace.query_id)
            error = ErrorBody(code="INTERNAL_ERROR", message="An internal error occurred.")
            yield ServerSentEvent(event="error", data=error.model_dump(mode="json"))
            return
        response = _response(trace.query_id, result)
        yield ServerSentEvent(event="answer", data=response.model_dump(mode="json"))
    finally:
        if not run.done():
            logger.info("The client left the stream of query %s; cancelling it", trace.query_id)
            run.cancel()
            with suppress(asyncio.CancelledError):
                await run
