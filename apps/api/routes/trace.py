"""Read the trace of an earlier query, or follow a running one live."""

from collections.abc import AsyncGenerator
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Path
from fastapi.sse import EventSourceResponse, ServerSentEvent

from apps.api.auth import Caller
from apps.api.routes.query import Traces
from apps.api.schemas import ErrorResponse, QueryTrace
from packages.observability.trace import TraceRecorder

router = APIRouter(prefix="/v1", tags=["Query"])

QueryId = Annotated[
    str, Path(pattern=r"^[0-9a-f]{32}$", description="The query_id from the query's response.")
]
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    HTTPStatus.UNAUTHORIZED: {"model": ErrorResponse},
    HTTPStatus.NOT_FOUND: {"model": ErrorResponse},
}


def owned_trace(query_id: QueryId, caller: Caller, traces: Traces) -> TraceRecorder:
    """The caller's own trace. A dependency, so a missing or foreign trace is a 404 even on the
    streaming endpoint, before its stream starts."""
    return traces.get(query_id, caller)


Trace = Annotated[TraceRecorder, Depends(owned_trace)]


@router.get("/query/{query_id}/trace", responses=ERROR_RESPONSES)
async def get_trace(trace: Trace) -> QueryTrace:
    """Every step of one of your queries: the plan, each search, the evidence found and how the
    answer was checked, with timings. Only the user who asked can read it."""
    return QueryTrace(
        query_id=trace.query_id,
        status=trace.status,
        started_at=trace.started_at,
        finished_at=trace.finished_at,
        events=list(trace.events),
    )


@router.get(
    "/query/{query_id}/trace/stream",
    response_class=EventSourceResponse,
    responses=ERROR_RESPONSES,
)
async def follow_trace(
    trace: Trace, last_event_id: Annotated[str | None, Header()] = None
) -> AsyncGenerator[ServerSentEvent]:
    """The trace as server-sent events: past events, then new ones live while the query runs,
    then `end` with the final status. Send `Last-Event-ID` to resume after a dropped connection."""
    after = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0
    async for event in trace.follow(after):
        yield ServerSentEvent(event="trace", data=event.model_dump(mode="json"), id=event.event_id)
    yield ServerSentEvent(event="end", data={"status": trace.status})
