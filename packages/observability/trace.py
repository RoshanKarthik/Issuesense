"""Trace events for one query: what each step did, in order, for the trace endpoint and live stream.

Events say what happened (the plan, the searches, which evidence was found, how the answer was
checked) with IDs, locations, counts and timings. They never carry evidence text, prompts or model
reasoning: the response holds the evidence, and private reasoning is not exposed.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Literal

from packages.common.models import TraceEvent, utc_now

logger = logging.getLogger(__name__)

TraceStatus = Literal["running", "completed", "failed", "cancelled"]


class TraceRecorder:
    def __init__(self, query_id: str, user_id: str) -> None:
        self.query_id = query_id
        self.user_id = user_id
        self.events: list[TraceEvent] = []
        self.status: TraceStatus = "running"
        self.started_at: datetime = utc_now()
        self.finished_at: datetime | None = None
        self._clock = time.perf_counter()
        self._subscribers: list[asyncio.Queue[TraceEvent | None]] = []

    def emit(
        self,
        node_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            event_id=str(len(self.events) + 1),
            node_id=node_id,
            event_type=event_type,
            payload=payload or {},
            duration_ms=round(duration_ms, 2) if duration_ms is not None else None,
        )
        self.events.append(event)
        logger.debug(
            "trace %s #%s %s %s",
            self.query_id,
            event.event_id,
            node_id,
            event_type,
            extra={"query_id": self.query_id, "trace_event": event.model_dump(mode="json")},
        )
        for queue in self._subscribers:
            queue.put_nowait(event)
        return event

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._clock) * 1000

    def close(self, status: TraceStatus) -> None:
        if self.status != "running":
            return
        self.status = status
        self.finished_at = utc_now()
        for queue in self._subscribers:
            queue.put_nowait(None)

    async def follow(self, after: int = 0) -> AsyncIterator[TraceEvent]:
        """Every event after the `after`-th, then new ones live, until the trace closes."""
        queue: asyncio.Queue[TraceEvent | None] = asyncio.Queue()
        backlog = self.events[after:]
        running = self.status == "running"
        if running:
            self._subscribers.append(queue)
        try:
            for event in backlog:
                yield event
            if not running:
                return
            seen = after + len(backlog)
            while (live := await queue.get()) is not None:
                if int(live.event_id) > seen:
                    yield live
        finally:
            if queue in self._subscribers:
                self._subscribers.remove(queue)
