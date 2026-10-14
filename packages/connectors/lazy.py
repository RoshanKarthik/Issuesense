"""Open a connector on first use and keep it open for the rest of one query run.

MCP sessions are anyio task groups: they must be entered and exited by the same asyncio task. Graph
nodes may run in different tasks, so a dedicated owner task opens the session, hands the connector
out, and closes it when the run ends. Nothing is opened for a run that never retrieves (for
example an unauthorized question).
"""

import asyncio
import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, suppress

from packages.common.errors import ConnectorError
from packages.connectors.base import Connector

logger = logging.getLogger(__name__)

ConnectorOpener = Callable[[], AbstractAsyncContextManager[Connector]]


class LazyConnector:
    def __init__(self, opener: ConnectorOpener, name: str) -> None:
        self._opener = opener
        self.name = name
        self._lock = asyncio.Lock()
        self._ready: asyncio.Future[Connector] | None = None
        self._closing = asyncio.Event()
        self._owner: asyncio.Task[None] | None = None

    @property
    def opened(self) -> bool:
        return self._owner is not None

    async def get(self) -> Connector:
        """The open connector; raises ConnectorError if it could not be opened (cached)."""
        async with self._lock:
            if self._ready is None:
                self._ready = asyncio.get_running_loop().create_future()
                self._owner = asyncio.create_task(self._own(self._ready))
        return await asyncio.shield(self._ready)

    async def _own(self, ready: asyncio.Future[Connector]) -> None:
        try:
            async with self._opener() as connector:
                ready.set_result(connector)
                await self._closing.wait()
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
            raise
        except Exception as exc:
            if ready.done():
                # The run already has its connector; a failed close must not mask its result.
                logger.warning("Closing the %s connector failed", self.name, exc_info=True)
            elif isinstance(exc, ConnectorError):
                ready.set_exception(exc)
            else:
                logger.exception("Opening the %s connector failed", self.name)
                error = ConnectorError("Could not connect to the source.", {"server": self.name})
                ready.set_exception(error)

    async def aclose(self) -> None:
        self._closing.set()
        if self._owner is not None:
            with suppress(asyncio.CancelledError):
                await self._owner
        if self._ready is not None and self._ready.done() and not self._ready.cancelled():
            self._ready.exception()  # mark a failed open as retrieved, so asyncio does not warn
