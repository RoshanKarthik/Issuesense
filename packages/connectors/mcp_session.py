"""Adapter from the official `mcp` SDK client to the connectors' `McpToolClient` protocol."""

import base64
import json
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

from packages.common.errors import ConnectorError, IssueSenseError
from packages.connectors.base import ToolResult

logger = logging.getLogger(__name__)


def basic_auth_header(email: str, api_token: str) -> dict[str, str]:
    """The `Authorization` header Atlassian's MCP server expects for personal API tokens."""
    encoded = base64.b64encode(f"{email}:{api_token}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def _content_of(result: CallToolResult) -> Any:
    """Structured content when the server sends it; otherwise text, parsed as JSON when possible."""
    if result.structured_content is not None:
        return result.structured_content
    text = "".join(block.text for block in result.content if isinstance(block, TextContent))
    try:
        return json.loads(text)
    except ValueError:
        return {"text": text}


class McpSession:
    """One connected MCP server, exposed through the `McpToolClient` protocol."""

    def __init__(self, client: Client, server: str) -> None:
        self._client = client
        self.server = server

    async def list_tools(self) -> dict[str, dict[str, Any]]:
        tools: dict[str, dict[str, Any]] = {}
        cursor: str | None = None
        try:
            while True:
                page = await self._client.list_tools(cursor=cursor)
                tools.update({tool.name: dict(tool.input_schema) for tool in page.tools})
                cursor = page.next_cursor
                if cursor is None:
                    return tools
        except Exception as exc:
            logger.exception("Listing tools on MCP server %r failed", self.server)
            raise ConnectorError(
                "Could not list the source's tools.", {"server": self.server}
            ) from exc

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            result = await self._client.call_tool(name, arguments)
        except Exception as exc:
            logger.exception("MCP tool %r on server %r raised", name, self.server)
            raise ConnectorError(
                "The source did not answer.", {"server": self.server, "tool": name}
            ) from exc
        return ToolResult(tool=name, content=_content_of(result), is_error=result.is_error)


@asynccontextmanager
async def connect(
    target: Any, *, server: str, headers: dict[str, str] | None = None, timeout: float = 30.0
) -> AsyncIterator[McpSession]:
    """Connect to an MCP server: a Streamable HTTP URL, or an in-process server in tests.

    Connection failures become `ConnectorError`; errors raised by the caller inside the block pass
    through unchanged.
    """
    stack = AsyncExitStack()
    if isinstance(target, str):
        # httpx2 is the HTTP client the mcp SDK is built on; the SDK leaves closing it to us.
        http = httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(timeout))
        stack.push_async_callback(http.aclose)
        target = streamable_http_client(target, http_client=http)
    try:
        client = await stack.enter_async_context(Client(target, read_timeout_seconds=timeout))
    except Exception as exc:
        await stack.aclose()
        if isinstance(exc, IssueSenseError):
            raise
        logger.exception("Connecting to MCP server %r failed", server)
        raise ConnectorError("Could not connect to the source.", {"server": server}) from exc
    try:
        yield McpSession(client, server)
    finally:
        try:
            await stack.aclose()
        except Exception:
            # Closing a finished session must not mask the result or the caller's own error.
            logger.warning("Closing the MCP session to %r failed", server, exc_info=True)
