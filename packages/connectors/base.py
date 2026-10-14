"""The connector boundary: access checks before retrieval, and read-only MCP tool allowlists.

Connectors reach live sources through MCP servers. Two rules hold for every connector:

1. `Connector.retrieve` checks the caller's scopes before any fetch, and drops any record for a
   resource the caller was not granted. Subclasses implement `_fetch` and cannot skip the check.
2. `AllowlistedToolClient` refuses every MCP tool that is not on the server's read-only allowlist,
   before the call leaves the process. Connector credentials never imply the caller's access.
"""

import logging
from abc import ABC, abstractmethod
from datetime import date
from typing import Annotated, Any, ClassVar, Protocol

from pydantic import BaseModel, Field, StringConstraints

from packages.common.errors import AuthorizationError
from packages.common.models import AuthorizationContext

logger = logging.getLogger(__name__)

WILDCARD = "*"

# A resource is one container in a source, such as a Jira project key. Plain identifiers only, so
# a request can never smuggle a wildcard or a list into a scope check.
ResourceId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]
# A single lowercase search word. Connectors may place it inside a source query (such as JQL), so
# only letters, digits and inner hyphens are allowed: nothing that could quote or escape.
SearchTerm = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{0,28}[a-z0-9]$")]
# The key of one item in a source, such as the Jira issue key "ATLAS-12".
ItemKey = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]


class ToolResult(BaseModel):
    """What an MCP tool call returned (structured content when the server provides it)."""

    tool: str
    content: Any = None
    is_error: bool = False


class McpToolClient(Protocol):
    """The slice of an MCP client session that connectors use."""

    async def list_tools(self) -> dict[str, dict[str, Any]]:
        """Every tool the server offers, mapped to its JSON input schema."""
        ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...


class ToolNotAllowedError(AuthorizationError):
    """A connector tried to call an MCP tool outside its read-only allowlist."""


class AllowlistedToolClient:
    """Wraps an MCP client so that only allowlisted (read-only) tools can be called."""

    def __init__(self, client: McpToolClient, server: str, allowed_tools: frozenset[str]) -> None:
        if not allowed_tools:
            raise ValueError(f"MCP server {server!r} needs at least one allowlisted tool")
        self._client = client
        self.server = server
        self.allowed_tools = allowed_tools

    async def tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Input schemas of the allowlisted tools the server offers (never the other tools)."""
        offered = await self._client.list_tools()
        return {name: schema for name, schema in offered.items() if name in self.allowed_tools}

    async def missing_tools(self) -> set[str]:
        """Allowlisted tools the server does not offer (a misconfiguration to report at startup)."""
        return set(self.allowed_tools) - set(await self.tool_schemas())

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in self.allowed_tools:
            logger.error("Refused MCP tool %r on server %r: not allowlisted", name, self.server)
            raise ToolNotAllowedError(
                "This tool is not allowed.", {"server": self.server, "tool": name}
            )
        return await self._client.call_tool(name, arguments)


class RetrievalRequest(BaseModel):
    """What to retrieve. Every filter is typed and validated, so a connector can place it in a
    source query without user text ever reaching that query."""

    query: str = Field(min_length=1)  # the (sub-)question; used for logging and ranking only
    resources: list[ResourceId] = Field(min_length=1, max_length=20)
    limit: int = Field(default=20, ge=1, le=100)
    terms: list[SearchTerm] = Field(default_factory=list, max_length=8)  # match any of them
    updated_since: date | None = None  # items with activity on or after this day
    keys: list[ItemKey] = Field(default_factory=list, max_length=20)  # fetch exactly these items


class SourceRecord(BaseModel):
    """One raw item from a source. Never passed to synthesis: normalized into Evidence first."""

    source: str
    resource: str
    tool: str
    payload: dict[str, Any]
    permission_context: str  # the scope that granted access to this record


class ToolFailure(BaseModel):
    """One source call that failed while others succeeded. Safe to show: no raw error text."""

    tool: str
    resource: str  # the project (or other container) the call was for
    item: str | None = None  # the item it was about, such as an issue key
    reason: str


class FetchResult(BaseModel):
    records: list[SourceRecord] = Field(default_factory=list)
    failures: list[ToolFailure] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    source: str
    records: list[SourceRecord] = Field(default_factory=list)
    # Calls that failed; their records are missing, every other record is kept.
    failures: list[ToolFailure] = Field(default_factory=list)


def read_scope(source: str, resource: str) -> str:
    return f"{source}:read:{resource}"


def grant_for(caller: AuthorizationContext, source: str, resource: str) -> str | None:
    """The caller's scope that grants reading `resource` (exact before wildcard), or None."""
    for scope in (read_scope(source, resource), read_scope(source, WILDCARD)):
        if scope in caller.scopes:
            return scope
    return None


class Connector(ABC):
    source: ClassVar[str]

    def authorize(self, caller: AuthorizationContext, request: RetrievalRequest) -> dict[str, str]:
        """Map each requested resource to the scope that grants it, or raise AuthorizationError."""
        grants: dict[str, str] = {}
        denied: list[str] = []
        for resource in request.resources:
            if scope := grant_for(caller, self.source, resource):
                grants[resource] = scope
            else:
                denied.append(resource)
        if denied:
            logger.warning(
                "Denied %s read for user %r on %s", self.source, caller.user_id, ", ".join(denied)
            )
            raise AuthorizationError(
                f"Not authorized to read these {self.source} resources.",
                {"source": self.source, "resources": denied},
            )
        return grants

    async def retrieve(
        self, caller: AuthorizationContext, request: RetrievalRequest
    ) -> RetrievalResult:
        grants = self.authorize(caller, request)
        fetched = await self._fetch(request, grants)
        records = fetched.records
        allowed = [record for record in records if record.resource in grants]
        if len(allowed) != len(records):
            # A connector bug or a server returning more than asked for: never pass it on.
            logger.error(
                "Dropped %d %s records for resources that were not granted",
                len(records) - len(allowed),
                self.source,
            )
        return RetrievalResult(source=self.source, records=allowed, failures=fetched.failures)

    @abstractmethod
    async def _fetch(self, request: RetrievalRequest, grants: dict[str, str]) -> FetchResult:
        """Fetch records for the granted resources. Called only after `authorize` succeeds.

        A call that fails is reported in `failures` while the other records are still returned;
        raise `ConnectorError` only when nothing could be fetched at all."""
