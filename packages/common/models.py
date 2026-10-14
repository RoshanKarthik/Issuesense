from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


class Answerability(StrEnum):
    """Answerability state of a query."""

    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    PARTIALLY_ANSWERABLE = "PARTIALLY_ANSWERABLE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    TOOL_FAILURE = "TOOL_FAILURE"
    UNAUTHORIZED_EVIDENCE = "UNAUTHORIZED_EVIDENCE"
    NOT_APPLICABLE = "NOT_APPLICABLE"  # no factual question was asked, so nothing is claimed


class AuthorizationContext(BaseModel):
    """User identity and permission scopes for a request."""

    user_id: str
    scopes: list[str] = Field(default_factory=list)
    tenant_id: str | None = None


class Query(BaseModel):
    """The user's request and associated metadata."""

    query_id: str
    text: str
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class Evidence(BaseModel):
    """
    Normalized grounding evidence.
    Raw connector payloads must be converted to this format before synthesis.
    """

    evidence_id: str
    source: str  # e.g., "github", "notion"
    authority: str  # e.g., "implementation", "architecture"
    title: str | None = None
    location: str  # e.g., "repository/path.py#L10-L30"
    content: str
    timestamp: datetime | None = None
    url: str | None = None
    permission_context: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """A pointer to specific evidence supporting a claim."""

    evidence_id: str
    quote: str | None = None
    location_offset: str | None = None


class Answer(BaseModel):
    """The final grounded synthesis."""

    text: str
    answerability: Answerability
    citations: list[Citation]
    reasoning_summary: str | None = None


class TraceEvent(BaseModel):
    """A structured event for system observability."""

    event_id: str
    timestamp: datetime = Field(default_factory=utc_now)
    node_id: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None
