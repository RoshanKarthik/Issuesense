"""HTTP request/response contracts for the public API."""

from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from packages.common.models import Answer, Evidence
from packages.evidence.citations import VerificationReport
from packages.guardrails.input import Triage

MAX_QUESTION_LENGTH = 4000

Question = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_LENGTH),
]


class QueryRequest(BaseModel):
    """A question for IssueSense.

    Identity and permission scopes are deliberately not part of the body: they come from the
    authenticated caller, so a client can never grant itself access by sending them.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"question": "What changed in Project Atlas this quarter, and what are the risks?"}
            ]
        },
    )

    question: Question

    @field_validator("question")
    @classmethod
    def _has_words(cls, question: str) -> str:
        if not any(char.isalnum() for char in question):
            raise ValueError("question must contain at least one letter or digit")
        return question


class QueryResponse(BaseModel):
    """A grounded answer together with every piece of evidence it cites."""

    query_id: str
    triage: Triage = Triage.RETRIEVAL
    answer: Answer
    evidence: list[Evidence] = Field(default_factory=list)
    # How the answer's citations were checked; absent when nothing was retrieved.
    verification: VerificationReport | None = None

    @model_validator(mode="after")
    def _citations_reference_returned_evidence(self) -> Self:
        ids = [item.evidence_id for item in self.evidence]
        duplicates = sorted({evidence_id for evidence_id in ids if ids.count(evidence_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate evidence_id values: {', '.join(duplicates)}")
        missing = sorted({citation.evidence_id for citation in self.answer.citations} - set(ids))
        if missing:
            raise ValueError(
                f"citations reference evidence that is not returned: {', '.join(missing)}"
            )
        return self


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """The single error shape returned by every endpoint."""

    error: ErrorBody
