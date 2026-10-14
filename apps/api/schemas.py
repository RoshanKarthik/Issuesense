"""HTTP request/response contracts for the public API."""

from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from packages.common.models import Answer, Evidence

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
        json_schema_extra={"examples": [{"question": "Where is request authentication handled?"}]},
    )

    question: Question


class QueryResponse(BaseModel):
    """A grounded answer together with every piece of evidence it cites."""

    query_id: str
    answer: Answer
    evidence: list[Evidence] = Field(default_factory=list)

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
