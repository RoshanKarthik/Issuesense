import pytest
from pydantic import ValidationError

from apps.api.schemas import MAX_QUESTION_LENGTH, QueryRequest, QueryResponse
from packages.common.models import Answer, Answerability, Citation, Evidence


def _evidence(evidence_id: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source="github",
        authority="implementation",
        location="org/repo/app/auth.py#L10-L30",
        content="def authenticate(request): ...",
    )


def _answer(*cited: str) -> Answer:
    return Answer(
        text="Authentication is handled in app/auth.py.",
        answerability=Answerability.SUFFICIENT,
        citations=[Citation(evidence_id=evidence_id) for evidence_id in cited],
    )


def test_question_is_stripped() -> None:
    assert QueryRequest(question="  Where is auth handled?\n").question == "Where is auth handled?"


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_blank_question_is_rejected(question: str) -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question=question)


def test_question_length_is_bounded() -> None:
    QueryRequest(question="x" * MAX_QUESTION_LENGTH)
    with pytest.raises(ValidationError):
        QueryRequest(question="x" * (MAX_QUESTION_LENGTH + 1))


def test_client_cannot_supply_identity_or_scopes() -> None:
    with pytest.raises(ValidationError, match="extra"):
        QueryRequest.model_validate(
            {"question": "Where is auth handled?", "user_id": "admin", "scopes": ["read:*"]}
        )


def test_response_accepts_citations_of_returned_evidence() -> None:
    response = QueryResponse(
        query_id="Q1", answer=_answer("E1", "E2"), evidence=[_evidence("E1"), _evidence("E2")]
    )
    assert [c.evidence_id for c in response.answer.citations] == ["E1", "E2"]


def test_response_allows_an_answer_without_evidence() -> None:
    answer = Answer(
        text="No authorized evidence was found.",
        answerability=Answerability.INSUFFICIENT_EVIDENCE,
        citations=[],
    )
    assert QueryResponse(query_id="Q1", answer=answer).evidence == []


def test_response_rejects_citation_of_missing_evidence() -> None:
    with pytest.raises(ValidationError, match="not returned: E9"):
        QueryResponse(query_id="Q1", answer=_answer("E1", "E9"), evidence=[_evidence("E1")])


def test_response_rejects_duplicate_evidence_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate evidence_id values: E1"):
        QueryResponse(query_id="Q1", answer=_answer("E1"), evidence=[_evidence("E1")] * 2)
