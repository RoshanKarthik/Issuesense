import logging
import uuid
from http import HTTPStatus

from fastapi import APIRouter, HTTPException

from apps.api.auth import Caller
from apps.api.schemas import ErrorResponse, QueryRequest, QueryResponse
from packages.common.errors import GuardrailError
from packages.common.models import Answer, Answerability
from packages.guardrails.input import FindingCategory, InputAssessment, Triage, assess_question

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


@router.post(
    "/query",
    responses={
        HTTPStatus.BAD_REQUEST: {"model": ErrorResponse},
        HTTPStatus.UNAUTHORIZED: {"model": ErrorResponse},
        HTTPStatus.UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED: {"model": ErrorResponse},
    },
)
async def query(request: QueryRequest, caller: Caller) -> QueryResponse:
    """Answer a question from authorized source evidence, citing every claim."""
    assessment = assess_question(request.question)
    if assessment.blocked:
        raise _reject(assessment)

    if assessment.triage is Triage.SMALL_TALK:
        return QueryResponse(
            query_id=uuid.uuid4().hex,
            triage=Triage.SMALL_TALK,
            answer=Answer(
                text=USAGE_HINT, answerability=Answerability.NOT_APPLICABLE, citations=[]
            ),
        )

    # Retrieval and synthesis arrive in later Phase 1 steps; until then the honest answer is that
    # the capability does not exist, never a placeholder response.
    raise HTTPException(
        HTTPStatus.NOT_IMPLEMENTED,
        "Query answering is not available yet: no source connector is configured.",
    )
