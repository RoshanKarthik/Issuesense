import logging
import uuid
from functools import lru_cache, partial
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends

from apps.api.auth import Caller
from apps.api.schemas import ErrorResponse, QueryRequest, QueryResponse
from packages.agent.llm import provider_from_settings
from packages.agent.pipeline import QueryPipeline
from packages.agent.planner import planner_for
from packages.agent.synthesis import GroundedSynthesizer
from packages.common.errors import GuardrailError
from packages.common.models import Answer, Answerability
from packages.common.settings import settings
from packages.connectors.jira import open_jira_connector
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


@lru_cache(maxsize=1)
def get_pipeline() -> QueryPipeline:
    """The configured pipeline (a FastAPI dependency, so tests can swap in fakes)."""
    provider = provider_from_settings(settings)
    return QueryPipeline(
        open_jira=partial(open_jira_connector, settings) if settings.jira_configured else None,
        site_url=settings.jira_base_url,
        projects=settings.jira_projects,
        synthesizer=GroundedSynthesizer(provider),
        planner=planner_for(provider),
        retrieval_timeout=settings.retrieval_timeout_seconds,
    )


@router.post(
    "/query",
    responses={
        HTTPStatus.BAD_REQUEST: {"model": ErrorResponse},
        HTTPStatus.UNAUTHORIZED: {"model": ErrorResponse},
        HTTPStatus.UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
    },
)
async def query(
    request: QueryRequest,
    caller: Caller,
    pipeline: Annotated[QueryPipeline, Depends(get_pipeline)],
) -> QueryResponse:
    """Answer a question from authorized source evidence, citing every claim."""
    assessment = assess_question(request.question)
    if assessment.blocked:
        raise _reject(assessment)

    query_id = uuid.uuid4().hex
    if assessment.triage is Triage.SMALL_TALK:
        return QueryResponse(
            query_id=query_id,
            triage=Triage.SMALL_TALK,
            answer=Answer(
                text=USAGE_HINT, answerability=Answerability.NOT_APPLICABLE, citations=[]
            ),
        )

    result = await pipeline.answer(assessment.normalized_text, caller)
    logger.info(
        "Answered query %s: %s, %d citation(s), projects %s",
        query_id,
        result.answer.answerability.value,
        len(result.answer.citations),
        ", ".join(result.projects) or "none",
    )
    return QueryResponse(
        query_id=query_id,
        answer=result.answer,
        evidence=result.evidence,
        verification=result.verification,
    )
