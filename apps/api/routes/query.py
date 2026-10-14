from http import HTTPStatus

from fastapi import APIRouter, HTTPException

from apps.api.schemas import ErrorResponse, QueryRequest, QueryResponse

router = APIRouter(prefix="/v1", tags=["Query"])


@router.post(
    "/query",
    responses={
        HTTPStatus.UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED: {"model": ErrorResponse},
    },
)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a question from authorized source evidence, citing every claim."""
    # Retrieval and synthesis arrive in later Phase 1 milestones; until then the honest answer
    # is that the capability does not exist, never a placeholder response.
    raise HTTPException(
        HTTPStatus.NOT_IMPLEMENTED,
        "Query answering is not available yet: no source connector is configured.",
    )
