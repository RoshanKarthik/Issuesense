"""Map exceptions to the API's single `ErrorResponse` shape with explicit status codes."""

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from apps.api.schemas import ErrorBody, ErrorResponse
from packages.common.errors import (
    AuthorizationError,
    ConnectorError,
    GuardrailError,
    IssueSenseError,
)

logger = logging.getLogger(__name__)

# Most specific first: the first matching class decides the status and code.
_DOMAIN_ERRORS: list[tuple[type[IssueSenseError], HTTPStatus, str]] = [
    (AuthorizationError, HTTPStatus.FORBIDDEN, "FORBIDDEN"),
    (GuardrailError, HTTPStatus.BAD_REQUEST, "GUARDRAIL_VIOLATION"),
    (ConnectorError, HTTPStatus.BAD_GATEWAY, "CONNECTOR_FAILURE"),
]


def error_response(
    status: int, code: str, message: str, details: dict[str, Any] | None = None
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details or {}))
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # Only location, message and type: the rejected input is never echoed back.
    errors = [
        {"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]} for err in exc.errors()
    ]
    return error_response(
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "INVALID_REQUEST",
        "Request validation failed.",
        {"errors": errors},
    )


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    status = HTTPStatus(exc.status_code)
    message = exc.detail if isinstance(exc.detail, str) else status.phrase
    return error_response(status, status.name, message)


async def _domain_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, IssueSenseError)
    for error_type, status, code in _DOMAIN_ERRORS:
        if isinstance(exc, error_type):
            if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
                # Upstream details can hold internal URLs or payloads: log them, don't return them.
                logger.warning("%s: %s", code, exc.message, extra={"details": exc.details})
                return error_response(status, code, exc.message)
            return error_response(status, code, exc.message, exc.details)
    return await _unexpected_error(request, exc)


async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return error_response(
        HTTPStatus.INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", "An internal error occurred."
    )


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(IssueSenseError, _domain_error)
    app.add_exception_handler(Exception, _unexpected_error)
