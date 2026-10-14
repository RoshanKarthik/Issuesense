import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from apps.api.errors import install_error_handlers
from apps.api.routes import query
from packages.common import __version__
from packages.common.settings import settings

app = FastAPI(
    title="IssueSense API",
    description="Evidence-grounded engineering intelligence platform API",
    version=__version__,
)
install_error_handlers(app)
app.include_router(query.router)


class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str


@app.get("/health", tags=["System"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint to verify the API is operational.
    Returns the current environment and status.
    """
    return HealthResponse(status="healthy", environment=settings.env, version=__version__)


def run() -> None:
    """Serve the API on the configured host and port (the `issuesense-api` command)."""
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )
