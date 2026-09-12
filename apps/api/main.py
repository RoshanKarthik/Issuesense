from fastapi import FastAPI

from packages.common.settings import settings

app = FastAPI(
    title="IssueSense API",
    description="Evidence-grounded engineering intelligence platform API"
)

@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint to verify the API is operational.
    Returns the current environment and status.
    """
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "version": "0.1.0"
    }
