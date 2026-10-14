from fastapi.testclient import TestClient

from apps.api.main import app
from packages.common import __version__
from packages.common.settings import settings

client = TestClient(app)


def test_health_endpoint() -> None:
    """Verify the /health endpoint returns 200 and the expected payload."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data == {"status": "healthy", "environment": settings.env, "version": __version__}


def test_openapi_reports_package_version() -> None:
    assert client.get("/openapi.json").json()["info"]["version"] == __version__


def test_health_endpoint_not_found() -> None:
    """Verify that non-existent endpoints return 404."""
    response = client.get("/not-exists")
    assert response.status_code == 404
