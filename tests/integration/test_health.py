from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Verify the /health endpoint returns 200 and the expected payload."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "environment" in data
    assert data["version"] == "0.1.0"

def test_health_endpoint_not_found():
    """Verify that non-existent endpoints return 404."""
    response = client.get("/not-exists")
    assert response.status_code == 404
