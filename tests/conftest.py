from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from apps.api.auth import ApiKeyStore, get_api_key_store, hash_api_key
from apps.api.main import app
from packages.common.settings import ApiKeyEntry

# A test-only key: it exists nowhere but in this process.
TEST_API_KEY = "isk_test_" + "k" * 43
TEST_KEY_ENTRY = ApiKeyEntry(
    key_id="test-key",
    sha256=hash_api_key(TEST_API_KEY),
    user_id="test-user",
    scopes=["jira:read:ATLAS"],
)


@pytest.fixture(autouse=True)
def _test_api_keys() -> Iterator[None]:
    """Serve the app with a known key store instead of whatever the environment configures."""
    app.dependency_overrides[get_api_key_store] = lambda: ApiKeyStore([TEST_KEY_ENTRY])
    yield
    app.dependency_overrides.pop(get_api_key_store, None)


@pytest.fixture
def client() -> TestClient:
    """A client authenticated as `test-user`."""
    return TestClient(app, headers={"Authorization": f"Bearer {TEST_API_KEY}"})


@pytest.fixture
def anonymous_client() -> TestClient:
    return TestClient(app)
