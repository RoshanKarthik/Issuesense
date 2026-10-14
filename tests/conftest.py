from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

import pytest
from fastapi.testclient import TestClient

from apps.api.auth import ApiKeyStore, get_api_key_store, hash_api_key
from apps.api.main import app
from apps.api.routes.query import get_pipeline
from packages.agent.llm import LlmProvider
from packages.agent.pipeline import QueryPipeline
from packages.agent.synthesis import GroundedSynthesizer
from packages.common.settings import ApiKeyEntry, Settings
from packages.connectors.jira import open_jira_connector
from packages.evaluation.atlas_dataset import ATLAS_ISSUES
from tests.integration.fake_atlassian import SITE, FakeJiraMcp

# A test-only key: it exists nowhere but in this process.
TEST_API_KEY = "isk_test_" + "k" * 43
TEST_KEY_ENTRY = ApiKeyEntry(
    key_id="test-key",
    sha256=hash_api_key(TEST_API_KEY),
    user_id="test-user",
    scopes=["jira:read:ATLAS"],
)


@pytest.fixture(autouse=True)
def _isolated_app() -> Iterator[None]:
    """Serve the app with a known key store and no live sources, whatever the environment says.

    Tests never reach real Jira or LLM APIs: end-to-end tests install a pipeline backed by the
    fake MCP server explicitly.
    """
    app.dependency_overrides[get_api_key_store] = lambda: ApiKeyStore([TEST_KEY_ENTRY])
    app.dependency_overrides[get_pipeline] = lambda: QueryPipeline(
        open_jira=None, site_url=None, projects=[], synthesizer=GroundedSynthesizer(None)
    )
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    """A client authenticated as `test-user`."""
    return TestClient(app, headers={"Authorization": f"Bearer {TEST_API_KEY}"})


@pytest.fixture
def anonymous_client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def client_for() -> Callable[..., TestClient]:
    """A client for a new key with the given scopes (the default test key keeps working)."""

    def make(*scopes: str, user_id: str = "other-user") -> TestClient:
        key = f"isk_{user_id}_" + "s" * 40
        entry = ApiKeyEntry(
            key_id=f"key-{user_id}", sha256=hash_api_key(key), user_id=user_id, scopes=list(scopes)
        )
        app.dependency_overrides[get_api_key_store] = lambda: ApiKeyStore([TEST_KEY_ENTRY, entry])
        return TestClient(app, headers={"Authorization": f"Bearer {key}"})

    return make


@pytest.fixture
def fake_jira() -> FakeJiraMcp:
    return FakeJiraMcp(ATLAS_ISSUES)


@pytest.fixture
def jira_settings() -> Callable[..., Settings]:
    def make(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "jira_base_url": SITE,
            "jira_email": "owner@example.com",
            "jira_api_token": "test-token",
            "jira_projects": ["ATLAS"],
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)

    return make


@pytest.fixture
def use_fake_jira(
    fake_jira: FakeJiraMcp, jira_settings: Callable[..., Settings]
) -> Callable[..., FakeJiraMcp]:
    """Serve /v1/query from the fake Atlassian MCP server, optionally with a fake LLM."""

    def install(
        provider: LlmProvider | None = None, projects: list[str] | None = None, **server: Any
    ) -> FakeJiraMcp:
        settings = jira_settings(jira_projects=projects or ["ATLAS"])
        pipeline = QueryPipeline(
            open_jira=partial(open_jira_connector, settings, fake_jira.server(**server)),
            site_url=SITE,
            projects=settings.jira_projects,
            synthesizer=GroundedSynthesizer(provider),
        )
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        return fake_jira

    return install
