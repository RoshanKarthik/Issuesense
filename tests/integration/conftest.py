"""Fixtures for connector and end-to-end tests."""

from collections.abc import Callable
from typing import Any

import pytest

from packages.common.settings import Settings
from packages.evaluation.atlas_dataset import ATLAS_ISSUES
from tests.integration.fake_atlassian import SITE, FakeJiraMcp


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
