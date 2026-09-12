
import pytest

from packages.common.errors import AuthorizationError, IssueSenseError
from packages.common.models import Answerability, AuthorizationContext, Evidence
from packages.common.settings import settings


def test_settings_defaults():
    assert settings.ENVIRONMENT in ["dev", "test", "prod"]
    assert settings.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

def test_error_hierarchy():
    with pytest.raises(IssueSenseError):
        raise AuthorizationError("Unauthorized access")

def test_evidence_model_validation():
    # Valid evidence
    ev = Evidence(
        evidence_id="E1",
        source="github",
        authority="implementation",
        location="repo/main.py#L1",
        content="print('hello')"
    )
    assert ev.evidence_id == "E1"

    # Invalid evidence (missing required fields)
    with pytest.raises(ValueError):
        Evidence(evidence_id="E2") # Missing source, authority, location, content

def test_authorization_context():
    ctx = AuthorizationContext(user_id="user123", scopes=["read:repo"])
    assert ctx.user_id == "user123"
    assert "read:repo" in ctx.scopes

def test_answerability_enum():
    assert Answerability.SUFFICIENT == "SUFFICIENT"
    assert "CONFLICTING_EVIDENCE" in Answerability.__members__.values()
