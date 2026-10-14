from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, SecretStr, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def _strip_trailing_slash(url: str) -> str:
    return url.rstrip("/")


# An https:// URL without a trailing slash, kept as a plain string.
HttpsUrl = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^https://[^\s/]+(/\S*)?$"),
    AfterValidator(_strip_trailing_slash),
]


class ApiKeyEntry(BaseModel):
    """One API key: only its SHA-256 hash is stored, never the key itself."""

    model_config = ConfigDict(extra="forbid")

    key_id: str
    sha256: Sha256Hex
    user_id: str
    scopes: list[str] = Field(default_factory=list)
    tenant_id: str | None = None


class Settings(BaseSettings):
    """Project-wide configuration, read from `ISSUESENSE_*` environment variables and `.env`."""

    env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # JSON list of ApiKeyEntry objects; create entries with `issuesense-create-key`. Empty means
    # every authenticated endpoint refuses all requests.
    api_keys: list[ApiKeyEntry] = Field(default_factory=list)

    # Jira (demo workspace). The site URL doubles as the MCP `cloudId`.
    jira_base_url: HttpsUrl | None = Field(default=None, validation_alias="JIRA_BASE_URL")
    jira_email: str | None = Field(default=None, validation_alias="JIRA_EMAIL")
    jira_api_token: SecretStr | None = Field(default=None, validation_alias="JIRA_API_TOKEN")
    jira_mcp_url: HttpsUrl = "https://mcp.atlassian.com/v2/mcp"
    # Project keys IssueSense searches when a question names none, e.g. ["ATLAS"].
    jira_projects: list[str] = Field(default_factory=list)

    # Resilience of calls to MCP servers: each tool call gets this long per attempt, failed or slow
    # calls are retried, and one whole retrieval (a search plus its details) is cut off after
    # `retrieval_timeout_seconds` without discarding what other retrievals found.
    mcp_call_timeout_seconds: float = Field(default=20.0, gt=0, le=300)
    mcp_call_attempts: int = Field(default=2, ge=1, le=5)
    mcp_retry_backoff_seconds: float = Field(default=0.5, ge=0, le=30)
    retrieval_timeout_seconds: float = Field(default=60.0, gt=0, le=600)

    # LLM provider for planning and written answers; empty means rule-based planning and the
    # deterministic evidence-listing answer. The model and key are required once a provider is set.
    llm_provider: str | None = Field(default=None, validation_alias="LLM_PROVIDER")
    llm_model: str | None = Field(default=None, validation_alias="LLM_MODEL")
    llm_api_key: SecretStr | None = Field(default=None, validation_alias="LLM_API_KEY")
    # Overrides the provider's default endpoint (for example a self-hosted NIM behind HTTPS).
    llm_base_url: HttpsUrl | None = Field(default=None, validation_alias="LLM_BASE_URL")
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=600)

    # Optional LangSmith tracing of graph runs; off unless the key is set.
    langsmith_api_key: SecretStr | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="issuesense", validation_alias="LANGSMITH_PROJECT")
    # Another region or a self-hosted LangSmith (default: LangSmith's own endpoint).
    langsmith_endpoint: HttpsUrl | None = Field(default=None, validation_alias="LANGSMITH_ENDPOINT")

    # Further connector and provider settings are added with the phase that introduces them.

    model_config = SettingsConfigDict(
        env_prefix="ISSUESENSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        # A blank `KEY=` line (as in .env.example) means "not set", not an invalid empty value.
        env_ignore_empty=True,
    )

    @property
    def jira_configured(self) -> bool:
        return bool(self.jira_base_url and self.jira_email and self.jira_api_token)


settings = Settings()
