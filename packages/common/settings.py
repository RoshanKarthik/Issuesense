from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


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

    # Connector and provider settings are added with the phase that introduces them.

    model_config = SettingsConfigDict(
        env_prefix="ISSUESENSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
