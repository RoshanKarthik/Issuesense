"""API-key authentication: turns a bearer key into the caller's `AuthorizationContext`."""

import argparse
import hashlib
import hmac
import json
import logging
import secrets
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from packages.common.errors import AuthenticationError
from packages.common.models import AuthorizationContext
from packages.common.settings import ApiKeyEntry, settings

logger = logging.getLogger(__name__)

KEY_PREFIX = "isk_"


def generate_api_key() -> str:
    """A new random key: 256 bits of entropy, prefixed so leaked keys are easy to scan for."""
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    # Keys are long random strings, not passwords, so a fast hash is enough: there is nothing to
    # brute-force from the hash.
    return hashlib.sha256(key.encode()).hexdigest()


class ApiKeyStore:
    def __init__(self, entries: list[ApiKeyEntry]) -> None:
        ids = [entry.key_id for entry in entries]
        if len(ids) != len(set(ids)):
            raise ValueError("API key entries must have unique key_id values")
        self._entries = list(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def authenticate(self, presented_key: str) -> AuthorizationContext | None:
        digest = hash_api_key(presented_key)
        match: ApiKeyEntry | None = None
        # Compare against every entry in constant time, so timing reveals neither which entry
        # matched nor how much of a hash did.
        for entry in self._entries:
            if hmac.compare_digest(digest, entry.sha256):
                match = entry
        if match is None:
            return None
        return AuthorizationContext(
            user_id=match.user_id,
            scopes=list(match.scopes),
            tenant_id=match.tenant_id,
            credential_id=match.key_id,
        )


@lru_cache(maxsize=1)
def get_api_key_store() -> ApiKeyStore:
    store = ApiKeyStore(settings.api_keys)
    if not len(store):
        logger.warning("No API keys are configured: every authenticated request will be refused.")
    return store


_bearer = HTTPBearer(auto_error=False, description="An IssueSense API key (`isk_...`).")


async def require_caller(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    store: Annotated[ApiKeyStore, Depends(get_api_key_store)],
) -> AuthorizationContext:
    """FastAPI dependency: the authenticated caller, or a 401."""
    if credentials is None:
        raise AuthenticationError("Missing API key: send `Authorization: Bearer <key>`.")
    caller = store.authenticate(credentials.credentials)
    if caller is None:
        logger.warning("Rejected an invalid API key")
        raise AuthenticationError("Invalid API key.")
    return caller


Caller = Annotated[AuthorizationContext, Depends(require_caller)]


def create_key_cli(argv: list[str] | None = None) -> None:
    """`issuesense-create-key`: print a new key once, plus its ISSUESENSE_API_KEYS entry."""
    parser = argparse.ArgumentParser(description="Create an IssueSense API key.")
    parser.add_argument("--user-id", required=True, help="Who the key belongs to.")
    parser.add_argument(
        "--scope",
        action="append",
        default=[],
        dest="scopes",
        help="A granted scope such as jira:read:ATLAS; repeat for several.",
    )
    parser.add_argument("--key-id", help="A name for the key (default: random).")
    args = parser.parse_args(argv)

    key = generate_api_key()
    entry = ApiKeyEntry(
        key_id=args.key_id or f"key_{secrets.token_hex(4)}",
        sha256=hash_api_key(key),
        user_id=args.user_id,
        scopes=args.scopes,
    )
    print(f"API key (shown once, store it safely): {key}")
    print("Add this entry to the ISSUESENSE_API_KEYS JSON list:")
    print(json.dumps(entry.model_dump(exclude_none=True)))
