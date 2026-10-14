import json

import pytest
from pydantic import ValidationError

from apps.api.auth import (
    KEY_PREFIX,
    ApiKeyStore,
    create_key_cli,
    generate_api_key,
    hash_api_key,
)
from packages.common.settings import ApiKeyEntry, Settings


def _entry(key: str, key_id: str = "k1", **overrides: object) -> ApiKeyEntry:
    fields: dict[str, object] = {
        "key_id": key_id,
        "sha256": hash_api_key(key),
        "user_id": "alice",
        "scopes": ["jira:read:ATLAS"],
    }
    fields.update(overrides)
    return ApiKeyEntry.model_validate(fields)


def test_generated_keys_are_prefixed_long_and_unique() -> None:
    keys = {generate_api_key() for _ in range(50)}
    assert len(keys) == 50
    assert all(key.startswith(KEY_PREFIX) and len(key) >= 40 for key in keys)


def test_hash_is_sha256_hex() -> None:
    assert hash_api_key("isk_abc") == hash_api_key("isk_abc")
    assert len(hash_api_key("isk_abc")) == 64
    assert hash_api_key("isk_abc") != hash_api_key("isk_abd")


def test_store_returns_the_matching_callers_context() -> None:
    alice, bob = generate_api_key(), generate_api_key()
    store = ApiKeyStore(
        [_entry(alice), _entry(bob, key_id="k2", user_id="bob", scopes=["jira:read:*"])]
    )
    caller = store.authenticate(bob)
    assert caller is not None
    assert (caller.user_id, caller.scopes, caller.credential_id) == ("bob", ["jira:read:*"], "k2")


@pytest.mark.parametrize("presented", ["", "isk_wrong", "Bearer isk_x", " "])
def test_store_rejects_unknown_keys(presented: str) -> None:
    assert ApiKeyStore([_entry(generate_api_key())]).authenticate(presented) is None


def test_empty_store_rejects_everything() -> None:
    assert ApiKeyStore([]).authenticate(generate_api_key()) is None


def test_store_rejects_duplicate_key_ids() -> None:
    with pytest.raises(ValueError, match="unique key_id"):
        ApiKeyStore([_entry("isk_a"), _entry("isk_b")])


def test_caller_scopes_are_a_copy() -> None:
    key = generate_api_key()
    entry = _entry(key)
    caller = ApiKeyStore([entry]).authenticate(key)
    assert caller is not None
    caller.scopes.append("jira:read:*")
    assert entry.scopes == ["jira:read:ATLAS"]


def test_entry_rejects_a_plaintext_key_in_place_of_a_hash() -> None:
    with pytest.raises(ValidationError):
        ApiKeyEntry(key_id="k", sha256="isk_plaintext", user_id="alice")


def test_settings_parse_api_keys_from_json(monkeypatch: pytest.MonkeyPatch) -> None:
    key = generate_api_key()
    monkeypatch.setenv(
        "ISSUESENSE_API_KEYS", json.dumps([_entry(key).model_dump(exclude_none=True)])
    )
    keys = Settings(_env_file=None).api_keys
    assert [(k.key_id, k.user_id, k.scopes) for k in keys] == [("k1", "alice", ["jira:read:ATLAS"])]


def test_create_key_cli_prints_a_key_matching_its_entry(capsys: pytest.CaptureFixture[str]) -> None:
    create_key_cli(["--user-id", "alice", "--scope", "jira:read:ATLAS", "--key-id", "laptop"])
    lines = capsys.readouterr().out.splitlines()
    key = lines[0].rsplit(" ", 1)[1]
    entry = ApiKeyEntry.model_validate(json.loads(lines[-1]))
    assert key.startswith(KEY_PREFIX)
    assert entry.sha256 == hash_api_key(key)
    assert (entry.key_id, entry.user_id, entry.scopes) == ("laptop", "alice", ["jira:read:ATLAS"])
    assert key not in lines[-1]
