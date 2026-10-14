from datetime import UTC, datetime

from packages.connectors.base import SourceRecord
from packages.evaluation.atlas_dataset import ATLAS_ISSUES
from packages.evidence.jira import (
    LOCATION_PATTERN,
    MAX_CONTENT_CHARS,
    TRUNCATION_MARK,
    adf_to_text,
    normalize_jira_records,
)
from tests.integration.fake_atlassian import SITE, FakeJiraMcp

FAKE = FakeJiraMcp(ATLAS_ISSUES)


def _record(ref: str, **payload_overrides: object) -> SourceRecord:
    key = FAKE.keys[ref]
    payload: dict[str, object] = {
        "issue": FAKE.issues[key],
        "comments": FAKE.comments[key],
        "changelog": FAKE.changelogs[key],
        "details_fetched": True,
    }
    payload.update(payload_overrides)
    return SourceRecord(
        source="jira",
        resource="ATLAS",
        tool="searchJiraIssuesUsingJql",
        payload=payload,
        permission_context="jira:read:ATLAS",
    )


def test_adf_to_text_handles_paragraphs_breaks_and_mentions() -> None:
    doc = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "Hello "},
                                              {"type": "mention", "attrs": {"text": "@Priya"}}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "a"}, {"type": "hardBreak"},
                                              {"type": "text", "text": "b"}]},
            {"type": "bulletList", "content": [{"type": "listItem", "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "item"}]}]}]},
        ],
    }  # fmt: skip
    assert adf_to_text(doc) == "Hello @Priya\na\nb\nitem\n"
    assert adf_to_text("plain") == "plain"
    assert adf_to_text(None) == ""


def test_one_issue_becomes_issue_comment_and_change_evidence() -> None:
    evidence = normalize_jira_records([_record("atlas-card-auth")], SITE)
    key = FAKE.keys["atlas-card-auth"]
    assert [e.metadata["kind"] for e in evidence] == [
        "issue",
        "comment",
        "comment",
        "change",
        "change",
    ]
    assert [e.evidence_id for e in evidence] == ["E1", "E2", "E3", "E4", "E5"]
    issue, first_comment, _, first_change, _ = evidence

    assert issue.location == key
    assert issue.url == f"{SITE}/browse/{key}"
    assert issue.title == f"{key}: Integrate the Payments v2 card authorization API"
    assert "Status: In Progress" in issue.content and "Due date: 2026-12-05" in issue.content
    assert "Labels: blocker" in issue.content
    assert issue.metadata["due_date"] == "2026-12-05"
    assert issue.permission_context == "jira:read:ATLAS"
    assert issue.source == "jira" and issue.authority == "delivery_status"
    assert issue.timestamp is not None and issue.timestamp.tzinfo is not None

    comment_id = FAKE.comments[key][0]["id"]
    assert first_comment.location == f"{key}#comment-{comment_id}"
    assert first_comment.url == f"{SITE}/browse/{key}?focusedCommentId={comment_id}"
    assert "PV2-4471" in first_comment.content
    assert first_comment.metadata["author"] == "Priya Shah"

    change_id = FAKE.changelogs[key][0]["id"]
    assert first_change.location == f"{key}#change-{change_id}"
    assert first_change.content == (
        f"{key}: Due date changed from 2026-11-07 to 2026-11-21 (by Priya Shah)."
    )


def test_every_location_matches_the_location_pattern() -> None:
    records = [_record(issue.ref) for issue in ATLAS_ISSUES]
    evidence = normalize_jira_records(records, SITE)
    assert evidence
    assert all(LOCATION_PATTERN.fullmatch(e.location) for e in evidence)
    ids = [e.evidence_id for e in evidence]
    assert len(ids) == len(set(ids))


def test_numbering_can_continue_from_earlier_evidence() -> None:
    evidence = normalize_jira_records([_record("atlas-runbook")], SITE, first_id=40)
    assert evidence[0].evidence_id == "E40"


def test_long_content_is_truncated_with_a_marker() -> None:
    record = _record("atlas-runbook", comments=[{"id": "1", "body": "x" * 5000}])
    comment = normalize_jira_records([record], SITE)[1]
    assert len(comment.content) == MAX_CONTENT_CHARS
    assert comment.content.endswith(TRUNCATION_MARK)
    assert comment.metadata["truncated"] is True


def test_items_without_valid_ids_are_skipped() -> None:
    record = _record(
        "atlas-runbook",
        comments=[{"id": "abc", "body": "no id"}, {"id": "7", "body": ""}],
        changelog=[{"id": "8", "items": []}, {"body": "no id"}],
    )
    evidence = normalize_jira_records([record], SITE)
    assert [e.metadata["kind"] for e in evidence] == ["issue"]
    bad_key = SourceRecord(
        source="jira", resource="ATLAS", tool="t", payload={"issue": {"key": "../etc"}},
        permission_context="jira:read:ATLAS",
    )  # fmt: skip
    assert normalize_jira_records([bad_key], SITE) == []


def test_jira_offsets_without_colons_are_parsed() -> None:
    issue = dict(FAKE.issues[FAKE.keys["atlas-runbook"]])
    issue["fields"] = {**issue["fields"], "updated": "2026-10-12T09:30:00.000+0530"}
    evidence = normalize_jira_records([_record("atlas-runbook", issue=issue)], SITE)
    assert evidence[0].timestamp == datetime(2026, 10, 12, 4, 0, tzinfo=UTC)


def test_injected_instructions_are_kept_as_plain_data() -> None:
    evidence = normalize_jira_records([_record("atlas-feedback-import")], SITE)
    assert "ignore all previous instructions" in evidence[0].content
    assert evidence[0].metadata["kind"] == "issue"


def test_planted_instructions_are_flagged_but_normal_requests_are_not() -> None:
    evidence = normalize_jira_records([_record("atlas-feedback-import")], SITE)
    assert evidence[0].metadata["injection_findings"] == ["ignore_instructions"]
    normal = _record(
        "atlas-runbook",
        comments=[{"id": "5", "body": "Please create a ticket for the rollback docs."}],
    )
    assert all(
        "injection_findings" not in e.metadata for e in normalize_jira_records([normal], SITE)
    )
