"""Normalize raw Jira records into `Evidence`: one item per issue, comment and history entry.

Splitting an issue into separate items lets an answer cite exactly what supports a claim: the
comment that explains a slip, or the history entry that records it, rather than "somewhere in
ATLAS-3". Content is untrusted data from the source and is never interpreted as instructions.
"""

import re
from datetime import datetime
from typing import Any

from packages.common.models import Evidence
from packages.connectors.base import SourceRecord
from packages.guardrails.input import scan_untrusted_text

SOURCE = "jira"
AUTHORITY = "delivery_status"
MAX_CONTENT_CHARS = 2000
TRUNCATION_MARK = " [...]"

ISSUE_KEY = r"[A-Z][A-Z0-9_]+-\d+"
# Every location a Jira evidence item can have; citation checks validate against this.
LOCATION_PATTERN = re.compile(rf"^{ISSUE_KEY}(?:#(?:comment|change)-\d+)?$")

_FIELD_LABELS = {
    "duedate": "Due date",
    "status": "Status",
    "assignee": "Assignee",
    "priority": "Priority",
    "labels": "Labels",
    "summary": "Summary",
    "resolution": "Resolution",
    "Sprint": "Sprint",
}


def adf_to_text(node: Any) -> str:
    """Plain text from an Atlassian Document Format node (or pass a plain string through)."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(adf_to_text(child) for child in node)
    if not isinstance(node, dict):
        return ""
    kind = node.get("type")
    if kind == "text":
        return str(node.get("text", ""))
    if kind == "hardBreak":
        return "\n"
    if kind in {"mention", "emoji", "status", "date", "inlineCard"}:
        attrs = node.get("attrs") or {}
        return str(attrs.get("text") or attrs.get("shortName") or attrs.get("url") or "")
    inner = adf_to_text(node.get("content", []))
    if kind in {"paragraph", "heading", "listItem", "blockquote", "codeBlock", "rule"}:
        return inner.strip("\n") + "\n"
    return inner


def _clean(text: str) -> tuple[str, bool]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= MAX_CONTENT_CHARS:
        return text, False
    return text[: MAX_CONTENT_CHARS - len(TRUNCATION_MARK)].rstrip() + TRUNCATION_MARK, True


def _name(value: Any) -> str | None:
    if isinstance(value, dict):
        found = value.get("name") or value.get("displayName") or value.get("key")
        return str(found) if found else None
    return str(value) if value else None


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    # Jira writes offsets as +0000; Python's parser wants +00:00.
    value = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", value)
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _browse_url(site_url: str, key: str) -> str:
    return f"{site_url}/browse/{key}"


def _issue_content(key: str, fields: dict[str, Any]) -> str:
    lines = [
        f"{_name(fields.get('issuetype')) or 'Issue'} {key}: {fields.get('summary', '')}",
        f"Status: {_name(fields.get('status')) or 'unknown'}",
    ]
    if priority := _name(fields.get("priority")):
        lines.append(f"Priority: {priority}")
    if due := fields.get("duedate"):
        lines.append(f"Due date: {due}")
    if labels := fields.get("labels"):
        lines.append(f"Labels: {', '.join(map(str, labels))}")
    if parent := _name(fields.get("parent")):
        lines.append(f"Parent: {parent}")
    if assignee := _name(fields.get("assignee")):
        lines.append(f"Assignee: {assignee}")
    if description := adf_to_text(fields.get("description")).strip():
        lines += ["Description:", description]
    return "\n".join(lines)


def _change_content(key: str, entry: dict[str, Any]) -> str | None:
    author = _name(entry.get("author")) or "someone"
    changes = []
    for item in entry.get("items") or []:
        field = str(item.get("field", ""))
        label = _FIELD_LABELS.get(field, field or "A field")
        old, new = item.get("fromString") or "none", item.get("toString") or "none"
        changes.append(f"{label} changed from {old} to {new}")
    if not changes:
        return None
    return f"{key}: {'; '.join(changes)} (by {author})."


class _Numbering:
    def __init__(self, start: int) -> None:
        self.next = start

    def __call__(self) -> str:
        evidence_id = f"E{self.next}"
        self.next += 1
        return evidence_id


def normalize_jira_records(
    records: list[SourceRecord], site_url: str, *, first_id: int = 1
) -> list[Evidence]:
    """Turn connector records into evidence, numbered E<first_id>, E<first_id + 1>, ..."""
    new_id = _Numbering(first_id)
    evidence: list[Evidence] = []
    for record in records:
        issue = record.payload.get("issue") or {}
        key = str(issue.get("key", ""))
        if not re.fullmatch(ISSUE_KEY, key):
            continue  # nothing citable without a valid key
        fields = issue.get("fields") or {}
        url = _browse_url(site_url, key)
        base_meta = {"issue_key": key, "project": record.resource}

        content, truncated = _clean(_issue_content(key, fields))
        evidence.append(
            Evidence(
                evidence_id=new_id(),
                source=SOURCE,
                authority=AUTHORITY,
                title=f"{key}: {fields.get('summary', '')}",
                location=key,
                content=content,
                timestamp=_timestamp(fields.get("updated")),
                url=url,
                permission_context=record.permission_context,
                metadata={
                    **base_meta,
                    "kind": "issue",
                    "status": _name(fields.get("status")),
                    "issue_type": _name(fields.get("issuetype")),
                    "due_date": fields.get("duedate"),
                    "labels": list(fields.get("labels") or []),
                    "created": fields.get("created"),
                    "truncated": truncated,
                },
            )
        )
        for comment in record.payload.get("comments") or []:
            comment_id = str(comment.get("id", ""))
            text, truncated = _clean(adf_to_text(comment.get("body")))
            if not comment_id.isdigit() or not text:
                continue
            author = _name(comment.get("author")) or "unknown"
            evidence.append(
                Evidence(
                    evidence_id=new_id(),
                    source=SOURCE,
                    authority=AUTHORITY,
                    title=f"Comment on {key} by {author}",
                    location=f"{key}#comment-{comment_id}",
                    content=text,
                    timestamp=_timestamp(comment.get("created")),
                    url=f"{url}?focusedCommentId={comment_id}",
                    permission_context=record.permission_context,
                    metadata={
                        **base_meta,
                        "kind": "comment",
                        "author": author,
                        "truncated": truncated,
                    },
                )
            )
        for entry in record.payload.get("changelog") or []:
            change_id = str(entry.get("id", ""))
            change_text = _change_content(key, entry)
            if not change_id.isdigit() or not change_text:
                continue
            evidence.append(
                Evidence(
                    evidence_id=new_id(),
                    source=SOURCE,
                    authority=AUTHORITY,
                    title=f"Change to {key}",
                    location=f"{key}#change-{change_id}",
                    content=change_text,
                    timestamp=_timestamp(entry.get("created")),
                    url=url,
                    permission_context=record.permission_context,
                    metadata={
                        **base_meta,
                        "kind": "change",
                        "fields": [str(i.get("field")) for i in entry.get("items") or []],
                    },
                )
            )
    for item in evidence:
        if findings := scan_untrusted_text(f"{item.title}\n{item.content}"):
            item.metadata["injection_findings"] = sorted({f.rule_id for f in findings})
    return evidence
