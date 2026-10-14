"""The evidence pool of one query run: every item retrieved so far, across sub-questions and hops.

Items are deduplicated by location (the same comment found by two retrievals is one item), keep
the ID they were first given, and record which sub-questions found them. New items are numbered
after the existing ones, so IDs shown in a live trace never change later in the run.
"""

import re
from datetime import date, datetime

from packages.common.models import Evidence


def merge_evidence(
    pool: list[Evidence], found: list[Evidence], tag: str
) -> tuple[list[Evidence], list[Evidence]]:
    """The pool with `found` merged in, and the items that were new to it."""
    merged = list(pool)
    index = {item.location: position for position, item in enumerate(merged)}
    added: list[Evidence] = []
    for item in found:
        position = index.get(item.location)
        if position is not None:
            existing = merged[position]
            tags = list(existing.metadata.get("sub_questions", []))
            if tag not in tags:
                merged[position] = existing.model_copy(
                    update={"metadata": {**existing.metadata, "sub_questions": [*tags, tag]}}
                )
            continue
        fresh = item.model_copy(
            update={
                "evidence_id": f"E{len(merged) + 1}",
                "metadata": {**item.metadata, "sub_questions": [tag]},
            }
        )
        index[fresh.location] = len(merged)
        merged.append(fresh)
        added.append(fresh)
    return merged, added


def within_period(evidence: list[Evidence], start: date, end: date) -> list[Evidence]:
    """The items dated inside [start, end]: comments and history entries by their own date, issues
    when created or last updated inside it.

    A source search can only bound activity from below ("updated since"): an issue updated after the
    period still matches, and its current fields describe now, not then. This drops what the period
    did not contain, so a question about last quarter is not answered with this quarter's state.
    """
    kept = []
    for item in evidence:
        dates = [item.timestamp]
        if item.metadata.get("kind") == "issue":
            dates.append(_parse(item.metadata.get("created")))
        if any(moment is not None and start <= moment.date() <= end for moment in dates):
            kept.append(item)
    return kept


def _parse(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", value))
    except ValueError:
        return None
