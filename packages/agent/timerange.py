"""Resolve relative time phrases in a question ("this quarter", "last month") into dates.

Deterministic on purpose: dates come from the calendar and the clock, never from a language model.
Quarters and months are calendar ones; weeks start on Monday; every range is inclusive.
"""

import re
from datetime import date, timedelta

from pydantic import BaseModel


class TimeRange(BaseModel):
    start: date
    end: date
    phrase: str  # the words in the question that set the range

    def describe(self) -> str:
        return f"{self.start.isoformat()} to {self.end.isoformat()}"


_UNITS = {"day": 1, "week": 7, "month": 30}
_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "a": 1, "an": 1}

_QUARTER_NAMED = re.compile(r"\bq([1-4])(?:\s*(?:of\s+)?((?:19|20)\d{2}))?\b", re.I)
_RELATIVE = re.compile(
    r"\b(this|current|last|previous|past|next)\s+(quarter|month|week|year)\b", re.I
)
_LAST_N = re.compile(
    r"\b(?:last|past|previous)\s+(\d{1,3}|one|two|three|four|five|six|a|an)\s+"
    r"(day|week|month)s?\b",
    re.I,
)
_SINCE_TODAY = re.compile(r"\b(today|yesterday)\b", re.I)


def _quarter(year: int, quarter: int) -> tuple[date, date]:
    start = date(year, 3 * quarter - 2, 1)
    end = _month_end(year, 3 * quarter)
    return start, end


def _month_end(year: int, month: int) -> date:
    first_of_next = date(year + month // 12, month % 12 + 1, 1)
    return first_of_next - timedelta(days=1)


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) + delta
    return index // 12, index % 12 + 1


def _relative(which: str, unit: str, today: date) -> tuple[date, date]:
    step = {"this": 0, "current": 0, "last": -1, "previous": -1, "past": -1, "next": 1}[which]
    if unit == "quarter":
        quarter = (today.month - 1) // 3 + 1
        year, month = _shift_month(today.year, 3 * quarter - 2, 3 * step)
        return _quarter(year, (month - 1) // 3 + 1)
    if unit == "month":
        year, month = _shift_month(today.year, today.month, step)
        return date(year, month, 1), _month_end(year, month)
    if unit == "week":
        monday = today - timedelta(days=today.weekday()) + timedelta(weeks=step)
        return monday, monday + timedelta(days=6)
    return date(today.year + step, 1, 1), date(today.year + step, 12, 31)


def resolve_time_range(question: str, today: date) -> TimeRange | None:
    """The first time phrase in the question, as dates; None when it names no period."""
    candidates: list[tuple[int, TimeRange]] = []
    if match := _QUARTER_NAMED.search(question):
        year = int(match.group(2)) if match.group(2) else today.year
        start, end = _quarter(year, int(match.group(1)))
        candidates.append((match.start(), TimeRange(start=start, end=end, phrase=match.group(0))))
    if match := _LAST_N.search(question):
        raw = match.group(1).lower()
        count = _NUMBERS.get(raw) or int(raw)
        days = count * _UNITS[match.group(2).lower()]
        start = today - timedelta(days=days)
        candidates.append((match.start(), TimeRange(start=start, end=today, phrase=match.group(0))))
    elif match := _RELATIVE.search(question):
        start, end = _relative(match.group(1).lower(), match.group(2).lower(), today)
        candidates.append((match.start(), TimeRange(start=start, end=end, phrase=match.group(0))))
    if match := _SINCE_TODAY.search(question):
        day = today if match.group(1).lower() == "today" else today - timedelta(days=1)
        candidates.append((match.start(), TimeRange(start=day, end=day, phrase=match.group(0))))
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate[0])[1]
