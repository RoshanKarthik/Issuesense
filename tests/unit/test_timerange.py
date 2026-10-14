from datetime import date

import pytest

from packages.agent.timerange import resolve_time_range

TODAY = date(2026, 10, 15)  # a Thursday in Q4


@pytest.mark.parametrize(
    ("question", "start", "end"),
    [
        ("What changed this quarter?", date(2026, 10, 1), date(2026, 12, 31)),
        ("What changed in the current quarter?", date(2026, 10, 1), date(2026, 12, 31)),
        ("What slipped last quarter?", date(2026, 7, 1), date(2026, 9, 30)),
        ("Plans for next quarter?", date(2027, 1, 1), date(2027, 3, 31)),
        ("Risks in Q2?", date(2026, 4, 1), date(2026, 6, 30)),
        ("Risks in Q4 2025?", date(2025, 10, 1), date(2025, 12, 31)),
        ("What happened this month?", date(2026, 10, 1), date(2026, 10, 31)),
        ("What happened last month?", date(2026, 9, 1), date(2026, 9, 30)),
        ("Anything this week?", date(2026, 10, 12), date(2026, 10, 18)),
        ("Anything last week?", date(2026, 10, 5), date(2026, 10, 11)),
        ("Changes in the past 2 weeks?", date(2026, 10, 1), TODAY),
        ("Changes in the last three days?", date(2026, 10, 12), TODAY),
        ("What changed this year?", date(2026, 1, 1), date(2026, 12, 31)),
        ("What changed yesterday?", date(2026, 10, 14), date(2026, 10, 14)),
    ],
)
def test_time_phrases_resolve_to_calendar_dates(question: str, start: date, end: date) -> None:
    resolved = resolve_time_range(question, TODAY)
    assert resolved is not None
    assert (resolved.start, resolved.end) == (start, end)
    assert resolved.phrase.lower() in question.lower()


def test_last_quarter_in_january_is_the_previous_year() -> None:
    resolved = resolve_time_range("What slipped last quarter?", date(2027, 1, 5))
    assert resolved is not None
    assert (resolved.start, resolved.end) == (date(2026, 10, 1), date(2026, 12, 31))


def test_questions_without_a_period_have_no_range() -> None:
    assert resolve_time_range("What blocks the card authorization work?", TODAY) is None


def test_the_first_phrase_wins() -> None:
    resolved = resolve_time_range("Compare this month with Q1", TODAY)
    assert resolved is not None and resolved.phrase == "this month"


def test_describe_is_iso_dates() -> None:
    resolved = resolve_time_range("this quarter", TODAY)
    assert resolved is not None and resolved.describe() == "2026-10-01 to 2026-12-31"
