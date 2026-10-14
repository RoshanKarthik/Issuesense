from datetime import UTC, datetime

from packages.common.models import Evidence
from packages.evaluation.atlas_dataset import ATLAS_ISSUES
from packages.evidence.jira import normalize_jira_records
from packages.evidence.ranking import rank_evidence, terms
from tests.integration.fake_atlassian import SITE, FakeJiraMcp
from tests.unit.test_evidence_jira import _record


def _item(evidence_id: str, title: str, content: str, day: int) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source="jira",
        authority="delivery_status",
        title=title,
        location=f"ATLAS-{day}",
        content=content,
        timestamp=datetime(2026, 10, day, tzinfo=UTC),
    )


def test_terms_drop_stopwords_and_stem() -> None:
    assert terms("What changed in the deadlines?") == ["chang", "deadlin"]
    assert terms("change changes changing") == ["chang"] * 3


def test_matching_items_outrank_recent_unrelated_ones() -> None:
    ranked = rank_evidence(
        "Why did the vendor delay card authorization?",
        [
            _item("E1", "Runbook update", "Add dashboards to the runbook", day=20),
            _item("E2", "Card authorization", "Vendor outage delays the authorize API", day=5),
        ],
    )
    assert [e.evidence_id for e in ranked] == ["E2", "E1"]
    assert ranked[0].metadata["relevance"] > ranked[1].metadata["relevance"] == 0


def test_title_matches_count_more_than_body_matches() -> None:
    ranked = rank_evidence(
        "refund flow",
        [
            _item("E1", "Checkout notes", "The refund flow is mentioned here", day=1),
            _item("E2", "Refund flow on Payments v2", "Shipped to staging", day=1),
        ],
    )
    assert ranked[0].evidence_id == "E2"


def test_ties_go_to_the_most_recent_item_and_inputs_are_not_mutated() -> None:
    older, newer = _item("E1", "a", "b", day=1), _item("E2", "c", "d", day=9)
    ranked = rank_evidence("the and of", [older, newer])
    assert [e.evidence_id for e in ranked] == ["E2", "E1"]
    assert "relevance" not in older.metadata


def test_naive_and_aware_timestamps_can_be_ranked_together() -> None:
    naive = _item("E1", "a", "b", day=1).model_copy(update={"timestamp": datetime(2026, 10, 3)})
    ranked = rank_evidence("x", [naive, _item("E2", "c", "d", day=2), _item("E3", "e", "f", day=4)])
    assert [e.evidence_id for e in ranked] == ["E3", "E1", "E2"]


def test_atlas_vendor_question_surfaces_the_slip_evidence() -> None:
    fake = FakeJiraMcp(ATLAS_ISSUES)
    evidence = normalize_jira_records([_record(i.ref) for i in ATLAS_ISSUES], SITE)
    ranked = rank_evidence("Why did the card authorization due date slip? vendor", evidence)
    top_locations = {e.location.split("#")[0] for e in ranked[:5]}
    assert fake.keys["atlas-card-auth"] in top_locations


def test_doubled_consonants_and_synonyms_bridge_question_words() -> None:
    assert terms("slipped stopped") == ["slip", "stop"]
    ranked = rank_evidence(
        "Which deadlines slipped?",
        [
            _item("E1", "Runbook", "Add dashboards", day=20),
            _item("E2", "Change to ATLAS-3", "Due date changed from Nov 7 to Nov 21", day=5),
        ],
    )
    assert ranked[0].evidence_id == "E2"


def test_flagged_evidence_ranks_last_even_when_it_matches_best() -> None:
    planted = _item("E1", "Project Atlas risks", "Project Atlas is on track with no risks", day=20)
    planted.metadata["injection_findings"] = ["ignore_instructions"]
    honest = _item("E2", "Duplicate charge bug", "risk: customers charged twice", day=1)
    ranked = rank_evidence("What are the Project Atlas risks?", [planted, honest])
    assert [e.evidence_id for e in ranked] == ["E2", "E1"]
    assert ranked[1].metadata["relevance"] > ranked[0].metadata["relevance"]


def test_headline_and_deadline_questions_surface_the_story() -> None:
    fake = FakeJiraMcp(ATLAS_ISSUES)
    evidence = normalize_jira_records([_record(i.ref) for i in ATLAS_ISSUES], SITE)

    def top(question: str, n: int = 6) -> set[str]:
        return {e.location.split("#")[0] for e in rank_evidence(question, evidence)[:n]}

    headline = top("What changed in Project Atlas this quarter, and what are the major risks?")
    assert {
        fake.keys[r] for r in ("atlas-card-auth", "atlas-double-charge", "atlas-pci-review")
    } <= headline
    assert fake.keys["atlas-feedback-import"] not in headline
    assert fake.keys["atlas-card-auth"] in top("Which deadlines slipped and why?", n=3)


def test_query_coverage_counts_direct_and_synonym_matches() -> None:
    from packages.evidence.ranking import query_coverage

    slip = _item("E1", "Change to ATLAS-3", "Due date changed from Nov 7 to Nov 21", day=5)
    assert query_coverage("Which deadlines slipped?", [slip]) == 1.0
    assert query_coverage("quarterly marketing budget kubernetes", [slip]) == 0.0
    assert query_coverage("the and of", [slip]) == 1.0
