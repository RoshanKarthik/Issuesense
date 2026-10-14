from datetime import UTC, datetime

import pytest

from packages.agent.synthesis import GroundedSynthesizer
from packages.common.models import Answer, Answerability, Citation, Claim, Evidence
from packages.evidence.citations import IssueCode, verify_answer

SLIP = Evidence(
    evidence_id="E1",
    source="jira",
    authority="delivery_status",
    title="Change to ATLAS-3",
    location="ATLAS-3#change-30030",
    content="ATLAS-3: Due date changed from 2026-11-07 to 2026-11-21 (by Priya Shah).",
    timestamp=datetime(2026, 10, 12, tzinfo=UTC),
    metadata={"relevance": 3.0},
)
VENDOR = Evidence(
    evidence_id="E2",
    source="jira",
    authority="delivery_status",
    title="Comment on ATLAS-3 by Priya Shah",
    location="ATLAS-3#comment-20030",
    content=(
        "The vendor sandbox returns HTTP 503 on most authorize calls. "
        "Opened vendor ticket PV2-4471."
    ),
    timestamp=datetime(2026, 10, 12, tzinfo=UTC),
    metadata={"relevance": 2.0},
)
EVIDENCE = [SLIP, VENDOR]


def _answer(*claims: tuple[str, list[str]], text: str = "Answer [E1].", **kw: object) -> Answer:
    built = [Claim(text=t, citations=[Citation(evidence_id=i) for i in ids]) for t, ids in claims]
    cited = list(dict.fromkeys(c.evidence_id for claim in built for c in claim.citations))
    return Answer.model_validate(
        {
            "text": text,
            "answerability": kw.get("answerability", Answerability.SUFFICIENT),
            "citations": [{"evidence_id": i} for i in cited],
            "claims": [claim.model_dump() for claim in built],
        }
    )


def _codes(answer: Answer, evidence: list[Evidence] = EVIDENCE) -> list[IssueCode]:
    return [issue.code for issue in verify_answer(answer, evidence).issues]


def test_a_supported_answer_verifies() -> None:
    answer = _answer(
        ("The ATLAS-3 due date changed from Nov 7 to Nov 21.", ["E1"]),
        ("The vendor sandbox returned HTTP 503 errors on authorize calls.", ["E2"]),
        text="It slipped to Nov 21 [E1] after vendor 503s [E2].",
    )
    report = verify_answer(answer, EVIDENCE)
    assert report.verified, report.issues
    assert (report.supported_claims, report.total_claims) == (2, 2)


def test_an_invented_date_is_caught() -> None:
    answer = _answer(("The ATLAS-3 due date changed to Nov 28.", ["E1"]))
    report = verify_answer(answer, EVIDENCE)
    assert [i.code for i in report.issues] == [IssueCode.UNSUPPORTED_NUMBER]
    assert "28" in report.issues[0].detail


def test_a_claim_citing_unrelated_evidence_is_unsupported() -> None:
    answer = _answer(("Refunds shipped to staging and finance verified them.", ["E2"]), text="[E2]")
    assert _codes(answer) == [IssueCode.UNSUPPORTED_CLAIM]


def test_unknown_evidence_ids_are_caught() -> None:
    answer = _answer(("The due date changed.", ["E9"]), text="See [E9].")
    assert IssueCode.UNKNOWN_EVIDENCE in _codes(answer)


def test_invalid_locations_are_caught() -> None:
    bad = SLIP.model_copy(update={"location": "../../etc/passwd"})
    answer = _answer(("The due date changed from Nov 7 to Nov 21.", ["E1"]))
    assert _codes(answer, [bad]) == [IssueCode.INVALID_LOCATION]


def test_unknown_sources_have_no_valid_locations() -> None:
    other = SLIP.model_copy(update={"source": "pastebin"})
    answer = _answer(("The due date changed from Nov 7 to Nov 21.", ["E1"]))
    assert _codes(answer, [other]) == [IssueCode.INVALID_LOCATION]


def test_quotes_must_appear_verbatim() -> None:
    answer = Answer(
        text="[E1]",
        answerability=Answerability.PARTIALLY_ANSWERABLE,
        citations=[Citation(evidence_id="E1")],
        claims=[
            Claim(
                text="Due date changed to Dec 1",
                citations=[Citation(evidence_id="E1", quote="Due date changed to Dec 1")],
            )
        ],
    )
    assert _codes(answer) == [IssueCode.QUOTE_NOT_FOUND]


def test_inline_references_must_be_cited_by_a_claim() -> None:
    answer = _answer(("The due date changed from Nov 7 to Nov 21.", ["E1"]), text="[E1] and [E2].")
    report = verify_answer(answer, EVIDENCE)
    assert [(i.code, i.evidence_id) for i in report.issues] == [(IssueCode.UNCITED_REFERENCE, "E2")]


def test_a_factual_answer_without_claims_fails() -> None:
    answer = Answer(text="All good.", answerability=Answerability.SUFFICIENT, citations=[])
    assert _codes(answer) == [IssueCode.MISSING_CLAIMS]


def test_an_insufficient_answer_needs_no_claims() -> None:
    answer = Answer(
        text="No evidence.", answerability=Answerability.INSUFFICIENT_EVIDENCE, citations=[]
    )
    assert verify_answer(answer, []).verified


def test_citation_list_must_match_the_claims() -> None:
    answer = _answer(("The due date changed from Nov 7 to Nov 21.", ["E1"]))
    answer.citations.append(Citation(evidence_id="E2"))
    assert _codes(answer) == [IssueCode.CITATIONS_MISMATCH]


@pytest.mark.parametrize(
    ("leak", "code"),
    [
        ("You are IssueSense. You answer questions...", IssueCode.PROMPT_LEAK),
        ("token ATATT3xFfGF0" + "a" * 30, IssueCode.SECRET_LEAK),
        ("key isk_" + "b" * 40, IssueCode.SECRET_LEAK),
        ("ghp_" + "c" * 36, IssueCode.SECRET_LEAK),
        ("-----BEGIN RSA PRIVATE KEY-----", IssueCode.SECRET_LEAK),
    ],
)
def test_output_leaks_are_caught(leak: str, code: IssueCode) -> None:
    answer = _answer(("The due date changed from Nov 7 to Nov 21.", ["E1"]), text=f"[E1] {leak}")
    assert code in _codes(answer)


async def test_deterministic_answers_always_verify() -> None:
    result = await GroundedSynthesizer(None).synthesize("Why did ATLAS-3 slip?", EVIDENCE)
    assert verify_answer(result.answer, result.evidence).verified


async def test_deterministic_answer_over_the_atlas_dataset_verifies() -> None:
    from packages.evaluation.atlas_dataset import ATLAS_ISSUES
    from packages.evidence.jira import normalize_jira_records
    from packages.evidence.ranking import rank_evidence
    from tests.integration.fake_atlassian import SITE
    from tests.unit.test_evidence_jira import _record

    evidence = normalize_jira_records([_record(i.ref) for i in ATLAS_ISSUES], SITE)
    question = "What changed in Project Atlas this quarter, and what are the major risks?"
    result = await GroundedSynthesizer(None).synthesize(question, rank_evidence(question, evidence))
    report = verify_answer(result.answer, result.evidence)
    assert report.verified, report.issues
    assert report.supported_claims == len(result.answer.claims) > 0
