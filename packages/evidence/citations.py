"""Deterministic verification of an answer against the evidence it cites.

Checks, per claim: every cited ID exists, its location is valid for its source, any quote appears
verbatim in the evidence, and the claim is supported — most of its content words, and every number
and date it states, appear in the cited evidence. Then whole-answer and output checks: inline
"[E#]" references must be cited, a factual answer must carry claims, and the text must not leak the
system prompt or credentials.

Lexical support is a floor, not proof of entailment: it catches invented numbers, dates and
off-topic citations cheaply. Phase 3 adds stronger support checks.
"""

import re
from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, Field

from packages.common.models import Answer, Answerability, Evidence
from packages.evidence import jira
from packages.evidence.ranking import terms

MIN_TERM_COVERAGE = 0.5

LOCATION_PATTERNS: dict[str, re.Pattern[str]] = {jira.SOURCE: jira.LOCATION_PATTERN}

_MONTHS = {
    name: number
    for number, names in enumerate(
        [
            ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
            ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
            ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for name in names
}  # fmt: skip
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_MONTH_WORD = re.compile(r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\b", re.I)
_INLINE_REF = re.compile(r"\[(E\d+)\]")
_SECRET_PATTERNS = {
    "atlassian_token": re.compile(r"\bATATT[A-Za-z0-9_\-=]{20,}"),
    "issuesense_key": re.compile(r"\bisk_[A-Za-z0-9_\-]{20,}"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
_PROMPT_MARKERS = ("You are IssueSense.", "Reply with a single JSON object")


class IssueCode(StrEnum):
    UNKNOWN_EVIDENCE = "UNKNOWN_EVIDENCE"
    INVALID_LOCATION = "INVALID_LOCATION"
    QUOTE_NOT_FOUND = "QUOTE_NOT_FOUND"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    UNSUPPORTED_NUMBER = "UNSUPPORTED_NUMBER"
    UNCITED_REFERENCE = "UNCITED_REFERENCE"
    MISSING_CLAIMS = "MISSING_CLAIMS"
    CITATIONS_MISMATCH = "CITATIONS_MISMATCH"
    PROMPT_LEAK = "PROMPT_LEAK"
    SECRET_LEAK = "SECRET_LEAK"


class VerificationIssue(BaseModel):
    code: IssueCode
    detail: str
    claim_index: int | None = None
    evidence_id: str | None = None


class VerificationReport(BaseModel):
    issues: list[VerificationIssue] = Field(default_factory=list)
    total_claims: int = 0
    supported_claims: int = 0

    @property
    def verified(self) -> bool:
        return not self.issues


def _numbers(text: str) -> set[str]:
    """Numbers stated in text, with month names read as month numbers ("Nov 21" -> {11, 21})."""
    found = {str(int(float(n))) if "." not in n else n for n in _NUMBER.findall(text)}
    found |= {str(_MONTHS[m.lower()]) for m in _MONTH_WORD.findall(text)}
    return found


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _check_outputs(text: str) -> Iterable[VerificationIssue]:
    for marker in _PROMPT_MARKERS:
        if marker in text:
            yield VerificationIssue(
                code=IssueCode.PROMPT_LEAK, detail="answer repeats the instructions"
            )
    for name, pattern in _SECRET_PATTERNS.items():
        if pattern.search(text):
            yield VerificationIssue(code=IssueCode.SECRET_LEAK, detail=f"answer contains a {name}")


def verify_answer(answer: Answer, evidence: list[Evidence]) -> VerificationReport:
    by_id = {item.evidence_id: item for item in evidence}
    report = VerificationReport(total_claims=len(answer.claims))
    issues = report.issues

    if answer.answerability in {Answerability.SUFFICIENT, Answerability.PARTIALLY_ANSWERABLE}:
        if not answer.claims:
            issues.append(
                VerificationIssue(
                    code=IssueCode.MISSING_CLAIMS, detail="a factual answer needs claims"
                )
            )

    for index, claim in enumerate(answer.claims):
        cited = []
        for citation in claim.citations:
            item = by_id.get(citation.evidence_id)
            if item is None:
                issues.append(
                    VerificationIssue(
                        code=IssueCode.UNKNOWN_EVIDENCE,
                        detail="cited evidence is not in the response",
                        claim_index=index,
                        evidence_id=citation.evidence_id,
                    )
                )
                continue
            pattern = LOCATION_PATTERNS.get(item.source)
            if pattern is None or not pattern.fullmatch(item.location):
                issues.append(
                    VerificationIssue(
                        code=IssueCode.INVALID_LOCATION,
                        detail=f"invalid location {item.location!r}",
                        claim_index=index,
                        evidence_id=item.evidence_id,
                    )
                )
                continue
            if citation.quote is not None and _squash(citation.quote) not in _squash(item.content):
                issues.append(
                    VerificationIssue(
                        code=IssueCode.QUOTE_NOT_FOUND,
                        detail="quote does not appear in the evidence",
                        claim_index=index,
                        evidence_id=item.evidence_id,
                    )
                )
                continue
            cited.append(item)
        if len(cited) != len(claim.citations):
            continue

        support_text = "\n".join(f"{item.title or ''}\n{item.content}" for item in cited)
        claim_terms = set(terms(claim.text))
        coverage = (
            len(claim_terms & set(terms(support_text))) / len(claim_terms) if claim_terms else 1.0
        )
        missing_numbers = sorted(_numbers(claim.text) - _numbers(support_text), key=int)
        if missing_numbers:
            issues.append(
                VerificationIssue(
                    code=IssueCode.UNSUPPORTED_NUMBER,
                    detail=f"numbers not in the cited evidence: {', '.join(missing_numbers)}",
                    claim_index=index,
                )
            )
        elif coverage < MIN_TERM_COVERAGE:
            issues.append(
                VerificationIssue(
                    code=IssueCode.UNSUPPORTED_CLAIM,
                    detail=f"only {coverage:.0%} of the claim's terms appear in the cited evidence",
                    claim_index=index,
                )
            )
        else:
            report.supported_claims += 1

    claim_ids = [c.evidence_id for claim in answer.claims for c in claim.citations]
    if set(claim_ids) != {c.evidence_id for c in answer.citations}:
        issues.append(
            VerificationIssue(
                code=IssueCode.CITATIONS_MISMATCH,
                detail="the answer's citation list differs from its claims' citations",
            )
        )
    for reference in sorted(set(_INLINE_REF.findall(answer.text)) - set(claim_ids)):
        issues.append(
            VerificationIssue(
                code=IssueCode.UNCITED_REFERENCE,
                detail="the text refers to evidence no claim cites",
                evidence_id=reference,
            )
        )
    issues.extend(_check_outputs(answer.text))
    return report
