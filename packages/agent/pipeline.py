"""The Phase 1 query pipeline: one question, Jira over MCP, a verified and cited answer.

    resolve projects -> check access -> retrieve -> normalize -> rank -> synthesize -> verify

Every outcome is an explicit answer: UNAUTHORIZED_EVIDENCE (the caller can read none of the
projects the question needs, so nothing is retrieved), TOOL_FAILURE (Jira unavailable),
INSUFFICIENT_EVIDENCE (nothing relevant), PARTIALLY_ANSWERABLE (quoted evidence, or some named
projects skipped). An answer that fails verification is never returned: an LLM answer falls back to
quoting, and if even that fails the answer is withheld. Phase 2 replaces this linear flow with a
typed graph.
"""

import logging
import re
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from pydantic import BaseModel, Field

from packages.agent.synthesis import GroundedSynthesizer, SynthesisResult
from packages.common.errors import ConnectorError
from packages.common.models import Answer, Answerability, AuthorizationContext, Evidence
from packages.connectors.base import Connector, RetrievalRequest, grant_for
from packages.evidence.citations import VerificationReport, verify_answer
from packages.evidence.jira import normalize_jira_records
from packages.evidence.ranking import rank_evidence

logger = logging.getLogger(__name__)

SOURCE = "jira"
_ISSUE_KEY = re.compile(r"\b([A-Z][A-Z0-9_]+)-\d+\b")
_WORD = re.compile(r"\b[A-Z][A-Z0-9_]+\b")

NOT_CONFIGURED_TEXT = "Jira is not configured on this server, so there is no answer."
UNAVAILABLE_TEXT = "Jira could not be reached, so there is no answer. Please try again later."
UNVERIFIED_TEXT = "The answer could not be verified against the evidence, so it was withheld."

JiraOpener = Callable[[], AbstractAsyncContextManager[Connector]]


class PipelineResult(BaseModel):
    answer: Answer
    evidence: list[Evidence] = Field(default_factory=list)
    verification: VerificationReport | None = None
    projects: list[str] = Field(default_factory=list)  # projects actually searched
    mode: str = "none"


def mentioned_projects(question: str, configured: list[str]) -> list[str]:
    """Configured project keys the question names directly ("ATLAS") or via an issue key."""
    named = [m.group(1) for m in _ISSUE_KEY.finditer(question)] + _WORD.findall(question)
    return [key for key in dict.fromkeys(named) if key in configured]


def _plain(text: str, answerability: Answerability, summary: str | None = None) -> Answer:
    return Answer(text=text, answerability=answerability, citations=[], reasoning_summary=summary)


class QueryPipeline:
    def __init__(
        self,
        *,
        open_jira: JiraOpener | None,
        site_url: str | None,
        projects: list[str],
        synthesizer: GroundedSynthesizer,
        retrieval_limit: int = 20,
    ) -> None:
        self._open_jira = open_jira
        self._site_url = site_url or ""
        self._projects = projects
        self._synthesizer = synthesizer
        self._limit = retrieval_limit

    async def answer(self, question: str, caller: AuthorizationContext) -> PipelineResult:
        if self._open_jira is None or not self._projects:
            return PipelineResult(answer=_plain(NOT_CONFIGURED_TEXT, Answerability.TOOL_FAILURE))

        named = mentioned_projects(question, self._projects)
        requested = named or self._projects
        granted = [p for p in requested if grant_for(caller, SOURCE, p)]
        skipped = [p for p in named if p not in granted]
        if not granted:
            logger.info("User %r cannot read any project the question needs", caller.user_id)
            # Name projects only when the caller named them; never reveal the configured list.
            text = (
                f"You are not authorized to read the Jira project(s) {', '.join(skipped)}."
                if skipped
                else "You are not authorized to read any of the Jira projects IssueSense searches."
            )
            return PipelineResult(answer=_plain(text, Answerability.UNAUTHORIZED_EVIDENCE))

        try:
            async with self._open_jira() as jira:
                retrieved = await jira.retrieve(
                    caller, RetrievalRequest(query=question, resources=granted, limit=self._limit)
                )
        except ConnectorError:
            logger.exception("Jira retrieval failed for projects %s", ", ".join(granted))
            return PipelineResult(
                answer=_plain(UNAVAILABLE_TEXT, Answerability.TOOL_FAILURE), projects=granted
            )

        evidence = normalize_jira_records(retrieved.records, self._site_url)
        ranked = rank_evidence(question, evidence)
        synthesis = await self._synthesizer.synthesize(question, ranked)
        report = verify_answer(synthesis.answer, synthesis.evidence)
        if not report.verified and synthesis.mode == "llm":
            logger.warning(
                "LLM answer failed verification (%s); quoting evidence instead",
                ", ".join(sorted({issue.code.value for issue in report.issues})),
            )
            synthesis = await self._synthesizer.synthesize(question, ranked, allow_llm=False)
            report = verify_answer(synthesis.answer, synthesis.evidence)
        if not report.verified:
            logger.error("Answer withheld: %s", [issue.code.value for issue in report.issues])
            return PipelineResult(
                answer=_plain(UNVERIFIED_TEXT, Answerability.INSUFFICIENT_EVIDENCE),
                verification=report,
                projects=granted,
            )
        return self._finish(synthesis, report, granted, skipped)

    @staticmethod
    def _finish(
        synthesis: SynthesisResult,
        report: VerificationReport,
        searched: list[str],
        skipped: list[str],
    ) -> PipelineResult:
        answer = synthesis.answer
        if skipped:
            note = f"Skipped Jira project(s) you cannot read: {', '.join(skipped)}."
            answer.reasoning_summary = f"{answer.reasoning_summary or ''} {note}".strip()
            if answer.answerability is Answerability.SUFFICIENT:
                answer.answerability = Answerability.PARTIALLY_ANSWERABLE
        return PipelineResult(
            answer=answer,
            evidence=synthesis.evidence,
            verification=report,
            projects=searched,
            mode=synthesis.mode,
        )
