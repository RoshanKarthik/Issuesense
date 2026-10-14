"""The query pipeline: runs the typed query graph (`packages.agent.graph`) for one question.

Every outcome is an explicit answer: UNAUTHORIZED_EVIDENCE (the caller can read none of the
projects the question needs, so nothing is retrieved), TOOL_FAILURE (Jira unavailable),
INSUFFICIENT_EVIDENCE (nothing relevant), PARTIALLY_ANSWERABLE (quoted evidence, or some named
projects skipped). An answer that fails verification is never returned: an LLM answer falls back to
quoting, and if even that fails the answer is withheld.
"""

from collections.abc import Callable
from datetime import date

from pydantic import BaseModel, Field

from packages.agent.graph import RunContext, build_graph
from packages.agent.planner import Planner, QueryPlan, RulePlanner
from packages.agent.state import GraphState
from packages.agent.synthesis import GroundedSynthesizer
from packages.common.models import Answer, AuthorizationContext, Evidence, utc_now
from packages.connectors.lazy import ConnectorOpener, LazyConnector
from packages.evidence.citations import VerificationReport

JiraOpener = ConnectorOpener


class PipelineResult(BaseModel):
    answer: Answer
    evidence: list[Evidence] = Field(default_factory=list)
    verification: VerificationReport | None = None
    projects: list[str] = Field(default_factory=list)  # projects actually searched
    mode: str = "none"
    plan: QueryPlan | None = None


class QueryPipeline:
    def __init__(
        self,
        *,
        open_jira: JiraOpener | None,
        site_url: str | None,
        projects: list[str],
        synthesizer: GroundedSynthesizer,
        planner: Planner | None = None,
        today: Callable[[], date] | None = None,
        retrieval_limit: int = 20,
    ) -> None:
        self._open_jira = open_jira
        self._site_url = site_url or ""
        self._projects = projects
        self._synthesizer = synthesizer
        self._planner = planner or RulePlanner()
        self._today = today or (lambda: utc_now().date())
        self._limit = retrieval_limit
        self._graph = build_graph()

    async def answer(self, question: str, caller: AuthorizationContext) -> PipelineResult:
        jira = LazyConnector(self._open_jira, "jira") if self._open_jira else None
        context = RunContext(
            jira=jira,
            site_url=self._site_url,
            projects=self._projects,
            synthesizer=self._synthesizer,
            planner=self._planner,
            today=self._today,
            retrieval_limit=self._limit,
        )
        try:
            output = await self._graph.ainvoke(
                GraphState(question=question, caller=caller), context=context
            )
        finally:
            if jira is not None:
                await jira.aclose()
        return self._result(GraphState.model_validate(output))

    @staticmethod
    def _result(state: GraphState) -> PipelineResult:
        assert state.answer is not None
        verified = state.verification is not None and state.verification.verified
        # Evidence is returned only with a verified synthesized answer, never with a withheld one.
        synthesis = state.synthesis if verified else None
        return PipelineResult(
            answer=state.answer,
            evidence=synthesis.evidence if synthesis else [],
            verification=state.verification,
            projects=state.projects,
            mode=synthesis.mode if synthesis else "none",
            plan=state.plan,
        )
