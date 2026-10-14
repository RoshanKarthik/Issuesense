"""The query pipeline: runs the typed query graph (`packages.agent.graph`) for one question.

Every outcome is an explicit answer: UNAUTHORIZED_EVIDENCE (the caller can read none of the
projects the question needs, so nothing is retrieved), TOOL_FAILURE (Jira unavailable),
INSUFFICIENT_EVIDENCE (nothing relevant), PARTIALLY_ANSWERABLE (quoted evidence, or some named
projects skipped). An answer that fails verification is never returned: an LLM answer falls back to
quoting, and if even that fails the answer is withheld.
"""

import asyncio
import uuid
from collections.abc import Callable
from datetime import date

from pydantic import BaseModel, Field

from packages.agent.context import RunContext
from packages.agent.graph import build_graph
from packages.agent.planner import Planner, QueryPlan, RulePlanner
from packages.agent.state import Budget, GraphState, RetrievalRecord
from packages.agent.synthesis import GroundedSynthesizer
from packages.common.models import Answer, AuthorizationContext, Evidence, utc_now
from packages.connectors.lazy import ConnectorOpener, LazyConnector
from packages.evidence.citations import VerificationReport
from packages.observability import langsmith
from packages.observability.langsmith import LangSmithTracing
from packages.observability.trace import TraceRecorder

JiraOpener = ConnectorOpener


class PipelineResult(BaseModel):
    answer: Answer
    evidence: list[Evidence] = Field(default_factory=list)
    verification: VerificationReport | None = None
    projects: list[str] = Field(default_factory=list)  # projects actually searched
    mode: str = "none"
    plan: QueryPlan | None = None
    retrievals: list[RetrievalRecord] = Field(default_factory=list)
    llm_tokens: int = 0


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
        budget: Budget | None = None,
        retrieval_limit: int = 20,
        retrieval_timeout: float = 60.0,
        tracing: LangSmithTracing | None = None,
    ) -> None:
        self._open_jira = open_jira
        self._site_url = site_url or ""
        self._projects = projects
        self._synthesizer = synthesizer
        self._planner = planner or RulePlanner()
        self._today = today or (lambda: utc_now().date())
        self._budget = budget or Budget()
        self._limit = retrieval_limit
        self._timeout = retrieval_timeout
        self._tracing = tracing or langsmith.disabled()
        self._graph = build_graph()

    @property
    def step_limit(self) -> int:
        """LangGraph's recursion limit for one run: the most node runs a run can need (two per
        retrieval round, the fixed nodes, one quoting fallback) plus slack, as a hard stop."""
        return 2 * (self._budget.max_follow_ups + 1) + 12

    async def answer(
        self, question: str, caller: AuthorizationContext, trace: TraceRecorder | None = None
    ) -> PipelineResult:
        """Answer one question. `trace` receives an event for every step; it is closed when the
        run ends (completed, failed or cancelled)."""
        trace = trace or TraceRecorder(uuid.uuid4().hex, caller.user_id)
        trace.emit("query", "query.started", {"question": question})
        try:
            result = await self._run(question, caller, trace)
        except asyncio.CancelledError:
            trace.emit("query", "query.cancelled", duration_ms=trace.elapsed_ms())
            trace.close("cancelled")
            raise
        except Exception as exc:
            trace.emit("query", "query.failed", {"error": type(exc).__name__}, trace.elapsed_ms())
            trace.close("failed")
            raise
        trace.emit(
            "query",
            "query.completed",
            {
                "answerability": result.answer.answerability.value,
                "citations": len(result.answer.citations),
                "evidence": len(result.evidence),
                "searches": len(result.retrievals),
                "llm_tokens": result.llm_tokens,
            },
            trace.elapsed_ms(),
        )
        trace.close("completed")
        return result

    async def _run(
        self, question: str, caller: AuthorizationContext, trace: TraceRecorder
    ) -> PipelineResult:
        jira = LazyConnector(self._open_jira, "jira") if self._open_jira else None
        context = RunContext(
            jira=jira,
            site_url=self._site_url,
            projects=self._projects,
            synthesizer=self._synthesizer,
            planner=self._planner,
            today=self._today,
            budget=self._budget,
            retrieval_limit=self._limit,
            retrieval_timeout=self._timeout,
            trace=trace,
        )
        try:
            with self._tracing.run(trace.query_id, caller.user_id):
                output = await self._graph.ainvoke(
                    GraphState(question=question, caller=caller),
                    config={
                        "recursion_limit": self.step_limit,
                        "run_name": "issuesense-query",
                        "metadata": {"query_id": trace.query_id},
                    },
                    context=context,
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
            retrievals=state.retrievals,
            llm_tokens=state.llm_tokens,
        )
