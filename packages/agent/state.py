"""Typed state for the query graph. Every node reads this state and returns updates to it.

The state holds data only (no connections or providers): those live in the run context, so a state
snapshot can always be logged or traced.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from packages.agent.planner import QueryPlan
from packages.agent.synthesis import SynthesisResult
from packages.common.models import Answer, AuthorizationContext, Evidence
from packages.connectors.base import ItemKey, SearchTerm
from packages.evidence.citations import VerificationReport


class Budget(BaseModel):
    """Limits on one run's work, so a question can never loop or spend without bound."""

    max_follow_ups: int = Field(default=2, ge=0)  # retrieval rounds after the first
    max_retrievals: int = Field(default=10, ge=1)  # connector searches in total
    max_llm_tokens: int = Field(default=24_000, ge=0)  # planner and synthesis, input + output


class RetrievalTask(BaseModel):
    """One search to run: for a sub-question, a broader retry of one, or a follow-up by key."""

    sub_question: str  # "Q1", ..., or "follow-up"
    kind: Literal["search", "broaden", "follow"]
    query: str
    terms: list[SearchTerm] = Field(default_factory=list)
    updated_since: date | None = None  # the period, for time-bound sub-questions
    updated_until: date | None = None
    keys: list[ItemKey] = Field(default_factory=list)


class RetrievalRecord(BaseModel):
    """What one retrieval did; the trace and the answer's notes are built from these."""

    task: RetrievalTask
    round: int  # 1 for the planned searches, 2+ for follow-ups
    items: int = 0  # evidence items it returned
    new_evidence: list[str] = Field(default_factory=list)  # IDs it added to the pool
    error: str | None = None  # a safe description when it failed


class GraphState(BaseModel):
    question: str
    caller: AuthorizationContext

    # authorize
    projects: list[str] = Field(default_factory=list)  # projects the caller may read and asked for
    skipped: list[str] = Field(default_factory=list)  # named projects the caller cannot read

    # plan
    plan: QueryPlan | None = None

    # retrieve / assess
    pending: list[RetrievalTask] = Field(default_factory=list)  # to run in the next round
    retrievals: list[RetrievalRecord] = Field(default_factory=list)
    rounds: int = 0
    evidence: list[Evidence] = Field(default_factory=list)  # the pool: every item retrieved
    ranked: list[Evidence] = Field(default_factory=list)

    # synthesize / verify
    synthesis: SynthesisResult | None = None
    verification: VerificationReport | None = None
    quote_only: bool = False  # set after an LLM answer fails verification

    # accounting, and notes for the answer's reasoning summary
    llm_tokens: int = 0
    notes: list[str] = Field(default_factory=list)

    # finish; a node that settles the outcome early sets `answer` and the graph skips to finish
    answer: Answer | None = None

    @property
    def failed_retrievals(self) -> list[RetrievalRecord]:
        return [record for record in self.retrievals if record.error]
