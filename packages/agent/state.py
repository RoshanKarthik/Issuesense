"""Typed state for the query graph. Every node reads this state and returns updates to it.

The state holds data only (no connections or providers): those live in the run context, so a state
snapshot can always be logged or traced.
"""

from pydantic import BaseModel, Field

from packages.agent.planner import QueryPlan
from packages.agent.synthesis import SynthesisResult
from packages.common.models import Answer, AuthorizationContext, Evidence
from packages.evidence.citations import VerificationReport


class GraphState(BaseModel):
    question: str
    caller: AuthorizationContext

    # authorize
    projects: list[str] = Field(default_factory=list)  # projects the caller may read and asked for
    skipped: list[str] = Field(default_factory=list)  # named projects the caller cannot read

    # plan
    plan: QueryPlan | None = None

    # retrieve / assess
    evidence: list[Evidence] = Field(default_factory=list)  # every item retrieved so far
    ranked: list[Evidence] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)  # safe descriptions of failed retrievals

    # synthesize / verify
    synthesis: SynthesisResult | None = None
    verification: VerificationReport | None = None
    quote_only: bool = False  # set after an LLM answer fails verification

    # finish; a node that settles the outcome early sets `answer` and the graph skips to finish
    answer: Answer | None = None
