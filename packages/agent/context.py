"""The run context: per-run dependencies of the query graph, kept out of the state so the state
stays plain data that can be traced."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from packages.agent.planner import Planner, RulePlanner
from packages.agent.state import Budget
from packages.agent.synthesis import GroundedSynthesizer
from packages.common.models import utc_now
from packages.connectors.lazy import LazyConnector
from packages.observability.trace import TraceRecorder


@dataclass
class RunContext:
    jira: LazyConnector | None
    site_url: str
    projects: list[str]  # configured project keys
    synthesizer: GroundedSynthesizer
    planner: Planner = field(default_factory=RulePlanner)
    budget: Budget = field(default_factory=Budget)
    today: Callable[[], date] = field(default=lambda: utc_now().date())
    retrieval_limit: int = 20
    retrieval_timeout: float = 60.0  # seconds for one retrieval, details included
    trace: TraceRecorder = field(default_factory=lambda: TraceRecorder("untraced", "unknown"))
