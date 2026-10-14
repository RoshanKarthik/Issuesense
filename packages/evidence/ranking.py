"""Deterministic lexical ranking of evidence against a question.

Phase 1 has no planner, so retrieval returns a project's recent items and this ranks them. Terms
are lightly stemmed and weighted by rarity (rarer terms count more), titles count double, and ties
go to the most recent item. A small synonym list bridges common question words to the words Jira
uses ("deadline slipped" -> "due date changed"); Phase 2's query rewriting replaces it.

Evidence flagged for prompt injection always ranks last: planted text is written to match likely
questions, so it must not win on relevance.
"""

import math
import re
from collections import Counter
from datetime import UTC, datetime

from packages.common.models import Evidence

_STOPWORDS = frozenset(
    """a about above after all also am an and any are as at be been before being below between
    both but by can could did do does doing down during each few for from further had has have
    having he her here hers him his how i if in into is it its itself just me more most my no nor
    not now of off on once only or other our ours out over own same she should so some such than
    that the their theirs them then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you your yours
    tell show give list summarize summarise explain please""".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")
_OLDEST = datetime.min.replace(tzinfo=UTC)
_DOUBLED = re.compile(r"([b-df-hj-np-tv-z])\1$")

# Question word -> words the source tends to use instead. Used for query expansion only.
_SYNONYM_WORDS: dict[str, tuple[str, ...]] = {
    "deadline": ("due", "date"),
    "slip": ("changed", "delay", "moved"),
    "delay": ("slip", "moved", "changed"),
    "postpone": ("moved", "changed", "delay"),
    "risk": ("blocker", "blocked"),
    "blocker": ("blocked", "risk"),
    "launch": ("release", "target"),
}
_SYNONYM_WEIGHT = 0.5


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            if suffix in ("ing", "ed"):
                token = _DOUBLED.sub(r"\1", token)  # slipped -> slipp -> slip
            break
    return token[:-1] if token.endswith("e") and len(token) > 3 else token


def _aware(moment: datetime | None) -> datetime:
    if moment is None:
        return _OLDEST
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def terms(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


_SYNONYMS = {
    _stem(word): tuple(_stem(alt) for alt in alternatives)
    for word, alternatives in _SYNONYM_WORDS.items()
}


def query_terms(question: str) -> dict[str, float]:
    """Stemmed question terms with weights: 1 for the question's own words, less for synonyms."""
    weights = {term: 1.0 for term in terms(question)}
    for term in list(weights):
        for synonym in _SYNONYMS.get(term, ()):
            weights.setdefault(synonym, _SYNONYM_WEIGHT)
    return weights


def rank_evidence(question: str, evidence: list[Evidence]) -> list[Evidence]:
    """Evidence sorted by relevance to the question; each copy carries `metadata["relevance"]`."""
    query = query_terms(question)
    documents = [
        (Counter(terms(item.title or "")), Counter(terms(item.content))) for item in evidence
    ]
    df = Counter(term for title, body in documents for term in set(title) | set(body))
    total = len(evidence)

    scored: list[tuple[bool, float, datetime, int, Evidence]] = []
    for index, (item, (title, body)) in enumerate(zip(evidence, documents, strict=True)):
        score = 0.0
        for term, query_weight in query.items():
            if term in title or term in body:
                rarity = math.log(1 + total / df[term])
                score += (
                    query_weight
                    * rarity
                    * (2.0 if term in title else 1.0)
                    * (1 + math.log(1 + body[term]))
                )
        ranked = item.model_copy(
            update={"metadata": {**item.metadata, "relevance": round(score, 4)}}
        )
        trusted = not item.metadata.get("injection_findings")
        scored.append((trusted, score, _aware(item.timestamp), -index, ranked))
    scored.sort(key=lambda entry: entry[:4], reverse=True)
    return [entry[4] for entry in scored]
