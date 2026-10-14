"""Deterministic input checks: hidden characters, prompt injection, tool intent, and triage.

Every rule is a plain pattern with a stable ID, so a block is explainable and testable. These checks
are a first line of defence, not a guarantee; Phase 5 extends them, and retrieved content is checked
separately because it is untrusted too.
"""

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field


class FindingCategory(StrEnum):
    HIDDEN_CHARACTERS = "HIDDEN_CHARACTERS"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    TOOL_INTENT = "TOOL_INTENT"


class Severity(StrEnum):
    BLOCK = "BLOCK"  # the question is refused
    WARN = "WARN"  # recorded, and the offending characters are stripped


class Triage(StrEnum):
    RETRIEVAL = "RETRIEVAL"  # needs source evidence
    SMALL_TALK = "SMALL_TALK"  # greeting or usage question: answered without retrieval


class Finding(BaseModel):
    rule_id: str
    category: FindingCategory
    severity: Severity


class InputAssessment(BaseModel):
    normalized_text: str
    findings: list[Finding] = Field(default_factory=list)
    triage: Triage

    @property
    def blocked(self) -> bool:
        return any(finding.severity is Severity.BLOCK for finding in self.findings)

    @property
    def blocking_findings(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.severity is Severity.BLOCK]


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    category: FindingCategory
    pattern: re.Pattern[str]


# --- Hidden characters ------------------------------------------------------------------------

# Unicode "tag" characters can smuggle invisible ASCII instructions; bidi controls reorder what a
# reviewer sees. Neither has a place in a question.
_SMUGGLING_CHARS = re.compile("[\U000e0000-\U000e007f‪-‮⁦-⁩]")
# Zero-width and soft-hyphen characters can split a keyword ("ig​nore") to dodge the rules.
# ZWJ/ZWNJ are left alone: emoji sequences and several scripts need them.
_INVISIBLE_CHARS = re.compile("[​⁠﻿­]")

# --- Prompt injection -------------------------------------------------------------------------

_I = re.IGNORECASE
_INJECTION_RULES = [
    _Rule(
        "ignore_instructions",
        FindingCategory.PROMPT_INJECTION,
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\b(?:\s+\w+){0,4}?\s+"
            r"(?:instructions?|prompts?|guardrails|safety\s+rules|system\s+rules)\b",
            _I,
        ),
    ),
    _Rule(
        "reveal_system_prompt",
        FindingCategory.PROMPT_INJECTION,
        re.compile(
            r"\b(?:reveal|show|print|repeat|output|display|leak|dump|tell\s+me|what\s+(?:is|are))\b"
            r"(?:\s+\w+){0,4}?\s+(?:system|hidden|initial|developer|secret)\s+"
            r"(?:prompt|instructions|message|rules)\b",
            _I,
        ),
    ),
    _Rule(
        "role_override",
        FindingCategory.PROMPT_INJECTION,
        re.compile(
            r"\b(?:you\s+are\s+now|from\s+now\s+on,?\s+you\s+(?:are|will|must)|"
            r"pretend\s+(?:to\s+be|you\s+are|that\s+you)|roleplay\s+as|"
            r"act\s+as\s+(?:an?\s+)?(?:unrestricted|unfiltered|jailbroken|evil|uncensored))\b",
            _I,
        ),
    ),
    _Rule(
        "jailbreak_phrase",
        FindingCategory.PROMPT_INJECTION,
        re.compile(
            r"\b(?:jailbreak|jailbroken|do\s+anything\s+now|developer\s+mode|god\s+mode)\b", _I
        ),
    ),
    _Rule(
        "fake_delimiter",
        FindingCategory.PROMPT_INJECTION,
        re.compile(
            r"<\|?\s*/?\s*(?:system|im_start|im_end|assistant|endoftext)\s*\|?>|\[/?INST\]|"
            r"<<\s*/?SYS\s*>>|^\s*#{0,3}\s*(?:system|assistant)\s*:",
            _I | re.MULTILINE,
        ),
    ),
    _Rule(
        "new_instructions",
        FindingCategory.PROMPT_INJECTION,
        re.compile(r"\b(?:new|updated|real|actual|revised)\s+instructions?\s*:", _I),
    ),
]

# Long runs of base64 alphabet hide instructions from both the rules and a human reviewer. Pure hex
# runs (commit SHAs, IDs) are not flagged.
_ENCODED_RUN = re.compile(r"[A-Za-z0-9+/]{48,}={0,2}")
_HEX_RUN = re.compile(r"[0-9a-fA-F]+")

# --- Tool intent ------------------------------------------------------------------------------

# A mutation is an imperative verb at the start of a clause (optionally after "please", "can you",
# ...) acting on a work item. Read questions about the same verbs ("which tickets were deleted?",
# "how do I create a ticket?", "send me a summary") do not match.
_CLAUSE_LEAD = (
    r"(?:^|[.!?;:\n]\s*|\b(?:please|then|and|also|now|next|first|finally|just)\s+|"
    r"\b(?:can|could|would|will)\s+you\s+(?:please\s+)?|\bgo\s+ahead\s+and\s+|"
    r"\bi\s+(?:want|need)\s+you\s+to\s+)"
)
_MUTATION_VERB = (
    r"(?:delete|remove|close|resolve|reopen|create|file|raise|assign|reassign|unassign|transition|"
    r"move|update|edit|change|modify|rename|set|send|email|post|reply\s+to|forward|archive|merge|"
    r"deploy|publish)"
)
_WORK_ITEM = (
    r"(?:(?:the|this|that|these|those|all|every|a|an|new|my|our|his|her|their)\s+)*"
    r"(?:\w+\s+){0,2}?"
    r"(?:tickets?|issues?|tasks?|stor(?:y|ies)|epics?|bugs?|comments?|pages?|emails?|mails?|"
    r"messages?|replies|sprints?|projects?|boards?|labels?|assignees?|[A-Z][A-Z0-9]+-\d+)\b"
)
_TOOL_INTENT_RULES = [
    _Rule(
        "mutation_request",
        FindingCategory.TOOL_INTENT,
        re.compile(
            _CLAUSE_LEAD + _MUTATION_VERB + r"\b(?!\s+(?:me|us)\b)\s+" + _WORK_ITEM,
            _I | re.MULTILINE,
        ),
    ),
    _Rule(
        "tool_invocation",
        FindingCategory.TOOL_INTENT,
        re.compile(
            r"\b(?:call|invoke|execute|run)\s+(?:the\s+)?\w+\s+(?:tool|function)\b|"
            r"\b(?:call|invoke)\s+(?:the\s+)?[a-z]+_[a-z_]+\b",
            _I,
        ),
    ),
    _Rule(
        "shell_command",
        FindingCategory.TOOL_INTENT,
        re.compile(
            r"\brm\s+-[rf]+\b|\bdrop\s+(?:table|database)\b|\bsudo\s+\w|"
            r"\b(?:curl|wget)\s+(?:-\S+\s+)*https?://|\bchmod\s+[0-7]{3}\b|;\s*(?:cat|ls|rm)\s",
            _I,
        ),
    ),
]

# --- Triage -----------------------------------------------------------------------------------

_SMALL_TALK = frozenset(
    {
        "hi", "hello", "hey", "hiya", "yo", "hi there", "hello there", "hey there",
        "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "thanks a lot", "thank you so much", "thx", "ty", "cheers",
        "ok", "okay", "cool", "great", "nice", "bye", "goodbye",
        "help", "who are you", "what are you", "what can you do", "what can i ask",
        "what can i ask you", "how do i use this",
    }
)  # fmt: skip
_NON_WORD = re.compile(r"[^\w\s]+")


def normalize(text: str) -> tuple[str, list[Finding]]:
    """Strip invisible characters, apply NFKC, and collapse runs of spaces."""
    findings: list[Finding] = []
    if _SMUGGLING_CHARS.search(text):
        findings.append(
            Finding(
                rule_id="smuggled_characters",
                category=FindingCategory.HIDDEN_CHARACTERS,
                severity=Severity.BLOCK,
            )
        )
    if _INVISIBLE_CHARS.search(text):
        findings.append(
            Finding(
                rule_id="invisible_characters",
                category=FindingCategory.HIDDEN_CHARACTERS,
                severity=Severity.WARN,
            )
        )
    text = _INVISIBLE_CHARS.sub("", _SMUGGLING_CHARS.sub("", text))
    # NFKC folds look-alikes such as fullwidth letters ("ｉｇｎｏｒｅ") into plain ASCII.
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip(), findings


def _rule_findings(text: str) -> list[Finding]:
    findings = [
        Finding(rule_id=rule.rule_id, category=rule.category, severity=Severity.BLOCK)
        for rule in (*_INJECTION_RULES, *_TOOL_INTENT_RULES)
        if rule.pattern.search(text)
    ]
    if any(not _HEX_RUN.fullmatch(run) for run in _ENCODED_RUN.findall(text)):
        findings.append(
            Finding(
                rule_id="encoded_payload",
                category=FindingCategory.PROMPT_INJECTION,
                severity=Severity.BLOCK,
            )
        )
    return findings


def triage(text: str) -> Triage:
    words = _NON_WORD.sub(" ", text.lower()).split()
    return Triage.SMALL_TALK if " ".join(words) in _SMALL_TALK else Triage.RETRIEVAL


def assess_question(text: str) -> InputAssessment:
    """Run every input check on a question and decide how it should be handled."""
    normalized, findings = normalize(text)
    findings.extend(_rule_findings(normalized))
    return InputAssessment(normalized_text=normalized, findings=findings, triage=triage(normalized))


def scan_untrusted_text(text: str) -> list[Finding]:
    """Injection signals in retrieved content (tickets, pages, emails).

    Only the injection and hidden-character rules apply: work items legitimately say "create a
    ticket" or contain encoded blobs, so tool-intent and encoded-payload rules would flag normal
    content. Findings mark the content as suspect; it is still data, never instructions.
    """
    normalized, findings = normalize(text)
    findings.extend(
        Finding(rule_id=rule.rule_id, category=rule.category, severity=Severity.BLOCK)
        for rule in _INJECTION_RULES
        if rule.pattern.search(normalized)
    )
    return [finding for finding in findings if finding.severity is Severity.BLOCK]
