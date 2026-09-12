# ADR 0001: Evidence-First Design and Grounding

## Status
Proposed

## Context
In engineering intelligence platforms, LLM hallucinations can lead to incorrect technical decisions. To prevent this, IssueSense must ensure that every material claim in a response is directly supported by a verifiable piece of evidence from an authorized source.

## Decision
We will implement an "Evidence-First" pipeline where:
1. **Normalization**: All data retrieved from connectors must be normalized into a standard `Evidence` model before reaching the synthesis stage.
2. **Mandatory Citations**: Every claim in the final `Answer` must be mapped to one or more `Evidence` IDs via `Citation` objects.
3. **Verification**: The system must be capable of verifying that a citation actually exists and that the cited content supports the claim.
4. **Truth Hierarchy**: Authorized source evidence is the primary truth. Internal model memory is used only for context and cannot override source evidence.

## Consequences
- **Pros**: Extremely high reliability, verifiable answers, and a clear audit trail.
- **Cons**: Increased complexity in the retrieval pipeline and potential for "insufficient evidence" responses where a general LLM might have guessed.
