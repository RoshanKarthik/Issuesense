# IssueSense — Claude Code build instructions

## Mission

Build IssueSense as an evidence-grounded engineering intelligence platform. It answers questions using authorized organizational sources and must show exactly which evidence supports each claim.

The core rule is non-negotiable:

> Current, authorized source evidence is truth. Memory is context only. Every material claim must map to verifiable evidence and a citation.

## How to work

Work iteratively. Complete **one approved milestone at a time**. Do not begin a future phase, add speculative abstractions, build a UI, add additional connectors, or make external service assumptions unless the project owner explicitly asks.

For every milestone:

1. Inspect the relevant existing files and preserve unrelated work.
2. State the smallest implementation plan before making changes.
3. Implement only the approved scope.
4. Add tests alongside the feature.
5. Run formatter/linter and relevant tests.
6. Report changed files, test results, known limitations, and the next smallest milestone.
7. Stop and wait for explicit approval before moving to that next milestone.

Never claim functionality works without executing relevant tests. If credentials, a connector, or a dependency are unavailable, implement a clear boundary or fallback and report the limitation; never fabricate a production response, citation, or external result.

## Technology and code standards

- Python 3.12, FastAPI, Pydantic, HTTPX, pytest, Ruff.
- Introduce LangGraph only in Phase 2.
- Introduce PostgreSQL/pgvector only in Phase 6.
- Keep external providers behind interfaces and configuration. Do not hardcode model names, API keys, or vendor-specific logic in business code.
- Use typed Pydantic models for important state and API contracts.
- Keep functions narrow; avoid giant agent modules and premature microservices.
- Use explicit error handling and structured trace events. Do not silently swallow exceptions.
- Keep read operations automatic. Any future mutation (create/update/send) requires a human approval gate.

## Required architecture boundaries

```text
apps/api             FastAPI application and HTTP schemas
apps/web             Future TypeScript client; do not build until Phase 7
packages/common      Shared models, settings, errors, authorization context
packages/agent       Orchestration and typed graph state
packages/connectors  Source adapters and authorization boundary
packages/retrieval   Retrieval planning and source querying
packages/evidence    Evidence normalization, ranking, citations, verification
packages/guardrails  Input and output safety checks
packages/memory      Session and durable memory interfaces
packages/observability  Trace events and optional provider adapters
packages/evaluation  Evaluation fixtures, scorers, and runners
tests/unit           Fast deterministic tests
tests/integration    HTTP/connector-boundary tests
tests/evals          Quality and regression evaluation cases
tests/adversarial    Guardrail, PII, and authorization abuse cases
```

## Evidence and security requirements

Raw connector payloads must never flow straight to synthesis. Normalize them into evidence with at least:

```json
{
  "evidence_id": "E1",
  "source": "github",
  "authority": "implementation",
  "title": "...",
  "location": "repository/path.py#L10-L30",
  "content": "...",
  "timestamp": "...",
  "url": "...",
  "permission_context": "...",
  "metadata": {}
}
```

- Every connector invocation must receive an authorization context and perform an application-level access check before retrieval.
- Never assume connector credentials mean the requester is authorized.
- Source authority is contextual: GitHub for implementation, Jira for delivery status, Notion for intended architecture, and Google for human decisions.
- Recency affects confidence. Older documentation disagreeing with recent implementation may be documentation drift, not an unexplained contradiction.
- Returned claims must cite evidence IDs that exist, have valid source locations, and support the claim.
- Clearly return `INSUFFICIENT_EVIDENCE`, `PARTIALLY_ANSWERABLE`, `CONFLICTING_EVIDENCE`, `TOOL_FAILURE`, or `UNAUTHORIZED_EVIDENCE` where appropriate.
- Do not expose private chain-of-thought; provide concise evidence-based reasoning summaries instead.

## Delivery sequence

### Phase 0 — Foundations and contracts (current)

Implement only these sub-milestones, stopping after each approval:

1. Add project settings, core Pydantic models (`Query`, `AuthorizationContext`, `Evidence`, `Citation`, `Answerability`, `TraceEvent`), and an explicit error taxonomy.
2. Add a minimal FastAPI app with `GET /health`; test it.
3. Configure Ruff, pytest, and a GitHub Actions workflow that runs linting and tests.
4. Add ADRs for evidence-first design and authorization boundaries.

**Phase 0 exit:** clean checkout passes CI checks; contracts, settings, errors, and health endpoint are tested; no connector, LLM, database, or UI has been added.

### Phase 1 — GitHub-only vertical slice

1. Add typed `POST /v1/query` request/response models.
2. Add deterministic input validation, injection/tool-intent checks, and low-cost triage.
3. Add a GitHub connector interface and authorization check.
4. Add real GitHub retrieval using configured credentials. Tests must mock the HTTP boundary; production code must not return fake connector data.
5. Normalize retrieved files/issues/PRs into evidence with URLs and locations.
6. Add provider-neutral grounded synthesis. When no LLM provider is configured, use a deterministic safe response that reports retrieved evidence without inventing claims.
7. Add citation verification and output checks.
8. Add end-to-end integration and adversarial tests.

**Phase 1 exit:** authorized GitHub questions return evidence-backed citations; unsafe, unauthorized, empty-result, and connector-failure cases are explicit and tested.

### Phase 2 — Typed graph and observability

Replace linear orchestration with a typed LangGraph workflow. Add timeout/retry/partial-result policy, trace events for each node, and optional LangSmith integration. A connector failure must not discard independent evidence.

### Phase 3 — Guardrail and citation hardening

Expand adversarial input checks, output PII checks with recorded redactions, unsupported-claim checks, unauthorized-data checks, and citation support/location validation.

### Phase 4 — Evaluation and CI gates

Create versioned evaluation fixtures and deterministic scorers for retrieval, citation, guardrails, and triage. Add smoke evaluations on push and a scheduled full suite. Set quality thresholds only after recording a baseline.

### Phase 5 — Notion and cross-source reasoning

Add Notion under the same authorization boundary. Implement contextual authority, temporal ranking, direct contradiction detection, documentation-drift detection, gap detection, and answerability assessment.

### Phase 6 — Memory and persistence

Add PostgreSQL + pgvector, session memory, scoped durable memory, metadata/retention, and memory evaluation. Prove via tests that memory cannot override current authoritative evidence.

### Phase 7 — Real UI

Build the TypeScript client against the real API. Show chat, streamed live trace events, citations/evidence inspection, sources, dashboard, and a memory drawer. Never hardcode trace steps.

### Phase 8 — Jira and Google

Add Jira and Google incrementally with authorization, normalization, capability detection, and failure isolation. Document Google integration availability limitations honestly.

### Phase 9 — Human-in-the-loop mutations

Add mutation tools with action previews, durable approvals, graph pause/resume, expiry/rejection behavior, and audit events. No mutation may execute without approval.

### Phase 10 — Deployment and operations

Add deployment configuration, health checks, database migrations, secrets management guidance, rate limits, dashboards, alerts, backup/retention, and production smoke tests. Keep providers replaceable.

### Phase 11 — Final evaluation and hardening

Run the full curated suite (target ~150 cases), measure quality, safety, latency, cost, and triage metrics, then remediate critical failures. Publish only actual measurements.

### Phase 12 — Documentation

Complete the engineering narrative: problem, architecture, evidence pipeline, conflict example, evaluation methodology/results, observability, security model, deployment, limitations, and future work.

## Definition of done

A significant feature is complete only with implementation, focused unit tests, an integration test where relevant, an evaluation case where relevant, observable events/logs, explicit error behavior, and documentation.

## First Claude Code task

Inspect this repository. Confirm the active milestone is **Phase 0, sub-milestone 1**. Propose a concise plan for settings, shared Pydantic contracts, and errors. Do not edit files until the project owner approves that plan.
