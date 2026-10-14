# IssueSense roadmap

IssueSense is an **agentic RAG system over live data, with memory, guardrails, evaluations in CI, and a
public deployment**. You ask a question such as:

> "Summarize what changed in Project Atlas this quarter and identify the major risks."

A basic RAG would search some documents once. IssueSense instead:

1. **Plans**: breaks the question into sub-questions (what were the original goals? which tasks were due
   this quarter? which deadlines moved? which blockers were reported?).
2. **Calls tools**: queries live, changing sources (Jira, Notion, Gmail) for each sub-question, and
   follows up when evidence is missing (multi-hop).
3. **Uses memory**: recalls durable facts from earlier sessions ("Project Atlas is the top priority"),
   but never lets memory override current source evidence.
4. **Answers with citations**: every claim links to the exact ticket, page or email that supports it,
   and every citation is verified before it is shown.
5. **Shows its work**: the demo shows the original question, the rewritten sub-questions, the tools
   called, the evidence retrieved, the memory used, and the final cited answer, traced in LangSmith.
6. **Stays safe**: blocks prompt injection (including malicious instructions hidden inside retrieved
   documents), redacts personal data, and never runs a risky action without human approval.
7. **Proves it**: an evaluation suite of ~150 cases runs in GitHub Actions on every push, and the README
   reports measured numbers (injection block rate, citation accuracy, latency, cost per task).

Each phase is split into small steps. A step is done when it has code, tests, and docs, and CI passes.

---

## Phase 0: Foundations ✅

- [x] Settings, core models (`Query`, `AuthorizationContext`, `Evidence`, `Citation`, `Answerability`,
      `TraceEvent`), error classes
- [x] FastAPI app with `GET /health`
- [x] Ruff, mypy, pytest, GitHub Actions CI
- [x] Design records: evidence-first design, authorization boundaries

## Phase 1: First live source, end to end (Jira)

One question → Jira → evidence → cited answer, with no agent loop yet.

- [x] 1.1 Typed `POST /v1/query` request/response and one error shape for every endpoint
- [x] 1.2 Input checks: validation, prompt-injection and tool-intent detection, cheap triage
      (reject or short-circuit questions that need no retrieval)
- [x] 1.3 Caller authentication (API keys mapped to a user and scopes) and the MCP connector
      interface, with an access check before every retrieval and an allowlist of read-only tools per
      MCP server
- [x] 1.4 Demo workspace: a seed script that fills a test Jira project with a realistic quarter of
      "Project Atlas" work (goals, slipped deadlines, blockers, status changes). Used by the demo and
      the evals; never your personal data. *Live run pending: needs the owner's Jira site.*
- [x] 1.5 Real Jira connector over MCP: IssueSense is the MCP client (official `mcp` SDK) and talks to
      Atlassian's official Rovo MCP server with API-token auth. Only allowlisted read-only tools are
      callable; tests run against a fake in-memory MCP server. *Live check pending:
      `issuesense-check-jira` once the owner's site is set up.*
- [x] 1.6 Normalize Jira issues, comments and changelogs into `Evidence` with URLs and locations
- [x] 1.7 LLM provider interface and grounded synthesis; without a configured LLM, a deterministic
      answer that lists retrieved evidence and invents nothing
- [x] 1.8 Citation verification: every cited ID exists, its location is valid, and it supports the claim
- [x] 1.9 End-to-end and adversarial tests for the slice

**Exit:** an authorized question about the demo Jira project returns a cited answer; unsafe,
unauthorized, empty-result and connector-failure cases each return an explicit, tested result.

**Status:** met against the fake Atlassian MCP server (in-process and over real Streamable HTTP
with Basic auth). The live run against the owner's Jira site is pending:
`issuesense-seed-jira --apply`, then `issuesense-check-jira`, then a question to `/v1/query`.

## Phase 2: Agentic planning, multi-hop, tracing

- [ ] 2.1 LangGraph typed workflow: plan → retrieve → assess evidence → synthesize → verify
- [ ] 2.2 Planner: query rewrite and decomposition into sub-questions, including time ranges
      ("this quarter")
- [ ] 2.3 Multiple tool calls per question, and follow-up retrieval when the assessment finds gaps
      (multi-hop), with a step and cost budget
- [ ] 2.4 Timeouts, retries and partial results: one failed tool call never discards other evidence
- [ ] 2.5 A trace event for every node, plus LangSmith tracing (optional, off without a key)
- [ ] 2.6 `GET /v1/query/{id}/trace` and live streaming of trace events (server-sent events)

## Phase 3: More live sources and cross-source reasoning

- [ ] 3.1 Notion connector over MCP: project goals, specs, decision docs. Candidates: Notion's
      official `@notionhq/notion-mcp-server` with an integration token, run as a sidecar container
      (Notion's hosted MCP server is OAuth-only and built for interactive chat); decided at this step
- [ ] 3.2 Gmail connector over MCP: status updates and decisions from a dedicated demo mailbox.
      Candidates: Google's official Gmail MCP server (in developer preview, OAuth; check that its
      scopes can be kept read-only) or a maintained community server; decided at this step
- [ ] 3.3 Source authority by question type: Jira for delivery status, Notion for intended plans, Gmail
      for human decisions
- [ ] 3.4 Recency ranking, contradiction detection ("the Notion plan says June, Jira says August"),
      documentation-drift and gap detection, answerability assessment
- [ ] 3.5 Demo workspace extended to Notion and Gmail with deliberate conflicts

## Phase 4: Memory

- [ ] 4.1 PostgreSQL + pgvector, with migrations
- [ ] 4.2 Short-term memory: the current session's conversation and working context, with expiry
- [ ] 4.3 Long-term memory: extract durable facts and preferences ("Project Atlas is the top priority"),
      with a write policy that decides what is worth keeping, deduplicates, updates and forgets.
      Never "store every sentence"
- [ ] 4.4 Memory recall in planning, shown as "memory used" in the trace
- [ ] 4.5 Tests proving memory never overrides current source evidence, plus memory evals

## Phase 5: Guardrails

- [ ] 5.1 Input: catalogue of injection techniques (direct, role-play, encoding, tool-intent)
- [ ] 5.2 Indirect injection: malicious instructions inside retrieved tickets, pages and emails are
      treated as data, never followed
- [ ] 5.3 Output: PII redaction with recorded redactions, structured-output validation,
      unsupported-claim detection
- [ ] 5.4 Risky actions: one or two write tools (comment on a Jira issue, draft an email). The agent
      prepares the plan and waits; nothing runs without approval; approvals can be rejected or expire;
      every decision is audit-logged

## Phase 6: Evaluation suite in CI

- [ ] 6.1 ~150 versioned cases: normal questions, edge cases, adversarial prompts, missing data,
      tool failures
- [ ] 6.2 Scorers: retrieval recall, citation accuracy, answer correctness, refusal correctness,
      injection block rate, PII leaks
- [ ] 6.3 Metrics per run: accuracy, refusal rate, latency, token usage, cost per task
- [ ] 6.4 GitHub Actions: a smoke eval on every push, the full suite on `main` and nightly, a pass-rate
      badge in the README; thresholds are set only after recording a baseline

## Phase 7: Demo UI

- [ ] 7.1 TypeScript web client (`apps/web`) against the real API
- [ ] 7.2 Ask a question and watch the live trace: rewritten sub-questions, tool calls, evidence,
      memory used, then the cited answer
- [ ] 7.3 Click a citation to inspect the evidence and open the source
- [ ] 7.4 Memory panel (view and delete remembered facts) and approval panel for risky actions

## Phase 8: Deployment

- [ ] 8.1 Production Dockerfile (non-root, health check) and `docker compose` for local Postgres
- [ ] 8.2 API on Google Cloud Run, database on Neon (Postgres + pgvector), web client on Vercel
- [ ] 8.3 Secrets in the platforms' secret stores; read-only connector tokens on the demo workspace only
- [ ] 8.4 Abuse protection: demo access keys, per-key rate limits, LLM spend cap, request size limits
- [ ] 8.5 GitHub Actions deploys on merge to `main` after checks and smoke evals pass; migrations run
      before the new version takes traffic; a post-deploy smoke test hits the live URL
- [ ] 8.6 Structured logs, LangSmith dashboards, an uptime check, and an alert on error rate

## Phase 9: Final evaluation and write-up

- [ ] 9.1 Run the full suite against the deployed system; fix critical failures
- [ ] 9.2 README with **measured** results only (e.g. "N of M injections blocked", citation accuracy,
      p50/p95 latency, cost per task, before/after numbers for optimizations)
- [ ] 9.3 The story: why it was built, architecture, the evidence pipeline, a conflict example, the
      trade-offs and why they were chosen, the hardest problems, limitations, future work

---

## Deployment design

| Part | Where | Why |
|---|---|---|
| API (FastAPI + LangGraph) | Google Cloud Run, one container | Runs the same Docker image as local, scales to zero when idle (low cost), starts in seconds (a recruiter clicking the link doesn't wait a minute), supports streaming |
| Database (Postgres + pgvector) | Neon | Managed Postgres with pgvector; database branches give CI a throwaway copy for migration tests |
| Web client | Vercel | Static/edge hosting for the TypeScript client, preview deploys per pull request |
| Traces | LangSmith (hosted) | Shareable trace links for the demo and the README |
| MCP servers | Atlassian's hosted server for Jira; self-hosted ones (e.g. Notion) as Cloud Run sidecar containers | The API reaches every source over MCP; sidecars are reachable only from the API container |
| Secrets | Cloud Run secrets / Vercel env vars | Never in the repo or the image |
| CI/CD | GitHub Actions | Lint, types, tests, smoke evals, then deploy on `main` |

Rules for the public demo:

- It reads a **demo workspace** (a test Jira project, Notion workspace and Gmail account filled by the
  seed scripts), never your real accounts. Connector tokens are read-only.
- Visitors need a demo access key, each key is rate-limited, and the LLM key has a hard spend cap, so a
  public link can't run up a bill.
- Write tools (Phase 5.4) only ever act on the demo workspace, and only after approval.

Simpler alternative: Render (one web service plus its managed Postgres) needs less setup, at the cost of
slower cold starts on the cheap tiers. Check current pricing and free tiers for any platform before
committing to it.

## Decisions still open

- LLM provider and model (kept behind an interface, configured in `.env`)
- Notion and Gmail MCP servers, decided at steps 3.1 and 3.2

## Decisions made

- Live sources connect over MCP; IssueSense is the MCP client, using the official `mcp` Python SDK
- Jira: Atlassian's official Rovo MCP server with API-token auth (an org admin enables API-token
  access in Atlassian Administration). Its tools can write, so IssueSense calls only an allowlist
  of read-only tools; the seed script (1.4) writes demo data with the Jira REST API, outside the agent
- No LLM provider yet: Phase 1 ships the provider interface and the deterministic no-LLM answer
- Web client framework (e.g. Next.js or React + Vite), decided at Phase 7
- Hosting accounts (Google Cloud, Neon, Vercel) are created by the project owner
