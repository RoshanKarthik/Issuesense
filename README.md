# IssueSense

IssueSense is an agentic RAG system over live data. Ask something like *"Summarize what changed in Project Atlas this quarter and identify the major risks"*: it breaks the question into sub-questions, queries Jira, Notion and Gmail, recalls relevant memory, and returns an answer where every claim cites the ticket, page or email that supports it. Guardrails block prompt injection and redact personal data, an evaluation suite runs in CI on every push, and the whole path is traced.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the plan and progress.


## Core rule

Current, authorized source evidence is truth for an answer. Memory is context, never an automatic override. Claims must map to normalized evidence and citations must be verified before output.

## What works today (Phase 1)

Ask a question over HTTP and IssueSense answers from a Jira project, read through Atlassian's
official MCP server:

- **Input guardrails** block prompt injection, hidden characters and write requests
  (IssueSense is read-only), and answer greetings without retrieval.
- **API-key authentication** with per-project scopes (`jira:read:ATLAS`); IssueSense checks access
  before any retrieval, and only allowlisted read-only MCP tools can be called.
- **Evidence**: every issue, comment and history entry becomes citable evidence with a location
  (`ATLAS-3#comment-20030`) and a link back to Jira. Retrieved text that tries to instruct the
  assistant is flagged and withheld.
- **Answers**: with no LLM configured, the most relevant evidence is quoted verbatim; with one
  (not yet chosen), it writes an answer as claims that each cite evidence.
- **Verification**: every claim's citations, quotes, numbers and dates are checked against the
  evidence before anything is returned. Unverifiable answers are replaced or withheld.
- Every outcome is explicit: `PARTIALLY_ANSWERABLE`, `SUFFICIENT`, `INSUFFICIENT_EVIDENCE`,
  `UNAUTHORIZED_EVIDENCE`, `TOOL_FAILURE`.

## Local setup

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
```

### 1. Jira demo workspace

Use a **dedicated demo Jira Cloud site**, never your real one:

1. Create a Scrum project named "Project Atlas" with the key `ATLAS`.
2. Create an API token at https://id.atlassian.com/manage-profile/security/api-tokens.
3. In Atlassian Administration, turn on API-token authentication for the Rovo MCP server.
4. Put `JIRA_BASE_URL`, `JIRA_EMAIL` and `JIRA_API_TOKEN` in `.env`.

Fill the project with the Project Atlas story, then check the MCP connection:

```bash
issuesense-seed-jira              # preview
issuesense-seed-jira --apply      # write the demo issues (safe to re-run)
issuesense-check-jira             # connect over MCP, verify tools, read a few issues
```

### 2. API keys

The key is printed once; only its hash goes into configuration:

```bash
issuesense-create-key --user-id you --scope jira:read:ATLAS
```

Add the printed entry to the `ISSUESENSE_API_KEYS` JSON list in `.env`, e.g.
`ISSUESENSE_API_KEYS=[{"key_id": "...", "sha256": "...", "user_id": "you", "scopes": ["jira:read:ATLAS"]}]`.
Scopes are per source and project: `jira:read:ATLAS` grants one project, `jira:read:*` all of them.

### 3. Ask

Run the API (serves on `ISSUESENSE_API_HOST:ISSUESENSE_API_PORT`, default `127.0.0.1:8000`):

```bash
issuesense-api
curl -H "Authorization: Bearer $ISSUESENSE_KEY" -H "Content-Type: application/json" \
  -d '{"question": "Which deadlines slipped in Project Atlas, and why?"}' http://127.0.0.1:8000/v1/query
```

## Checks

CI runs the same four commands on every push and pull request:

```bash
ruff check .
ruff format --check .
mypy
pytest
```

Later phases add optional dependency groups: `.[persistence]`, `.[observability]`.
