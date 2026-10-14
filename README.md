# IssueSense

IssueSense is an agentic RAG system over live data. Ask something like *"Summarize what changed in Project Atlas this quarter and identify the major risks"*: it breaks the question into sub-questions, queries Jira, Notion and Gmail, recalls relevant memory, and returns an answer where every claim cites the ticket, page or email that supports it. Guardrails block prompt injection and redact personal data, an evaluation suite runs in CI on every push, and the whole path is traced.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the plan and progress.


## Core rule

Current, authorized source evidence is truth for an answer. Memory is context, never an automatic override. Claims must map to normalized evidence and citations must be verified before output.

## Local setup

Requires Python 3.12. No external connector is implemented yet, so no credentials are needed.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env             # only the ISSUESENSE_* runtime settings are read so far
```

Run the API (serves on `ISSUESENSE_API_HOST:ISSUESENSE_API_PORT`, default `127.0.0.1:8000`):

```bash
issuesense-api
curl http://127.0.0.1:8000/health
```

## Checks

CI runs the same four commands on every push and pull request:

```bash
ruff check .
ruff format --check .
mypy
pytest
```

Later phases add optional dependency groups: `.[agent]`, `.[persistence]`, `.[observability]`.
