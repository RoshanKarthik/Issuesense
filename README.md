# IssueSense

IssueSense is an evidence-grounded engineering intelligence platform. It answers engineering questions across code, documentation, delivery systems, and decision records while preserving source locations, authorization context, uncertainty, and verifiable citations.


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
