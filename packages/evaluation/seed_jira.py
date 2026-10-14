"""Seed a Jira Cloud project with the Project Atlas dataset (`issuesense-seed-jira`).

This is an owner-run setup tool, not part of the agent: it writes through the Jira REST API with
the owner's credentials. It previews by default and writes only with `--apply`. Re-running is safe:
seeded issues carry a `seed-<ref>` label and are skipped when they already exist.

Jira does not let clients backdate history, so every seeded change is timestamped "now"; the dates
that matter to the story (due dates and their slips) live in fields and comments.
"""

import argparse
import asyncio
import sys
from datetime import date
from typing import Any

import httpx
from pydantic import BaseModel, Field

from packages.common.settings import Settings
from packages.evaluation.atlas_dataset import ATLAS_ISSUES, SeedIssue, Status

SEED_LABEL = "issuesense-seed"


def seed_label(ref: str) -> str:
    return f"seed-{ref}"


def adf(text: str) -> dict[str, Any]:
    """Plain text as an Atlassian Document Format document, one paragraph per line."""
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": line}]}
            for line in text.splitlines()
            if line.strip()
        ],
    }


class SeedError(Exception):
    pass


class JiraRestClient:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    @classmethod
    def connect(
        cls,
        base_url: str,
        email: str,
        api_token: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> "JiraRestClient":
        http = httpx.AsyncClient(
            base_url=base_url,
            auth=(email, api_token),
            headers={"Accept": "application/json"},
            timeout=30,
            transport=transport,
        )
        return cls(http)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self._http.request(method, path, **kwargs)
        if response.is_error:
            # Jira error bodies name the offending field; the request itself carries no secrets.
            raise SeedError(
                f"{method} {path} failed with HTTP {response.status_code}: {response.text[:500]}"
            )
        return response.json() if response.content else None

    async def seeded_issues(self, project: str) -> dict[str, str]:
        """Map dataset ref -> issue key for every issue this tool already created in the project."""
        found: dict[str, str] = {}
        token: str | None = None
        while True:
            params: dict[str, Any] = {
                "jql": f'project = "{project}" AND labels = "{SEED_LABEL}"',
                "fields": "labels",
                "maxResults": 100,
            }
            if token:
                params["nextPageToken"] = token
            page = await self._request("GET", "/rest/api/3/search/jql", params=params)
            for issue in page.get("issues", []):
                for label in issue["fields"].get("labels", []):
                    if label.startswith("seed-"):
                        found[label.removeprefix("seed-")] = issue["key"]
            token = page.get("nextPageToken")
            if not token or page.get("isLast", True):
                return found

    async def create_issue(self, fields: dict[str, Any]) -> str:
        created = await self._request("POST", "/rest/api/3/issue", json={"fields": fields})
        return str(created["key"])

    async def add_comment(self, key: str, text: str) -> None:
        await self._request("POST", f"/rest/api/3/issue/{key}/comment", json={"body": adf(text)})

    async def set_due_date(self, key: str, due: date) -> None:
        await self._request(
            "PUT", f"/rest/api/3/issue/{key}", json={"fields": {"duedate": due.isoformat()}}
        )

    async def transition(self, key: str, status: str) -> None:
        options = await self._request("GET", f"/rest/api/3/issue/{key}/transitions")
        for option in options.get("transitions", []):
            if option["to"]["name"].casefold() == status.casefold():
                await self._request(
                    "POST",
                    f"/rest/api/3/issue/{key}/transitions",
                    json={"transition": {"id": option["id"]}},
                )
                return
        names = sorted(option["to"]["name"] for option in options.get("transitions", []))
        raise SeedError(f"{key} has no transition to {status!r}; available: {names}")


class SeedReport(BaseModel):
    project: str
    applied: bool
    created: dict[str, str] = Field(default_factory=dict)  # ref -> key ("" in a preview)
    skipped: dict[str, str] = Field(default_factory=dict)  # ref -> existing key
    actions: list[str] = Field(default_factory=list)


def _fields(project: str, issue: SeedIssue, parent_key: str | None) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "project": {"key": project},
        "issuetype": {"name": issue.type.value},
        "summary": issue.summary,
        "description": adf(issue.description),
        "labels": [SEED_LABEL, seed_label(issue.ref), *issue.labels],
    }
    if issue.due:
        fields["duedate"] = issue.due.isoformat()
    if parent_key:
        fields["parent"] = {"key": parent_key}
    return fields


async def seed(
    client: JiraRestClient, project: str, issues: list[SeedIssue], *, apply: bool
) -> SeedReport:
    report = SeedReport(project=project, applied=apply)
    keys = await client.seeded_issues(project)
    report.skipped = {ref: key for ref, key in keys.items() if ref in {i.ref for i in issues}}

    # Parents first, so children can point at them.
    ordered = sorted(issues, key=lambda issue: issue.parent_ref is not None)
    for issue in ordered:
        if issue.ref in keys:
            continue
        parent_key = keys.get(issue.parent_ref) if issue.parent_ref else None
        if issue.parent_ref and parent_key is None and apply:
            raise SeedError(f"{issue.ref}: parent {issue.parent_ref} was not created")
        report.actions.append(f"create {issue.type.value} {issue.ref}: {issue.summary}")
        report.actions += [f"  comment on {issue.ref}" for _ in issue.comments]
        report.actions += [f"  move {issue.ref} due date to {due}" for due in issue.due_changes]
        if issue.status is not Status.TO_DO:
            report.actions.append(f"  transition {issue.ref} to {issue.status.value}")
        if not apply:
            report.created[issue.ref] = ""
            keys[issue.ref] = f"<{issue.ref}>"
            continue

        key = await client.create_issue(_fields(project, issue, parent_key))
        keys[issue.ref] = report.created[issue.ref] = key
        for comment in issue.comments:
            await client.add_comment(key, comment)
        for due in issue.due_changes:
            await client.set_due_date(key, due)
        if issue.status is not Status.TO_DO:
            await client.transition(key, issue.status.value)
    return report


async def _main(project: str, apply: bool, settings: Settings) -> SeedReport:
    if not (settings.jira_base_url and settings.jira_email and settings.jira_api_token):
        raise SeedError("Set JIRA_BASE_URL, JIRA_EMAIL and JIRA_API_TOKEN in .env first.")
    client = JiraRestClient.connect(
        settings.jira_base_url, settings.jira_email, settings.jira_api_token.get_secret_value()
    )
    try:
        return await seed(client, project, ATLAS_ISSUES, apply=apply)
    finally:
        await client.aclose()


def seed_jira_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Seed a Jira project with the Project Atlas dataset."
    )
    parser.add_argument("--project", default="ATLAS", help="Jira project key (default: ATLAS).")
    parser.add_argument(
        "--apply", action="store_true", help="Write to Jira (default: preview only)."
    )
    args = parser.parse_args(argv)
    try:
        report = asyncio.run(_main(args.project, args.apply, Settings()))
    except (SeedError, httpx.HTTPError) as exc:
        print(f"Seeding failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    verb = "Created" if report.applied else "Would create"
    created, skipped = len(report.created), len(report.skipped)
    print(f"{verb} {created} issues in {report.project}; {skipped} already seeded.")
    for action in report.actions:
        print(f"  {action}")
    if report.applied:
        for ref, key in report.created.items():
            print(f"  {ref} -> {key}")
    else:
        print("Preview only. Re-run with --apply to write these changes to Jira.")
