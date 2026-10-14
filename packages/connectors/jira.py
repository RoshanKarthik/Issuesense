"""Jira over MCP: Atlassian's official Rovo MCP server, read-only.

Retrieval is injection-proof: JQL is built only from validated values (a project key, plain
lowercase search words, a date and issue keys), never from user text. Issues come back newest
first, and the evidence step ranks them against each sub-question.

The argument names below follow Atlassian's documented tools. `verify()` checks them against the
live server's input schemas at connection time, so a mismatch fails loudly instead of silently
returning nothing.
"""

import argparse
import asyncio
import logging
import re
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pydantic import ValidationError

from packages.common.errors import ConnectorError
from packages.common.models import AuthorizationContext
from packages.common.settings import Settings
from packages.connectors.base import (
    AllowlistedToolClient,
    Connector,
    RetrievalRequest,
    SourceRecord,
    read_scope,
)
from packages.connectors.mcp_session import basic_auth_header, connect

logger = logging.getLogger(__name__)

SERVER_NAME = "atlassian"
SEARCH_TOOL = "searchJiraIssuesUsingJql"
COMMENTS_TOOL = "listJiraIssueComments"
CHANGELOG_TOOL = "listJiraIssueChangelogs"
# Read tools only. Atlassian's server also offers create/edit/transition/delete tools; they are
# refused before any call leaves the process.
JIRA_READ_TOOLS = frozenset({SEARCH_TOOL, COMMENTS_TOOL, CHANGELOG_TOOL})

Details = tuple[list[dict[str, Any]], list[dict[str, Any]]]

SEARCH_FIELDS = [
    "summary",
    "description",
    "status",
    "issuetype",
    "priority",
    "labels",
    "duedate",
    "parent",
    "assignee",
    "reporter",
    "resolution",
    "created",
    "updated",
]
MAX_COMMENTS = 20


_ISSUE_KEY = re.compile(r"^[A-Z][A-Z0-9_]+-\d+$")


def project_jql(project: str) -> str:
    # `project` is a validated ResourceId (letters, digits, - and _), so quoting is enough.
    return f'project = "{project}" ORDER BY updated DESC'


def search_jql(project: str, request: RetrievalRequest) -> str | None:
    """JQL for one project and the request's filters; None when no requested key is in it.

    Every value is validated before it gets here: search terms are lowercase letters, digits and
    hyphens, keys match the Jira key format, and the date is a `date`. So quoting is enough.
    """
    clauses = [f'project = "{project}"']
    if request.keys:
        keys = [k for k in request.keys if _ISSUE_KEY.match(k) and k.startswith(f"{project}-")]
        if not keys:
            return None
        clauses.append(f"key in ({', '.join(keys)})")
    if request.updated_since is not None:
        clauses.append(f'updated >= "{request.updated_since.isoformat()}"')
    if request.terms:
        clauses.append("(" + " OR ".join(f'text ~ "{term}"' for term in request.terms) + ")")
    return f"{' AND '.join(clauses)} ORDER BY updated DESC"


def _items(content: Any, *keys: str) -> list[dict[str, Any]]:
    """The list inside a tool result, whichever of the usual keys the server used."""
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    if isinstance(content, dict):
        for key in keys:
            value = content.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


class JiraConnector(Connector):
    source = "jira"

    def __init__(
        self,
        tools: AllowlistedToolClient,
        cloud_id: str,
        *,
        detail_limit: int = 25,
        concurrency: int = 4,
    ) -> None:
        self._tools = tools
        self._cloud_id = cloud_id
        self._detail_limit = detail_limit
        self._concurrency = asyncio.Semaphore(concurrency)
        # Comments and history per issue, fetched once per connector (one query run).
        self._detail_cache: dict[str, asyncio.Task[Details]] = {}

    def _search_args(self, project: str, limit: int, jql: str | None = None) -> dict[str, Any]:
        return {
            "cloudId": self._cloud_id,
            "jql": jql or project_jql(project),
            "maxResults": limit,
            "fields": SEARCH_FIELDS,
        }

    def _comments_args(self, key: str) -> dict[str, Any]:
        return {"cloudId": self._cloud_id, "issueIdOrKey": key, "maxResults": MAX_COMMENTS}

    def _changelog_args(self, key: str) -> dict[str, Any]:
        return {"cloudId": self._cloud_id, "issueIdOrKey": key}

    async def verify(self) -> list[str]:
        """Problems with the server's tools: missing tools, or arguments their schemas reject."""
        schemas = await self._tools.tool_schemas()
        expected = {
            SEARCH_TOOL: self._search_args("ATLAS", 1),
            COMMENTS_TOOL: self._comments_args("ATLAS-1"),
            CHANGELOG_TOOL: self._changelog_args("ATLAS-1"),
        }
        problems: list[str] = []
        for tool, arguments in sorted(expected.items()):
            if tool not in schemas:
                problems.append(f"{tool}: not offered by the server")
                continue
            properties = schemas[tool].get("properties") or {}
            if properties:
                unknown = sorted(set(arguments) - set(properties))
                if unknown:
                    problems.append(f"{tool}: server does not accept {', '.join(unknown)}")
            missing = sorted(set(schemas[tool].get("required") or []) - set(arguments))
            if missing:
                problems.append(f"{tool}: server requires {', '.join(missing)}")
        return problems

    async def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        async with self._concurrency:
            result = await self._tools.call_tool(tool, arguments)
        if result.is_error:
            # The error text may echo internals; log it, return only which tool failed.
            logger.error("Jira MCP tool %s failed: %s", tool, str(result.content)[:500])
            raise ConnectorError("Jira returned an error.", {"server": SERVER_NAME, "tool": tool})
        return result.content

    async def _details(self, key: str) -> Details:
        task = self._detail_cache.get(key)
        if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
            task = asyncio.ensure_future(self._fetch_details(key))
            self._detail_cache[key] = task
        return await asyncio.shield(task)

    async def _fetch_details(self, key: str) -> Details:
        comments, changelog = await asyncio.gather(
            self._call(COMMENTS_TOOL, self._comments_args(key)),
            self._call(CHANGELOG_TOOL, self._changelog_args(key)),
        )
        return (
            _items(comments, "comments", "values"),
            _items(changelog, "values", "histories", "changelogs"),
        )

    async def _fetch(self, request: RetrievalRequest, grants: dict[str, str]) -> list[SourceRecord]:
        records: list[SourceRecord] = []
        for project in grants:
            jql = search_jql(project, request)
            if jql is None:
                continue
            found = _items(
                await self._call(SEARCH_TOOL, self._search_args(project, request.limit, jql)),
                "issues",
            )
            issues = [
                issue for issue in found if str(issue.get("key", "")).startswith(f"{project}-")
            ]
            if len(issues) != len(found):
                logger.error(
                    "Jira returned %d issues outside %s", len(found) - len(issues), project
                )
            detailed = issues[: self._detail_limit]
            details = await asyncio.gather(
                *(self._details(str(issue["key"])) for issue in detailed)
            )
            for index, issue in enumerate(issues):
                comments, changelog = details[index] if index < len(details) else ([], [])
                records.append(
                    SourceRecord(
                        source=self.source,
                        resource=project,
                        tool=SEARCH_TOOL,
                        payload={
                            "issue": issue,
                            "comments": comments,
                            "changelog": changelog,
                            "details_fetched": index < len(details),
                        },
                        permission_context=grants[project],
                    )
                )
        return records


@asynccontextmanager
async def open_jira_connector(
    settings: Settings, target: Any = None
) -> AsyncIterator[JiraConnector]:
    """Connect to the configured Jira MCP server and check its tools before any retrieval.

    `target` overrides the server URL (tests pass an in-process server).
    """
    if not (settings.jira_base_url and settings.jira_email and settings.jira_api_token):
        raise ConnectorError("Jira is not configured.", {"server": SERVER_NAME})
    headers = basic_auth_header(settings.jira_email, settings.jira_api_token.get_secret_value())
    async with connect(
        target or settings.jira_mcp_url, server=SERVER_NAME, headers=headers
    ) as session:
        connector = JiraConnector(
            AllowlistedToolClient(session, SERVER_NAME, JIRA_READ_TOOLS), settings.jira_base_url
        )
        problems = await connector.verify()
        if problems:
            logger.error("Jira MCP server tools do not match: %s", "; ".join(problems))
            raise ConnectorError(
                "The Jira MCP server's tools do not match what IssueSense expects.",
                {"server": SERVER_NAME, "problems": problems},
            )
        yield connector


async def check_jira(settings: Settings, project: str, target: Any = None) -> list[str]:
    """Connect, verify the tools and read a few issues; returns report lines (owner tool)."""
    lines: list[str] = []
    request = RetrievalRequest(query="connectivity check", resources=[project], limit=5)
    async with open_jira_connector(settings, target) as jira:
        server = settings.jira_mcp_url if target is None else "test server"
        tools = ", ".join(sorted(await jira._tools.tool_schemas()))
        lines += [f"Connected to {server}.", f"Read tools available: {tools}."]
        owner = AuthorizationContext(user_id="owner-check", scopes=[read_scope("jira", project)])
        result = await jira.retrieve(owner, request)
    lines.append(f"Read {len(result.records)} recent issues from {project}:")
    for record in result.records:
        issue = record.payload["issue"]
        summary = (issue.get("fields") or {}).get("summary", "")
        lines.append(
            f"  {issue.get('key')}: {summary} ({len(record.payload['comments'])} comments, "
            f"{len(record.payload['changelog'])} changes)"
        )
    return lines


def check_jira_cli(argv: list[str] | None = None) -> None:
    """`issuesense-check-jira`: a live end-to-end check of the Jira MCP connection."""
    settings = Settings()
    parser = argparse.ArgumentParser(description="Check the live Jira MCP connection.")
    default_project = settings.jira_projects[0] if settings.jira_projects else "ATLAS"
    parser.add_argument("--project", default=default_project, help="Jira project key.")
    args = parser.parse_args(argv)
    try:
        lines = asyncio.run(check_jira(settings, args.project))
    except ValidationError:
        print(f"Not a Jira project key: {args.project!r}", file=sys.stderr)
        raise SystemExit(2) from None
    except ConnectorError as exc:
        print(f"Jira check failed: {exc.message}", file=sys.stderr)
        for problem in exc.details.get("problems", []):
            print(f"  - {problem}", file=sys.stderr)
        print(
            "Check JIRA_BASE_URL, JIRA_EMAIL and JIRA_API_TOKEN in .env, and that API-token "
            "authentication is enabled for the Rovo MCP server in Atlassian Administration.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    for line in lines:
        print(line)
