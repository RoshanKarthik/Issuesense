"""A fake Atlassian Rovo MCP server holding the Project Atlas dataset, for connector tests."""

import re
from datetime import UTC, datetime, timedelta
from typing import Any

from mcp.server.mcpserver import MCPServer

from packages.evaluation.atlas_dataset import SeedIssue

SITE = "https://atlas-demo.atlassian.net"
SEEDED_AT = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


def _adf(text: str) -> dict[str, Any]:
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": line}]}
            for line in text.splitlines()
            if line.strip()
        ],
    }


def _timestamp(step: int) -> str:
    return (SEEDED_AT + timedelta(minutes=step)).isoformat()


class FakeJiraMcp:
    def __init__(self, issues: list[SeedIssue], project: str = "ATLAS") -> None:
        self.project = project
        self.keys = {issue.ref: f"{project}-{n}" for n, issue in enumerate(issues, start=1)}
        self.issues: dict[str, dict[str, Any]] = {}
        self.comments: dict[str, list[dict[str, Any]]] = {}
        self.changelogs: dict[str, list[dict[str, Any]]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.failing_tools: set[str] = set()
        for n, issue in enumerate(issues, start=1):
            self._add(n, issue)

    def _add(self, n: int, issue: SeedIssue) -> None:
        key = self.keys[issue.ref]
        due_dates = [d.isoformat() for d in [issue.due, *issue.due_changes] if d]
        self.issues[key] = {
            "id": str(10000 + n),
            "key": key,
            "self": f"{SITE}/rest/api/3/issue/{10000 + n}",
            "fields": {
                "summary": issue.summary,
                "description": _adf(issue.description),
                "status": {"name": issue.status.value},
                "issuetype": {"name": issue.type.value},
                "priority": {"name": "High" if "risk" in issue.labels else "Medium"},
                "labels": list(issue.labels),
                "duedate": due_dates[-1] if due_dates else None,
                "parent": {"key": self.keys[issue.parent_ref]} if issue.parent_ref else None,
                "created": _timestamp(n),
                "updated": _timestamp(n + 30),
            },
        }
        self.comments[key] = [
            {
                "id": f"{20000 + n * 10 + i}",
                "author": {"displayName": "Priya Shah"},
                "body": _adf(text),
                "created": _timestamp(n + 10 + i),
            }
            for i, text in enumerate(issue.comments)
        ]
        self.changelogs[key] = [
            {
                "id": f"{30000 + n * 10 + i}",
                "author": {"displayName": "Priya Shah"},
                "created": _timestamp(n + 20 + i),
                "items": [{"field": "duedate", "fromString": old, "toString": new}],
            }
            for i, (old, new) in enumerate(zip(due_dates, due_dates[1:], strict=False))
        ]

    def _record(self, tool: str, arguments: dict[str, Any]) -> None:
        self.calls.append((tool, arguments))
        if tool in self.failing_tools:
            raise RuntimeError(f"{tool} is down (internal host jira-db-7)")

    def tools_called(self) -> set[str]:
        return {tool for tool, _ in self.calls}

    def server(self, *, rename_issue_arg: bool = False) -> MCPServer:
        server = MCPServer("fake-atlassian")
        fake = self

        @server.tool()
        def searchJiraIssuesUsingJql(
            cloudId: str,
            jql: str,
            maxResults: int = 50,
            fields: list[str] | None = None,
            nextPageToken: str | None = None,
        ) -> dict[str, Any]:
            fake._record(
                "searchJiraIssuesUsingJql",
                {"cloudId": cloudId, "jql": jql, "maxResults": maxResults},
            )
            match = re.match(r'project = "([A-Za-z0-9_-]+)"', jql)
            project = match.group(1) if match else ""
            issues = [i for k, i in fake.issues.items() if k.startswith(f"{project}-")]
            issues.sort(key=lambda i: i["fields"]["updated"], reverse=True)
            return {"issues": issues[:maxResults], "isLast": True}

        if rename_issue_arg:

            @server.tool(name="listJiraIssueComments")
            def comments_renamed(cloudId: str, issueKey: str) -> dict[str, Any]:
                return {"comments": []}

        else:

            @server.tool()
            def listJiraIssueComments(
                cloudId: str, issueIdOrKey: str, maxResults: int = 50
            ) -> dict[str, Any]:
                fake._record(
                    "listJiraIssueComments", {"cloudId": cloudId, "issueIdOrKey": issueIdOrKey}
                )
                return {"comments": fake.comments.get(issueIdOrKey, [])[:maxResults]}

        @server.tool()
        def listJiraIssueChangelogs(cloudId: str, issueIdOrKey: str) -> dict[str, Any]:
            fake._record(
                "listJiraIssueChangelogs", {"cloudId": cloudId, "issueIdOrKey": issueIdOrKey}
            )
            return {"values": fake.changelogs.get(issueIdOrKey, [])}

        # Write tools the real server also offers: IssueSense must never call them.
        @server.tool()
        def createJiraIssue(cloudId: str, projectKey: str, summary: str) -> dict[str, Any]:
            fake._record("createJiraIssue", {"projectKey": projectKey})
            return {"key": "ATLAS-999"}

        @server.tool()
        def deleteJiraIssue(cloudId: str, issueIdOrKey: str) -> dict[str, Any]:
            fake._record("deleteJiraIssue", {"issueIdOrKey": issueIdOrKey})
            return {"deleted": True}

        return server
