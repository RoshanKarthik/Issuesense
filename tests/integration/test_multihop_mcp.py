from collections import Counter
from collections.abc import Callable
from functools import partial

from packages.agent.pipeline import QueryPipeline
from packages.agent.synthesis import GroundedSynthesizer
from packages.common.models import AuthorizationContext
from packages.common.settings import Settings
from packages.connectors.jira import open_jira_connector
from tests.integration.fake_atlassian import SITE, TODAY, FakeJiraMcp

READER = AuthorizationContext(user_id="alice", scopes=["jira:read:ATLAS"])
HEADLINE = "What changed in Project Atlas this quarter, and what are the risks?"


def _pipeline(fake: FakeJiraMcp, settings: Settings) -> QueryPipeline:
    return QueryPipeline(
        open_jira=partial(open_jira_connector, settings, fake.server()),
        site_url=SITE,
        projects=["ATLAS"],
        synthesizer=GroundedSynthesizer(None),
        today=lambda: TODAY,
    )


async def test_headline_question_runs_targeted_searches_over_mcp(
    fake_jira: FakeJiraMcp, jira_settings: Callable[..., Settings]
) -> None:
    result = await _pipeline(fake_jira, jira_settings()).answer(HEADLINE, READER)

    searches = fake_jira.searches()
    assert 'project = "ATLAS" AND updated >= "2026-10-01" ORDER BY updated DESC' in searches
    assert any('text ~ "risk"' in jql for jql in searches)
    # Comments and history are fetched once per issue, however many searches return it.
    details = Counter(
        (tool, args["issueIdOrKey"])
        for tool, args in fake_jira.calls
        if tool != "searchJiraIssuesUsingJql"
    )
    assert details and max(details.values()) == 1

    card_auth = fake_jira.keys["atlas-card-auth"]
    cited = {item.location.split("#")[0] for item in result.evidence}
    assert card_auth in cited and fake_jira.keys["atlas-double-charge"] in cited
    assert result.verification is not None and result.verification.verified


async def test_a_question_about_the_past_quarter_finds_nothing_in_this_one(
    fake_jira: FakeJiraMcp, jira_settings: Callable[..., Settings]
) -> None:
    result = await _pipeline(fake_jira, jira_settings()).answer(
        "What changed in Atlas last quarter?", READER
    )
    assert all('updated >= "2026-07-01"' in jql for jql in fake_jira.searches())
    assert result.answer.answerability == "INSUFFICIENT_EVIDENCE"
    assert result.evidence == []
