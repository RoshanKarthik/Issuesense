import json
from datetime import date
from typing import Any

import pytest

from packages.agent.llm import LlmProviderError, LlmRequest, LlmResponse
from packages.agent.planner import LlmPlanner, RulePlanner, clean_terms, planner_for
from tests.unit.agent_fakes import ScriptedProvider

TODAY = date(2026, 10, 15)
HEADLINE = "Summarize what changed in Project Atlas this quarter and identify the major risks."


async def test_rules_decompose_a_compound_question_by_aspect() -> None:
    plan = await RulePlanner().plan(HEADLINE, TODAY, ["ATLAS"])
    assert plan.planner == "rules"
    assert [q.id for q in plan.sub_questions] == ["Q1", "Q2"]
    changes, risks = plan.sub_questions
    assert changes.text.startswith("What changed") and changes.time_bound
    assert risks.text == "What are the open risks?"
    assert not risks.time_bound  # risks are current, not bound to the period
    assert "risk" in risks.terms and "blocked" in risks.terms
    assert plan.rewritten.endswith(
        "this quarter (2026-10-01 to 2026-12-31) and identify the major risks."
    )
    assert plan.time_range is not None and plan.time_range.start == date(2026, 10, 1)


async def test_rules_keep_a_single_aspect_question_whole() -> None:
    question = "What is the status of the card authorization work?"
    plan = await RulePlanner().plan(question, TODAY, ["ATLAS"])
    assert len(plan.sub_questions) == 1
    assert plan.sub_questions[0].text == question
    assert plan.sub_questions[0].terms == ["card", "authorization"]
    assert plan.time_range is None and plan.rewritten == question


async def test_rules_carry_the_subject_into_each_aspect() -> None:
    plan = await RulePlanner().plan(
        "Which payments deadlines slipped and why is it risky?", TODAY, []
    )
    assert len(plan.sub_questions) == 2
    assert all("payments" in q.text and "payments" in q.terms for q in plan.sub_questions)


async def test_project_keys_are_not_search_terms() -> None:
    plan = await RulePlanner().plan("What blocks ATLAS checkout?", TODAY, ["ATLAS"])
    assert "atlas" not in plan.sub_questions[0].terms


def test_clean_terms_drop_anything_that_could_escape_a_query() -> None:
    assert clean_terms(['risk" OR project = SECRET', "due-date", "x", "the", "Risk"]) == [
        "risk",
        "project",
        "secret",
        "due-date",
    ]


def _plan_reply(**overrides: Any) -> str:
    body: dict[str, Any] = {
        "rewritten_question": "What changed in Atlas in Q4 2026, and what are the risks?",
        "sub_questions": [
            {"question": "Which due dates moved?", "search_terms": ["due", "Moved", 'date"x']},
            {"question": "Which risks are open?", "search_terms": []},
        ],
    }
    body.update(overrides)
    return json.dumps(body)


async def test_llm_plan_is_validated_and_terms_sanitized() -> None:
    provider = ScriptedProvider("<think>hmm</think>```json\n" + _plan_reply() + "\n```")
    plan = await LlmPlanner(provider).plan(HEADLINE, TODAY, ["ATLAS"])
    assert plan.planner == "llm" and plan.note is None
    assert plan.rewritten.startswith("What changed in Atlas")
    assert plan.sub_questions[0].terms == ["due", "moved", "date"]
    assert plan.sub_questions[1].terms == ["risks", "open"]  # derived from its text
    assert plan.time_range is not None and plan.time_range.end == date(2026, 12, 31)
    assert (plan.input_tokens, plan.output_tokens) == (100, 50)


async def test_llm_planner_receives_the_period_but_no_evidence() -> None:
    provider = ScriptedProvider(_plan_reply())
    await LlmPlanner(provider).plan(HEADLINE, TODAY, ["ATLAS"])
    user = provider.requests[0].messages[1].content
    assert "2026-10-01 to 2026-12-31" in user and "Today: 2026-10-15" in user
    assert "<evidence" not in user


@pytest.mark.parametrize(
    "reply",
    [
        "I think you should look at Jira.",
        json.dumps({"rewritten_question": "x?", "sub_questions": []}),
        _plan_reply(sub_questions=[{"question": f"Q{i}?"} for i in range(6)]),
    ],
)
async def test_invalid_llm_plan_falls_back_to_rules(reply: str) -> None:
    plan = await LlmPlanner(ScriptedProvider(reply)).plan(HEADLINE, TODAY, ["ATLAS"])
    assert plan.planner == "rules"
    assert plan.note == "Planned with rules because the language model returned an invalid plan."
    assert len(plan.sub_questions) == 2
    assert plan.input_tokens == 100  # the discarded call still cost tokens


async def test_provider_failure_falls_back_to_rules() -> None:
    class Down:
        name = "down"

        async def complete(self, request: LlmRequest) -> LlmResponse:
            raise LlmProviderError("The language model could not be reached.")

    plan = await LlmPlanner(Down()).plan(HEADLINE, TODAY, ["ATLAS"])
    assert plan.planner == "rules"
    assert plan.note is not None and "unavailable" in plan.note
    assert plan.input_tokens == 0


def test_planner_for_uses_rules_without_a_provider() -> None:
    assert isinstance(planner_for(None), RulePlanner)
    assert isinstance(planner_for(ScriptedProvider()), LlmPlanner)
