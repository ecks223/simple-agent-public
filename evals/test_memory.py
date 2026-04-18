"""Parameterized pytest harness comparing memory strategies.

Each test is one (strategy, scenario) pair. The scenario's turns run in order;
every turn is asserted against its own `expect` / `forbid` checks. Per-turn
results are accumulated into the session-scoped `METRICS` table (see
evals/conftest.py), which is printed at the end of the run.

A scenario is considered failed (in pytest's sense) if ANY turn inside it
fails its check. The aggregate metric counts each turn independently.
"""

from __future__ import annotations

import pytest
from dotenv import load_dotenv

from agent.memory import STRATEGIES

from ._metrics import record_turn
from .conversations import SCENARIOS

load_dotenv()


@pytest.fixture(params=list(STRATEGIES), ids=list(STRATEGIES))
def strategy(request):
    return STRATEGIES[request.param]()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_scenario(strategy, scenario):
    agent = strategy.build_agent()
    messages: list[dict] = []
    failures: list[str] = []

    for i, turn in enumerate(scenario.turns):
        if turn.reset_before:
            messages = []

        user_msg = {
            "role": "user",
            "content": turn.prompt,
            "user": turn.user,
            "department": turn.department,
        }

        invoke_messages = strategy.pre_turn(user_msg, messages)
        result = agent.invoke({"messages": invoke_messages})
        ai_content = result["messages"][-1].content
        if isinstance(ai_content, list):
            # Anthropic can return content as a list of blocks; stringify.
            ai_content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in ai_content
            )

        # Record this turn in the running conversation history.
        messages.append({"role": "user", "content": turn.prompt})
        messages.append({"role": "assistant", "content": ai_content})

        # Update the store (no-op for baseline).
        strategy.post_turn(user_msg, ai_content)

        passed = True
        if turn.expect:
            if not all(s.lower() in ai_content.lower() for s in turn.expect):
                passed = False
        if turn.forbid:
            if any(s.lower() in ai_content.lower() for s in turn.forbid):
                passed = False

        record_turn(strategy.name, scenario.category, passed)

        if not passed:
            failures.append(
                f"  turn {i} (user={turn.user!r}): "
                f"expect={turn.expect} forbid={turn.forbid}\n"
                f"    reply: {ai_content[:240]!r}"
            )

    if failures:
        pytest.fail(
            f"{scenario.name}: {len(failures)}/{len(scenario.turns)} turns failed\n"
            + "\n".join(failures),
            pytrace=False,
        )
