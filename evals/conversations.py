"""Scripted conversations for the memory-strategy harness.

Each Scenario is a sequence of Turns. Every Turn is one user message followed
by one assistant reply; turn-level `expect` / `forbid` checks are asserted
against that reply.

Scenarios are grouped by `category`:
  - per_user:    learn about a specific user across sessions
  - cross_user:  patterns learned from one user benefit another user
  - regression:  existing behavior the memory layer must not break

A Turn with `reset_before=True` clears the local conversation history before
the turn runs (simulating a fresh conversation). The strategy's Store is NOT
cleared — that's the whole point of the comparison.

Each Turn also carries:
  - `user`:       speaker identifier; drives per-user memory namespace.
  - `department`: the speaker's department; drives pattern-memory namespace
                  in `per_user_plus_patterns`. Team patterns are shared within
                  a department, not globally — pytest being an engineering
                  convention doesn't make it a marketing convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Category = Literal["per_user", "cross_user", "regression"]


@dataclass
class Turn:
    prompt: str
    user: str = "alice"
    department: str = "engineering"
    expect: list[str] = field(default_factory=list)
    forbid: list[str] = field(default_factory=list)
    reset_before: bool = False


@dataclass
class Scenario:
    name: str
    category: Category
    turns: list[Turn]


SCENARIOS: list[Scenario] = [
    # -----------------------------------------------------------------
    # per_user — learn about specific users across sessions
    # -----------------------------------------------------------------
    Scenario(
        name="preference_recall",
        category="per_user",
        turns=[
            Turn(prompt="I prefer pytest over unittest for Python testing.", user="alice"),
            Turn(
                prompt="What testing framework do I prefer?",
                user="alice",
                reset_before=True,
                expect=["pytest"],
            ),
        ],
    ),
    Scenario(
        name="preference_update",
        category="per_user",
        turns=[
            Turn(prompt="I use macOS for development.", user="alice"),
            Turn(prompt="Actually I switched to Linux last week.", user="alice"),
            Turn(
                prompt="What operating system do I use?",
                user="alice",
                reset_before=True,
                expect=["Linux"],
            ),
        ],
    ),
    Scenario(
        name="multi_fact_recall",
        category="per_user",
        turns=[
            Turn(
                prompt=(
                    "I work on data pipelines, I live in Seattle, and my main "
                    "language is Python."
                ),
                user="alice",
            ),
            # Reset before each question so no single turn can be satisfied by
            # re-reading the transcript — every question must come from the store.
            Turn(
                prompt="Where do I live?",
                user="alice",
                reset_before=True,
                expect=["Seattle"],
            ),
            Turn(
                prompt="What do I work on?",
                user="alice",
                reset_before=True,
                expect=["pipeline"],
            ),
            Turn(
                prompt="What is my primary programming language?",
                user="alice",
                reset_before=True,
                expect=["Python"],
            ),
        ],
    ),
    Scenario(
        name="two_users_no_crosstalk",
        category="per_user",
        turns=[
            Turn(prompt="I prefer Rust for systems programming.", user="alice"),
            Turn(prompt="I prefer Go for systems programming.", user="bob"),
            Turn(
                prompt="What language do I prefer for systems programming?",
                user="alice",
                reset_before=True,
                expect=["Rust"],
                forbid=["Go"],
            ),
            Turn(
                prompt="What language do I prefer for systems programming?",
                user="bob",
                reset_before=True,
                expect=["Go"],
                forbid=["Rust"],
            ),
        ],
    ),
    Scenario(
        # Tier-2 review add: a third-person statement ("Bob's manager is Sue")
        # must NOT be stored as a personal fact about the speaker. The extractor
        # prompt explicitly rejects non-first-person statements; this scenario
        # pins that behavior.
        name="third_person_not_extracted_as_personal",
        category="per_user",
        turns=[
            Turn(prompt="Bob's manager is Sue.", user="alice"),
            Turn(
                prompt="Who is my manager?",
                user="alice",
                reset_before=True,
                forbid=["Sue"],
            ),
        ],
    ),
    # -----------------------------------------------------------------
    # cross_user — patterns one user teaches benefit another user of the
    # same department, but must NOT leak to a different department.
    # -----------------------------------------------------------------
    Scenario(
        name="team_convention_propagates_within_department",
        category="cross_user",
        turns=[
            Turn(
                prompt="At our team we use pytest for all Python projects.",
                user="alice",
                department="engineering",
            ),
            Turn(
                prompt="What testing framework should I use for my Python project?",
                user="bob",
                department="engineering",
                reset_before=True,
                expect=["pytest"],
            ),
        ],
    ),
    Scenario(
        name="team_cadence_propagates_within_department",
        category="cross_user",
        turns=[
            Turn(
                prompt="We do sprint planning every other Monday.",
                user="alice",
                department="engineering",
            ),
            Turn(
                prompt="When does our team do sprint planning?",
                user="carol",
                department="engineering",
                reset_before=True,
                # "Monday" alone is too generic — a model with no memory can
                # guess "Monday" from sprint-planning priors. "every other" is
                # the distinctive phrase from Alice's seed.
                expect=["every other"],
            ),
        ],
    ),
    Scenario(
        name="pattern_isolation_across_departments",
        category="cross_user",
        turns=[
            Turn(
                prompt="At our team we use pytest for all Python projects.",
                user="alice",
                department="engineering",
            ),
            Turn(
                prompt=(
                    "What testing framework does our marketing team use for "
                    "our landing page scripts?"
                ),
                user="dan",
                department="marketing",
                reset_before=True,
                forbid=["pytest"],
            ),
        ],
    ),
    Scenario(
        name="personal_does_not_leak_across_users",
        category="cross_user",
        turns=[
            Turn(
                prompt="I personally prefer dark mode in my editor.",
                user="alice",
                department="engineering",
            ),
            Turn(
                prompt="What editor theme do I use?",
                user="bob",
                department="engineering",
                reset_before=True,
                # A bare "dark" substring is too noisy (e.g. "I don't know if
                # you prefer light or dark"); check the distinctive phrase.
                forbid=["dark mode"],
            ),
        ],
    ),
    Scenario(
        # Tier-2 review add: when a personal fact and a team pattern cover the
        # same topic, the injected block tells the model personal wins for
        # personal questions. This scenario pins that behavior in evals rather
        # than only in prose.
        name="personal_overrides_team_when_conflict",
        category="cross_user",
        turns=[
            Turn(
                prompt="We standardize on Google Docs for all team documents.",
                user="alice",
                department="engineering",
            ),
            Turn(
                prompt="I personally prefer Notion for organizing my own notes.",
                user="alice",
                department="engineering",
            ),
            Turn(
                prompt="Which tool do I personally use for my own notes?",
                user="alice",
                department="engineering",
                reset_before=True,
                expect=["Notion"],
            ),
        ],
    ),
    Scenario(
        # Tier-2 review add: the pattern extractor has enable_updates=True;
        # this scenario exercises the update path by having the same user
        # retract and replace a team pattern. A different user of the same
        # department should see the new value, not the retracted one.
        name="team_pattern_supersession",
        category="cross_user",
        turns=[
            Turn(
                prompt="We use Jenkins for our CI/CD pipelines.",
                user="alice",
                department="engineering",
            ),
            Turn(
                prompt=(
                    "Actually we migrated from Jenkins to GitHub Actions last "
                    "quarter — Jenkins is no longer in use on our team."
                ),
                user="alice",
                department="engineering",
            ),
            Turn(
                prompt="What CI/CD tool does our team use?",
                user="bob",
                department="engineering",
                reset_before=True,
                expect=["GitHub Actions"],
                forbid=["Jenkins"],
            ),
        ],
    ),
    # -----------------------------------------------------------------
    # regression — existing behavior the memory layer must not break
    # -----------------------------------------------------------------
    Scenario(
        name="within_session_recall",
        category="regression",
        turns=[
            Turn(prompt="My name is Alice.", user="alice"),
            Turn(prompt="What is my name?", user="alice", expect=["Alice"]),
        ],
    ),
    Scenario(
        name="simple_math_no_memory",
        category="regression",
        turns=[
            Turn(prompt="What is 2 + 2?", user="alice", expect=["4"]),
        ],
    ),
]
