"""Per-user semantic memory + shared team patterns clustered by department.

Adds a second bucket on top of `PerUserStrategy`: facts that describe a
team/department convention (rather than a personal preference) are written
to a namespace scoped by the speaker's department and surfaced to other
members of the same department.

This is "goal 2" from the take-home prompt — learn patterns across users.

Design choices (the what/when/how knobs):
  what:  two schemas. `UserFact` (personal, in per_user.py) and `TeamPattern`
         (shared within a department). Classification happens inside the
         extractor prompts.
  when:  both extractors run after each turn. Two LLM extraction calls per
         turn (the cost of splitting what from where).
  how:   two namespaces in the same local dict-backed store. Personal facts
         live at `("facts", <user_id>)` and patterns at
         `("patterns", <department>)`. Retrieval returns all items from
         both namespaces; the LLM picks relevance from the injected system
         block. The block ends with an explicit precedence instruction so
         the model prefers personal facts over team patterns when they
         conflict.
"""

from collections import defaultdict

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from agent.core import make_agent
from agent.memory._store import DictStore
from agent.memory.per_user import UserFact

DEFAULT_DEPARTMENT = "default"


class TeamPattern(BaseModel):
    """A convention, practice, or pattern that applies to a team or
    department, not to a single individual.
    """

    topic: str = Field(
        description=(
            "What area this pattern concerns — e.g. 'tooling', 'process', "
            "'convention', 'cadence', 'terminology'."
        )
    )
    statement: str = Field(description="The pattern itself, stated concisely.")


class _ExtractedUserFact(UserFact):
    update_id: str | None = Field(
        default=None,
        description=(
            "If this fact replaces an existing record, the id of that "
            "record. Leave null for a new record."
        ),
    )


class _ExtractedTeamPattern(TeamPattern):
    update_id: str | None = Field(
        default=None,
        description=(
            "If this pattern replaces an existing record, the id of that "
            "record. Leave null for a new record."
        ),
    )


class _UserFactExtraction(BaseModel):
    extracted: list[_ExtractedUserFact] = Field(default_factory=list)


class _TeamPatternExtraction(BaseModel):
    extracted: list[_ExtractedTeamPattern] = Field(default_factory=list)


PERSONAL_EXTRACTOR_INSTRUCTIONS = """\
You are extracting durable personal facts about the user from a conversation.

RULES:
- Only first-person statements — signaled by "I", "my", "me", "mine".
- DO NOT extract team/group statements — "we", "our", "the team",
  "everyone", "our org", "we all", "standard practice". Those are patterns,
  not personal facts, and are handled elsewhere.
- Always capture specific nouns the user mentions: tools, languages,
  frameworks, operating systems, locations, companies, people, cadences.
  A user asking "what X do I prefer?" later should be able to match a stored
  record whose value contains the specific name they're asking about.
- Use a short CATEGORY as the subject (what kind of fact), and put the
  specific noun in the value. Do not use the user's sentence phrasing.
- Produce concise, atomic facts — one record per distinct fact. A single
  sentence may contain multiple facts; decompose when they stand alone.
- If a new fact supersedes an existing one, set `update_id` to the existing
  fact's id so we overwrite it in place.
- If there is nothing worth storing, return an empty list.

EXAMPLES:
  "I prefer Rust for systems programming"
    → {subject: "programming language", value: "Rust (for systems programming)"}
  "I use macOS for development"
    → {subject: "operating system", value: "macOS"}
  "I work on data pipelines, I live in Seattle, my main language is Python"
    → {subject: "work area",          value: "data pipelines"}
      {subject: "location",           value: "Seattle"}
      {subject: "primary language",   value: "Python"}
  "Bob's manager is Sue"  (third-person — NOT about the speaker)
    → extract nothing
"""


PATTERN_EXTRACTOR_INSTRUCTIONS = """\
You are extracting team/department patterns and conventions from a conversation.

RULES:
- Only group statements — signaled by "we", "our", "the team", "everyone",
  "standard practice", "we all". Examples: "the team uses pytest",
  "we do sprint planning every other Monday", "our standard is Terraform".
- DO NOT extract purely personal preferences — "I", "my", "me". Those are
  handled elsewhere and must not end up here.
- These patterns will be SHARED with every member of the speaker's
  department, so be conservative: only extract a pattern when the statement
  is clearly about a group, not an individual. When in doubt, do not extract.
- If a new pattern supersedes an existing one (team migrated, changed
  process, etc.), set `update_id` to the existing pattern's id so we
  overwrite it in place — do not leave both records.
- If there is nothing worth storing, return an empty list.
"""


class PerUserPlusPatternsStrategy:
    """Per-user personal facts + department-scoped team patterns."""

    name = "per_user_plus_patterns"

    def __init__(
        self,
        model_str: str = "anthropic:claude-haiku-4-5-20251001",
        system_prompt: str | None = None,
    ):
        self.model_str = model_str
        self.system_prompt = system_prompt
        self.store = DictStore()
        model = init_chat_model(model_str)
        self._personal_extractor = model.with_structured_output(_UserFactExtraction)
        self._pattern_extractor = model.with_structured_output(_TeamPatternExtraction)
        self._agent = None
        self._fact_keys: dict[str, set[str]] = defaultdict(set)
        self._pattern_keys: dict[str, set[str]] = defaultdict(set)

    def build_agent(self):
        if self._agent is None:
            self._agent = make_agent(self.model_str, self.system_prompt)
        return self._agent

    def pre_turn(self, user_msg: dict, messages: list[dict]) -> list[dict]:
        user_id = user_msg.get("user") or "unknown"
        department = user_msg.get("department") or DEFAULT_DEPARTMENT

        personal_hits = self.store.list(("facts", user_id), limit=20)
        pattern_hits = self.store.list(("patterns", department), limit=20)

        blocks: list[str] = []
        if personal_hits:
            blocks.append(self._render_personal(user_id, personal_hits))
        if pattern_hits:
            blocks.append(self._render_patterns(department, pattern_hits))

        out: list[dict] = list(messages)
        if blocks:
            block = "\n\n".join(blocks)
            block += (
                "\n\nWhen answering, personal facts about the current user "
                "take precedence over team patterns."
            )
            out = [{"role": "system", "content": block}, *out]
        out.append({"role": "user", "content": user_msg.get("content", "")})
        return out

    def post_turn(self, user_msg: dict, ai_content: str) -> None:
        user_id = user_msg.get("user") or "unknown"
        department = user_msg.get("department") or DEFAULT_DEPARTMENT
        exchange_block = (
            f"user: {user_msg.get('content', '')}\n"
            f"assistant: {ai_content}"
        )

        personal_existing = self._render_existing(
            ("facts", user_id), "subject", "value"
        )
        p_result = self._personal_extractor.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        f"{PERSONAL_EXTRACTOR_INSTRUCTIONS}\n\n"
                        f"Existing facts for this user:\n{personal_existing}"
                    ),
                },
                {"role": "user", "content": exchange_block},
            ]
        )
        for item in p_result.extracted:
            key = item.update_id or f"{user_id}-fact-{len(self._fact_keys[user_id])}"
            self.store.put(
                ("facts", user_id),
                key=key,
                value=UserFact(subject=item.subject, value=item.value).model_dump(),
            )
            self._fact_keys[user_id].add(key)

        pattern_existing = self._render_existing(
            ("patterns", department), "topic", "statement"
        )
        t_result = self._pattern_extractor.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        f"{PATTERN_EXTRACTOR_INSTRUCTIONS}\n\n"
                        f"Existing patterns for the {department} department:\n"
                        f"{pattern_existing}"
                    ),
                },
                {"role": "user", "content": exchange_block},
            ]
        )
        for item in t_result.extracted:
            key = (
                item.update_id
                or f"{department}-pattern-{len(self._pattern_keys[department])}"
            )
            self.store.put(
                ("patterns", department),
                key=key,
                value=TeamPattern(
                    topic=item.topic, statement=item.statement
                ).model_dump(),
            )
            self._pattern_keys[department].add(key)

    def _render_existing(
        self, namespace: tuple, field_a: str, field_b: str
    ) -> str:
        items = self.store.list(namespace, limit=50)
        if not items:
            return "(none)"
        lines = []
        for item in items:
            v = item.value if isinstance(item.value, dict) else {}
            lines.append(
                f"- id={item.key}: {v.get(field_a, '')} = {v.get(field_b, '')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_personal(user_id: str, hits) -> str:
        lines = [f"[Personal facts about the current user ({user_id})]"]
        for hit in hits:
            value = hit.value if isinstance(hit.value, dict) else {}
            lines.append(
                f"- {value.get('subject', 'fact')}: {value.get('value', '')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_patterns(department: str, hits) -> str:
        lines = [f"[Conventions shared across the {department} department]"]
        for hit in hits:
            value = hit.value if isinstance(hit.value, dict) else {}
            lines.append(
                f"- {value.get('topic', 'pattern')}: {value.get('statement', '')}"
            )
        return "\n".join(lines)
