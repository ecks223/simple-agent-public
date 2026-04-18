"""Per-user semantic memory (collection shape, background extraction).

What is stored: atomic `UserFact` records extracted from each exchange.
When:           after each turn, via a direct `model.with_structured_output`
                call that returns new facts and optional update-ids.
How:            one namespace per user (`("facts", <user_id>)`) in a local
                dict-backed store; retrieval returns all stored facts, and
                the LLM picks the relevant one from the injected system block.
"""

from collections import defaultdict

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from agent.core import make_agent
from agent.memory._store import DictStore


class UserFact(BaseModel):
    """A fact the user stated about themselves personally."""

    subject: str = Field(
        description=(
            "What aspect this fact concerns — e.g. 'preference', 'identity', "
            "'location', 'work', 'skill', 'goal'."
        )
    )
    value: str = Field(description="The fact itself, stated concisely.")


class _ExtractedUserFact(UserFact):
    """A UserFact the extractor decided to write. Carries an optional
    `update_id` so the model can supersede an existing record in place.
    """

    update_id: str | None = Field(
        default=None,
        description=(
            "If this fact replaces an existing record, the id of that "
            "record (from the 'Existing facts' list). Leave null for a new "
            "record."
        ),
    )


class _UserFactExtraction(BaseModel):
    """Structured output produced by the extractor on each turn."""

    extracted: list[_ExtractedUserFact] = Field(
        default_factory=list,
        description="Facts to write to the store (new or updates).",
    )


PER_USER_EXTRACTOR_INSTRUCTIONS = """\
You are extracting durable personal facts about the user from a conversation.

RULES:
- Only first-person statements — signaled by "I", "my", "me", "mine".
- DO NOT extract team/group statements — "we", "our", "the team",
  "everyone", "our org", "standard practice" — those belong elsewhere.
- Always capture specific nouns the user mentions: tools, languages,
  frameworks, operating systems, locations, companies, people, cadences.
  A user asking "what X do I prefer?" later should be able to match a stored
  record whose value contains the specific name they're asking about.
- Use a short CATEGORY as the subject (what kind of fact), and put the
  specific noun in the value. Do not use the user's sentence phrasing.
- Produce concise, atomic facts — one record per distinct fact. A single
  sentence may contain multiple facts; decompose when they stand alone.
- If a new fact supersedes an existing one (the user changed their mind,
  moved, switched tools, etc.), set `update_id` to the existing fact's id
  so we overwrite it in place rather than accumulating stale records.
- If there is nothing worth storing, return an empty `extracted` list.

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


class PerUserStrategy:
    """Semantic collection memory, scoped per user."""

    name = "per_user"

    def __init__(
        self,
        model_str: str = "anthropic:claude-haiku-4-5-20251001",
        system_prompt: str | None = None,
    ):
        self.model_str = model_str
        self.system_prompt = system_prompt
        self.store = DictStore()
        self._extractor = init_chat_model(model_str).with_structured_output(
            _UserFactExtraction
        )
        self._agent = None
        self._fact_keys: dict[str, set[str]] = defaultdict(set)

    def build_agent(self):
        if self._agent is None:
            self._agent = make_agent(self.model_str, self.system_prompt)
        return self._agent

    def pre_turn(self, user_msg: dict, messages: list[dict]) -> list[dict]:
        user_id = user_msg.get("user") or "unknown"
        content = user_msg.get("content", "")

        hits = self.store.list(("facts", user_id), limit=20)
        out: list[dict] = list(messages)
        if hits:
            out = [
                {"role": "system", "content": self._render_facts(user_id, hits)},
                *out,
            ]
        out.append({"role": "user", "content": content})
        return out

    def post_turn(self, user_msg: dict, ai_content: str) -> None:
        user_id = user_msg.get("user") or "unknown"
        existing = self._render_existing(user_id)
        prompt = [
            {
                "role": "system",
                "content": (
                    f"{PER_USER_EXTRACTOR_INSTRUCTIONS}\n\n"
                    f"Existing facts for this user:\n{existing}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"user: {user_msg.get('content', '')}\n"
                    f"assistant: {ai_content}"
                ),
            },
        ]
        result = self._extractor.invoke(prompt)
        for item in result.extracted:
            key = item.update_id or self._next_key(user_id)
            self.store.put(
                ("facts", user_id),
                key=key,
                value=UserFact(subject=item.subject, value=item.value).model_dump(),
            )
            self._fact_keys[user_id].add(key)

    def _next_key(self, user_id: str) -> str:
        return f"{user_id}-{len(self._fact_keys[user_id])}"

    def _render_existing(self, user_id: str) -> str:
        items = self.store.list(("facts", user_id), limit=50)
        if not items:
            return "(none)"
        lines = []
        for item in items:
            v = item.value if isinstance(item.value, dict) else {}
            lines.append(
                f"- id={item.key}: {v.get('subject', 'fact')} = {v.get('value', '')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_facts(user_id: str, hits) -> str:
        lines = [f"[Known facts about the current user ({user_id})]"]
        for hit in hits:
            value = hit.value if isinstance(hit.value, dict) else {}
            subject = value.get("subject", "fact")
            v = value.get("value", "")
            lines.append(f"- {subject}: {v}")
        lines.append(
            "Use these facts when answering the current user's question. "
            "If the question is not about the user, you may ignore them."
        )
        return "\n".join(lines)
