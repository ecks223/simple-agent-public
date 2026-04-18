"""Memory strategy protocol + baseline (no long-term memory).

Every strategy exposes the same three hooks:

    build_agent()          -> the compiled LangGraph agent to invoke
    pre_turn(user_msg,     -> the list[dict] of messages to pass to .invoke()
             messages)        on this turn (may prepend a memory system block)
    post_turn(user_msg,    -> writes extracted memories to the store
              ai_content)     (no-op for strategies without long-term memory)

This keeps the harness oblivious to what a strategy does internally.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryStrategy(Protocol):
    name: str

    def build_agent(self): ...

    def pre_turn(
        self, user_msg: dict, messages: list[dict]
    ) -> list[dict]: ...

    def post_turn(self, user_msg: dict, ai_content: str) -> None: ...


class BaselineStrategy:
    """No long-term memory. The agent sees only messages from the current session.

    This is the existing behavior of the CLI prior to any memory work — kept as
    an explicit strategy so the harness can measure what everything else wins
    relative to it.
    """

    name = "baseline"

    def __init__(
        self,
        model_str: str = "anthropic:claude-haiku-4-5-20251001",
        system_prompt: str | None = None,
    ):
        self.model_str = model_str
        self.system_prompt = system_prompt
        self._agent = None

    def build_agent(self):
        if self._agent is None:
            from agent.core import make_agent

            self._agent = make_agent(self.model_str, self.system_prompt)
        return self._agent

    def pre_turn(self, user_msg: dict, messages: list[dict]) -> list[dict]:
        return [
            *messages,
            {"role": "user", "content": user_msg.get("content", "")},
        ]

    def post_turn(self, user_msg: dict, ai_content: str) -> None:
        return
