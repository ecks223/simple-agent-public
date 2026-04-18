"""In-process dict-backed store used by the memory strategies.

Why not `langgraph.store.memory.InMemoryStore`?
  - Its added value over a plain dict is the optional vector index for
    semantic search, which requires a separate embedding provider API key
    (OpenAI / Voyage / Cohere) beyond what the agent itself needs.
  - At this demo's scale — a handful of facts per user — "return all
    namespace items and let the LLM pick relevant ones" is enough; the
    injected system block is short.
  - Fewer moving parts, easier to explain, no extra credentials.

The interface deliberately mirrors the subset of `BaseStore` the strategies
used (`put` / `get` / `list`), so swapping back to `InMemoryStore` later is
a drop-in change if production scale ever demands semantic retrieval.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class StoreItem:
    namespace: tuple
    key: str
    value: dict


class DictStore:
    """Dict-backed namespace KV. In-process only — dies with the process."""

    def __init__(self):
        self._data: dict[tuple, dict[str, dict]] = defaultdict(dict)

    def put(self, namespace: tuple, key: str, value: dict) -> None:
        self._data[namespace][key] = value

    def get(self, namespace: tuple, key: str) -> StoreItem | None:
        value = self._data[namespace].get(key)
        return StoreItem(namespace, key, value) if value is not None else None

    def list(self, namespace: tuple, limit: int = 50) -> list[StoreItem]:
        return [
            StoreItem(namespace, k, v)
            for k, v in list(self._data[namespace].items())[:limit]
        ]
