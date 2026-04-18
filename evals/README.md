# Memory-strategy evaluation harness

This harness runs a fixed set of scripted conversations against each memory
strategy and reports a quantitative comparison.

## Running

```bash
uv sync
uv run pytest evals/test_memory.py -v
```

> **This hits real LLM APIs and costs tokens.** All three strategies need
> only `ANTHROPIC_API_KEY` in `.env`.
>
> **Note on the simpler-than-you-might-expect store.** The first cut of the
> memory strategies used LangChain's LangMem + LangGraph's `InMemoryStore`
> with an OpenAI embedding index (what the LangGraph memory docs recommend).
> That worked but needed a second API key (`OPENAI_API_KEY`) for the
> embeddings, which our development environment didn't have — and Anthropic
> doesn't offer embeddings directly. Rather than require a second provider,
> we swapped to a local dict-backed store (see
> [src/agent/memory/_store.py](../src/agent/memory/_store.py)) and replaced
> LangMem's `create_memory_manager` with a direct
> `model.with_structured_output(...)` call. At this scale — a handful of
> facts per user — this is sufficient; the LLM does the relevance filtering
> the vector index would have done. Swapping back to the embedding-indexed
> store is a one-file change if production scale ever needs semantic
> retrieval.

Run only the matrix for one strategy:

```bash
uv run pytest evals/test_memory.py -v -k "per_user_plus_patterns"
```

Run only one scenario across all strategies:

```bash
uv run pytest evals/test_memory.py -v -k "team_convention_propagates"
```

## What it does

For every `(strategy, scenario)` pair:

1. Instantiate a fresh strategy — a fresh in-process `DictStore` — so
   scenarios don't pollute each other.
2. Walk the scenario's turns in order. Each turn is one user message followed
   by one agent reply.
3. When a turn is marked `reset_before=True`, the local `messages` list is
   cleared before the turn runs. The strategy's Store is **not** cleared —
   that's what makes cross-session recall testable.
4. After each reply, the strategy's `post_turn` hook runs — for memory
   strategies this is where facts are extracted and written to the store.
5. Assert `expect` substrings (all must appear, case-insensitive) and
   `forbid` substrings (none may appear) against the reply.

## Metric

Results are accumulated in `evals/_metrics.py` and printed at session end,
grouped by strategy × category × overall. The same table is written to
`evals/out/metrics.md`.

```
Memory Strategy Evaluation
======================================================================
strategy                cross_user  per_user  regression  overall
----------------------  ----------  --------  ----------  ------------
baseline                11/14       9/15      3/3         23/32 (72%)
per_user                12/14       15/15     3/3         30/32 (94%)
per_user_plus_patterns  14/14       15/15     3/3         32/32 (100%)
```

(Numbers drift ±1 between runs on flaky turns. The shape — baseline flat
on persistence, per_user solid on per-user and weak on cross-user,
per_user_plus_patterns strong on both — is stable across runs.)

Categories:

- **per_user** — facts a user shares in one session must be recalled in a later
  session by the same user.
- **cross_user** — patterns stated by one user should propagate to other users
  **of the same department** (team conventions, cadences); the same patterns
  must NOT bleed into other departments; personal preferences must NOT leak
  across users at all.
- **regression** — existing same-session behavior that the memory layer must
  not break.

## Scenario structure

Defined in `evals/conversations.py` as `Scenario(name, category, turns)` where
each `Turn` has:

| Field          | Meaning                                                           |
| -------------- | ----------------------------------------------------------------- |
| `prompt`       | The user's message content                                        |
| `user`         | Speaker identifier (drives per-user memory namespace)             |
| `department`   | Speaker's department (drives team-pattern namespace in strategy 2)|
| `expect`       | Substrings that MUST all appear in the reply                      |
| `forbid`       | Substrings that MUST NOT appear in the reply                      |
| `reset_before` | If true, clear the local history before this turn (keep Store)    |

Add scenarios by appending to `SCENARIOS` in `conversations.py`. No harness
changes required.

## Interpreting failures

Pytest fails a scenario as soon as any turn in it fails. The failure message
lists every failing turn with its `expect` / `forbid` and the first ~240 chars
of the actual reply. Use `-v` to see the full matrix even when some pass.
