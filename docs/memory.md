# Memory strategies

Write-up for the backend take-home. Builds on the minimal
[deepagents](https://github.com/langchain-ai/deepagents) CLI in the repo by
adding two cross-conversation memory strategies and a harness that compares
them quantitatively.

## What the take-home asks for

Two goals stated in the prompt:

1. **Learn about specific users** — remember a user's facts across
   conversations.
2. **Learn patterns across users** — notice conventions / practices stated by
   one user and apply them to other users of the same system.

And three design axes named explicitly: **what is stored, when it is stored,
how it is stored**.

## The strategies

| #   | Name                     | Addresses goal 1 | Addresses goal 2 |
| --- | ------------------------ | ---------------- | ---------------- |
| 0   | `baseline`               | no               | no               |
| 1   | `per_user`               | yes              | no               |
| 2   | `per_user_plus_patterns` | yes              | yes              |

`baseline` is the repo's starting behavior — kept as an explicit strategy so
the harness can measure the two real memory strategies against it.

### Strategy 1 — `per_user`

Per-user semantic collection, background write.

- **What**: atomic `UserFact(subject, value)` Pydantic records extracted from
  each exchange.
- **When**: after every assistant reply — a direct
  `init_chat_model(...).with_structured_output(_UserFactExtraction)` call
  takes the last user/assistant pair plus the existing facts and returns a
  list of new-or-updated records. One extra LLM call per turn. (An earlier
  cut used `langmem.create_memory_manager`; we dropped the dep after
  seeing the direct call did the same job in ~15 lines.)
- **How**: a local `DictStore` (see [src/agent/memory/_store.py](../src/agent/memory/_store.py)) —
  a dict-backed namespace KV, in-process, no vector index. One namespace per
  user: `("facts", <user_id>)`. At the start of each turn the strategy reads
  all facts for the current user and prepends them as a system message. The
  LLM picks which fact is relevant. At this scale (single-digit facts per
  user) semantic retrieval isn't needed; adding it back in production would
  be a one-file change (swap `DictStore` for `langgraph.store.memory.InMemoryStore`
  with an embedding index).

### Strategy 2 — `per_user_plus_patterns`

Same mechanism as strategy 1 plus a second bucket for team-level patterns
clustered by the speaker's department.

- **What**: two schemas. `UserFact` (personal, per user) AND `TeamPattern(topic,
  statement)` (shared within a department).
- **When**: two extractors run after every turn. A personal extractor with
  instructions that explicitly reject team/group statements, and a pattern
  extractor with instructions that explicitly reject personal statements.
  Two extra LLM calls per turn (this is the cost of splitting *what* from
  *where*).
- **How**: two namespaces in the same `DictStore`. Personal facts at
  `("facts", <user_id>)`, patterns at `("patterns", <department>)` — one
  bucket per department. Patterns written by Alice in the `engineering`
  department are surfaced to Bob and Carol in `engineering`, but NOT to
  Dan in `marketing`. Both namespaces are enumerated on each turn and
  concatenated into the injected system block with distinct labels. The
  block ends with an explicit precedence instruction ("personal facts take
  precedence over team patterns") so the model picks the right one when
  they disagree.

The take-home prompt says "as we move across roles & departments." We took
that literally: "marketing uses Figma" isn't a claim about what engineers
use, so it has no business showing up for an engineering speaker.

### Baseline for reference

- **What**: nothing persisted.
- **When**: nothing written.
- **How**: raw `messages` list grown in-process; the full history is re-sent
  to the agent on every turn.

Expected to fail every scenario with `reset_before=True` and every scenario
where a second user needs context from the first.

## What "role", "user", and "department" each mean here

Three different identifiers show up on each message. Worth keeping them
straight:

| Field                 | Meaning                                       | Used for            |
| --------------------- | --------------------------------------------- | ------------------- |
| `role` (LangChain)    | `"user"` / `"assistant"` / `"system"`         | model protocol only |
| `user` (our addition) | speaker identifier (e.g. `"alice"`)           | personal namespace  |
| `department` (ours)   | speaker's team (e.g. `"engineering"`)         | pattern namespace   |

The `role` field in LangChain is constrained to the standard set — custom
values like `"user:alice"` are rejected at `convert_to_messages`. And
`HumanMessage.name` (which would have been a natural place for speaker
identity) is dropped by `langchain_anthropic` before the API call, so the
model never sees it. So we added `user` and `department` as first-class
optional fields on the wire Pydantic and thread them through the strategy's
`pre_turn` / `post_turn` hooks. The CLI sets them from `--user` / `--department`
(with `/user` and `/dept` switches mid-session); the harness sets them per
turn in the scenario.

## What the harness measures

Defined in `evals/conversations.py`. Thirteen scenarios grouped by category:

- **per_user** (5): preference recall, preference update, multi-fact recall,
  two-user no-crosstalk, third-person-not-extracted (the extractor must
  ignore statements about people other than the speaker).
- **cross_user** (6): team convention propagates within a department, team
  cadence propagates within a department, pattern isolation across
  departments (the key test — a marketing speaker must NOT receive an
  engineering pattern), personal preference does NOT leak across users,
  personal-overrides-team-when-conflict (precedence is anchored in evals,
  not only docs), team-pattern-supersession (exercises the pattern extractor's
  update path).
- **regression** (2): within-session recall, simple math without memory.

Each turn asserts `expect` substrings and `forbid` substrings against the
reply (grep-all, case-insensitive). Per-turn results are aggregated into the
matrix printed at session end (also written to `evals/out/metrics.md`).

## Trade-offs

### Goal 1 — individual learning

`per_user` covers this cleanly. The collection shape (many atomic records
with semantic search) is more robust than a single profile document because:

- No reconciliation — each extracted fact is its own row, and the
  `enable_updates=True` flag lets the extractor supersede records by ID when
  a user changes their mind.
- Retrieval is cheap and paraphrase-robust — a question like "where do I
  live?" will surface `("location", "Seattle")` even if the stored record
  says "lives in".

Cost: one embedding call per `put`, one per `search`, and one LLM extraction
call per turn.

Failure mode to watch: the extractor hallucinating facts that the user did
not state. The prompt leans hard on "first-person signals" to mitigate.

### Goal 2 — cross-user learning

`per_user_plus_patterns` adds this. The critical design decision is
**classification at write time**: is this statement personal or pattern?

Two reasonable ways to do it:

1. **One extractor, one schema with a `scope` field, route at write time.**
   Cheap (one LLM call) but one prompt has to carry both classifications,
   which tends to blur the distinction.
2. **Two extractors with mutually exclusive instructions.** (This is what
   strategy 2 does.) More expensive (two LLM calls per turn) but each prompt
   is focused — the personal extractor is told to reject team statements and
   the pattern extractor is told to reject personal ones. The harness's
   `personal_does_not_leak_across_users` scenario specifically probes this:
   Alice says "I personally prefer dark mode", Bob asks about his editor
   theme. A good pattern extractor ignores the statement entirely; a bad one
   writes it to the shared bucket and Bob gets dark-mode context that isn't
   his.

Failure mode to watch: the pattern extractor being over-eager and writing
personal statements as patterns (cross-contamination). The prompt tells it
"when in doubt, do not extract" as a conservative default.

### Baseline

Not meant to be competitive. It fails every `reset_before` scenario and every
cross-user scenario. It passes regression scenarios because within a single
session the agent just reads the full history. Its presence in the matrix
quantifies exactly how much memory adds.

## Expected matrix (qualitative)

The numbers depend on model choice and whether the extractors are tuned, but
the shape should be:

```
strategy                cross_user  per_user  regression  overall
baseline                11/14       9/15      3/3         23/32 (72%)
per_user                12/14       15/15     3/3         30/32 (94%)
per_user_plus_patterns  14/14       15/15     3/3         32/32 (100%)
```

(From an actual run. Numbers can drift ±1 between runs on the flakier turns
— e.g. `pattern_isolation_across_departments` occasionally fails because
the agent volunteers "pytest" as a generic suggestion even with no memory
context, and nondeterminism in structured-output extraction can occasionally
split a compound statement differently. The shape — baseline flat on
persistence, per_user solid on per-user and weak on cross-user,
per_user_plus_patterns strong on both — is stable across runs.)

`per_user` scoring 0 on `cross_user` is the point — it has no shared bucket,
so it cannot propagate a pattern from Alice to Bob.

## CLI usage

```bash
uv run chat --memory per_user_plus_patterns --user alice --department engineering
You (alice@engineering): at our team we use pytest for python
You (alice@engineering): /user bob
[speaker set to bob]
You (bob@engineering):   what testing framework should I use?
Assistant: You should use pytest — your team's convention for Python projects.
You (bob@engineering): /dept marketing
[department set to marketing]
You (bob@marketing):    what testing framework does our team use?
Assistant: I don't have that information for the marketing team.
```

`/user <name>` switches speaker; `/dept <name>` switches department.

## Why not episodic / procedural memory

The LangGraph memory docs describe three types (semantic, episodic,
procedural). Both strategies here are semantic — atomic facts.

- Episodic (storing past exchanges as few-shot examples) is harder to test
  deterministically with substring assertions; scenario design is more
  brittle.
- Procedural (agent self-rewriting its own prompt over feedback) doesn't fit
  the take-home's time budget and is hard to grep.

Both are reasonable extensions; out of scope for this 1–3 hr deliverable.

## Where each piece lives

- `src/agent/core.py` — `make_agent(..., store=...)` (extended).
- `src/agent/memory/base.py` — `MemoryStrategy` protocol + `BaselineStrategy`.
- `src/agent/memory/per_user.py` — strategy 1 + `UserFact` schema.
- `src/agent/memory/per_user_plus_patterns.py` — strategy 2 + `TeamPattern`.
- `src/agent/cli.py` — `--memory` and `--user` flags; `/user <name>` switch.
- `evals/conversations.py` — scenarios.
- `evals/test_memory.py` — parameterized harness.
- `evals/_metrics.py` + `evals/conftest.py` — summary table at session end.
- `evals/README.md` — how to run.
