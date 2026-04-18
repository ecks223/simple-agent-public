# simple-agent

A minimal LLM agent built on [LangChain Deep Agents](https://github.com/langchain-ai/deepagents). Supports OpenAI, Anthropic, and Google models out of the box. Two ways to run it — pick one:

- [CLI guide](docs/cli.md) — interactive terminal chat
- [Fullstack guide](docs/fullstack.md) — FastAPI server + React frontend

---

## Core agent

The agent lives in `src/agent/core.py` and exposes a single factory:

```python
from agent.core import make_agent

agent = make_agent(
    model_str="anthropic:claude-haiku-4-5-20251001",  # provider:model
    system_prompt=None,                                # optional override
)
```

It wraps LangChain's `init_chat_model` + `create_deep_agent` and returns a compiled LangGraph agent that supports `.invoke()`, `.stream()`, and `.astream()`.

## Supported providers

| Provider  | Model string example                            | Required env var    |
|-----------|-------------------------------------------------|---------------------|
| Anthropic | `anthropic:claude-haiku-4-5-20251001` (default) | `ANTHROPIC_API_KEY` |
| OpenAI    | `openai:gpt-4o`                                 | `OPENAI_API_KEY`    |
| Google    | `google_genai:gemini-2.5-flash`                 | `GOOGLE_API_KEY`    |

Any model supported by LangChain's [`init_chat_model`](https://python.langchain.com/docs/how_to/chat_models_universal_init/) works — just pass the `provider:model` string.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- At least one LLM provider API key

## Initial setup

```bash
git clone https://github.com/valkai-tech/simple-agent-public.git
cd simple-agent-public
uv sync
cp .env.example .env
# Fill in your API key(s) in .env
```

## Running evals

```bash
uv run pytest evals/ -v
```

Evals make real LLM calls (not mocked) to verify provider integration end-to-end.

## Project structure

```
simple-agent/
├── README.md               # this file — core concepts
├── docs/
│   ├── cli.md              # CLI usage guide
│   └── fullstack.md        # server + frontend guide
├── pyproject.toml          # uv project config and dependencies
├── .env.example            # API key template
├── src/
│   └── agent/
│       ├── core.py         # agent factory (shared by both approaches)
│       ├── cli.py          # CLI entry point
│       └── server.py       # FastAPI server entry point
├── frontend/               # React chat UI
└── evals/
    └── test_agent.py       # pytest evals
```

---

## Memory implementation

See also [docs/memory.md](docs/memory.md) for the strategy design + trade-offs and [evals/README.md](evals/README.md) for the harness walkthrough.

### Why not `langmem` + `InMemoryStore`

The first attempt followed the LangGraph memory docs exactly: [`langmem.create_memory_manager`](https://langchain-ai.github.io/langmem/) for extraction + `langgraph.store.memory.InMemoryStore` with an OpenAI embedding index for semantic retrieval. It worked, but required a second provider credential (`OPENAI_API_KEY`) that wasnt provided, and Anthropic doesn't offer a first-party embedding model (their sanctioned partner, Voyage AI, would also have needed a separate key).

Rather than add a second provider just to store a handful of facts per user, we swapped to:

- A local dict-backed store in [src/agent/memory/_store.py](src/agent/memory/_store.py) — a ~40-line `DictStore` with the same `put` / `get` / `list` surface as `BaseStore`.
- Extraction via direct `init_chat_model(model_str).with_structured_output(...)` calls against Claude — a ~15-line replacement for the LangMem wrapper.

Semantic search falls away, but at this scale (single-digit facts per user) the LLM does the relevance filtering on its own: every turn injects all of the current speaker's stored facts into the system prompt as a small labeled block, and the model picks which ones to use. Swapping back to the embedding-indexed store is a one-file change if production scale ever needs it.

### Tests-first workflow

To make iteration on the memory layer cheap, we wrote the evaluation harness **before** the strategies. [evals/test_memory.py](evals/test_memory.py) is a parameterized pytest suite driven by [evals/conversations.py](evals/conversations.py) — scripted multi-turn scenarios, each turn carrying `expect` / `forbid` substring assertions. The harness accumulates per-turn pass/fail into a matrix keyed by `(strategy, category)` and prints a summary table at session end.

That let us run the whole suite as a fast feedback loop while building: edit extractor prompt → re-run → see which scenarios moved on the matrix. The baseline (no memory) and both strategies all go through the same harness, so the matrix directly quantifies what memory adds.

Turn-level checks also meant that every scenario produces N data points, not one — a scenario with a bad prompt isn't a binary red/green, it's N-of-M, which is a much better signal for prompt tuning.

### Wire schema: `user` and `department`

The starter repo's `Message` Pydantic in [src/agent/server.py](src/agent/server.py) carried only `{role, content}`. For long-term memory to scope correctly, we extended it with two optional identity fields:

```python
class Message(BaseModel):
    role: str
    content: str
    user: str | None = None          # speaker identifier  → personal namespace
    department: str | None = None    # speaker's team      → pattern namespace
```

These are treated as **authenticated user metadata** — in a real deployment they'd be populated from whatever auth layer sits in front of the server (SSO claim, JWT, session lookup, directory service), not from anything the end user types. The client never trusts them; the server writes them on the incoming request before the agent sees it. For this take-home we stop short of the auth layer itself and just thread the fields through: the CLI exposes `--user` / `--department` flags plus `/user <name>` and `/dept <name>` switches for multi-user demos, and the harness sets them per-turn in each scenario.

Note that LangChain's message-level `role` field (`"user"` / `"assistant"` / `"system"`) is unrelated — that's the protocol role. The `user` field we added is the speaker identity. We briefly considered overloading `role` with values like `"user:alice"`, but LangChain's `convert_to_messages` rejects non-standard roles, so a new wire field was the cleanest option.

### `MemoryStrategy` interface

Every strategy exposes the same three hooks ([src/agent/memory/base.py](src/agent/memory/base.py)):

```python
class MemoryStrategy(Protocol):
    name: str
    def build_agent(self): ...
    def pre_turn(self, user_msg: dict, messages: list[dict]) -> list[dict]: ...
    def post_turn(self, user_msg: dict, ai_content: str) -> None: ...
```

- `build_agent()` — returns a compiled LangGraph agent. Baseline and memory strategies both use [src/agent/core.py::make_agent](src/agent/core.py) under the hood.
- `pre_turn(user_msg, messages)` — called with the incoming user turn and the running conversation history. Returns the message list to pass to `.invoke()`. This is the **retrieval** hook.
- `post_turn(user_msg, ai_content)` — called with the exchange that just happened. Writes any extracted facts to the store. This is the **extraction** hook.

The CLI and the harness are both just loops that call these three methods, so any `MemoryStrategy` plugs in without touching either.

### Extraction (post-turn): what, when, how

Runs once per assistant reply, out-of-band from the model-facing turn. Each strategy holds a Claude client wrapped in `with_structured_output(ExtractionSchema)`, where the schema is a Pydantic model like:

```python
class _ExtractedUserFact(UserFact):
    update_id: str | None = None    # set to existing id to overwrite in place

class _UserFactExtraction(BaseModel):
    extracted: list[_ExtractedUserFact] = []
```

`post_turn` renders the existing stored facts into the extractor's system prompt (so the model can decide whether to insert new or update an existing one), plus the just-completed exchange, then calls the structured-output model. Returned records are written to the dict store under the namespace that matches the strategy:

- `per_user`: one extractor for personal facts → `("facts", <user_id>)`
- `per_user_plus_patterns`: two extractors with mutually-exclusive prompts (first-person vs. team-statement) → `("facts", <user_id>)` and `("patterns", <department>)`

The extractor prompts are in [src/agent/memory/per_user.py](src/agent/memory/per_user.py) and [src/agent/memory/per_user_plus_patterns.py](src/agent/memory/per_user_plus_patterns.py). They explicitly include examples — e.g. *"I prefer Rust for systems programming" → {subject: "programming language", value: "Rust (for systems programming)"}* — because we found the extractor would otherwise miss specific nouns when they were embedded in compound sentences.

### Retrieval (pre-turn): what, when, how

Runs at the start of every user turn. The strategy uses the `user` field (and for patterns, the `department` field) on the incoming message to scope reads to the right namespace, then enumerates that namespace's items via `store.list(...)` and formats them into a short labeled block:

```
[Known facts about the current user (alice)]
- programming language: Rust (for systems programming)
- location: Seattle
- primary language: Python
```

That block is prepended as a single `{"role": "system", "content": ...}` message and the result is passed to `.invoke()`. For `per_user_plus_patterns`, two blocks are concatenated — personal first, then team patterns — and the block ends with an explicit precedence instruction so the model prefers personal facts over team patterns when they conflict.

No vector search, no embedding API call, no top-K filtering. The LLM reads the question, reads the facts block, and selects relevance itself. This is adequate because the block stays small (a handful of records per user); at larger scale you'd want a real retriever in front of this.

### Running the memory evals

The full memory-strategy comparison (3 strategies × 13 scenarios = 39 tests, ~4 min wall clock):

```bash
uv run pytest evals/test_memory.py -v
```

> Hits real LLM APIs. Roughly tens of cents of Haiku usage per full run. Only `ANTHROPIC_API_KEY` is required.

Just one strategy across all scenarios:

```bash
uv run pytest evals/test_memory.py -v -k "per_user_plus_patterns"
```

Just one scenario across all strategies:

```bash
uv run pytest evals/test_memory.py -v -k "team_convention_propagates"
```

At the end of any `test_memory.py` run, the session summary prints a matrix and writes it to `evals/out/metrics.md`. Example from an actual run:

```
strategy                cross_user  per_user  regression  overall
baseline                11/14       9/15      3/3         23/32 (72%)
per_user                12/14       15/15     3/3         30/32 (94%)
per_user_plus_patterns  14/14       15/15     3/3         32/32 (100%)
```

The shape is what the harness is designed to expose: baseline fails every cross-session scenario (no persistence), `per_user` solves personal-fact persistence but has no shared bucket so it can't propagate team patterns, `per_user_plus_patterns` covers both.
