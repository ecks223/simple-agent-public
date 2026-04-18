# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Python tooling is `uv` (Python 3.13+). API keys live in `.env` (copy from `.env.example`).

```bash
uv sync                                          # install deps
uv run chat                                      # CLI chat (default: Anthropic Haiku)
uv run chat --model openai:gpt-4o                # override model (provider:model)
uv run chat --system "..."                       # override system prompt
uv run serve                                     # FastAPI on :8000 (reload enabled)
uv run pytest evals/ -v                          # run all evals
uv run pytest evals/test_agent.py::test_agent_responds -v   # single eval
```

Frontend (Node 18+, Vite dev server on :3000):

```bash
cd frontend && npm install && npm run dev        # dev
npm run build                                    # production build
```

**Evals hit real LLM APIs — they are not mocked.** They require a valid `ANTHROPIC_API_KEY` (the default model is `anthropic:claude-haiku-4-5-20251001`) and cost tokens to run.

## Architecture

The whole agent is a single factory, [src/agent/core.py](src/agent/core.py), wrapping `langchain.chat_models.init_chat_model` + `deepagents.create_deep_agent`. Everything else is a thin entry point around it:

- [src/agent/cli.py](src/agent/cli.py) — argparse + REPL loop. Maintains `messages` locally and feeds the returned `result["messages"]` back in on each turn (so tool/intermediate messages accumulate).
- [src/agent/server.py](src/agent/server.py) — FastAPI with permissive CORS, one `POST /chat` endpoint. The agent is instantiated **once at startup with the default model** — there is no per-request model override and no session storage; clients send the full history every request.
- [frontend/src/App.jsx](frontend/src/App.jsx) — React client hardcoded to `http://localhost:8000`, holds the conversation array in component state, POSTs the whole array on every send.
- [evals/test_agent.py](evals/test_agent.py) — pytest-based integration tests that make real LLM calls to verify provider wiring end-to-end.

Model selection is a string: `"<provider>:<model>"` (e.g. `anthropic:claude-haiku-4-5-20251001`, `openai:gpt-4o`, `google_genai:gemini-2.5-flash`). Anything supported by LangChain's `init_chat_model` works; no code change needed to switch providers, only the env var for the matching key.

`pyproject.toml` declares the internal project name as `take-home` but the installed Python package is `agent` (see `[tool.hatch.build.targets.wheel]`). Console scripts `chat` and `serve` are defined there.
