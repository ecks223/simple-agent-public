# Fullstack guide

Run the agent as a FastAPI server with a React chat frontend. See the [core README](../README.md) for initial setup.

## Prerequisites

- Everything in the core README
- Node.js 18+

## Start

Run both processes in separate terminals.

**Terminal 1 — backend:**

```bash
uv run serve                                    # baseline (no memory)
# or, to run the server with memory:
MEMORY_STRATEGY=per_user_plus_patterns uv run serve
```

Server starts at `http://localhost:8000`. The `MEMORY_STRATEGY` env var picks the strategy (`baseline`, `per_user`, `per_user_plus_patterns`) and is read once at startup.

**Terminal 2 — frontend:**

```bash
cd frontend
npm install   # first time only
npm run dev
```

UI opens at `http://localhost:3000`.

## API

```
POST /chat
Content-Type: application/json

{
  "messages": [
    {
      "role": "user",
      "content": "Hello!",
      "user": "alice",          // optional — scopes per-user memory
      "department": "engineering"  // optional — scopes team-pattern memory
    }
  ]
}
```

```json
{
  "reply": "Hi! How can I help you?"
}
```

The frontend sends the full conversation history on each request. The server is stateless at the session level — it doesn't track per-conversation state — but the selected memory strategy holds a process-wide store, so facts extracted from one request persist to the next.

## How it works

[`src/agent/server.py`](../src/agent/server.py) reads `MEMORY_STRATEGY` at startup, builds the chosen `MemoryStrategy` instance, and holds it for the lifetime of the process. Each `/chat` request routes through the same three-hook pattern used by the CLI and the harness:

1. `pre_turn(user_msg, history)` — retrieval. Returns the message list to invoke.
2. `agent.invoke(...)` — model call.
3. `post_turn(user_msg, ai_content)` — extraction. Writes new/updated facts to the store.

The React frontend (`frontend/src/App.jsx`) manages conversation state locally, posts the full message list on every send, and exposes editable `user` / `department` inputs so you can switch speakers (and their department) to demo cross-user / cross-department memory behavior from the UI. The "New conversation" button clears the local transcript without touching the server's store — useful for simulating a fresh conversation against a remembered user.

## Relevant files

```
src/agent/
├── core.py       # make_agent(...) factory (shared)
├── server.py     # FastAPI app, POST /chat, memory-strategy routing
└── memory/       # MemoryStrategy implementations (see docs/memory.md)

frontend/
├── src/
│   ├── App.jsx   # chat UI — user/dept inputs, "New conversation" button
│   └── main.jsx  # React entry point
├── index.html
├── vite.config.js
└── package.json
```
