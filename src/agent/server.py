import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.memory import STRATEGIES

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Memory strategy is picked at startup via MEMORY_STRATEGY env var and held
# for the lifetime of the process. The strategy's own DictStore lives on
# this single instance, so memory persists across HTTP requests but dies
# with the process (same semantics as the CLI — see docs/memory.md for the
# durability caveat).
STRATEGY_NAME = os.environ.get("MEMORY_STRATEGY", "baseline")
if STRATEGY_NAME not in STRATEGIES:
    raise ValueError(
        f"Unknown MEMORY_STRATEGY={STRATEGY_NAME!r}. "
        f"Choose from {list(STRATEGIES)}."
    )
strategy = STRATEGIES[STRATEGY_NAME]()
agent = strategy.build_agent()


class Message(BaseModel):
    role: str
    content: str
    user: str | None = None
    department: str | None = None


class ChatRequest(BaseModel):
    messages: list[Message]


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.messages:
        return {"reply": ""}

    # The client sends the whole conversation on every request (the server
    # is stateless w.r.t. session). The latest message is the new user turn
    # we're responding to; everything before it is history the agent sees
    # verbatim. We only extract from the new exchange in post_turn so we
    # don't re-extract already-processed turns.
    history = [
        {"role": m.role, "content": m.content} for m in req.messages[:-1]
    ]
    last = req.messages[-1]
    user_msg = {
        "role": last.role,
        "content": last.content,
        "user": last.user,
        "department": last.department,
    }

    invoke_messages = strategy.pre_turn(user_msg, history)
    result = agent.invoke({"messages": invoke_messages})
    ai_content = result["messages"][-1].content
    if isinstance(ai_content, list):
        ai_content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in ai_content
        )

    strategy.post_turn(user_msg, ai_content)
    return {"reply": ai_content}


def main():
    import uvicorn

    uvicorn.run("agent.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
