import argparse

from dotenv import load_dotenv

from agent.memory import STRATEGIES


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="CLI Chat Agent")
    parser.add_argument(
        "--model",
        default="anthropic:claude-haiku-4-5-20251001",
        help=(
            "Model string, e.g. openai:gpt-4o, "
            "anthropic:claude-haiku-4-5-20251001, google_genai:gemini-2.5-flash"
        ),
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Custom system prompt override.",
    )
    parser.add_argument(
        "--memory",
        default="baseline",
        choices=list(STRATEGIES),
        help=(
            "Memory strategy. 'baseline' = no long-term memory (original "
            "behavior). 'per_user' = per-user semantic collection. "
            "'per_user_plus_patterns' = per-user facts plus shared team "
            "patterns scoped by department."
        ),
    )
    parser.add_argument(
        "--user",
        default="cli_user",
        help=(
            "Speaker identifier used for scoping personal memory. Type "
            "'/user <name>' at the prompt to switch speaker mid-session."
        ),
    )
    parser.add_argument(
        "--department",
        default="default",
        help=(
            "Speaker's department, used for scoping team patterns. Team "
            "patterns are shared within a department, not across. Type "
            "'/dept <name>' to switch."
        ),
    )
    args = parser.parse_args()

    strategy = STRATEGIES[args.memory](
        model_str=args.model, system_prompt=args.system
    )
    agent = strategy.build_agent()
    messages: list[dict] = []
    current_user = args.user
    current_dept = args.department

    print(
        f"Chat started (model={args.model}, memory={args.memory}, "
        f"user={current_user}, department={current_dept}). "
        "Type 'quit' to exit. "
        "Type '/user <name>' or '/dept <name>' to switch.\n"
    )

    while True:
        try:
            user_input = input(f"You ({current_user}@{current_dept}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.startswith("/user "):
            current_user = user_input[len("/user ") :].strip() or current_user
            print(f"[speaker set to {current_user}]\n")
            continue
        if user_input.startswith("/dept "):
            current_dept = user_input[len("/dept ") :].strip() or current_dept
            print(f"[department set to {current_dept}]\n")
            continue

        user_msg = {
            "role": "user",
            "content": user_input,
            "user": current_user,
            "department": current_dept,
        }
        invoke_messages = strategy.pre_turn(user_msg, messages)
        result = agent.invoke({"messages": invoke_messages})
        ai_msg = result["messages"][-1]
        ai_content = ai_msg.content
        if isinstance(ai_content, list):
            ai_content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in ai_content
            )
        print(f"\nAssistant: {ai_content}\n")

        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": ai_content})
        strategy.post_turn(user_msg, ai_content)


if __name__ == "__main__":
    main()
