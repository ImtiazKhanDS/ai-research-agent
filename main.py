import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Research & Social Media Agent")
    parser.add_argument("--verbose", action="store_true", help="Show agent tool calls")
    parser.add_argument(
        "--llm-mode",
        choices=["api", "local"],
        help="Override LLM_MODE: 'api' = Groq, 'local' = Ollama (default: from .env)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Research a specific topic instead of general AI news (e.g. 'mixture of experts', 'AI agents')",
    )
    args = parser.parse_args()

    # Override LLM_MODE before any agent imports read it
    if args.llm_mode:
        os.environ["LLM_MODE"] = args.llm_mode

    _check_env()

    from agents.orchestrator import run
    run(verbose=args.verbose, topic=args.topic)


def _check_env() -> None:
    required = [
        "TAVILY_API_KEY",
        "LINKEDIN_ACCESS_TOKEN",
        "LINKEDIN_PERSON_URN",
        "X_API_KEY",
        "X_API_SECRET",
        "X_ACCESS_TOKEN",
        "X_ACCESS_TOKEN_SECRET",
    ]
    # Only require GROQ_API_KEY when running in api mode
    if os.environ.get("LLM_MODE", "api").lower() == "api":
        required.insert(0, "GROQ_API_KEY")

    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print("Error: Missing required environment variables:\n  " + "\n  ".join(missing))
        print("\nCopy .env.example to .env and fill in the values.")
        sys.exit(1)


if __name__ == "__main__":
    main()
