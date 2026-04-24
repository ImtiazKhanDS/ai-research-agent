import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from agents.research import run_research_agent
from agents.report import run_report_agent
from agents.poster import post_linkedin, post_tweet

DIVIDER = "=" * 60


def run(verbose: bool = False, topic: str | None = None) -> None:
    run_dir = _make_run_dir()

    from tools.llm import current_provider
    print(f"\n  LLM provider: {current_provider()}")
    if topic:
        print(f"  Topic mode: {topic}")

    # Step 1: Research
    print(f"\n{DIVIDER}")
    step1_label = f"STEP 1/3 — Researching topic: {topic}" if topic else "STEP 1/3 — Researching AI developments (last 24h)..."
    print(f"  {step1_label}")
    print(DIVIDER)
    research_brief = run_research_agent(verbose=verbose, topic=topic)
    _save(run_dir / "research.md", research_brief)
    print(f"  Research saved → {run_dir / 'research.md'}")

    # Step 2: Generate reports
    print(f"\n{DIVIDER}")
    print("  STEP 2/3 — Generating LinkedIn post and X tweet...")
    print(DIVIDER)
    linkedin_post, tweet = run_report_agent(research_brief, verbose=verbose, topic=topic)
    _save(run_dir / "linkedin.md", linkedin_post)
    _save(run_dir / "x.md", tweet)
    print(f"  Reports saved → {run_dir}/")

    # Step 3: Approval flow
    print(f"\n{DIVIDER}")
    print("  STEP 3/3 — Review and approve")
    print(DIVIDER)

    _approval_flow(
        platform="LINKEDIN",
        content_path=run_dir / "linkedin.md",
        research_brief=research_brief,
        post_fn=post_linkedin,
        verbose=verbose,
        topic=topic,
    )

    _x_manual_flow(
        content_path=run_dir / "x.md",
        research_brief=research_brief,
        verbose=verbose,
        topic=topic,
    )

    print(f"\nDone. All reports saved in: {run_dir}\n")


def _approval_flow(
    platform: str,
    content_path: Path,
    research_brief: str,
    post_fn,
    verbose: bool,
    topic: str | None = None,
) -> None:
    while True:
        content = content_path.read_text()

        print(f"\n{DIVIDER}")
        print(f"  {platform} DRAFT")
        print(DIVIDER)
        print(content)
        print(f"\n{'-' * 60}")
        print(f"Saved to: {content_path}")
        if platform.startswith("X"):
            print(f"Character count: {len(content)}/280")
        elif platform.startswith("LINKEDIN"):
            print(f"Character count: {len(content)}/4000 (target: 3500–3999)")
        print()

        choice = input("[A]pprove  [R]eject  [G]enerate again  [E]dit  > ").strip().upper()

        if choice == "A":
            print(f"  Posting to {platform}...")
            try:
                url = post_fn(content)
                print(f"  Posted! {url}")
            except Exception as e:
                print(f"  Post failed: {e}")
            return

        elif choice == "R":
            print(f"  Skipped {platform}.")
            return

        elif choice == "G":
            print(f"  Regenerating {platform} report...")
            from agents.report import run_report_agent
            if platform.startswith("LINKEDIN"):
                new_linkedin, _ = run_report_agent(research_brief, verbose=verbose, topic=topic)
                _save(content_path, new_linkedin)
            else:
                _, new_tweet = run_report_agent(research_brief, verbose=verbose, topic=topic)
                _save(content_path, new_tweet)

        elif choice == "E":
            _open_in_editor(content_path)

        else:
            print("  Invalid choice. Enter A, R, G, or E.")


def _x_manual_flow(content_path: Path, research_brief: str, verbose: bool, topic: str | None = None) -> None:
    while True:
        content = content_path.read_text()

        print(f"\n{DIVIDER}")
        print("  X (TWEET) DRAFT")
        print(DIVIDER)
        print(content)
        print(f"\n{'-' * 60}")
        print(f"Saved to: {content_path}")
        print(f"Character count: {len(content)}/280")
        print("Note: X API posting requires a paid plan. Copy the tweet above to post manually.")
        print()

        choice = input("[A]cknowledge  [G]enerate again  [E]dit  > ").strip().upper()

        if choice == "A":
            print("  Tweet saved. Post it manually at x.com.")
            return
        elif choice == "G":
            print("  Regenerating tweet...")
            from agents.report import run_report_agent
            _, new_tweet = run_report_agent(research_brief, verbose=verbose, topic=topic)
            _save(content_path, new_tweet)
        elif choice == "E":
            _open_in_editor(content_path)
        else:
            print("  Invalid choice. Enter A, G, or E.")


def _open_in_editor(path: Path) -> None:
    editor = os.environ.get("EDITOR", "nano")
    subprocess.call([editor, str(path)])


def _save(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _make_run_dir() -> Path:
    date_dir = Path("reports") / datetime.now().strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(date_dir.glob("run-*"))
    next_num = len(existing) + 1
    run_dir = date_dir / f"run-{next_num}"
    run_dir.mkdir()
    return run_dir
