import os
from tools.llm import complete

LINKEDIN_CHAR_MIN = 3500
LINKEDIN_CHAR_LIMIT = 4000
_MAX_ATTEMPTS = 3

LINKEDIN_PROMPT = """You are a senior AI practitioner writing a daily LinkedIn post for a technical audience.

Today's research brief identifies what is gaining the most attention in AI right now. Your job is to pick the 1-3 most significant trending items and write a post that goes BEYOND summarising them — apply genuine technical thought leadership.

What this means:
- Explain WHY these specific developments matter technically, not just that they happened
- Connect the dots: how does this fit into broader trends, architectural shifts, or competitive dynamics?
- Share a concrete technical perspective or opinion — agree, push back, or add nuance that others aren't saying
- If there's hype, cut through it with specifics (benchmarks, architectural choices, real-world implications)
- If it's genuinely significant, explain the technical mechanism that makes it so

Format requirements:
- Length: STRICTLY between {char_min} and {char_limit} characters. Count carefully before finishing.
- Open with a hook that names the specific development, not a generic "AI is moving fast" opener
- No Markdown headers (##, ###) — LinkedIn renders them as literal text
- Use short paragraphs and line breaks for readability
- End with 3–5 hashtags on their own line
- Write in first person as a senior AI/ML professional

Research Brief:
{{research_brief}}

Write only the LinkedIn post, nothing else."""

TOPIC_LINKEDIN_PROMPT = """You are a senior AI practitioner writing a LinkedIn article on a specific topic.

Topic: "{topic}"

Using the research brief below, write a comprehensive thought leadership article on this topic for a technical LinkedIn audience.

What this means:
- Go deep on the technical substance — explain mechanisms, architectures, tradeoffs
- Share a clear point of view: what do you believe about where this topic is heading and why?
- Reference specific papers, models, benchmarks, or results from the research brief
- Address the open problems and what they mean practically
- Make it educational AND opinionated — not just a summary

Format requirements:
- Length: STRICTLY between {char_min} and {char_limit} characters. Count carefully before finishing.
- No Markdown headers (##, ###) — LinkedIn renders them as literal text
- Use short paragraphs and line breaks for readability
- End with 3–5 relevant hashtags on their own line
- Write in first person as a senior AI/ML professional

Research Brief:
{{research_brief}}

Write only the LinkedIn article, nothing else."""

TWEET_PROMPT = """You are a sharp tech communicator writing for X (formerly Twitter).

Based on the research brief below, write a single tweet.

Requirements:
- Maximum 280 characters (HARD LIMIT — count carefully)
- Punchy, insight-driven, sparks curiosity or conversation
- Include 1–2 hashtags
- Optionally reference the single most important story with a short URL if available
- No thread, just one tweet

Research Brief:
{research_brief}

Write only the tweet text, nothing else. Double-check it is under 280 characters."""


def run_report_agent(research_brief: str, verbose: bool = False, topic: str | None = None) -> tuple[str, str]:
    if verbose:
        print("  [report] generating LinkedIn post...")

    if topic:
        linkedin_prompt = TOPIC_LINKEDIN_PROMPT.format(
            topic=topic, char_min=LINKEDIN_CHAR_MIN, char_limit=LINKEDIN_CHAR_LIMIT
        ).replace("{research_brief}", research_brief)
    else:
        linkedin_prompt = LINKEDIN_PROMPT.format(
            char_min=LINKEDIN_CHAR_MIN, char_limit=LINKEDIN_CHAR_LIMIT
        ).replace("{research_brief}", research_brief)

    linkedin_post = _generate_linkedin(linkedin_prompt, verbose)

    if verbose:
        print(f"  [report] LinkedIn final: {len(linkedin_post)} chars")
        print("  [report] generating X tweet...")

    tweet = _generate(TWEET_PROMPT.format(research_brief=research_brief))
    if len(tweet) > 280:
        tweet = _truncate_tweet(tweet)

    return linkedin_post, tweet


def _generate_linkedin(prompt: str, verbose: bool) -> str:
    """
    Retry loop: ask model to overshoot, then hard-truncate to land in [3500, 4000].
    Hard-truncate is deterministic — guarantees ≤ 4000.
    Retries handle the rare case where the model under-generates despite the prompt.
    """
    post = ""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        post = _generate(prompt)
        char_count = len(post)

        if verbose:
            print(f"  [report] LinkedIn attempt {attempt}: {char_count} chars")

        if char_count > LINKEDIN_CHAR_LIMIT:
            return _hard_truncate(post)

        if char_count >= LINKEDIN_CHAR_MIN:
            return post

        # Under minimum — retry, but warn the model it was too short
        prompt = (
            f"Your previous response was only {char_count} characters. "
            f"You MUST write between {LINKEDIN_CHAR_MIN} and {LINKEDIN_CHAR_LIMIT} characters. "
            f"Be much more detailed and thorough.\n\n"
        ) + prompt

    # All retries exhausted and still short — hard-truncate whatever we have
    # (if post is somehow longer than limit) or return as-is with a warning
    if len(post) > LINKEDIN_CHAR_LIMIT:
        return _hard_truncate(post)

    if verbose:
        print(f"  [report] Warning: could not reach {LINKEDIN_CHAR_MIN} chars after {_MAX_ATTEMPTS} attempts. Returning {len(post)} chars.")
    return post


def _generate(prompt: str) -> str:
    response = complete(messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip()


def _hard_truncate(post: str) -> str:
    """Deterministic cut at the last sentence boundary before LINKEDIN_CHAR_LIMIT, preserving hashtags."""
    lines = post.rstrip().splitlines()
    hashtag_lines = []
    for line in reversed(lines):
        if line.strip().startswith("#") or (hashtag_lines and line.strip() == ""):
            hashtag_lines.insert(0, line)
        else:
            break
    body_lines = lines[: len(lines) - len(hashtag_lines)]
    hashtags = "\n".join(hashtag_lines).strip()
    body = "\n".join(body_lines)

    reserve = len(hashtags) + 2
    budget = LINKEDIN_CHAR_LIMIT - reserve

    truncated = body[:budget]
    last_stop = max(truncated.rfind(". "), truncated.rfind(".\n"))
    if last_stop != -1:
        truncated = truncated[: last_stop + 1]

    return f"{truncated.rstrip()}\n\n{hashtags}" if hashtags else truncated.rstrip()


def _truncate_tweet(tweet: str) -> str:
    prompt = f"""This tweet is too long ({len(tweet)} characters). Rewrite it to be under 280 characters while keeping the key insight and hashtags:

{tweet}

Write only the shortened tweet, nothing else."""
    return _generate(prompt)
