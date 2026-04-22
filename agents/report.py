import os
from tools.llm import complete

LINKEDIN_CHAR_LIMIT = 4000

LINKEDIN_PROMPT = """You are a senior AI practitioner writing a daily LinkedIn post for a technical audience.

Today's research brief identifies what is gaining the most attention in AI right now. Your job is to pick the 1-3 most significant trending items and write a post that goes BEYOND summarising them — apply genuine technical thought leadership.

What this means:
- Explain WHY these specific developments matter technically, not just that they happened
- Connect the dots: how does this fit into broader trends, architectural shifts, or competitive dynamics?
- Share a concrete technical perspective or opinion — agree, push back, or add nuance that others aren't saying
- If there's hype, cut through it with specifics (benchmarks, architectural choices, real-world implications)
- If it's genuinely significant, explain the technical mechanism that makes it so

Format requirements:
- HARD LIMIT: {char_limit} characters total (count carefully — stay under)
- Open with a hook that names the specific development, not a generic "AI is moving fast" opener
- No Markdown headers (##, ###) — LinkedIn renders them as literal text
- Use short paragraphs and line breaks for readability
- End with 3–5 hashtags on their own line
- Write in first person as a senior AI/ML professional

Research Brief:
{{research_brief}}

Write only the LinkedIn post, nothing else."""

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


def run_report_agent(research_brief: str, verbose: bool = False) -> tuple[str, str]:
    if verbose:
        print("  [report] generating LinkedIn post...")
    linkedin_prompt = LINKEDIN_PROMPT.format(char_limit=LINKEDIN_CHAR_LIMIT).replace(
        "{research_brief}", research_brief
    )
    linkedin_post = _generate(linkedin_prompt)
    if len(linkedin_post) > LINKEDIN_CHAR_LIMIT:
        linkedin_post = _truncate_linkedin(linkedin_post)

    if verbose:
        print("  [report] generating X tweet...")
    tweet = _generate(TWEET_PROMPT.format(research_brief=research_brief))
    if len(tweet) > 280:
        tweet = _truncate_tweet(tweet)

    return linkedin_post, tweet


def _generate(prompt: str) -> str:
    response = complete(messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip()


def _truncate_linkedin(post: str) -> str:
    prompt = f"""This LinkedIn post is too long ({len(post)} characters). Rewrite it to be under {LINKEDIN_CHAR_LIMIT} characters while preserving the key insights, structure, and hashtags:

{post}

Write only the shortened post, nothing else."""
    return _generate(prompt)


def _truncate_tweet(tweet: str) -> str:
    prompt = f"""This tweet is too long ({len(tweet)} characters). Rewrite it to be under 280 characters while keeping the key insight and hashtags:

{tweet}

Write only the shortened tweet, nothing else."""
    return _generate(prompt)
