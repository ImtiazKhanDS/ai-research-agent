import json
import os
from datetime import datetime, timedelta
from tools.llm import complete
from tools.web_search import web_search_tool, run_web_search
from tools.arxiv_search import arxiv_search_tool, run_arxiv_search

TOOLS = [web_search_tool, arxiv_search_tool]
MAX_TOOL_RESULT_CHARS = 1500  # keeps each tool result ~375 tokens

TOOL_HANDLERS = {
    "web_search": run_web_search,
    "arxiv_search": run_arxiv_search,
}

TOPIC_RESEARCH_PROMPT = """You are an AI research analyst. Today's date is {today} ({weekday}).

The user wants a deep-dive research brief on this specific topic: "{topic}"

Search comprehensively across all the sources below — prioritise quality and depth over breadth. Do NOT limit to last 24 hours; include the best recent content available on this topic.

Sources to prioritise:
- Research & Papers: arXiv cs.AI / cs.LG / cs.CL, Papers With Code, Semantic Scholar
- Newsletters & Blogs: The Rundown AI, The Batch, Lilian Weng's Blog, The AI Edge, Hugging Face Blog
- News: TechCrunch AI, VentureBeat AI
- Company blogs: Anthropic, OpenAI, Google DeepMind, Meta AI
- Community: r/MachineLearning, Hacker News

Run ALL of the following searches:
1. "{topic} latest research 2025 OR 2026"
2. "{topic} site:arxiv.org"
3. "{topic} site:huggingface.co"
4. "{topic} site:paperswithcode.com"
5. "{topic} site:venturebeat.com OR site:techcrunch.com"
6. "{topic} Hacker News OR r/MachineLearning"
7. "{topic} OpenAI OR Anthropic OR Google DeepMind OR Meta AI"
8. "{topic} benchmark OR evaluation OR state of the art"

arXiv searches:
9. arxiv_search: "{topic}"
10. arxiv_search: "{topic} survey"

After gathering results, produce a structured research brief:

# Topic Research Brief: {topic} — {today}

## What Is It / Current State
[Clear explanation of the topic and where it stands today technically]

## Key Developments & Papers
For each item (aim for 5):
### [Title]
- **Why it matters**: [technical significance]
- **Key finding / contribution**: [specifics — numbers, methods, claims]
- **Source/URL**: [url]

## Open Problems & Debates
[2-3 bullet points on what's unsolved, contested, or actively being worked on]

## Signal: Where This Is Heading
[2-3 bullet points on the technical trajectory — what the research suggests about the next 6-12 months]

Be specific. Name models, papers, authors, benchmarks, and numbers."""

RESEARCH_PROMPT = """You are an AI research analyst. Today's date is {today} ({weekday}).

Your job is to identify what is gaining the MOST attention in AI right now — the stories, releases, and papers from the LAST 24 HOURS (since {yesterday}) that people are actively discussing, sharing, and reacting to.

Prioritise content from these sources when searching:
- Research & Papers: arXiv cs.AI / cs.LG / cs.CL, Papers With Code, Semantic Scholar
- Newsletters & Blogs: The Rundown AI, The Batch, Lilian Weng's Blog, The AI Edge, Hugging Face Blog
- News: AI News, TechCrunch AI, VentureBeat AI
- Company blogs: Anthropic, OpenAI, Google DeepMind, Meta AI
- Community: r/MachineLearning, Hacker News (AI topics), Hugging Face Discord

Use the available tools to run ALL of the following searches — do not skip any:

Web searches (run each one):
1. "OpenAI announcement OR release {today} OR {yesterday}"
2. "Anthropic announcement OR release {today} OR {yesterday}"
3. "Google DeepMind announcement OR release {today} OR {yesterday}"
4. "Meta AI announcement OR release {today} OR {yesterday}"
5. "site:techcrunch.com/category/artificial-intelligence AI {today}"
6. "site:venturebeat.com AI {today} OR {yesterday}"
7. "site:huggingface.co blog {today} OR {yesterday}"
8. "site:paperswithcode.com {today} OR {yesterday} new method"
9. "The Rundown AI {today}" OR "The Batch AI {today}"
10. "site:news.ycombinator.com AI OR LLM OR machine learning {today}"
11. "r/MachineLearning discussion {today} trending AI"
12. "Semantic Scholar AI papers trending {today}"

arXiv searches (run each one):
13. arxiv_search: "large language models reasoning"
14. arxiv_search: "multimodal vision language"
15. arxiv_search: "AI agents reinforcement learning"
16. arxiv_search: "diffusion generative model"

For each item you find, assess its attention level — is it going viral, being widely cited, sparking debate, or just a routine update?

Produce a research brief in this exact Markdown format:

# AI Research Brief — {today}

## Trending Now (Highest Attention)
The 2-3 items gaining the most traction right now. These are the ones people are actively talking about.
### [Item Title]
- **Why it's trending**: [what's driving attention — controversy, surprise, significance, community reaction]
- **What happened**: [concise factual summary]
- **Technical depth**: [what makes this technically interesting or significant]
- **Source/URL**: [url]

## Other Notable Developments
Additional stories and papers worth knowing about (aim for 3-4 items).
### [Item Title]
- **Summary**: [2-3 sentences]
- **Source/URL**: [url]

## Signal vs Noise
[2-3 bullet points: what today's developments actually signal about where AI is heading — cut through hype to the real technical and industry implications]

Be specific. Avoid vague summaries — name the models, the numbers, the claims, the researchers."""


def run_research_agent(verbose: bool = False, topic: str | None = None) -> str:
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    weekday = now.strftime("%A")

    if topic:
        prompt = TOPIC_RESEARCH_PROMPT.format(topic=topic, today=today, weekday=weekday)
    else:
        prompt = RESEARCH_PROMPT.format(today=today, yesterday=yesterday, weekday=weekday)
    messages = [{"role": "user", "content": prompt}]
    tools_were_called = False

    while True:
        response = complete(messages, tools=TOOLS)
        choice = response.choices[0]

        if verbose:
            print(f"  [research] finish_reason={choice.finish_reason}")

        if choice.finish_reason == "stop":
            # Local models often skip tool calls entirely — detect and fall back
            if not tools_were_called:
                if verbose:
                    print("  [research] no tool calls made — running searches manually (local model fallback)")
                raw_results = _run_all_searches(today, yesterday, verbose)
                return _synthesise(raw_results, today, yesterday, weekday)
            return choice.message.content

        # Handle tool calls
        tool_calls = choice.message.tool_calls or []
        if tool_calls:
            tools_were_called = True
        messages.append(choice.message)

        for tc in tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            if verbose:
                print(f"  [research] calling tool: {fn_name}({fn_args})")

            handler = TOOL_HANDLERS.get(fn_name)
            if handler is None:
                result = f"Unknown tool: {fn_name}"
            else:
                try:
                    result = handler(**fn_args)
                except Exception as e:
                    result = f"Tool error: {e}"

            # Truncate to keep context size manageable across many tool calls
            if len(result) > MAX_TOOL_RESULT_CHARS:
                result = result[:MAX_TOOL_RESULT_CHARS] + "\n...[truncated]"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })


# --- Local model fallback: run all searches manually then synthesise ---

_MANUAL_SEARCHES = [
    ("web_search", {"query": "OpenAI announcement OR release {today} OR {yesterday}"}),
    ("web_search", {"query": "Anthropic announcement OR release {today} OR {yesterday}"}),
    ("web_search", {"query": "Google DeepMind announcement OR release {today} OR {yesterday}"}),
    ("web_search", {"query": "Meta AI announcement OR release {today} OR {yesterday}"}),
    ("web_search", {"query": "site:techcrunch.com artificial-intelligence {today}"}),
    ("web_search", {"query": "site:venturebeat.com AI {today} OR {yesterday}"}),
    ("web_search", {"query": "site:huggingface.co blog {today} OR {yesterday}"}),
    ("web_search", {"query": "site:paperswithcode.com {today} OR {yesterday}"}),
    ("web_search", {"query": "The Rundown AI OR The Batch newsletter {today}"}),
    ("web_search", {"query": "Hacker News AI OR LLM {today}"}),
    ("web_search", {"query": "r/MachineLearning trending {today}"}),
    ("web_search", {"query": "Semantic Scholar AI trending papers {today}"}),
    ("arxiv_search", {"query": "large language models reasoning"}),
    ("arxiv_search", {"query": "multimodal vision language"}),
    ("arxiv_search", {"query": "AI agents reinforcement learning"}),
    ("arxiv_search", {"query": "diffusion generative model"}),
]


def _run_all_searches(today: str, yesterday: str, verbose: bool) -> str:
    parts = []
    for tool_name, raw_args in _MANUAL_SEARCHES:
        args = {k: v.format(today=today, yesterday=yesterday) for k, v in raw_args.items()}
        if verbose:
            print(f"  [research/manual] {tool_name}({args})")
        try:
            result = TOOL_HANDLERS[tool_name](**args)
        except Exception as e:
            result = f"Error: {e}"
        parts.append(f"### {tool_name}: {args}\n{result}")
    return "\n\n".join(parts)


def _synthesise(raw_results: str, today: str, yesterday: str, weekday: str) -> str:
    synthesis_prompt = f"""You are an AI research analyst. Today is {today} ({weekday}).

Below are raw search results collected from various AI news sources and arXiv for the last 24 hours (since {yesterday}).

Based ONLY on these results, produce a structured research brief in this exact format:

# AI Research Brief — {today}

## Trending Now (Highest Attention)
The 2-3 items gaining the most traction. Include why they are trending, what happened, and the technical significance.

## Other Notable Developments
3-4 additional stories or papers worth knowing about.

## Signal vs Noise
2-3 bullet points on what today's developments signal about where AI is heading.

RAW SEARCH RESULTS:
{raw_results}

Write the research brief now. Be specific — name models, numbers, researchers."""

    response = complete([{"role": "user", "content": synthesis_prompt}])
    return response.choices[0].message.content
