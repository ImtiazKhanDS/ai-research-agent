import arxiv
from datetime import datetime, timedelta, timezone


arxiv_search_tool = {
    "type": "function",
    "function": {
        "name": "arxiv_search",
        "description": "Search arXiv for recent AI research papers submitted in the last 24 hours. Covers cs.AI (Artificial Intelligence), cs.LG (Machine Learning), and cs.CL (Computation and Language).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for arXiv papers",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of papers to return (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


def run_arxiv_search(query: str, max_results: int = 5) -> str:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    category_filter = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"
    full_query = f"({query}) AND ({category_filter})"

    client = arxiv.Client()
    search = arxiv.Search(
        query=full_query,
        max_results=max_results * 3,  # fetch extra to filter by date
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    results = []
    for paper in client.results(search):
        if paper.published < cutoff:
            continue
        authors = ", ".join(a.name for a in paper.authors[:3])
        if len(paper.authors) > 3:
            authors += " et al."
        results.append(
            f"Title: {paper.title}\n"
            f"Authors: {authors}\n"
            f"Published: {paper.published.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Abstract: {paper.summary[:400]}...\n"
            f"URL: {paper.entry_id}\n"
        )
        if len(results) >= max_results:
            break

    return "\n---\n".join(results) if results else "No recent arXiv papers found in the last 24 hours."
