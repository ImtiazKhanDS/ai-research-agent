import os
from tavily import TavilyClient

def _client() -> TavilyClient:
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


web_search_tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for recent AI news, announcements, and developments. Use this to find the latest AI trends, model releases, and research breakthroughs from the past 24 hours.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


def run_web_search(query: str, max_results: int = 5) -> str:
    client = _client()
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_answer=True,
        days=1,
    )
    results = []
    if response.get("answer"):
        results.append(f"Summary: {response['answer']}\n")
    for r in response.get("results", []):
        results.append(
            f"Title: {r.get('title', 'N/A')}\n"
            f"URL: {r.get('url', 'N/A')}\n"
            f"Content: {r.get('content', 'N/A')}\n"
        )
    return "\n---\n".join(results) if results else "No results found."
