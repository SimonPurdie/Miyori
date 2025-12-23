from ddgs import DDGS
from src.core.tools import Tool, ToolParameter

def web_search(query: str, num_results: int = 3) -> str:
    """
    Search the web using DDGS (Dux Distributed Global Search).
    
    Args:
        query: Search query string
        num_results: Number of results to return (1-10)
    
    Returns:
        Formatted search results as a string
    """
    print(f"🔍 Searching web: {query}")
    
    try:
        results = []
        with DDGS() as ddgs:
            # text() returns a generator of results
            ddgs_results = ddgs.text(query, max_results=num_results)
            for r in ddgs_results:
                results.append({
                    'title': r.get('title', 'No Title'),
                    'snippet': r.get('body', 'No snippet available'),
                    'url': r.get('href', 'N/A')
                })
        
        if not results:
            return f"No results found for '{query}'"
        
        # Format output
        formatted = f"Search results for '{query}':\n"
        for i, result in enumerate(results, 1):
            formatted += f"\n{i}. {result['title']}\n"
            formatted += f"   {result['snippet']}\n"
            formatted += f"   {result['url']}\n"
        
        return formatted
        
    except Exception as e:
        return f"Search failed: {str(e)}"


# Create the tool definition
web_search_tool = Tool(
    name="web_search",
    description="Search the web for current information. Use this when you need up-to-date facts, news, or information not in your training data.",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="The search query",
            required=True
        ),
        ToolParameter(
            name="num_results",
            type="number",
            description="Number of results to return (1-10)",
            required=False
        )
    ],
    function=web_search
)
