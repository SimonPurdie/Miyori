import requests
from bs4 import BeautifulSoup
from src.core.tools import Tool, ToolParameter

def web_search(query: str, num_results: int = 3) -> str:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: Search query string
        num_results: Number of results to return (1-10)
    
    Returns:
        Formatted search results as a string
    """
    print(f"🔍 Searching web: {query}")
    
    try:
        # Use DuckDuckGo HTML search (no API key needed)
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return f"Search failed: DuckDuckGo returned status {response.status_code}"
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Parse results - try multiple selectors
        results = []
        result_elements = soup.select('.result') or soup.select('.links_main')
        
        for result in result_elements[:num_results]:
            title_elem = result.select_one('.result__title') or result.select_one('.result__a')
            snippet_elem = result.select_one('.result__snippet') or result.select_one('.result__snippet')
            url_elem = result.select_one('.result__url')
            
            if title_elem and snippet_elem:
                results.append({
                    'title': title_elem.get_text(strip=True),
                    'snippet': snippet_elem.get_text(strip=True),
                    'url': url_elem.get_text(strip=True) if url_elem else 'N/A'
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
