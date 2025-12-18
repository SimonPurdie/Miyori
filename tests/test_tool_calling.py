import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.implementations.llm.google_ai_backend import GoogleAIBackend
from src.core.tool_registry import ToolRegistry
from src.tools.web_search import web_search_tool
from src.utils import logger

def test_tool_calling():
    print("Setting up logging...")
    logger.setup_logging()
    
    print("Initializing Backend...")
    backend = GoogleAIBackend()
    
    print("Setting up tools...")
    registry = ToolRegistry()
    registry.register(web_search_tool)
    
    prompt = "What is the current population of Tokyo? If your tool fails or you see internal logs, tell me exactly what they were. I want to see the '🔍 Searching web' log message."
    print(f"\nUser: {prompt}")
    
    def on_chunk(text: str):
        print(text, end="", flush=True)
        
    def on_tool_call(name, args):
        print(f"\n[INTERNAL] Executing tool: {name} with args: {args}")
        # Capture logs during tool execution
        with logger.capture_session() as buffer:
            result = registry.execute(name, **args)
            logs = buffer.getvalue().strip()
            
        if logs:
            print(f"[INTERNAL] Captured logs:\n{logs}")
            return f"TOOL LOGS:\n{logs}\n\nTOOL RESULT:\n{result}"
        return result
        
    backend.generate_stream_with_tools(
        prompt=prompt,
        tools=registry.get_all(),
        on_chunk=on_chunk,
        on_tool_call=on_tool_call
    )
    print("\n\nTest complete.")

if __name__ == "__main__":
    test_tool_calling()
