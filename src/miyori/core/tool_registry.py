from typing import Dict, List, Optional, Any
from miyori.core.tools import Tool

class ToolRegistry:
    """Central registry for all available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool by its name."""
        self._tools[tool.name] = tool
        print(f"✓ Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_all(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def execute(self, tool_name: str, **parameters) -> str:
        """Execute a tool and return its result."""
        tool = self.get_tool(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            result = tool.execute(**parameters)
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
