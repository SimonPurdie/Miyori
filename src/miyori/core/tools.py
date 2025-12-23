from typing import Any, Dict, List, Callable, Optional
from dataclasses import dataclass

@dataclass
class ToolParameter:
    """Describes a single parameter for a tool."""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None  # For restricted values

@dataclass
class Tool:
    """A tool that the LLM can call."""
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable[..., str]  # The actual function to execute
    
    def execute(self, **kwargs) -> str:
        """Execute the tool's function."""
        return self.function(**kwargs)
