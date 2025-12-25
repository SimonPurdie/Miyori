import time
from typing import Dict, Any, Callable, List, Optional
from miyori.interfaces.speech_input import ISpeechInput
from miyori.interfaces.speech_output import ISpeechOutput
from miyori.interfaces.llm_backend import ILLMBackend
from miyori.core.tool_registry import ToolRegistry
from miyori.utils.config import Config
from miyori.core.state_manager import StateManager

class MiyoriCore:
    def __init__(self, 
                 speech_output: ISpeechOutput,
                 llm: ILLMBackend,
                 state_manager: StateManager,
                 tool_registry: ToolRegistry = None):
        self.speech_output = speech_output
        self.llm = llm
        self.state_manager = state_manager
        self.tool_registry = tool_registry
        
        self.last_interaction_time = 0
        # Load config for active listen timeout
        self.active_listen_timeout = Config.data.get("speech_input", {}).get("active_listen_timeout", 30)

    def process_input(self, text: str, source: str, on_chunk: Callable[[str], None]) -> None:
        """
        Process user input and generate response.
        
        Args:
            text: User input text
            source: "voice" or "text"
            on_chunk: Callback for streaming response chunks
        """
        # Update interaction time for both voice and text
        self.last_interaction_time = time.time()
        
        # Handle special commands
        if "go to sleep" in text.lower():            
            self.last_interaction_time = 0
            on_chunk("ok goodnight")
            return
        
        if set(['exit', 'goodbye']).intersection(text.lower().split()):
            on_chunk("Goodbye!")
            return
        
        # Use tool-enabled generation when tools are available
        if self.tool_registry and self.tool_registry.get_all():
             self._handle_with_tools(text, on_chunk)
        else:
            # Fallback if no tools (though usually should have tools)
            self.llm.llm_chat(
                prompt=text,
                tools=[],
                on_chunk=on_chunk,
                on_tool_call=lambda n, p: "",
                interrupt_check=self.state_manager.should_interrupt
            )

    def _handle_with_tools(self, user_input: str, on_chunk: Callable[[str], None]) -> None:
        """Handle user input with tool support."""
        from miyori.utils import logger
        
        def on_tool_call(tool_name: str, parameters: Dict[str, Any]) -> str:
            """Execute tool and return result."""
            print(f"🔧 AI requested tool: {tool_name}")
            print(f"   Parameters: {parameters}")
            
            with logger.capture_session() as buffer:
                result = self.tool_registry.execute(tool_name, **parameters)
                logs = buffer.getvalue().strip()
            
            if logs:
                print(f"✓ Tool result: {result[:100]}...")
                return f"TOOL LOGS:\n{logs}\n\nTOOL RESULT:\n{result}"
            else:
                print(f"✓ Tool result: {result[:100]}...")
                return result
        
        # Get all registered tools
        tools = self.tool_registry.get_all()
        
        # Generate response with tool support
        self.llm.llm_chat(
            prompt=user_input,
            tools=tools,
            on_chunk=on_chunk,
            on_tool_call=on_tool_call,
            interrupt_check=self.state_manager.should_interrupt
        )

    def needs_wake_word(self) -> bool:
        """Determine if wake word is required for voice input."""
        if self.last_interaction_time == 0:
            return True
        return (time.time() - self.last_interaction_time) >= self.active_listen_timeout
