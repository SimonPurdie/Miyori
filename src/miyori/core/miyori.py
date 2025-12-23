import time
import json
from pathlib import Path
from src.interfaces.speech_input import ISpeechInput
from src.interfaces.speech_output import ISpeechOutput
from src.interfaces.llm_backend import ILLMBackend
from src.core.tool_registry import ToolRegistry
from typing import Dict, Any, Callable
from src.utils.config import Config


class MiyoriCore:
    def __init__(self, 
                 speech_input: ISpeechInput,
                 speech_output: ISpeechOutput,
                 llm: ILLMBackend,
                 tool_registry: ToolRegistry = None): # NEW
        self.speech_input = speech_input
        self.speech_output = speech_output
        self.llm = llm
        self.tool_registry = tool_registry # NEW
        
        # Load config for active listen timeout
        self.active_listen_timeout = Config.data.get("speech_input", {}).get("active_listen_timeout", 30)

    def run(self) -> None:
        print("Miyori starting up...")

        last_interaction_time = 0
        
        while True:
            # Determine if we need wake word
            require_wake_word = True
            if last_interaction_time > 0 and (time.time() - last_interaction_time < self.active_listen_timeout):
                require_wake_word = False
                
            text = self.speech_input.listen(require_wake_word=require_wake_word)
            if text is None:
                continue

            # Update last interaction time on successful speech detection
            last_interaction_time = time.time()
            
            # Check for "go to sleep" command
            if "go to sleep" in text.lower():
                self.speech_output.speak("ok goodnight")
                last_interaction_time = 0 # Force wake word requirement next time
                continue

            # Check for exit commands (case-insensitive)
            if set(['exit', 'goodbye']).intersection(text.lower().split()):
                self.speech_output.speak("Goodbye!")
                break
            
            # Define callback for streaming chunks - speak immediately
            def speak_chunk(chunk: str) -> None:
                self.speech_output.speak(chunk)
            
            # Use tool-enabled generation when tools are available
            if self.tool_registry and self.tool_registry.get_all():
                self._handle_with_tools(text, speak_chunk)
            else:
                print("Problem with Tools")
        
        print("Miyori shutting down...")
        time.sleep(2)

    def _handle_with_tools(self, user_input: str, on_chunk: Callable[[str], None]) -> None:
        """Handle user input with tool support."""
        from src.utils import logger
        
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
            on_tool_call=on_tool_call
        )
