from google import genai
from google.genai import types
import json
from pathlib import Path
from typing import Callable, List, Dict, Any
from src.core.tools import Tool
from src.interfaces.llm_backend import ILLMBackend

class GoogleAIBackend(ILLMBackend):
    def __init__(self):
        # e:/_Projects/Miyori/src/implementations/llm/google_ai_backend.py
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / "config.json"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        llm_config = config.get("llm", {})
        self.api_key = llm_config.get("api_key")
        self.model_name = llm_config.get("model", "gemini-2.0-flash-exp")

        # Load System Instructions
        system_instruction_file = llm_config.get("system_instruction_file", "system_instructions.txt")
        self.system_instruction_path = project_root / system_instruction_file
        self.system_instruction = None

        if self.system_instruction_path.exists():
            try:
                with open(self.system_instruction_path, "r", encoding="utf-8") as f:
                    self.system_instruction = f.read().strip()
            except Exception as e:
                print(f"Error reading system instruction file: {e}")
        else:
            print(f"Warning: System instruction file not found at {self.system_instruction_path}")
        
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            # Handle missing API key gracefully or let it fail later
            self.client = None
            
        self.chat = None

    def reset_context(self) -> None:
        """Resets the conversation history."""
        print("Resetting conversation context...")
        self.chat = None

    def generate_stream(self, prompt: str, on_chunk: Callable[[str], None]) -> None:
        if not self.client:
            print("Error: API Key not configured.")
            return

        print("Thinking...")
        
        # Initialize chat if not already active
        if self.chat is None:
            # We can create a new chat session
            try:
                config = None
                if self.system_instruction:
                    config = types.GenerateContentConfig(
                        system_instruction=self.system_instruction
                    )
                self.chat = self.client.chats.create(model=self.model_name, config=config)
            except Exception as e:
                print(f"Error creating chat session: {e}")
                # Fallback to direct generation if chat fails (though unexpected)
                # But better to just re-raise or handle gracefully
                return

        try:
            # Use chat.send_message with streaming
            response = self.chat.send_message_stream(prompt)
            
            for chunk in response:
                if chunk.text:
                    on_chunk(chunk.text)
                    
        except Exception as e:
            print(f"Error during streaming generation: {e}")
            self.chat = None # Invalidate chat on error?

    def generate_stream_with_tools(
        self,
        prompt: str,
        tools: List[Tool],
        on_chunk: Callable[[str], None],
        on_tool_call: Callable[[str, Dict[str, Any]], str]
    ) -> None:
        if not self.client:
            print("Error: API Key not configured.")
            return

        # Convert tools to Gemini format
        google_tools = self._convert_tools_to_gemini_format(tools)
        
        # Initialize chat if not already active
        if self.chat is None:
            try:
                config = types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    tools=google_tools if google_tools else None
                )
                self.chat = self.client.chats.create(model=self.model_name, config=config)
            except Exception as e:
                print(f"Error creating chat session: {e}")
                return
        else:
            # If chat exists, we might need to update the tools in the config for this session
            # However, google-genai chat sessions usually keep their config.
            # For simplicity, if tools change, we might need a new chat or update config.
            # In Miyori, tools are usually static after startup.
            pass

        max_turns = 10
        turn_count = 0

        try:
            # First turn: Send user prompt
            response = self.chat.send_message(prompt)
            
            while turn_count < max_turns:
                turn_count += 1
                has_tool_call = False
                tool_response_parts = []
                
                # Process parts of the current response
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.text:
                            on_chunk(part.text)
                        
                        if part.function_call:
                            has_tool_call = True
                            tool_name = part.function_call.name
                            args = part.function_call.args or {}
                            
                            # Execute tool
                            result = on_tool_call(tool_name, args)
                            
                            # Prepare result part
                            tool_response_parts.append(types.Part.from_function_response(
                                name=tool_name,
                                response={"result": result}
                            ))
                
                if has_tool_call:
                    # Send all tool results back to the model in one go
                    # This triggers the next turn
                    response = self.chat.send_message(tool_response_parts)
                else:
                    # No more tool calls in this response, we are finished
                    break

            if turn_count >= max_turns:
                print(f"⚠️ Warning: Max tool turns ({max_turns}) reached.")

        except Exception as e:
            print(f"Error during tool-enabled generation: {e}")
            self.chat = None

    def _convert_tools_to_gemini_format(self, tools: List[Tool]) -> List[types.Tool]:
        if not tools:
            return []
        
        function_declarations = []
        for tool in tools:
            properties = {}
            required = []
            
            for param in tool.parameters:
                # Map our types to Gemini Schema types
                # Gemini expects uppercase strings or types.Type enum
                param_type = param.type.upper()
                if param_type == "NUMBER": param_type = "NUMBER"
                elif param_type == "INTEGER": param_type = "INTEGER"
                elif param_type == "BOOLEAN": param_type = "BOOLEAN"
                elif param_type == "ARRAY": param_type = "ARRAY"
                elif param_type == "OBJECT": param_type = "OBJECT"
                else: param_type = "STRING"

                prop = {
                    "type": param_type,
                    "description": param.description
                }
                if param.enum:
                    prop["enum"] = param.enum
                
                properties[param.name] = prop
                if param.required:
                    required.append(param.name)
            
            function_declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=types.Schema(
                        type="OBJECT",
                        properties=properties,
                        required=required
                    )
                )
            )
            
        return [types.Tool(function_declarations=function_declarations)]
