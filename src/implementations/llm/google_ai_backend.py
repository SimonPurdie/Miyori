from google import genai
from google.genai import types
import json
from pathlib import Path
from typing import Callable, List, Dict, Any
from src.core.tools import Tool
from src.interfaces.llm_backend import ILLMBackend
import asyncio
import threading

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
        
        # Memory Components
        try:
            from src.memory.sqlite_store import SQLiteMemoryStore
            from src.utils.embeddings import EmbeddingService
            from src.memory.episodic import EpisodicMemoryManager
            from src.memory.summarizer import Summarizer
            from src.memory.context import ContextBuilder
            from src.memory.gates import MemoryGate
            
            self.store = SQLiteMemoryStore()
            self.embedding_service = EmbeddingService()
            self.episodic_manager = EpisodicMemoryManager(self.store, self.embedding_service)
            self.summarizer = Summarizer(self.client)
            self.context_builder = ContextBuilder(self.store, self.episodic_manager)
            self.gate = MemoryGate(self.client)
            
            # Phase 3 Components
            from src.memory.deep_layers import EmotionalTracker
            from src.memory.consolidation import RelationalManager, ConsolidationManager, SemanticExtractor
            
            self.emotional_tracker = EmotionalTracker(self.store)
            self.relational_manager = RelationalManager(self.client, self.store)
            self.semantic_extractor = SemanticExtractor(self.client, self.store)
            self.consolidation = ConsolidationManager(
                self.store, self.episodic_manager, 
                self.semantic_extractor, self.relational_manager
            )
            
            memory_config = config.get("memory", {})
            self.memory_enabled = memory_config.get("enabled", True)
            self.feature_flags = memory_config.get("feature_flags", {})
            
            # Async Background Loop for memory tasks
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._loop_thread.start()
            
            if self.memory_enabled:
                print("Memory system initialized and enabled.")
        except Exception as e:
            print(f"Warning: Failed to initialize memory system: {e}")
            self.memory_enabled = False

    def _run_async(self, coro):
        """Run a coroutine in the background thread's event loop."""
        if hasattr(self, '_loop'):
            return asyncio.run_coroutine_threadsafe(coro, self._loop)
        return None

    async def _store_turn(self, user_msg: str, assistant_msg: str):
        """Summarize and store the conversation turn."""
        try:
            # Phase 2: Memory Gating
            if self.feature_flags.get("enable_gating", False):
                if not await self.gate.should_remember(user_msg, assistant_msg):
                    print("Memory: Turn skipped by Gate.")
                    return

            print(f"Memory: Summarizing turn...")
            summary = await self.summarizer.create_summary(user_msg, assistant_msg)
            full_text = {"user": user_msg, "assistant": assistant_msg}
            
            print(f"Memory: Storing episode...")
            await self.episodic_manager.add_episode(summary, full_text)
            print(f"Memory: Episode stored successfully.")
            
            # Phase 3: Emotional Continuity
            self.emotional_tracker.update_thread(user_msg, assistant_msg)
            
        except Exception as e:
            import sys
            sys.stderr.write(f"Memory Error: Failure in _store_turn: {e}\n")

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
            # Fetch Context
            context = ""
            if self.memory_enabled:
                try:
                    context = self.context_builder.build_context(prompt)
                except Exception as e:
                    print(f"Memory retrieval failed: {e}")

            try:
                config = None
                effective_system_instruction = self.system_instruction or ""
                if context:
                    effective_system_instruction += f"\n\n[PAST CONTEXT]\n{context}\n"
                
                config = types.GenerateContentConfig(
                    system_instruction=effective_system_instruction if effective_system_instruction else None
                )
                self.chat = self.client.chats.create(model=self.model_name, config=config)
            except Exception as e:
                print(f"Error creating chat session: {e}")
                return

        full_response = []

        try:
            # Use chat.send_message with streaming
            response = self.chat.send_message_stream(prompt)
            
            for chunk in response:
                if chunk.text:
                    full_response.append(chunk.text)
                    on_chunk(chunk.text)
            
            # Store turn asynchronously
            if self.memory_enabled:
                self._run_async(self._store_turn(prompt, "".join(full_response)))
                    
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
            # Fetch Context
            context = ""
            if self.memory_enabled:
                try:
                    context = self.context_builder.build_context(prompt)
                except Exception as e:
                    print(f"Memory retrieval failed: {e}")

            try:
                effective_system_instruction = self.system_instruction or ""
                if context:
                    effective_system_instruction += f"\n\n[PAST CONTEXT]\n{context}\n"

                config = types.GenerateContentConfig(
                    system_instruction=effective_system_instruction if effective_system_instruction else None,
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
        full_response = []

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
                            full_response.append(part.text)
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
                    if self.memory_enabled:
                        self._run_async(self._store_turn(prompt, "".join(full_response)))
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
