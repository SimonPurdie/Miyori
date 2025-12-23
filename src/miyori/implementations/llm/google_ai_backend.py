from google import genai
from google.genai import types
import json
from pathlib import Path
from typing import Callable, List, Dict, Any, Union
import asyncio
import threading

from src.utils.config import Config
from src.core.tools import Tool
from src.interfaces.llm_backend import ILLMBackend

class GoogleAIBackend(ILLMBackend):
    def __init__(self):
        llm_config = Config.data.get("llm", {})
        self.api_key = llm_config.get("api_key")
        self.model_name = llm_config.get("model")

        # Load System Instructions
        # e:/_Projects/Miyori/src/implementations/llm/google_ai_backend.py
        project_root = Path(__file__).parent.parent.parent.parent
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

        self.google_tools = None
        self.chat_session = None
        
        # Memory Components
        try:
            from src.memory.sqlite_store import SQLiteMemoryStore
            from src.utils.embeddings import EmbeddingService
            from src.memory.episodic import EpisodicMemoryManager
            from src.memory.summarizer import Summarizer
            from src.memory.context import ContextBuilder
            from src.memory.gates import MemoryGate
            from src.memory.memory_retriever import MemoryRetriever
            from src.memory.async_memory_stream import AsyncMemoryStream

            memory_config = Config.data.get("memory", {})
            self.memory_enabled = memory_config.get("enabled", True)
            self.feature_flags = memory_config.get("feature_flags", {})

            self.store = SQLiteMemoryStore()
            self.embedding_service = EmbeddingService()
            self.episodic_manager = EpisodicMemoryManager(self.store, self.embedding_service)
            self.summarizer = Summarizer(self.client)

            # Dual-mode memory components
            self.memory_retriever = MemoryRetriever(self.store)
            self.async_memory_stream = AsyncMemoryStream(
                self.memory_retriever,
                self.embedding_service
            )

            self.context_builder = ContextBuilder(
                self.store,
                self.episodic_manager,
                self.async_memory_stream
            )
            self.gate = MemoryGate(self.client)
            
            # Phase 3 Components
            from src.memory.consolidation import ConsolidationManager, SemanticExtractor
            
            self.semantic_extractor = SemanticExtractor(self.client, self.store)
            self.consolidation = ConsolidationManager(
                self.store, self.episodic_manager, 
                self.semantic_extractor
            )
            # Async Background Loop for memory tasks
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._loop_thread.start()
            # Start async memory streaming
            if self.memory_enabled:
                self._run_async(self.async_memory_stream.start())
                print("Memory system initialized and enabled with dual-mode retrieval.")
        except Exception as e:
            print(f"Warning: Failed to initialize memory system: {e}")
            self.memory_enabled = False

    def _run_async(self, coro):
        """Run a coroutine in the background thread's event loop."""
        if hasattr(self, '_loop'):
            return asyncio.run_coroutine_threadsafe(coro, self._loop)
        return None

    def _send_to_log(self, logname: str, content: Union[str, List[str]]):
        try:
            logs_dir = Path(__file__).parent.parent.parent.parent / "logs"
            log_path = logs_dir / f"{logname}.log"

            logs_dir.mkdir(exist_ok=True)

            # Handle both string and list inputs
            if isinstance(content, list):
                content_to_write = "\n".join(content)
            else:
                content_to_write = str(content)

            # Write the content to the log file (overwrite previous content)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(content_to_write)

        except Exception as e:
            print(f"Warning: Failed to log to {logname}: {e}")

    async def _store_turn(self, user_msg: str, miyori_msg: str):
        """Summarize and store the conversation turn."""
        try:
            # Phase 2: Memory Gating
            should_remember = True
            if self.feature_flags.get("enable_gating", False):
                should_remember = await self.gate.should_remember(user_msg, miyori_msg)

            # Get recent context from async memory stream (up to 3 previous turns)
            recent_context = None
            if hasattr(self, 'async_memory_stream'):
                # Get the recent turns but exclude the current turn being summarized
                recent_turns = self.async_memory_stream._recent_turns.copy()
                if recent_turns:
                    recent_context = recent_turns
            if should_remember:
                summary = await self.summarizer.create_summary(user_msg, miyori_msg, recent_context)
                full_text = {"user": user_msg, "miyori": miyori_msg}
                
                print(f"Memory: Storing episode...")
                await self.episodic_manager.add_episode(summary, full_text)
                print(f"Memory: Episode stored successfully.")

            # Add to async memory stream for next turn prefetching
            if hasattr(self, 'async_memory_stream'):
                self.async_memory_stream.add_turn_context(user_msg, miyori_msg)
                await self.async_memory_stream.refresh_cache()
            
        except Exception as e:
            import sys
            sys.stderr.write(f"Memory Error: Failure in _store_turn: {e}\n")
        

    def reset_context(self) -> None:
        """Resets the conversation history."""
        print("Resetting conversation context...")
        self.chat_session = None
        # Clear async memory stream context
        if hasattr(self, 'async_memory_stream'):
            self.async_memory_stream._recent_turns.clear()

    async def _cleanup_async_memory(self):
        """Cleanup async memory stream."""
        if hasattr(self, 'async_memory_stream'):
            await self.async_memory_stream.stop()

    def llm_chat(
        self,
        prompt: str,
        tools: List[Tool],
        on_chunk: Callable[[str], None],
        on_tool_call: Callable[[str, Dict[str, Any]], str]
    ) -> None:
        if not self.client:
            print("Error: API Key not configured.")
            return

        if self.google_tools is None:
            self.google_tools = self._convert_tools_to_gemini_format(tools)
        
        google_tools = self.google_tools
        effective_system_instruction = self.system_instruction or ""
        
        if self.memory_enabled:
            passive_memories = ""
            try:
                passive_memories = self.context_builder.build_context(prompt)
            except Exception as e:
                print(f"Memory retrieval failed: {e}")

            if passive_memories:
                effective_system_instruction += f"\n\n[Non-linear fragments of your life as Miyori:]\n{passive_memories}\n"

        config = types.GenerateContentConfig(
            system_instruction=effective_system_instruction if effective_system_instruction else None,
            tools=google_tools if google_tools else None
        )
        try:
            if self.chat_session is None:
                self.chat_session = self.client.chats.create(model=self.model_name, config=config)
        except Exception as e:
            print(f"Error creating chat session: {e}")
            return

        max_turns = 5
        turn_count = 0
        full_response = []

        try:
            
            response = self.chat_session.send_message(message=prompt, config=config)
            self._send_to_log("system_instruction", config.system_instruction)
            self._send_to_log("prompt", prompt)

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
                    response = self.chat_session.send_message(tool_response_parts)
                else:
                    # No more tool calls in this response, we are finished
                    if self.memory_enabled:
                        self._run_async(self._store_turn(prompt, "".join(full_response)))
                    break

            if turn_count >= max_turns:
                print(f"⚠️ Warning: Max tool turns ({max_turns}) reached.")

        except Exception as e:
            print(f"Error during tool-enabled generation: {e}")
            self.chat_session = None

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
