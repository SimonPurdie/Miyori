from google import genai
from google.genai import types
from typing import Callable, List, Dict, Any, Union, Optional
import asyncio
import threading

from miyori.utils.config import Config
from miyori.core.tools import Tool
from miyori.interfaces.llm_backend import ILLMBackend
from miyori.core.chat_history import ChatHistory
from miyori.core.llm_coordinator import LLMCoordinator
import uuid

class GoogleAIBackend(ILLMBackend):
    def __init__(self):
        llm_config = Config.data.get("llm", {})
        self.api_key = llm_config.get("api_key")
        self.model_name = llm_config.get("model")

        # Load System Instructions
        # e:/_Projects/Miyori/src/implementations/llm/google_ai_backend.py
        project_root = Config.get_project_root()
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
        
        # LLM Limits and Coordination
        # around 8:1 ratio for token limit to chunk size is sensible
        # losing about 20% of conversation with each trim
        self.MAX_HISTORY_TOKENS = 128000
        self.MAX_TOOL_TURNS = 12
        self.TRIM_CHUNK_SIZE = 16000

        self.chat_history = ChatHistory(
            max_tokens=self.MAX_HISTORY_TOKENS,
            trim_chunk_size=self.TRIM_CHUNK_SIZE
        )

        self.coordinator = LLMCoordinator(
            chat_history=self.chat_history,
            translate_to_provider_callback=self._translate_to_provider_format,
            call_provider_api_callback=self._call_provider_api,
            parse_provider_response_callback=self._parse_provider_response,
            format_tool_result_callback=self._format_tool_result,
            max_tool_turns=self.MAX_TOOL_TURNS
        )
        
        # Memory Components
        try:
            from miyori.memory.sqlite_store import SQLiteMemoryStore
            from miyori.utils.embeddings import EmbeddingService
            from miyori.memory.episodic import EpisodicMemoryManager
            from miyori.memory.summarizer import Summarizer
            from miyori.memory.context import ContextBuilder
            from miyori.memory.gates import MemoryGate
            from miyori.memory.memory_retriever import MemoryRetriever
            from miyori.memory.async_memory_stream import AsyncMemoryStream

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
            from miyori.memory.consolidation import ConsolidationManager, SemanticExtractor
            
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
            import traceback
            print(f"Warning: Failed to initialize memory system: {e}")
            self.memory_enabled = False

    def _run_async(self, coro):
        """Run a coroutine in the background thread's event loop."""
        if hasattr(self, '_loop'):
            return asyncio.run_coroutine_threadsafe(coro, self._loop)
        return None

    def _send_to_log(self, logname: str, content: Union[str, List[str]]):
        try:
            logs_dir = Config.get_project_root() / "logs"
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
        self.chat_history.clear()
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
        on_tool_call: Callable[[str, Dict[str, Any]], str],
        interrupt_check: Callable[[], bool] = None,
        source: str = "text",
        agentic_state: Optional[Any] = None
    ) -> None:
        if not self.client:
            print("Error: API Key not configured.")
            return

        if self.google_tools is None:
            self.google_tools = self._convert_tools_to_gemini_format(tools)
        
        google_tools = self.google_tools
        
        # Static config for Gemini
        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction if self.system_instruction else None,
            tools=google_tools if google_tools else None
        )

        def store_turn_wrapper(u, m):
            if self.memory_enabled:
                self._run_async(self._store_turn(u, m))

        try:
            # Delegate to coordinator
            self.coordinator.run(
                prompt=prompt,
                tools=tools,
                on_chunk=on_chunk,
                on_tool_call=on_tool_call,
                interrupt_check=interrupt_check,
                source=source,
                context_builder=self.context_builder if self.memory_enabled else None,
                store_turn_callback=store_turn_wrapper,
                generate_config=config,
                agentic_state=agentic_state
            )

            # Log system instruction and prompt (from the first turn usually)
            self._send_to_log("system_instruction", self.system_instruction or "NONE")
            self._send_to_log("prompt", prompt)

        except Exception as e:
            print(f"Error during LLM coordination: {e}")
            import traceback
            traceback.print_exc()

    # --- Provider Specific Callbacks ---

    def _translate_to_provider_format(self, messages: List[Dict]) -> List[types.Content]:
        """Converts internal format to Google's Content format, grouping adjacent messages with same role."""
        provider_messages = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            # Map internal role to provider role
            current_provider_role = "model" if msg["role"] == "miyori" else "user"
            
            # Group consecutive messages that share the same provider role
            group_parts = []
            while i < len(messages):
                m = messages[i]
                m_provider_role = "model" if m["role"] == "miyori" else "user"
                
                if m_provider_role != current_provider_role:
                    break
                
                # Add parts from this message
                if m.get("content"):
                    if m["role"] == "tool":
                        # Tool results use function_response part
                        group_parts.append(types.Part.from_function_response(
                            name=m["name"],
                            response={"result": m["content"]}
                        ))
                    else:
                        # Regular text part
                        group_parts.append(types.Part.from_text(text=m["content"]))
                
                if m["role"] == "miyori" and m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        # Include thought_signature if present (required for some Gemini models)
                        group_parts.append(types.Part(
                            function_call=types.FunctionCall(
                                name=tc["name"],
                                args=tc["arguments"]
                            ),
                            thought_signature=tc.get("thought_signature")
                        ))
                
                i += 1
            
            if group_parts:
                provider_messages.append(types.Content(
                    role=current_provider_role, 
                    parts=group_parts
                ))
        
        return provider_messages

    def _call_provider_api(self, provider_messages: List[types.Content], config: types.GenerateContentConfig) -> Any:
        """Makes stateless API call to Google's generate_content."""
        return self.client.models.generate_content(
            model=self.model_name,
            contents=provider_messages,
            config=config
        )

    def _parse_provider_response(self, response: Any) -> Dict:
        """Extracts text and tool calls from Google response."""
        text_parts = []
        tool_calls = []
        
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
                
                if hasattr(part, 'function_call') and part.function_call:
                    tc_data = {
                        "id": str(uuid.uuid4())[:8],
                        "name": part.function_call.name,
                        "arguments": part.function_call.args or {}
                    }
                    # Capture thought_signature if present on the Part
                    if hasattr(part, 'thought_signature') and part.thought_signature:
                        tc_data["thought_signature"] = part.thought_signature
                    
                    tool_calls.append(tc_data)
        
        return {
            "text": "".join(text_parts),
            "tool_calls": tool_calls
        }

    def _format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> Any:
        """Handled within _translate_to_provider_format for Google."""
        # Coordinator calls this but we also need it for internal history storage
        # which is handled by LLMCoordinator. 
        # For Google, the actual provider mapping happens in _translate_to_provider_format.
        return result

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
