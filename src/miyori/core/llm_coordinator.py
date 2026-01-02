import datetime
import json
import os
from typing import List, Dict, Any, Callable, Optional
import uuid

class LLMCoordinator:
    """
    Provider-agnostic orchestration of LLM conversations.
    Handles tool-calling loops, memory context injection, and history management.
    """

    METADATA_KEYS_TO_STRIP = ["thought_signature"]

    def __init__(
        self,
        chat_history,
        translate_to_provider_callback: Callable,
        call_provider_api_callback: Callable,
        parse_provider_response_callback: Callable,
        format_tool_result_callback: Callable,
        max_tool_turns: int = 12
    ):
        self.chat_history = chat_history
        self._translate_to_provider = translate_to_provider_callback
        self._call_provider_api = call_provider_api_callback
        self._parse_provider_response = parse_provider_response_callback
        self._format_tool_result = format_tool_result_callback
        self.max_tool_turns = max_tool_turns

    def run(
        self,
        prompt: str,
        tools: List[Any],
        on_chunk: Callable[[str], None],
        on_tool_call: Callable[[str, Dict[str, Any]], str],
        interrupt_check: Optional[Callable[[], bool]] = None,
        source: str = "text",
        context_builder: Optional[Any] = None,
        store_turn_callback: Optional[Callable[[str, str], Any]] = None,
        generate_config: Optional[Any] = None
    ) -> None:
        """
        Main orchestration loop.
        """
        
        # Retrieve memory context (if enabled)
        memory_summary = ""
        if context_builder:
            try:
                memory_summary = context_builder.build_context(prompt)
            except Exception as e:
                print(f"LLMCoordinator: Memory retrieval failed: {e}")

        # Build contextualized prompt
        context_prefix = ""
        if memory_summary:
            context_prefix += f", {memory_summary}"
        
        # Log context prefix
        self._log_to_file("prefix.log", context_prefix)
        
        # 4. Add ORIGINAL user message to history with source prefix
        prefixed_prompt = f"({source} input)]: {prompt}"
        self.chat_history.add_message("user", prefixed_prompt)
        
        turn_count = 0
        full_response_text = []
        
        # Start tool calling loop
        while turn_count < self.max_tool_turns:
            if interrupt_check and interrupt_check():
                print("LLMCoordinator: Interrupt requested, stopping generation.")
                break
                
            turn_count += 1
            
            # 5. Get history and inject context prefix transiently into the LATEST user prompt
            history = self.chat_history.get_history().copy()
            # Find the most recent user prompt in history to prepend context to it
            for i in range(len(history) - 1, -1, -1):
                if history[i]["role"] == "user":
                    # Create a copy of the message so we don't modify the one in history
                    history[i] = history[i].copy()
                    history[i]["content"] = f"{context_prefix}\n\n[user {history[i]['content']}"
                    break
            
            # 6. Translate history to provider format
            provider_messages = self._translate_to_provider(history)
            # Log history to file
            self._log_to_file("history.log", history)
            
            # Call provider API
            try:
                response = self._call_provider_api(provider_messages, generate_config)
            except Exception as e:
                print(f"LLMCoordinator: API call failed: {e}")
                break
                
            # Parse response
            parsed = self._parse_provider_response(response)
            text = parsed.get("text", "")
            tool_calls = parsed.get("tool_calls", [])
            
            # Stream text chunks
            if text:
                full_response_text.append(text)
                on_chunk(text)
            
            # Handle tool calls
            if tool_calls:
                # Store miyori's turn WITH tool calls in history
                self.chat_history.add_message("miyori", text, tool_calls=tool_calls)
                
                # Execute all tool calls
                for tc in tool_calls:
                    tc_id = tc["id"]
                    tc_name = tc["name"]
                    tc_args = tc["arguments"]
                    
                    # Execute via callback
                    result = on_tool_call(tc_name, tc_args)
                    
                    # Store tool result in history
                    self.chat_history.add_message(
                        "tool", 
                        result, 
                        name=tc_name, 
                        tool_call_id=tc_id
                    )
                
                # Continue loop to send tool results back
                continue
            else:
                # No more tool calls, store final miyori response and finish
                self.chat_history.add_message("miyori", text)
                
                # Trigger memory storage if callback provided
                if store_turn_callback:
                    # We store the ORIGINAL prompt, not the contextualized one
                    # And the full aggregated text response
                    try:
                        # Since store_turn_callback might be async, we might need to handle it.
                        # However, GoogleAIBackend used _run_async.
                        store_turn_callback(prompt, "".join(full_response_text))
                    except Exception as e:
                        print(f"LLMCoordinator: Memory storage failed: {e}")
                
                break
        
        if turn_count >= self.max_tool_turns:
            print(f"LLMCoordinator: Max tool turns ({self.max_tool_turns}) reached.")

    def _strip_metadata(self, data: Any) -> Any:
        """
        Recursively strip metadata keys from the data.
        """
        if isinstance(data, dict):
            return {
                k: self._strip_metadata(v)
                for k, v in data.items()
                if k not in self.METADATA_KEYS_TO_STRIP
            }
        elif isinstance(data, list):
            return [self._strip_metadata(item) for item in data]
        else:
            return data

    def _log_to_file(self, filename: str, data: Any):
        """
        Generic logger for LLM coordinator.
        """
        try:
            log_dir = "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            log_path = os.path.join(log_dir, filename)
            
            # Strip metadata if it's a structural log
            if isinstance(data, (dict, list)):
                data = self._strip_metadata(data)
            
            with open(log_path, "w", encoding="utf-8") as f:
                if isinstance(data, (dict, list)):
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    f.write(str(data))
        except Exception as e:
            print(f"LLMCoordinator: Failed to log {filename}: {e}")
