import datetime
import json
import os
from typing import List, Dict, Any, Callable, Optional
import uuid
from miyori.core.agentic_state import AgenticExitSignal

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
        generate_config: Optional[Any] = None,
        agentic_state: Optional[Any] = None
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
        
        try:
            # Start tool calling / agentic loop
            while True:
                # 1. CHECK INTERRUPT
                if interrupt_check and interrupt_check():
                    print("LLMCoordinator: Interrupt requested, stopping generation.")
                    if agentic_state:
                         agentic_state.reset()
                    break
                    
                # 2. CHECK LOOP LIMITS
                # In agentic mode, we use agentic_state.iteration
                # In standard mode, we use turn_count
                if agentic_state and agentic_state.is_active:
                    if agentic_state.iteration > agentic_state.max_iterations:
                        print(f"LLMCoordinator: Agentic iteration limit ({agentic_state.max_iterations}) reached.")
                        on_chunk(f"\n[SYSTEM: Iteration limit reached. Current progress summarized in state.]")
                        agentic_state.reset()
                        break
                elif turn_count >= self.max_tool_turns:
                    print(f"LLMCoordinator: Max tool turns ({self.max_tool_turns}) reached.")
                    break

                turn_count += 1
                if agentic_state and agentic_state.is_active:
                    print(f"--- Agentic Iteration {agentic_state.iteration}/{agentic_state.max_iterations} ---")
                
                # 3. BUILD CONTEXT
                history = self.chat_history.get_history().copy()
                
                # Build agentic context if active
                agentic_prefix = ""
                if agentic_state and agentic_state.is_active:
                    agentic_prefix = self._build_agentic_context(agentic_state)

                # Find the most recent user prompt in history to prepend context to it
                for i in range(len(history) - 1, -1, -1):
                    if history[i]["role"] == "user":
                        # Create a copy of the message so we don't modify the one in history
                        history[i] = history[i].copy()
                        content = history[i]["content"]
                        
                        # Combine contexts
                        # Order: Memory Context -> Agentic State -> User Prompt
                        full_prefix = ""
                        if context_prefix:
                            full_prefix += f"{context_prefix}\n\n"
                        if agentic_prefix:
                            full_prefix += f"{agentic_prefix}\n\n"
                        
                        history[i]["content"] = f"{full_prefix}[user {content}"
                        break
                
                # 4. GET RESPONSE
                provider_messages = self._translate_to_provider(history)
                # Log history to file for debugging
                self._log_to_file("history.log", history)
                
                try:
                    response = self._call_provider_api(provider_messages, generate_config)
                except Exception as e:
                    print(f"LLMCoordinator: API call failed: {e}")
                    if agentic_state:
                        agentic_state.reset()
                    break
                    
                # Parse response
                parsed = self._parse_provider_response(response)
                text = parsed.get("text", "")
                thought = parsed.get("thought", "")
                tool_calls = parsed.get("tool_calls", [])
                
                # Stream thought chunks if present
                if thought:
                    on_chunk(f"\nmiyori (thinking): {thought}\n")

                # Stream text chunks
                if text:
                    full_response_text.append(text)
                    on_chunk(text)
                text = parsed.get("text", "")
                tool_calls = parsed.get("tool_calls", [])
                
                # Stream text chunks
                if text:
                    full_response_text.append(text)
                    on_chunk(text)
                
                # 5. HANDLE TOOL CALLS
                if tool_calls:
                    # Store miyori's turn WITH tool calls in history
                    self.chat_history.add_message("miyori", text, tool_calls=tool_calls)
                    
                    # Execute all tool calls
                    for tc in tool_calls:
                        tc_id = tc["id"]
                        tc_name = tc["name"]
                        tc_args = tc["arguments"]
                        
                        # Execute via callback
                        # Note: on_tool_call might raise AgenticExitSignal
                        result = on_tool_call(tc_name, tc_args)
                        
                        # Store tool result in history
                        self.chat_history.add_message(
                            "tool", 
                            result, 
                            name=tc_name, 
                            tool_call_id=tc_id
                        )
                    
                    # Increment agentic iteration if we just executed tools in agentic mode
                    if agentic_state and agentic_state.is_active:
                        agentic_state.iteration += 1
                    
                    # Continue loop to send tool results back
                    continue
                else:
                    # No more tool calls
                    # If we are in agentic mode, and it didn't call exit_loop, 
                    # we keep looping as long as it's still active.
                    # Wait: if it didn't call tool_calls, how does it know to keep going?
                    # The LLM must call agentic_loop to start, and exit_loop to end.
                    # If it's just talking, we consider it a reasoning turn in the loop.
                    
                    if agentic_state and agentic_state.is_active:
                        self.chat_history.add_message("miyori", text)
                        agentic_state.iteration += 1
                        # Small delay to prevent tight loops if it keeps "reasoning"
                        #os.sleep(0.5) 
                        continue
                    else:
                        # Standard turn finish
                        self.chat_history.add_message("miyori", text)
                        
                        # Trigger memory storage if callback provided
                        if store_turn_callback:
                            try:
                                store_turn_callback(prompt, "".join(full_response_text))
                            except Exception as e:
                                print(f"LLMCoordinator: Memory storage failed: {e}")
                        break
        
        except AgenticExitSignal as aes:
            print(f"LLMCoordinator: Agentic loop exited: {aes.status}")
            final_msg = f"\n\n[OBJECTIVE COMPLETE: {aes.status.upper()}]\n{aes.result}"
            on_chunk(final_msg)
            self.chat_history.add_message("miyori", final_msg)
            if agentic_state:
                agentic_state.reset()
        
        except Exception as e:
            print(f"LLMCoordinator: Unexpected error in run(): {e}")
            import traceback
            traceback.print_exc()
            if agentic_state:
                agentic_state.reset()

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

    def _build_agentic_context(self, state: Any) -> str:
        """Constructs the agentic state context block."""
        lines = [
            f"[SYSTEM: AGENTIC MODE ACTIVE - Iteration {state.iteration}/{state.max_iterations}]",
            f"Original User Prompt: {state.original_prompt}",
            f"Set Objective: {state.objective}"
        ]
        
        if state.working_directory:
            lines.append(f"Working Directory: {state.working_directory}")
            
        if state.last_command:
            lines.append(f"Last Command: {state.last_command}")
            if state.last_exit_code is not None:
                lines.append(f"Last Exit Code: {state.last_exit_code}")
            if state.last_output:
                lines.append(f"Last Output (truncated): {state.last_output}")
                
        if state.modified_files:
            lines.append(f"Files Modified This Session: {', '.join(state.modified_files)}")
            
        lines.append("[You are acting autonomously. Call 'exit_loop' when finished or if impossible.]")
        
        return "\n".join(lines)
