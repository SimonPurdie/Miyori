import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from miyori.core.chat_history import ChatHistory
from miyori.core.llm_coordinator import LLMCoordinator

def test_chat_history():
    print("Testing ChatHistory...")
    history = ChatHistory(max_tokens=100, trim_chunk_size=20)
    
    # 1. Add messages
    history.add_message("user", "Hello")
    history.add_message("miyori", "Hi there!", tool_calls=[{"id": "1", "name": "test", "arguments": {}}])
    history.add_message("tool", "result", name="test", tool_call_id="1")
    
    msgs = history.get_history()
    assert len(msgs) == 3
    assert msgs[0]["role"] == "user"
    assert msgs[1]["tool_calls"][0]["id"] == "1"
    
    # 2. Token counting (very rough check)
    count = history.get_token_count()
    print(f"Token count: {count}")
    assert count > 0
    
    # 3. Trimming
    # Add a huge message to trigger trimming
    huge_content = "A" * 500 # ~125 tokens
    history.add_message("user", huge_content)
    
    new_msgs = history.get_history()
    assert len(new_msgs) < 4 # Should have trimmed something
    print("ChatHistory tests passed!")

def test_llm_coordinator():
    print("Testing LLMCoordinator...")
    
    class MockHistory:
        def __init__(self): self.msgs = []
        def add_message(self, role, content, **kwargs):
            msg = {"role": role, "content": content}
            msg.update(kwargs)
            self.msgs.append(msg)
        def get_history(self): return self.msgs
        def clear(self): self.msgs = []

    def mock_translate(msgs): return msgs
    def mock_call(msgs, config): return {"text": "Hello world", "tool_calls": []}
    def mock_parse(resp): return resp
    def mock_format(id, name, res): return res

    history = MockHistory()
    coordinator = LLMCoordinator(
        chat_history=history,
        translate_to_provider_callback=mock_translate,
        call_provider_api_callback=mock_call,
        parse_provider_response_callback=mock_parse,
        format_tool_result_callback=mock_format
    )

    def on_chunk(t): print(f"CHUNK: {t}")
    def on_tool(n, a): return "tool_result"

    coordinator.run(
        prompt="Test prompt",
        tools=[],
        on_chunk=on_chunk,
        on_tool_call=on_tool
    )

    assert len(history.msgs) == 2 # User (contextualized), Miyori (response)
    # Wait, history.msgs[0] is User (contextualized), history.msgs[1] is Miyori
    # Ah, I added "user" contextualized, then "miyori" text response.
    # Total 2 messages if no tool calls.
    # Actually my coordinator adds 'user' then loop then 'miyori'.
    # If no tool calls: 1. user, 2. call, 3. parse, 4. add miyori.
    assert len(history.msgs) == 2
    assert "Test prompt" in history.msgs[0]["content"]
    # Context is now added at generation time, not stored in history
    # assert "[CONTEXT:" in history.msgs[0]["content"]
    assert history.msgs[1]["content"] == "Hello world"
    
    print("LLMCoordinator tests passed!")

if __name__ == "__main__":
    try:
        test_chat_history()
        test_llm_coordinator()
        print("\nAll core unit tests passed! ✅")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
