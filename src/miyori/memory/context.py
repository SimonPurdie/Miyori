from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

def count_tokens_approx(text: str) -> int:
    """Approximate token count (chars / 4)."""
    return len(text) // 4

def format_section(label: str, content: Any) -> str:
    """Format a context section for the prompt."""
    if not content:
        return ""

    header = f"--- {label} ---"

    if label == 'FACTS':
        # Semantic facts
        items = [f"- {item.get('fact')}" for item in content]
        body = "\n".join(items)

    elif label == 'RECENT' or label == 'RELEVANT' or label == 'EPISODIC':
        # Episodic memories
        items = []
        for item in content:
            ts = item.get('timestamp', '')
            # Simple timestamp formatting if possible
            try:
                dt = datetime.fromisoformat(ts)
                ts_str = dt.strftime("%Y-%m-%d")
                items.append(f"[{ts_str}] {item.get('summary')}")
            except:
                items.append(f"{item.get('summary')}")
        body = "\n".join(items)

    elif label == 'TOOL_RESULTS':
        # Tool-based memory search results
        body = content if isinstance(content, str) else str(content)

    else:
        body = str(content)

    return f"{header}\n{body}\n\n"

def truncate_to_budget(text: str, max_tokens: int) -> str:
    """Truncate text to fit token budget by removing entire items (paragraphs)."""
    items = text.split('\n')
    result = []
    current_tokens = 0

    for item in items:
        item_tokens = count_tokens_approx(item + '\n')
        if current_tokens + item_tokens <= max_tokens:
            result.append(item)
            current_tokens += item_tokens
        else:
            break

    return "\n".join(result)

class ContextBuilder:
    def __init__(self, store, episodic_manager, async_memory_stream=None, token_budget=1500):
        self.store = store
        self.episodic_manager = episodic_manager
        self.async_memory_stream = async_memory_stream
        self.token_budget = token_budget

    def build_context(self, user_message: str, tool_results: Optional[str] = None) -> str:
        """
        Assemble context using dual-mode memory retrieval.

        Args:
            user_message: Current user message (for backward compatibility)
            tool_results: Results from active memory search tool (privileged budget)
        """
        from src.utils.memory_logger import memory_logger

        context_parts = []
        tokens_used = 0

        # 1. Tool Results (Privileged - up to 400 tokens)
        if tool_results:
            section_text = format_section('TOOL_RESULTS', tool_results)
            section_tokens = count_tokens_approx(section_text)

            # Tool results get priority allocation (up to 400 tokens)
            tool_budget = min(400, self.token_budget // 3)  # Max 400 or 1/3 of total budget

            if section_tokens <= tool_budget:
                context_parts.append(section_text)
                tokens_used += section_tokens
                memory_logger.log_event("context_section", {
                    "label": "TOOL_RESULTS",
                    "tokens": section_tokens,
                    "status": "full",
                    "privileged": True
                })
            else:
                # Truncate tool results if needed
                truncated = truncate_to_budget(section_text, tool_budget)
                if truncated.strip():
                    context_parts.append(truncated + "\n\n")
                    added_tokens = count_tokens_approx(truncated + "\n\n")
                    tokens_used += added_tokens
                    memory_logger.log_event("context_section", {
                        "label": "TOOL_RESULTS",
                        "tokens": added_tokens,
                        "status": "truncated",
                        "privileged": True
                    })

        cached_memories = None
        if self.async_memory_stream:
            try:
                # This is now synchronous - just returns cached data
                cached_memories = self.async_memory_stream.get_cached_memories()
            except Exception as e:
                memory_logger.log_event("async_memory_access_error", {"error": str(e)})

        if cached_memories:
            episodic_memories = cached_memories.get('episodic', [])
            semantic_facts = cached_memories.get('semantic', [])

            memory_logger.log_event("using_cached_memories", {
                "episodic_count": len(episodic_memories),
                "semantic_count": len(semantic_facts)
            })
        else:
            # Fallback to synchronous search (backward compatibility)
            memory_logger.log_event("cache_miss_fallback", {})

            # Use recent high importance episodes instead of zero vector
            from datetime import datetime, timedelta
            seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
            
            # Get recent episodes directly without vector search
            all_episodes = self.store.search_episodes([0.0]*768, limit=100, status='active')
            episodic_memories = [
                e for e in all_episodes 
                if e.get('timestamp', '') >= seven_days_ago 
                and e.get('importance', 0) >= 0.7
            ][:5]  # Limit to 5
            
            # Get high-confidence facts
            all_facts = self.store.get_semantic_facts()
            semantic_facts = [
                f for f in all_facts 
                if f.get('confidence', 0) >= 0.5
            ][:5]  # Limit to 5

        # 4. Regular Memory Sections (remaining budget)
        remaining_budget = self.token_budget - tokens_used

        sections = [
            ('EPISODIC', episodic_memories, 400, False),
            ('FACTS', semantic_facts, 300, False)
        ]

        for label, content, target_tokens, required in sections:
            if not content:
                continue

            section_text = format_section(label, content)
            section_tokens = count_tokens_approx(section_text)

            if required:
                # Ensure required sections fit
                if tokens_used + section_tokens <= self.token_budget:
                    context_parts.append(section_text)
                    tokens_used += section_tokens
                continue

            # Optional sections - fit within remaining budget
            if tokens_used >= self.token_budget:
                memory_logger.log_event("context_skip", {"label": label, "reason": "budget_exhausted"})
                break

            available = self.token_budget - tokens_used
            if section_tokens <= available:
                context_parts.append(section_text)
                tokens_used += section_tokens
                memory_logger.log_event("context_section", {
                    "label": label,
                    "tokens": section_tokens,
                    "status": "full"
                })
            else:
                # Partial inclusion
                if available > 50:  # Minimum useful space
                    truncated = truncate_to_budget(section_text, available)
                    if truncated.strip():
                        context_parts.append(truncated + "\n\n")
                        added_tokens = count_tokens_approx(truncated + "\n\n")
                        tokens_used += added_tokens
                        memory_logger.log_event("context_section", {
                            "label": label,
                            "tokens": added_tokens,
                            "status": "truncated"
                        })
                else:
                    memory_logger.log_event("context_skip", {"label": label, "reason": "budget_too_low"})
                break

        memory_logger.log_event("context_build_complete", {
            "total_tokens": tokens_used,
            "used_cached_memories": cached_memories is not None,
            "has_tool_results": tool_results is not None
        })

        built_context = "".join(context_parts).strip()
        memory_logger.log_event("context_final", {"context": built_context})
        return built_context
