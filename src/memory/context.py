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

    elif label == 'RECENT' or label == 'RELEVANT':
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
    def __init__(self, store, episodic_manager, token_budget=1000):
        self.store = store
        self.episodic_manager = episodic_manager
        self.token_budget = token_budget

    def build_context(self, user_message: str) -> str:
        """Assemble context with strict priority ordering."""       
        # Recent high importance (last 7 days)
        # For Phase 1, we'll just get all and filter in Python
        all_episodes = self.store.search_episodes([0.0]*768, limit=100, status='active')
        seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
        recent_important = [e for e in all_episodes if e['timestamp'] >= seven_days_ago and e['importance'] >= 0.7]        
        semantic_facts = self.store.get_semantic_facts(limit=10) # Facts
        relevant_episodes = self.episodic_manager.retrieve_relevant(user_message, limit=5) # Semantic Search
        
        # 2. Priority Allocation
        # (Label, Content, Target Tokens, Required)
        sections = [
            ('RECENT', recent_important, 400, False),
            ('FACTS', semantic_facts, 300, False),
            ('RELEVANT', relevant_episodes, 300, False)
        ]
        
        context_parts = []
        tokens_used = 0
        
        for label, content, target_tokens, required in sections:
            if not content:
                continue
                
            section_text = format_section(label, content)
            section_tokens = count_tokens_approx(section_text)
            
            if required:
                context_parts.append(section_text)
                tokens_used += section_tokens
                continue
            
            # Optional sections
            remaining = self.token_budget - tokens_used
            if remaining <= 0:
                from src.utils.memory_logger import memory_logger
                memory_logger.log_event("context_skip", {"label": label, "reason": "budget_exhausted"})
                break
                
            if section_tokens <= remaining:
                context_parts.append(section_text)
                tokens_used += section_tokens
                from src.utils.memory_logger import memory_logger
                memory_logger.log_event("context_section", {"label": label, "tokens": section_tokens, "status": "full"})
            else:
                # Partial inclusion
                if remaining > 50: # Minimum useful space
                    truncated = truncate_to_budget(section_text, remaining)
                    if truncated.strip():
                        context_parts.append(truncated + "\n\n")
                        added_tokens = count_tokens_approx(truncated + "\n\n")
                        tokens_used += added_tokens
                        from src.utils.memory_logger import memory_logger
                        memory_logger.log_event("context_section", {"label": label, "tokens": added_tokens, "status": "truncated"})
                else:
                    from src.utils.memory_logger import memory_logger
                    memory_logger.log_event("context_skip", {"label": label, "reason": "budget_too_low"})
                break
        
        from src.utils.memory_logger import memory_logger
        memory_logger.log_event("context_build_complete", {"total_tokens": tokens_used})
        built_context = "".join(context_parts).strip()
        memory_logger.log_event("context_final", {"context": built_context})
        return built_context
