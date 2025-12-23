import sys
import os
import uuid
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from miyori.implementations.llm.google_ai_backend import GoogleAIBackend
from miyori.core.tool_registry import ToolRegistry
from miyori.tools.web_search import web_search_tool
from miyori.tools.file_ops import file_ops_tool
from miyori.utils import logger
from miyori.utils.config import Config
Config.load()

def run_test_case(backend, registry, prompt):
    print(f"\n[USER]: {prompt}")
    
    # Store complete response for debugging/validation
    miyori_response = []
    
    def on_chunk(text: str):
        miyori_response.append(text)
        print(text, end="", flush=True)
        
    def on_tool_call(name: str, args: Dict[str, Any]) -> str:
        print(f"\n[INTERNAL] Executing tool: {name} with args: {args}")
        
        with logger.capture_session() as buffer:
            result = registry.execute(name, **args)
            logs = buffer.getvalue().strip()
            
        if logs:
            print(f"[INTERNAL] Captured logs:\n{logs}")
            return f"TOOL LOGS:\n{logs}\n\nTOOL RESULT:\n{result}"
        return result

    backend.llm_chat(
        prompt=prompt,
        tools=registry.get_all(),
        on_chunk=on_chunk,
        on_tool_call=on_tool_call
    )
    print("\n" + "-"*50)
    return "".join(miyori_response)

def test_programmatic_verification():
    print("🚀 Starting Programmatic Truth Verification Test...")
    
    # Initialize logging
    logger.setup_logging()
    
    # Setup Registry
    registry = ToolRegistry()
    registry.register(web_search_tool)
    registry.register(file_ops_tool)
    
    # Initialize Backend
    backend = GoogleAIBackend()
    
    # --- PHASE 0: SECRET INJECTION ---
    secret_key = str(uuid.uuid4())
    project_root = Path(__file__).parent.parent
    workspace_dir = project_root / "workspace"
    workspace_dir.mkdir(exist_ok=True)
    truth_file = workspace_dir / "truth.txt"
    verification_file = workspace_dir / "verification.txt"
    
    # Clean up old verification file
    if verification_file.exists():
        verification_file.unlink()
        
    print(f"🔑 [SYSTEM] Generating secret key...")
    with open(truth_file, "w", encoding="utf-8") as f:
        f.write(secret_key)
    print(f"✅ [SYSTEM] Secret injected into {truth_file.name}")

    # --- PHASE 1: WEB SEARCH & LOG MEMORY ---
    print("\n--- Phase 1: Web Search & Log Memory ---")
    run_test_case(
        backend, 
        registry, 
        "Search for the current population of Tokyo. After the search, tell me exactly which internal log message you saw starting with a magnifying glass emoji."
    )
    
    # --- PHASE 2: PROGRAMMATIC TRUTH VERIFICATION ---
    print("\n--- Phase 2: Programmatic Truth Verification ---")
    prompt_truth = (
        "There is a file in the workspace named 'truth.txt' that I created manually. "
        "Use your tools to read it and tell me the exact content. "
        "Then, write that same content into a new file named 'verification.txt' in the workspace."
    )
    
    run_test_case(backend, registry, prompt_truth)

    # --- PHASE 3: VALIDATION LOGIC ---
    print("\n--- Phase 3: Validation Logic ---")
    time.sleep(1) # Ensure file handles are closed
    
    if not verification_file.exists():
        print("❌ [FAIL] verification.txt was never created.")
    else:
        with open(verification_file, "r", encoding="utf-8") as f:
            written_content = f.read().strip()
        
        print(f"Expected: {secret_key}")
        print(f"Actually Written: {written_content}")
        
        if written_content == secret_key:
            print("🏆 [SUCCESS] Tool-use is AUTHENTIC. Secret key matches.")
        else:
            print("🚫 [FAIL] Hallucination detected or incorrect data written.")

    print("\n✅ Integrated test sequence complete.")

if __name__ == "__main__":
    test_programmatic_verification()
