import uvicorn
from miyori.implementations.speech.porcupine_cobra_vosk import PorcupineCobraVosk
from miyori.implementations.tts.kokoro_tts_output import KokoroTTSOutput
from miyori.implementations.llm.google_ai_backend import GoogleAIBackend
from miyori.core.miyori import MiyoriCore
from miyori.core.tool_registry import ToolRegistry
from miyori.core.state_manager import StateManager
from miyori.server.sse_manager import SSEManager
from miyori.server import app as server_module
from miyori.tools.web_search import web_search_tool
from miyori.tools.file_ops import file_ops_tool
from miyori.tools.memory_search import create_memory_search_tool
from miyori.utils.logger import setup_logging
from miyori.utils.config import Config

def main():
    Config.load()
    setup_logging()
    
    # Initialize Core Components
    state_manager = StateManager()
    sse_manager = SSEManager()
    
    speech_input = PorcupineCobraVosk()
    speech_output = KokoroTTSOutput()
    llm_backend = GoogleAIBackend()
    
    # Setup tools
    tool_registry = ToolRegistry()
    tools_config = Config.data.get("tools", {})

    # Create memory search tool dependencies
    memory_search_tool = None
    if hasattr(llm_backend, 'memory_retriever') and hasattr(llm_backend, 'embedding_service'):
        memory_search_tool = create_memory_search_tool(
            llm_backend.memory_retriever,
            llm_backend.embedding_service
        )

    if tools_config.get("enabled", False):
        if tools_config.get("web_search", {}).get("enabled", False):
            tool_registry.register(web_search_tool)
        if tools_config.get("file_ops", {}).get("enabled", False):
            tool_registry.register(file_ops_tool)
        if memory_search_tool and tools_config.get("memory_search", {}).get("enabled", True):
            tool_registry.register(memory_search_tool)
    
    # Initialize Core
    miyori_core = MiyoriCore(
        speech_output=speech_output, 
        llm=llm_backend,
        state_manager=state_manager,
        tool_registry=tool_registry
    )
    
    # Configure Server Module Globals
    server_module.state_manager = state_manager
    server_module.miyori_core = miyori_core
    server_module.sse_manager = sse_manager
    server_module.speech_output = speech_output
    
    # Start Voice Input Thread
    server_module.start_voice_thread(speech_input)
    
    # Start FastAPI Server
    server_config = Config.data.get("server", {})
    port = server_config.get("port", 8000)
    host = server_config.get("host", "127.0.0.1")
    
    print(f"🚀 Miyori Server starting on http://{host}:{port}")
    uvicorn.run(server_module.app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
