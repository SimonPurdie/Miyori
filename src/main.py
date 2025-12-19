from src.implementations.speech.porcupine_speech_input import PorcupineSpeechInput
from src.implementations.tts.kokoro_tts_output import KokoroTTSOutput
from src.implementations.llm.google_ai_backend import GoogleAIBackend
from src.core.assistant import VoiceAssistant
from src.core.tool_registry import ToolRegistry
from src.tools.web_search import web_search_tool
from src.tools.file_ops import file_ops_tool
import json
from pathlib import Path
from src.utils.logger import setup_logging
from src.utils.config import Config

def main():
    setup_logging()
            
    speech_input = PorcupineSpeechInput()
    speech_output = KokoroTTSOutput()
    llm_backend = GoogleAIBackend()
    
    # Setup tools
    tool_registry = ToolRegistry()
    tools_config = Config.data.get("tools", {})
    if tools_config.get("enabled", False):
        if tools_config.get("web_search", {}).get("enabled", False):
            tool_registry.register(web_search_tool)
        if tools_config.get("file_ops", {}).get("enabled", False):
            tool_registry.register(file_ops_tool)
    
    assistant = VoiceAssistant(
        speech_input=speech_input, 
        speech_output=speech_output, 
        llm=llm_backend,
        tool_registry=tool_registry
    )
    assistant.run()

if __name__ == "__main__":
    main()
