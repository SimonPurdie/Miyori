# Miyori - Phase 1 Architecture

## Project Structure
```
miyori/
├── main.py
├── config.json
├── config.json.example          # Template config file
├── requirements.txt
├── AGENTS.md                    # AI agent instructions (minimal)
├── ARCHITECTURE.md              # This file
└── src/
    ├── interfaces/
    │   ├── speech_input.py      # ISpeechInput interface
    │   ├── speech_output.py     # ISpeechOutput interface
    │   └── llm_backend.py       # ILLMBackend interface
    ├── implementations/
    │   ├── speech/
    │   │   ├── SPEECH_INPUT_PLAN.md     # How to implement speech input
    │   │   └── google_speech_input.py
    │   ├── tts/
    │   │   ├── TTS_OUTPUT_PLAN.md       # How to implement TTS
    │   │   └── pyttsx_output.py
    │   └── llm/
    │       ├── LLM_BACKEND_PLAN.md      # How to implement LLM
    │       └── google_ai_backend.py
    └── core/
        ├── ASSISTANT_PLAN.md            # How to implement assistant
        └── assistant.py
```

## Core Interfaces

### ISpeechInput
```python
def listen(self) -> str | None:
    """Listen and return transcribed text"""
```

### ISpeechOutput
```python
def speak(self, text: str) -> None:
    """Convert text to speech"""
```

### ILLMBackend
```python
def generate(self, prompt: str) -> str:
    """Generate AI response"""
```

## Design Rules

1. **All implementations read from config.json** (in root)
2. **No error handling this phase unless explicitly stated**
3. **Use print() for output** (no logging framework)
4. **Full type hints everywhere**
5. **Each implementation folder contains a file ending in _PLAN.md with specific instructions**
6. **main.py just wires components together**

## Implementation Order

1. **Interfaces are already defined** in `src/interfaces/` (speech_input.py, speech_output.py, llm_backend.py)
2. Copy `config.json.example` to `config.json` and add your Google AI API key
3. Implement GoogleSpeechInput (see `src/implementations/speech/SPEECH_INPUT_PLAN.md`)
4. Implement PyttsxOutput (see `src/implementations/tts/TTS_OUTPUT_PLAN.md`)
5. Implement GoogleAIBackend (see `src/implementations/llm/LLM_BACKEND_PLAN.md`)
6. Build VoiceAssistant (see `src/core/ASSISTANT_PLAN.md`)
7. Wire in main.py

## Phase 1 Goal
Speak → AI responds → Hear response (looping)