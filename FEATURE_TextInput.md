# Miyori Stateful Server Implementation Plan

## 1. Architecture Overview

### State Machine
The system operates as a finite state machine with three states:

- **IDLE**: Waiting for input. Accepts both voice and text inputs.
- **PROCESSING**: LLM is generating response and/or executing tools. New inputs are rejected.
- **SPEAKING**: Audio output is playing. Text input can interrupt; voice input is rejected.

### State Transitions
```
IDLE → PROCESSING: Input received (voice or text)
PROCESSING → SPEAKING: LLM generation complete, audio starts
PROCESSING → IDLE: Text-only response complete (no audio)
SPEAKING → IDLE: Audio playback complete
SPEAKING → PROCESSING: Text interrupt received, audio cancelled
```

---

## 2. Project Structure

```
miyori/
├── pyproject.toml           # uv project config with multiple entry points
├── src/
│   └── miyori/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── state_manager.py    # NEW: State machine logic
│       │   ├── miyori_core.py      # MODIFIED: Extracted core logic
│       │   └── tool_registry.py    # EXISTING
│       ├── server/
│       │   ├── __init__.py
│       │   ├── app.py              # NEW: FastAPI server
│       │   ├── models.py           # NEW: Pydantic request/response models
│       │   └── sse_manager.py      # NEW: Server-Sent Events management
│       ├── interfaces/
│       │   ├── speech_input.py     # MODIFIED: Add interrupt awareness
│       │   ├── speech_output.py    # MODIFIED: Add .stop() method
│       │   └── llm_backend.py      # MODIFIED: Add .check_interrupt() method
│       ├── utils/
│       │   ├── config.py           # EXISTING
│       │   └── logger.py           # EXISTING
│       └── scripts/
│           └── run_server.py       # NEW: Server entry point
```

### Entry Points (pyproject.toml)
```toml
[project.scripts]
miyori-server = "miyori.scripts.run_server:main"
# Future: miyori-terminal, miyori-gui
```

---

## 3. Core Components

### 3.1 State Manager (`src/miyori/core/state_manager.py`)

```python
from enum import Enum
from threading import Lock
from typing import Optional

class SystemState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    SPEAKING = "speaking"

class StateManager:
    def __init__(self):
        self._state = SystemState.IDLE
        self._lock = Lock()
        self._interrupt_flag = False
    
    def get_state(self) -> SystemState:
        with self._lock:
            return self._state
    
    def transition_to(self, new_state: SystemState) -> bool:
        """Attempt state transition. Returns True if successful."""
        with self._lock:
            self._state = new_state
            return True
    
    def can_accept_input(self, is_text: bool) -> bool:
        """Check if input can be accepted."""
        with self._lock:
            if self._state == SystemState.IDLE:
                return True
            if is_text and self._state == SystemState.SPEAKING:
                return True  # Text can interrupt speech
            return False
    
    def request_interrupt(self) -> None:
        """Set interrupt flag for LLM to check."""
        with self._lock:
            self._interrupt_flag = True
    
    def clear_interrupt(self) -> None:
        """Clear interrupt flag."""
        with self._lock:
            self._interrupt_flag = False
    
    def should_interrupt(self) -> bool:
        """Check if interrupt was requested."""
        with self._lock:
            return self._interrupt_flag
```

### 3.2 Modified Core Logic (`src/miyori/core/miyori_core.py`)

Extract the processing logic from the run loop:

```python
class MiyoriCore:
    def __init__(self, 
                 speech_output: ISpeechOutput,
                 llm: ILLMBackend,
                 tool_registry: ToolRegistry,
                 state_manager: StateManager):
        self.speech_output = speech_output
        self.llm = llm
        self.tool_registry = tool_registry
        self.state_manager = state_manager
        self.last_interaction_time = 0
        self.active_listen_timeout = Config.data.get("speech_input", {}).get("active_listen_timeout", 30)
    
    def process_input(self, text: str, source: str, on_chunk: Callable[[str], None]) -> None:
        """
        Process user input and generate response.
        
        Args:
            text: User input text
            source: "voice" or "text"
            on_chunk: Callback for streaming response chunks
        """
        # Update interaction time for both voice and text
        self.last_interaction_time = time.time()
        
        # Handle special commands
        if "go to sleep" in text.lower():
            self.speech_output.speak("ok goodnight")
            on_chunk("ok goodnight")
            self.last_interaction_time = 0
            return
        
        if set(['exit', 'goodbye']).intersection(text.lower().split()):
            self.speech_output.speak("Goodbye!")
            on_chunk("Goodbye!")
            return
        
        # Process with tools
        self._handle_with_tools(text, on_chunk)
    
    def _handle_with_tools(self, user_input: str, on_chunk: Callable[[str], None]) -> None:
        """Handle user input with tool support."""
        from src.miyori.utils import logger
        
        def on_tool_call(tool_name: str, parameters: Dict[str, Any]) -> str:
            print(f"🔧 AI requested tool: {tool_name}")
            print(f"   Parameters: {parameters}")
            
            with logger.capture_session() as buffer:
                result = self.tool_registry.execute(tool_name, **parameters)
                logs = buffer.getvalue().strip()
            
            if logs:
                print(f"✓ Tool result: {result[:100]}...")
                return f"TOOL LOGS:\n{logs}\n\nTOOL RESULT:\n{result}"
            else:
                print(f"✓ Tool result: {result[:100]}...")
                return result
        
        tools = self.tool_registry.get_all()
        
        # Pass state manager for interrupt checking
        self.llm.llm_chat(
            prompt=user_input,
            tools=tools,
            on_chunk=on_chunk,
            on_tool_call=on_tool_call,
            interrupt_check=self.state_manager.should_interrupt
        )
    
    def needs_wake_word(self) -> bool:
        """Determine if wake word is required for voice input."""
        if self.last_interaction_time == 0:
            return True
        return (time.time() - self.last_interaction_time) >= self.active_listen_timeout
```

### 3.3 FastAPI Server (`src/miyori/server/app.py`)

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import asyncio
from typing import Optional
from .models import InputRequest, StatusResponse, InputResponse
from ..core.state_manager import StateManager, SystemState
from ..core.miyori_core import MiyoriCore
from .sse_manager import SSEManager

app = FastAPI(title="Miyori Server")

# Global state (initialized in run_server.py)
state_manager: Optional[StateManager] = None
miyori_core: Optional[MiyoriCore] = None
sse_manager: Optional[SSEManager] = None

@app.post("/input", response_model=InputResponse)
async def receive_input(request: InputRequest, background_tasks: BackgroundTasks):
    """
    Accept text input from client.
    
    Returns 423 Locked if system is PROCESSING.
    Interrupts if system is SPEAKING.
    """
    is_text = request.source == "text"
    
    if not state_manager.can_accept_input(is_text):
        raise HTTPException(status_code=423, detail="System is busy processing")
    
    # Handle interrupt case
    if state_manager.get_state() == SystemState.SPEAKING:
        speech_output.stop()  # Cancel current audio
        state_manager.request_interrupt()
    
    # Transition to PROCESSING
    state_manager.transition_to(SystemState.PROCESSING)
    state_manager.clear_interrupt()
    
    # Notify clients of state change
    await sse_manager.broadcast_state(SystemState.PROCESSING)
    
    # Process in background
    background_tasks.add_task(process_request, request.text, request.source)
    
    return InputResponse(status="accepted", message="Processing input")

async def process_request(text: str, source: str):
    """Background task to process input."""
    response_chunks = []
    
    def on_chunk(chunk: str):
        # Send to SSE clients
        asyncio.create_task(sse_manager.broadcast_chunk(chunk))
        # Send to speech output
        speech_output.speak(chunk)
        response_chunks.append(chunk)
    
    # Process input
    miyori_core.process_input(text, source, on_chunk)
    
    # Transition to SPEAKING or IDLE
    if response_chunks:
        state_manager.transition_to(SystemState.SPEAKING)
        await sse_manager.broadcast_state(SystemState.SPEAKING)
        
        # Wait for speech to complete (if not interrupted)
        # This is a simplified version - actual implementation needs coordination
        await asyncio.sleep(0.1)  # Placeholder
    
    # Return to IDLE
    state_manager.transition_to(SystemState.IDLE)
    await sse_manager.broadcast_state(SystemState.IDLE)

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current system status."""
    return StatusResponse(
        state=state_manager.get_state().value,
        needs_wake_word=miyori_core.needs_wake_word()
    )

@app.get("/stream")
async def stream_events():
    """SSE endpoint for real-time updates."""
    return EventSourceResponse(sse_manager.event_generator())
```

### 3.4 SSE Manager (`src/miyori/server/sse_manager.py`)

```python
import asyncio
from typing import Set, AsyncGenerator
from ..core.state_manager import SystemState

class SSEManager:
    def __init__(self):
        self._clients: Set[asyncio.Queue] = set()
    
    async def event_generator(self) -> AsyncGenerator[dict, None]:
        """Generator for SSE events."""
        queue = asyncio.Queue()
        self._clients.add(queue)
        
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self._clients.remove(queue)
    
    async def broadcast_state(self, state: SystemState):
        """Broadcast state change to all clients."""
        event = {
            "event": "state",
            "data": state.value
        }
        await self._broadcast(event)
    
    async def broadcast_chunk(self, chunk: str):
        """Broadcast response chunk to all clients."""
        event = {
            "event": "chunk",
            "data": chunk
        }
        await self._broadcast(event)
    
    async def _broadcast(self, event: dict):
        """Send event to all connected clients."""
        for queue in self._clients:
            await queue.put(event)
```

### 3.5 Request/Response Models (`src/miyori/server/models.py`)

```python
from pydantic import BaseModel
from typing import Literal

class InputRequest(BaseModel):
    text: str
    source: Literal["text", "voice"] = "text"

class InputResponse(BaseModel):
    status: str
    message: str

class StatusResponse(BaseModel):
    state: str
    needs_wake_word: bool
```

---

## 4. Modified Interfaces

### 4.1 Speech Output Interface (`src/miyori/interfaces/speech_output.py`)

Add interrupt capability:

```python
from abc import ABC, abstractmethod

class ISpeechOutput(ABC):
    @abstractmethod
    def speak(self, text: str) -> None:
        """Queue text for speech output."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """
        Stop current speech output immediately.
        
        For now, this is a dummy method. Further calls to speak()
        will queue outputs (existing functionality).
        """
        pass
```

### 4.2 LLM Backend Interface (`src/miyori/interfaces/llm_backend.py`)

Add interrupt checking:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional

class ILLMBackend(ABC):
    @abstractmethod
    def llm_chat(self, 
                 prompt: str,
                 tools: Optional[list] = None,
                 on_chunk: Optional[Callable[[str], None]] = None,
                 on_tool_call: Optional[Callable[[str, Dict[str, Any]], str]] = None,
                 interrupt_check: Optional[Callable[[], bool]] = None) -> None:
        """
        Generate response with streaming and tool support.
        
        Args:
            prompt: User input
            tools: List of available tools
            on_chunk: Callback for streaming response chunks
            on_tool_call: Callback for tool execution
            interrupt_check: Function to check if processing should stop.
                            Should be called between tool execution rounds.
                            If returns True, complete current tool but stop
                            further processing.
        """
        pass
```

---

## 5. Background Voice Input Thread

### 5.1 Voice Thread Implementation (`src/miyori/server/app.py` - continued)

```python
import threading

def start_voice_thread(speech_input: ISpeechInput):
    """Start background thread for voice input."""
    def voice_loop():
        while True:
            # Determine wake word requirement
            require_wake_word = miyori_core.needs_wake_word()
            
            # Listen for voice input
            text = speech_input.listen(require_wake_word=require_wake_word)
            
            if text is None:
                continue
            
            # Check if we can accept voice input
            if not state_manager.can_accept_input(is_text=False):
                print("Voice input ignored: Miyori is busy")
                continue
            
            # Submit to processing
            asyncio.run_coroutine_threadsafe(
                submit_voice_input(text),
                asyncio.get_event_loop()
            )
    
    thread = threading.Thread(target=voice_loop, daemon=True)
    thread.start()
    return thread

async def submit_voice_input(text: str):
    """Submit voice input through the same pipeline as text."""
    request = InputRequest(text=text, source="voice")
    try:
        await receive_input(request, BackgroundTasks())
    except HTTPException as e:
        print(f"Voice input rejected: {e.detail}")
```

---

## 6. Server Entry Point (`src/miyori/scripts/run_server.py`)

```python
import uvicorn
from src.miyori.server.app import app, state_manager, miyori_core, sse_manager, start_voice_thread
from src.miyori.core.state_manager import StateManager
from src.miyori.core.miyori_core import MiyoriCore
from src.miyori.server.sse_manager import SSEManager
# Import your implementations
from src.implementations.speech_input_impl import SpeechInputImpl
from src.implementations.speech_output_impl import SpeechOutputImpl
from src.implementations.llm_backend_impl import LLMBackendImpl
from src.miyori.core.tool_registry import ToolRegistry

def main():
    """Initialize and start the Miyori server."""
    
    # Initialize components
    global state_manager, miyori_core, sse_manager
    
    state_manager = StateManager()
    sse_manager = SSEManager()
    
    speech_input = SpeechInputImpl()
    speech_output = SpeechOutputImpl()
    llm = LLMBackendImpl()
    tool_registry = ToolRegistry()
    
    miyori_core = MiyoriCore(
        speech_output=speech_output,
        llm=llm,
        tool_registry=tool_registry,
        state_manager=state_manager
    )
    
    # Start voice input thread
    start_voice_thread(speech_input)
    
    # Start server
    print("🚀 Miyori Server starting...")
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
```

---

## 7. Client-Server Protocol

### 7.1 HTTP Endpoints

| Method | Endpoint | Purpose | Response |
|--------|----------|---------|----------|
| POST | `/input` | Submit text input | `{"status": "accepted"}` or 423 Locked |
| GET | `/status` | Query system state | `{"state": "idle", "needs_wake_word": false}` |
| GET | `/stream` | SSE connection | Real-time events |

### 7.2 SSE Event Types

**State Change Event:**
```
event: state
data: processing
```

**Response Chunk Event:**
```
event: chunk
data: Here is the answer to your question.
```

### 7.3 Client Example (Pseudo-code)

```python
import requests
from sseclient import SSEClient

# Connect to SSE stream
messages = SSEClient('http://127.0.0.1:8000/stream')

for msg in messages:
    if msg.event == 'state':
        print(f"State: {msg.data}")
    elif msg.event == 'chunk':
        print(f"Response: {msg.data}")

# Submit text input
response = requests.post('http://127.0.0.1:8000/input', 
                        json={"text": "Hello", "source": "text"})

if response.status_code == 423:
    print("System is busy!")
```

---

## 8. Execution Flow

### 8.1 Startup Sequence
1. Server initializes `StateManager`, `MiyoriCore`, `SSEManager`
2. Background voice thread starts, begins listening loop
3. FastAPI server opens on port 8000
4. System enters IDLE state

### 8.2 Voice Input Flow
1. Voice thread detects speech → calls `listen()`
2. Returns text → checks `state_manager.can_accept_input(False)`
3. If IDLE → submits to `/input` endpoint internally
4. If not IDLE → prints "Voice input ignored" and continues listening

### 8.3 Text Input Flow
1. Client POSTs to `/input` with text
2. Server checks state:
   - IDLE → Accept, transition to PROCESSING
   - SPEAKING → Accept, call `speech_output.stop()`, set interrupt flag, transition to PROCESSING
   - PROCESSING → Return 423 Locked
3. Background task processes input
4. LLM generates response, streams chunks via SSE and speech output
5. Transition to SPEAKING → wait for audio completion → transition to IDLE

### 8.4 Interrupt Handling
1. Text input arrives during SPEAKING
2. Server calls `speech_output.stop()`
3. Server sets `state_manager.request_interrupt()`
4. Transition to PROCESSING
5. LLM checks `interrupt_check()` between tool execution rounds
6. Current tool completes, but no further turns are processed
7. Response generation stops, system returns to IDLE

---

## 9. Migration Path

### Phase 1: Core Refactoring
- Extract `MiyoriCore` logic from `miyori.py` run loop
- Implement `StateManager`
- Add `.stop()` to `ISpeechOutput` (dummy implementation)
- Add `interrupt_check` parameter to `ILLMBackend.llm_chat()`

### Phase 2: Server Implementation
- Implement FastAPI app with `/input`, `/status`, `/stream`
- Implement `SSEManager` for event broadcasting
- Create `run_server.py` entry point

### Phase 3: Voice Thread Integration
- Move voice listening to background thread
- Implement voice submission logic with state checking

### Phase 4: Testing & Client Development
- Test state transitions with multiple clients
- Implement terminal and/or GUI clients
- Test interrupt behavior during speech and tool execution

---

## 10. Key Design Decisions

1. **SSE over WebSockets**: Simpler client implementation, HTTP-compatible, sufficient for one-way streaming with occasional status updates.

2. **Background Voice Thread**: Prevents voice input from blocking the server, allows concurrent listening while processing text inputs.

3. **State Machine**: Clear state boundaries prevent race conditions and make the system's behavior predictable.

4. **Interrupt Flag**: Allows LLM implementations to gracefully handle interrupts between tool execution rounds without complex cancellation logic.

5. **Multiple Client Support**: Stateless HTTP endpoints allow multiple clients to monitor state via SSE and submit inputs when available.

6. **Speech Output Queuing**: Existing behavior is preserved; `.stop()` is a placeholder for future implementation.
