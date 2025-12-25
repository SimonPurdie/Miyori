#!/usr/bin/env python3
"""
Dummy test server for Miyori API testing.
Mocks the core functionality without hitting real speech/LLM endpoints.
"""

import asyncio
import random
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks
from sse_starlette.sse import EventSourceResponse
from typing import Optional, Set
from pydantic import BaseModel

from contextlib import asynccontextmanager

# Mock models (simplified from server/models.py)
class InputRequest(BaseModel):
    text: str
    source: str = "text"

class StatusResponse(BaseModel):
    state: str
    needs_wake_word: bool

class InputResponse(BaseModel):
    status: str
    message: str

# Mock system states
class SystemState:
    IDLE = "idle"
    PROCESSING = "processing"
    SPEAKING = "speaking"

# Mock SSE Manager
class MockSSEManager:
    def __init__(self):
        self._clients: Set[asyncio.Queue] = set()

    async def event_generator(self):
        """Generator for SSE events - matches real SSEManager pattern."""
        queue = asyncio.Queue()
        self._clients.add(queue)

        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self._clients.remove(queue)

    async def broadcast_state(self, state: str):
        """Broadcast state change to all connected clients."""
        print(f"[MOCK SSE] Broadcasting state: {state}")
        event = {
            "event": "state",
            "data": state
        }
        await self._broadcast(event)

    async def broadcast_chunk(self, chunk: str):
        """Broadcast response chunk to all connected clients."""
        print(f"[MOCK SSE] Broadcasting chunk: {chunk}")
        event = {
            "event": "chunk",
            "data": chunk
        }
        await self._broadcast(event)

    async def _broadcast(self, event: dict):
        """Send event to all connected clients."""
        # Create tasks for all clients to broadcast concurrently
        tasks = []
        for queue in list(self._clients):  # Use list to avoid modification during iteration
            tasks.append(asyncio.create_task(queue.put(event)))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

class MockStateManager:
    def __init__(self):
        self.current_state = SystemState.IDLE
        self.interrupt_requested = False

    def get_state(self) -> str:
        return self.current_state

    def transition_to(self, state: str):
        print(f"[MOCK STATE] {self.current_state} -> {state}")
        self.current_state = state

    def can_accept_input(self, is_text: bool) -> bool:
        # Mock logic: accept input if idle or if text input (allows interruption)
        return self.current_state == SystemState.IDLE or is_text

    def request_interrupt(self):
        self.interrupt_requested = True
        print("[MOCK STATE] Interrupt requested")

    def clear_interrupt(self):
        self.interrupt_requested = False

class MockMiyoriCore:
    def __init__(self):
        self.needs_wake_word_flag = True

    def needs_wake_word(self) -> bool:
        return self.needs_wake_word_flag

    def process_input(self, text: str, source: str, on_chunk_callback):
        """Mock processing that simulates LLM response with delays"""
        print(f"[MOCK CORE] Processing input: '{text}' from {source}")

        # Simulate processing delay
        time.sleep(random.uniform(0.5, 1.5))

        # Mock responses based on input
        responses = {
            "hello": ["Hello! How can I help you today?", "Hi there! Nice to meet you."],
            "how are you": ["I'm doing well, thank you for asking!", "I'm functioning optimally!"],
            "goodbye": ["Goodbye! Have a great day!", "Farewell! Until next time."],
            "exit": ["Shutting down now.", "Goodbye!"],
        }

        # Find matching response or use default
        response_text = "I understand you said: " + text
        for key, options in responses.items():
            if key in text.lower():
                response_text = random.choice(options)
                break

        # Simulate streaming response chunks
        words = response_text.split()
        for i, word in enumerate(words):
            chunk = word + " " if i < len(words) - 1 else word
            on_chunk_callback(chunk)
            time.sleep(random.uniform(0.1, 0.3))  # Simulate typing/streaming delay

        print(f"[MOCK CORE] Finished processing input")

# Mock speech output (no-op)
class MockSpeechOutput:
    def speak(self, text: str):
        print(f"[MOCK SPEECH] Speaking: {text}")

    def stop(self):
        print("[MOCK SPEECH] Stopped speaking")

# Global instances
state_manager: Optional[MockStateManager] = None
miyori_core: Optional[MockMiyoriCore] = None
sse_manager: Optional[MockSSEManager] = None
speech_output: Optional[MockSpeechOutput] = None
main_loop: Optional[asyncio.AbstractEventLoop] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_loop, state_manager, miyori_core, sse_manager, speech_output

    # Initialize components
    state_manager = MockStateManager()
    miyori_core = MockMiyoriCore()
    sse_manager = MockSSEManager()
    speech_output = MockSpeechOutput()

    # Get event loop
    main_loop = asyncio.get_running_loop()
    print(f"[MOCK SERVER] Started with event loop {id(main_loop)}")
    print("[MOCK SERVER] All components initialized")

    yield
    print("[MOCK SERVER] Shutting down")

app = FastAPI(title="Miyori Mock Server", lifespan=lifespan)

@app.post("/input", response_model=InputResponse)
async def receive_input(request: InputRequest, background_tasks: BackgroundTasks):
    """
    Accept text input from client (mocked).
    """
    return await handle_input_logic(request.text, request.source, background_tasks)

async def handle_input_logic(text: str, source: str, background_tasks: Optional[BackgroundTasks] = None):
    """
    Core logic for handling input (mocked).
    """
    if not state_manager or not miyori_core or not sse_manager or not speech_output:
        raise HTTPException(status_code=500, detail="Server not fully initialized")

    is_text = source == "text"

    if not state_manager.can_accept_input(is_text):
        if not is_text:
            print(f"[MOCK] Voice input ignored (System busy): {text}")
            return None
        raise HTTPException(status_code=423, detail="System is busy processing")

    # Handle interrupt case
    if state_manager.get_state() == SystemState.SPEAKING:
        speech_output.stop()
        state_manager.request_interrupt()
        await asyncio.sleep(0.1)

    # Transition to PROCESSING
    state_manager.transition_to(SystemState.PROCESSING)
    await sse_manager.broadcast_state(SystemState.PROCESSING)

    # Process in background
    if background_tasks:
        background_tasks.add_task(process_request, text, source)
    else:
        if main_loop:
            main_loop.create_task(process_request(text, source))
        else:
            asyncio.create_task(process_request(text, source))

    return InputResponse(status="accepted", message="Processing input")

async def process_request(text: str, source: str):
    """Mock background task to process input."""
    response_chunks = []

    def on_chunk(chunk: str):
        # Send to SSE clients (mocked)
        if main_loop:
            asyncio.run_coroutine_threadsafe(sse_manager.broadcast_chunk(chunk), main_loop)
        # Send to speech output (mocked)
        speech_output.speak(chunk)
        response_chunks.append(chunk)

    try:
        # Mock processing
        miyori_core.process_input(text, source, on_chunk)

        # Transition to SPEAKING if we have chunks
        if response_chunks:
            # Check for exit commands
            full_response = "".join(response_chunks).lower()
            is_exit = any(word in full_response for word in ["goodbye", "exit", "shutting down"])

            state_manager.transition_to(SystemState.SPEAKING)
            if main_loop:
                asyncio.run_coroutine_threadsafe(sse_manager.broadcast_state(SystemState.SPEAKING), main_loop)

            # Simulate speaking time
            await asyncio.sleep(0.5)

            if is_exit:
                print("[MOCK SERVER] Exit command detected, shutting down...")
                # In real server this would trigger shutdown

    finally:
        # Return to IDLE
        state_manager.transition_to(SystemState.IDLE)
        if main_loop:
            asyncio.run_coroutine_threadsafe(sse_manager.broadcast_state(SystemState.IDLE), main_loop)

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current system status."""
    if not state_manager or not miyori_core:
        raise HTTPException(status_code=500, detail="Server not fully initialized")

    return StatusResponse(
        state=state_manager.get_state(),
        needs_wake_word=miyori_core.needs_wake_word()
    )

@app.get("/stream")
async def stream_events():
    """SSE endpoint for real-time updates."""
    if not sse_manager:
        raise HTTPException(status_code=500, detail="Server not fully initialized")

    return EventSourceResponse(sse_manager.event_generator())

if __name__ == "__main__":
    import uvicorn

    print("[MOCK SERVER] Starting test server on http://localhost:8069")

    uvicorn.run(app, host="localhost", port=8069)
