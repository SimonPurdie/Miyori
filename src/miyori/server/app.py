import asyncio
import threading
import time
from fastapi import FastAPI, HTTPException, BackgroundTasks
from sse_starlette.sse import EventSourceResponse
from typing import Optional

from miyori.core.state_manager import StateManager, SystemState
from miyori.core.miyori import MiyoriCore
from miyori.server.sse_manager import SSEManager
from miyori.server.models import InputRequest, StatusResponse, InputResponse
from miyori.interfaces.speech_input import ISpeechInput
from miyori.interfaces.speech_output import ISpeechOutput

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Capture the actual running loop
    global main_loop
    main_loop = asyncio.get_running_loop()
    print(f"Server Lifespan: Captured running event loop {id(main_loop)}")
    yield
    # Cleanup if needed

app = FastAPI(title="Miyori Server", lifespan=lifespan)

# Global state (to be set after initialization)
state_manager: Optional[StateManager] = None
miyori_core: Optional[MiyoriCore] = None
sse_manager: Optional[SSEManager] = None
speech_output: Optional[ISpeechOutput] = None
main_loop: Optional[asyncio.AbstractEventLoop] = None

@app.post("/input", response_model=InputResponse)
async def receive_input(request: InputRequest, background_tasks: BackgroundTasks):
    """
    Accept text input from client.
    """
    return await handle_input_logic(request.text, request.source, background_tasks)

async def handle_input_logic(text: str, source: str, background_tasks: Optional[BackgroundTasks] = None):
    """
    Core logic for handling input, shared between API and voice thread.
    """
    if not state_manager or not miyori_core or not sse_manager or not speech_output:
        raise HTTPException(status_code=500, detail="Server not fully initialized")

    is_text = source == "text"
    
    # IMMEDIATE SYNC: Update interaction time as soon as input is accepted (if voice)
    # This prevents the next voice loop iteration from thinking it needs a wake word
    # if it checks just before process_input starts.
    #if not is_text:
    #    miyori_core.last_interaction_time = time.time()

    if not state_manager.can_accept_input(is_text):
        if not is_text: # Don't raise HTTP exception for background voice tasks
            print(f"Voice input ignored (System busy): {text}")
            return None
        raise HTTPException(status_code=423, detail="System is busy processing")
    
    # Handle interrupt case
    if state_manager.get_state() == SystemState.SPEAKING:
        speech_output.stop()  # Cancel current audio
        state_manager.request_interrupt()
        # Wait a small amount for the interruption to propagate if needed
        await asyncio.sleep(0.1)
    
    # Transition to PROCESSING
    state_manager.transition_to(SystemState.PROCESSING)
    state_manager.clear_interrupt()
    
    # Notify clients of state change
    await sse_manager.broadcast_state(SystemState.PROCESSING)
    
    # Handle the processing
    if background_tasks:
        background_tasks.add_task(process_request, text, source)
    else:
        # If no background_tasks (e.g. from voice thread), we still want it to run in "background" relative to the loop
        # We use main_loop if available to ensure it runs on the server loop
        if main_loop:
            main_loop.create_task(process_request(text, source))
        else:
            asyncio.create_task(process_request(text, source))
    
    return InputResponse(status="accepted", message="Processing input")

async def process_request(text: str, source: str):
    """Background task to process input."""
    response_chunks = []
    
    def on_chunk(chunk: str):
        # 1. Send to SSE clients
        if main_loop:
            asyncio.run_coroutine_threadsafe(sse_manager.broadcast_chunk(chunk), main_loop)
        # 2. Send to speech output
        speech_output.speak(chunk)
        response_chunks.append(chunk)

    # Note: Transition to SPEAKING should happen as soon as we start getting chunks
    # But let's keep it simple for now and rely on the finally block for IDLE.
    
    try:
        # Check for exit via core processing results
        # MiyoriCore.process_input handles the actual command logic
        miyori_core.process_input(text, source, on_chunk)
        
        # Transition to SPEAKING or IDLE
        if response_chunks:
            # Check for goodbye/exit chunks to trigger shutdown
            is_exit = any(word in "".join(response_chunks).lower() for word in ["goodbye", "exit"])
            
            state_manager.transition_to(SystemState.SPEAKING)
            if main_loop:
                asyncio.run_coroutine_threadsafe(sse_manager.broadcast_state(SystemState.SPEAKING), main_loop)
            
            # Wait for speech to complete (simplified)
            # In a later iteration, ISpeechOutput should signal completion
            await asyncio.sleep(0.5) 
            
            if is_exit:
                print("Flow: Exit command detected, shutting down...")
                if main_loop:
                    main_loop.call_later(2, os._exit, 0)
        
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
        state=state_manager.get_state().value,
        needs_wake_word=miyori_core.needs_wake_word()
    )

@app.get("/stream")
async def stream_events():
    """SSE endpoint for real-time updates."""
    if not sse_manager:
        raise HTTPException(status_code=500, detail="Server not fully initialized")
    return EventSourceResponse(sse_manager.event_generator())

import os

def start_voice_thread(speech_input: ISpeechInput):
    """Start background thread for voice input."""
    def voice_loop():
        print("Voice input loop started.")
        while True:
            try:
                # 1. WAIT until system is IDLE before listening
                # This ensures we don't pick up Miyori's own voice or previous turn noise
                while state_manager and state_manager.get_state() != SystemState.IDLE:
                    time.sleep(0.2)

                # 2. Check wake word requirement
                require_wake_word = miyori_core.needs_wake_word()
                
                # 3. Listen for voice input (blocking)
                text = speech_input.listen(require_wake_word=require_wake_word)
                
                if text is None:
                    continue
                
                #print(f"Voice input detected: {text}")

                # 4. Submit to main loop for async processing
                if main_loop:
                    # Use a future to wait for the handoff to complete
                    # This ensures handle_input_logic (and interaction time sync) 
                    # is finished before we loop back and call needs_wake_word()
                    future = asyncio.run_coroutine_threadsafe(
                        handle_input_logic(text, "voice"),
                        main_loop
                    )
                    # Block voice thread until handoff is acknowledged
                    future.result() 
                else:
                    # Fallback if loop not yet captured (startup race)
                    print("Waiting for server loop capture...")
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Error in voice loop: {e}")
                time.sleep(1)
    
    thread = threading.Thread(target=voice_loop, daemon=True, name="VoiceInputThread")
    thread.start()
    return thread
