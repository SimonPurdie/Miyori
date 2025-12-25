## Critical Loose Ends

- [ ] Speech Output Lifecycle & State Coordination
- [ ] LLM Backend Interrupt Implementation
- [ ] Voice Thread Lifecycle Management
- [ ] Wake Word State Synchronization
- [ ] SSE Connection Management

### 1. **Speech Output Lifecycle & State Coordination**
**Current Issue**: The plan has `await asyncio.sleep(0.1)` as a placeholder for detecting when speech completes.

**What's needed**:
- The `ISpeechOutput` interface needs a way to signal completion (callback or async method)
- The server needs to know when SPEAKING → IDLE transition should happen
- The `.stop()` method needs actual implementation to cancel queued audio

**Impact**: Without this, the state machine can't transition properly from SPEAKING to IDLE.

---

### 2. **LLM Backend Interrupt Implementation**
**Current Issue**: `interrupt_check` parameter was added to the interface but implementations need to actually use it.

**What's needed**:
- LLM implementations must check `interrupt_check()` between tool execution rounds
- Need to decide: Should in-progress tool execution be cancellable, or only prevent *next* tool call?
- Handle the case where LLM wants to make multiple tool calls after an interrupt is set

**Impact**: Interrupts won't actually work until this is implemented in your LLM backend.

---

### 3. **Voice Thread Lifecycle Management**
**Current Issue**: Background voice thread starts but has no clean shutdown mechanism.

**What's needed**:
- Graceful shutdown signal for the voice thread
- Join thread on server shutdown to prevent orphaned processes
- Handle the case where `speech_input.listen()` is blocking during shutdown

**Impact**: Server won't shut down cleanly; may leave hanging threads.

---

## Important But Not Blocking

### 4. **Wake Word State Synchronization**
**Current Issue**: Voice thread calls `miyori_core.needs_wake_word()` but doesn't account for text inputs refreshing the timeout.

**What's needed**:
- The voice thread is already checking this on each loop iteration, which should work
- However, if `listen()` is a long-blocking call, it won't see timeout refreshes until it completes
- May need to pass timeout info differently or make `listen()` non-blocking

**Impact**: Minor UX issue - wake word requirement may feel inconsistent after text inputs.

---

### 5. **SSE Connection Management**
**Current Issue**: No cleanup of disconnected SSE clients in `SSEManager`.

**What's needed**:
- Detect and remove stale client queues
- Handle exceptions in the event generator
- Prevent memory leaks from abandoned connections

**Impact**: Server memory will grow over time with abandoned client connections.

---

## My Recommendation for Priority Order

1. **First**: Fix speech output lifecycle (#1) - this blocks proper state machine operation
2. **Second**: Implement LLM interrupt checking (#2) - core feature won't work without it
3. **Third**: Add basic error handling (#6) - prevents crashes
4. **Fourth**: Fix voice thread shutdown (#3) - important for clean operation
5. **Fifth**: SSE cleanup (#5) - prevents memory leaks