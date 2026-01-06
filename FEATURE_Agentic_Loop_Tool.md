# Agentic Behavior Tool - Specification Document

## 1. Overview

### Purpose
Transform the AI from a single-turn prompt-response system into an autonomous agent capable of pursuing multi-step objectives without user intervention between steps.

### Core Concept
The agent autonomously decides when a user prompt requires multi-step agentic behavior. When needed, the agent invokes its agentic tool, providing both the original user prompt and its own formulated objective. It then enters a loop where it uses available tools (including terminal access) and continues iterating until it either completes the objective or determines completion is impossible.

## 2. Architecture Components

### 2.1 Agentic Loop Harness
The execution environment that manages the agent's iterative behavior. Responsibilities:
- Receive agent-initiated agentic invocation with objective
- Execute iterations until termination
- Track iteration count and enforce limits
- Monitor for user interrupt signals
- Manage context between iterations

### 2.2 State Tracker
Maintains environmental state across iterations for terminal-aware context:
- Original user prompt
- Agent-formulated objective
- Current working directory
- Files modified during session
- Last command executed and its output
- Iteration counter

### 2.3 Completion Tool
A special tool the agent calls to signal objective completion. Parameters:
- `result`: Summary of what was accomplished
- `status`: success/failure/partial

Absence of this tool call indicates the agent intends to continue.

### 2.4 Agentic Invocation Tool
Tool the agent calls to enter agentic mode. Parameters:
- `user_prompt`: The original user request
- `objective`: The agent's formulated goal for autonomous pursuit

This tool initiates the agentic loop with the agent's own objective framing.

## 3. Context Management

### 3.1 State Injection Point
Insert agentic state metadata after the existing per-turn context section that appears directly before the user's most recent prompt.

### 3.2 State Format
Include in context:
- An indication that the agent is currently in an agentic loop
- The original user prompt
- The agent-formulated objective
- Current iteration number and maximum
- If a terminal instance has been opened:
  - Working directory (if a terminal instance is open)
  - Last command executed (truncated to ~500 chars of output)
- List of files modified this session
- Any other relevant variables impacting process state

### 3.3 History Management
The agent's prior turns already contain full command and tool use history. The state block should be concise and focus on **current** environmental context, not duplicate conversation history.

## 4. Loop Mechanics

### 4.1 Execution Flow
1. Agent receives user prompt in standard mode
2. Agent decides agentic behavior needed, invokes agentic tool with objective
3. System enters agentic mode, inserting state context
4. Agent generates response with tool calls
5. If agent calls `complete()` tool: exit loop with result
6. If agent calls other tools: execute them, update state
7. Check for user interrupt signal
8. Increment iteration counter
9. Check termination conditions
10. If not terminated: repeat from step 4

### 4.2 Agent Decision Making
The agent autonomously decides each iteration whether to:
- Execute tool calls (including terminal commands)
- Call `complete()` to terminate successfully
- Provide reasoning about progress

No forced planning phases. Agent determines its own strategy per-turn based on task requirements.

### 4.3 Iteration Limits
Hard maximum iteration count (recommended: 25) to prevent runaway loops. When limit reached:
- Loop terminates automatically
- Agent's last response is returned to user
- Output indicates that iteration limit was reached

# 6. Terminal Tool Integration

## 6.1 Terminal Tool Design

**Tool signature:** `terminal(command=None, persistent=False, close=False)`

**Execution modes:**
- Ephemeral (default): One-off execution, no state retention
- Persistent (`persistent=True`): Maintains working directory and environment across calls
- Close (`close=True`): Closes persistent session

**Rules:**
- Only one persistent session can exist
- First `persistent=True` call opens session automatically
- Ephemeral commands only work when no persistent session is open

## 6.2 Command Approval System

**Approval required for:**
- File deletion: `rm -rf`, `rm -r`
- Privilege escalation: `sudo`, `su`
- Network operations to external hosts: `curl`, `wget`, `scp`, `rsync`
- System modifications: `chmod 777`, `mkfs`, `dd`
- File overwrites to system paths

## 6.3 Approval Flow

When approval-required command detected:
1. Pause loop, output: "Approve command: [command]? (yes/no)"
2. Wait for user input
3. Parse yes/no response
4. Execute or reject, return result to agent
5. Input containing "stop" triggers interrupt

## 6.4 State Tracking

**Persistent session open:**
- Working directory
- Last command and output (truncated ~500 chars)
- Exit code

**Session closed:**
- Minimal state indication

## 6.5 Error Handling

Agent receives exit codes, stderr, and stdout. Timeouts kill process and return error. Agent decides whether to retry, adapt, or complete with failure.

## 7. Termination Conditions

Agent loop terminates when any condition met:

### 7.1 Success Termination
- Agent calls `complete()` tool with success status
- Return agent's result to user

### 7.2 Iteration Limit
- Maximum iterations reached
- Return partial results with indication of incompletion

### 7.3 User Interrupt
- User inputs "stop" (case-insensitive, can be part of longer input)
- Current operation completes, then loop exits
- Return state at interruption point

### 7.4 Unrecoverable Error
- Critical system failure
- Repeated tool execution failures
- Agent explicitly declares objective impossible

## 8. User Interaction Patterns

### 8.1 Interrupt Mechanism
- Monitor user input stream during agentic execution
- Input containing "stop" triggers interrupt flag
- Check interrupt flag before each iteration
- Graceful shutdown: completes current step, then exits
- Does not abort mid-tool-execution

### 8.2 Progress Output
Agent should output to terminal:
- Current iteration number at start of each iteration
- What action it's taking (before tool execution)
- Tool execution results
- Reasoning about progress when relevant

### 8.3 Approval Prompts
For terminal commands requiring approval:
- Output: "Approve command: [full command]? (yes/no)"
- Wait for user text input
- Parse response for affirmative/negative
- Continue or reject based on response

### 8.4 Output Format
Terminal output should clearly distinguish:
- Agent reasoning/commentary
- Tool execution output
- System messages (iteration counts, approvals, termination)
- User input prompts

## 9. Safety Considerations

### 9.1 Resource Limits
- Maximum iterations (prevents infinite loops)
- Timeout per tool execution (prevents hanging)
- Token budget per iteration (prevents context explosion)

### 9.2 Destructive Operation Handling
- No automatic rollback/backup system
- User explicitly approves dangerous operations
- Agent must handle failures gracefully
- Document assumes user accepts potential harm

### 9.3 Command Execution Constraints
For terminal tool specifically:
- Execute in user's actual environment (not sandboxed)
- Commands inherit user's permissions
- No automatic sudo/elevation
- Standard timeout enforcement (60-300 seconds per command)

