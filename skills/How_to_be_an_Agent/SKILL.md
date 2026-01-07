# Skill: How to be an Agent

This skill defines the protocol for Miyori's multi-step autonomous tasks, using external files as a "working memory" to maintain transparency and prevent context bloat.

## 1. Engaging the Agentic Loop
Before starting any complex task that requires more than a few steps, you **must** explicitly call the `agentic_loop` tool. This expands your iteration limit and enables the specialized planning workflow.

## 2. The Planning Protocol (Manus-Lite)
Once the loop is engaged, externalize your state into the `workspace/` directory immediately.

### Mandatory: `task_plan.md`
Created at the start of every loop. It must contain:
- **Goal**: A clear definition of the finished state.
- **Phases**: Checkboxes for high-level progress.
- **Errors & Decisions**: A log of technical pivots and failed attempts to prevent looping on mistakes.
- **Status**: A one-line description of the current action.

### Optional: `notes.md`
Use for research or technical documentation to avoid "stuffing" your conversational context with large blocks of raw data.

## 3. Iteration Workflow
1. **Initialize**: Call `agentic_loop`, then create `task_plan.md`.
2. **Read Before Act**: At the start of every turn, read the plan to refresh your focus.
3. **Act & Update**: After every tool execution, update the plan with results or errors.
4. **Finalize**: Use `exit_loop` once the objective is reached.

## 4. Critical Reminder
Do not rely on standard tool-calling for multi-step projects. **Explicitly engage the agentic loop tool** unless the task is definitely only going to take a very small number of turns.

This skill defines the protocol for Miyori's multi-step autonomous tasks, using external files as a "working memory" to maintain transparency and prevent context bloat.

## 1. The Planning Protocol (Manus-Lite)
For any task requiring 3 or more steps, Miyori must externalize her state into the `workspace/` directory.

### Mandatory: `task_plan.md`
Created at the start of every complex loop. It must contain:
- **Goal**: A clear definition of the finished state.
- **Phases**: Checkboxes for high-level progress.
- **Errors & Decisions**: A log of technical pivots and failed attempts to prevent looping.
- **Status**: A one-line description of the current action.

### Optional: `notes.md`
Used for research-heavy tasks. Store technical documentation, code snippets, or long tool outputs here instead of keeping them in the chat history. "Store, don't stuff."

## 2. Iteration Workflow
1. **Initialize**: Create `task_plan.md` before taking any other action.
2. **Read Before Act**: At the start of a turn, read the plan to refresh goals and avoid "attention drift."
3. **Act & Update**: After a significant action or error, update the plan.
4. **Finalize**: Summarize the result and delete or archive the plan as appropriate.

## 3. Best Practices
- **Verbal Leaness**: Keep conversational responses brief. Point the user to the plan file for detailed progress.
- **Error Logging**: Every failed command or rejected write must be logged in the plan with a "Resolution" attempt.
- **Context Management**: If a file read returns massive data, summarize it in the chat and save the full version to `notes.md`.
