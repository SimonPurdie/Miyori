# Skill: Tooling & Terminal Operations

This skill covers Miyori's ability to interact with the file system and execute shell commands safely and effectively.

## 1. File Operations (`file_ops`)
Provides comprehensive access to the workspace and project files.

### Key Actions:
- **`list`**: Recursively lists directory contents. Directories are listed first, then files. Shows file sizes and modification dates.
- **`read`**: Reads file content. Supports:
    - **Paginated Reading**: Use `offset` and `limit` for large files.
    - **Binary Detection**: Automatically detects and profiles binary files (images, executables) instead of dumping raw data.
- **`write`**: Creates or overwrites files.

### Safety:
- Operations are restricted to the project root and specific allowed directories.
- Binary files require `force=True` to read.

## 2. Terminal (`terminal`)
Allows execution of shell commands (bash/zsh/cmd) in the local environment.

### Features:
- **Persistent Sessions**: Setting `persistent=True` keeps track of the Current Working Directory (CWD) across tool calls.
- **State Synchronization**: Automatically updates `AgenticState` with the `last_command`, `last_exit_code`, and a truncated `last_output`.
- **Directory Management**: Handles `cd` commands internally to maintain the persistent CWD.

### Security:
- **Approval Loop**: All commands are checked against "dangerous patterns" (e.g., `rm -rf /`, `curl | sh`). If a pattern matches, the user must explicitly approve the command in the console.
- **Timeouts**: Commands are automatically killed if they exceed the configured timeout (default 120s).

## 3. Best Practices
- **Diagnostic Listing**: Always `list` a directory before searching for or creating a file to confirm the path.
- **Read-Modify-Write-Verify**: Always read a file completely, modify in memory, write back, and then read again to verify changes.
- **Verification**: Use `terminal` to run tests (e.g., `pytest` or `uv run tests/...`) after modifying core files.
