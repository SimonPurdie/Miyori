# Agent Guidelines

## Project Environment

**Target OS:** Windows 11

---

## Text and Encoding Standards

### Character Set
- Use ASCII only for PowerShell commands and file paths
- No smart quotes, em/en dashes, Unicode symbols in commands

### Source Code
- UTF-8 encoding for Python files (with proper `# -*- coding: utf-8 -*-` if needed)
- ASCII preferred for identifiers, but UTF-8 allowed in strings and comments
- Still avoid smart quotes/fancy dashes even in strings (use `"` not `"`)

---

## Python Environment

### Virtual Environment
- Always use the project `.venv`
- Do not use system Python or global installs

### Dependency Management
- Use `uv` for dependency management and execution
- **Do not use `pip`**

---

## Terminal Configuration

### Operating System
- Windows 11

### Shell
- PowerShell

---

## Quick Reference

| Component | Requirement |
|-----------|-------------|
| Encoding | ASCII only |
| Python Env | Project `.venv` |
| Package Manager | `uv` (not pip) |
| OS | Windows 11 |
| Shell | PowerShell |

Paths: Windows-style (\)
**Read `ARCHITECTURE.md` first** - it contains all conventions, design decisions, and extension patterns.

## Quick Start
- Interfaces define contracts in `src/interfaces/`
- Implementations in `src/implementations/`
- All config in `config.json` at root
- Check for `FEATURE_*.md` files for active work