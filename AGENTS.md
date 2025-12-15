# Miyori AI Agent Instructions

## Project Overview
Voice-activated AI assistant. Currently in Phase 1: basic voice loop.

## Architecture
- **Interfaces** define contracts in `src/interfaces/`
- **Implementations** fulfill contracts in `src/implementations/`
- **Each implementation folder has IMPLEMENTATION.md** with specific instructions
- Read `ARCHITECTURE.md` in root for full structure

## Rules
- Full type hints required
- Read config from `config.json` in root
- No error handling this phase (let exceptions bubble)
- Use `print()` for output

## When Implementing
1. Check the interface in `src/interfaces/` for the contract
2. Read `IMPLEMENTATION.md` in the folder you're working in
3. Follow the pseudocode provided

## Current Phase
Phase 1: Basic voice conversation loop