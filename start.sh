#!/bin/bash

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found. Please install it first."
    exit 1
fi

echo "Starting Miyori..."
# Using 'uv run' handles venv activation automatically
uv run python -m miyori.main
