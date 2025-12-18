@echo off
setlocal
cd /d %~dp0

if not exist .venv (
    echo Virtual environment .venv not found.
    echo Please create it first using: py 3.11 -m venv .venv
    pause
    exit /b
)

echo Activating virtual environment...
call .venv\Scripts\activate

echo Starting Miyori...
title Miyori
python main.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%
    pause
)

pause
