@echo off
REM Opens the window: API key box, max loss / min profit, modes, kill switch.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (set PY=.venv\Scripts\python.exe) else (set PY=python)
%PY% main.py --gui
if errorlevel 1 pause
