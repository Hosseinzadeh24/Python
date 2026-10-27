@echo off
REM Runs the advisor once and shows the suggestions.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (set PY=.venv\Scripts\python.exe) else (set PY=python)
%PY% main.py
echo.
pause
