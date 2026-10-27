@echo off
REM Opens the window: API key box, max loss / min profit, modes, kill switch.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] The .venv folder was not found. Double-click setup_windows.bat first.
  pause
  exit /b 1
)
set PY=.venv\Scripts\python.exe
%PY% main.py --gui
if errorlevel 1 pause
