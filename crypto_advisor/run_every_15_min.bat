@echo off
REM Runs the advisor at every quarter hour (:00 :15 :30 :45) until you close this window.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] The .venv folder was not found. Double-click setup_windows.bat first.
  pause
  exit /b 1
)
set PY=.venv\Scripts\python.exe
%PY% main.py --loop
pause
