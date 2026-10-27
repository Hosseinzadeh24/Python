@echo off
REM EMERGENCY: cancels every open order this app placed and stops new buys.
REM Stop-loss and take-profit stay active in the window. To sell, use the window kill switch.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] The .venv folder was not found. Double-click setup_windows.bat first.
  pause
  exit /b 1
)
set PY=.venv\Scripts\python.exe
%PY% main.py --kill
pause
