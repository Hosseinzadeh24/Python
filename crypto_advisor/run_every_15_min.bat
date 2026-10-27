@echo off
REM Runs the advisor at every quarter hour (:00 :15 :30 :45) until you close this window.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (set PY=.venv\Scripts\python.exe) else (set PY=python)
%PY% main.py --loop
pause
