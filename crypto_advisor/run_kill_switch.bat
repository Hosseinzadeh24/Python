@echo off
REM EMERGENCY: cancels every open order this app placed and stops new buys.
REM Stop-loss and take-profit stay active in the window. To sell, use the window kill switch.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (set PY=.venv\Scripts\python.exe) else (set PY=python)
%PY% main.py --kill
pause
