@echo off
REM Walk-forward test on real Binance history. Creates charts in output\backtest.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] The .venv folder was not found. Double-click setup_windows.bat first.
  pause
  exit /b 1
)
set PY=.venv\Scripts\python.exe
%PY% main.py --backtest
pause
