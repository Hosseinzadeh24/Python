@echo off
REM Walk-forward test on real Binance history. Creates charts in output\backtest.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (set PY=.venv\Scripts\python.exe) else (set PY=python)
%PY% main.py --backtest
pause
