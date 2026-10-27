@echo off
REM One-time setup: creates the .venv folder and installs the libraries.
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 (
  echo Python was not found. Install Python 3.12 from python.org and tick "Add python.exe to PATH".
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
echo Setup finished. Now double-click run_once.bat or run_every_15_min.bat
pause
