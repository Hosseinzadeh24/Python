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
if errorlevel 1 (
  echo [ERROR] Installing the libraries failed. Copy the messages above and ask for help.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -c "import numpy, pandas, sklearn, matplotlib, requests, keyring"
if errorlevel 1 (
  echo [ERROR] The libraries were installed but cannot be imported. Copy the messages above and ask for help.
  pause
  exit /b 1
)
echo.
echo Setup finished. Now double-click run_once.bat or run_every_15_min.bat
pause
