@echo off
setlocal
set PYTHON=%~dp0..\venv\Scripts\python.exe
if not exist "%PYTHON%" (
  echo Python venv not found at %PYTHON%
  exit /b 1
)
"%PYTHON%" -m pip install -r "%~dp0..\requirements.txt"
