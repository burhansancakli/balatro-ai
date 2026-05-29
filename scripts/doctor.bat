@echo off
setlocal

echo == Python launcher ==
py -0p
echo.

echo == Project venv ==
set PYTHON=%~dp0..\venv\Scripts\python.exe
if not exist "%PYTHON%" (
  echo Missing: %PYTHON%
  exit /b 1
)

"%PYTHON%" --version
if errorlevel 1 (
  echo.
  echo The venv exists, but its base Python is missing.
  echo Recreate it from an installed Python 3.11:
  echo   rmdir /s /q venv
  echo   py -3.11 -m venv venv
  echo   scripts\setup.bat
  exit /b 1
)

echo.
echo == Key packages ==
"%PYTHON%" -c "import gymnasium, numpy, pytest, requests, stable_baselines3; print('dependencies ok')"
