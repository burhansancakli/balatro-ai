@echo off
setlocal
set TENSORBOARD=%~dp0..\venv\Scripts\tensorboard.exe
if not exist "%TENSORBOARD%" (
  echo TensorBoard not found at %TENSORBOARD%. Install requirements first.
  exit /b 1
)
"%TENSORBOARD%" --logdir "%~dp0..\logs"
