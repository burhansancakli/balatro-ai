$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python venv not found at $python"
}

& $python -m pip install -r (Join-Path $PSScriptRoot "..\requirements.txt")
