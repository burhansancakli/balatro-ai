$ErrorActionPreference = "Stop"

$tensorboard = Join-Path $PSScriptRoot "..\venv\Scripts\tensorboard.exe"
if (-not (Test-Path $tensorboard)) {
    throw "TensorBoard not found at $tensorboard. Install requirements first."
}

& $tensorboard --logdir (Join-Path $PSScriptRoot "..\logs")
