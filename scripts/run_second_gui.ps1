param(
    [int]$Port = 8770,
    [string]$Backend = "http://127.0.0.1:7860",
    [switch]$Proxy,
    [switch]$Listen,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $Root "venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

$ArgsList = @("-m", "aiwf.second_gui", "--port", "$Port", "--backend", $Backend)
if ($Proxy) { $ArgsList += "--proxy" }
if ($Listen) { $ArgsList += "--listen" }
if ($NoBrowser) { $ArgsList += "--no-browser" }

Push-Location $Root
try {
    Write-Host "Starting AIWF Studio Second GUI preview..." -ForegroundColor Cyan
    Write-Host "Python: $Python"
    Write-Host "Backend: $Backend"
    & $Python @ArgsList
}
finally {
    Pop-Location
}
