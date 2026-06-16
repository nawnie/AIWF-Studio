param(
    [Parameter(Mandatory = $true)]
    [string]$Name,

    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EngineDir = Join-Path $Root "engines\$Name"
$VenvDir = Join-Path $EngineDir ".venv"
$Requirements = Join-Path $EngineDir "requirements.txt"
$TorchIndex = if ($env:TORCH_INDEX_URL) { $env:TORCH_INDEX_URL } else { "https://download.pytorch.org/whl/cu124" }
$TorchVersion = if ($env:TORCH_CUDA_VERSION) { $env:TORCH_CUDA_VERSION } else { "2.6.0+cu124" }
$TorchvisionVersion = if ($env:TORCHVISION_CUDA_VERSION) { $env:TORCHVISION_CUDA_VERSION } else { "0.21.0+cu124" }

if (!(Test-Path $EngineDir)) {
    throw "Engine folder not found: $EngineDir"
}

if (!(Test-Path $VenvDir)) {
    Write-Host "[AIWF] Creating $Name engine venv: $VenvDir"
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($Launcher) {
        py -3.10 -m venv $VenvDir
    } else {
        python -m venv $VenvDir
    }
}

$Python = Join-Path $VenvDir "Scripts\python.exe"
if (!(Test-Path $Python)) {
    throw "Engine python was not created: $Python"
}

& $Python -m pip install --upgrade pip

& $Python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.version.cuda else 1)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[AIWF] Installing CUDA torch for $Name engine"
    & $Python -m pip install --disable-pip-version-check --upgrade --force-reinstall "torch==$TorchVersion" "torchvision==$TorchvisionVersion" --index-url $TorchIndex
}

if (!$SkipInstall -and (Test-Path $Requirements)) {
    Write-Host "[AIWF] Installing $Name engine requirements"
    & $Python -m pip install --disable-pip-version-check -r $Requirements
}

Write-Host "[AIWF] $Name engine bootstrap complete: $Python"
