param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EngineDir = Join-Path $Root "engines\audio"
$RepoDir = Join-Path $EngineDir "MMAudio"
$VenvDir = Join-Path $EngineDir ".venv"
$TorchIndex = if ($env:TORCH_INDEX_URL) { $env:TORCH_INDEX_URL } else { "https://download.pytorch.org/whl/cu124" }
$TorchVersion = if ($env:TORCH_CUDA_VERSION) { $env:TORCH_CUDA_VERSION } else { "2.6.0+cu124" }
$TorchvisionVersion = if ($env:TORCHVISION_CUDA_VERSION) { $env:TORCHVISION_CUDA_VERSION } else { "0.21.0+cu124" }
$TorchaudioVersion = if ($env:TORCHAUDIO_CUDA_VERSION) { $env:TORCHAUDIO_CUDA_VERSION } else { "2.6.0+cu124" }

New-Item -ItemType Directory -Force -Path $EngineDir | Out-Null

if (!(Test-Path $RepoDir)) {
    Write-Host "[AIWF] Cloning MMAudio into $RepoDir"
    git clone https://github.com/hkchengrex/MMAudio.git $RepoDir
}

if (!(Test-Path $VenvDir)) {
    Write-Host "[AIWF] Creating audio engine venv: $VenvDir"
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($Launcher) {
        py -3.10 -m venv $VenvDir
    } else {
        python -m venv $VenvDir
    }
}

$Python = Join-Path $VenvDir "Scripts\python.exe"
if (!(Test-Path $Python)) {
    throw "Audio engine python was not created: $Python"
}

& $Python -m pip install --upgrade pip

& $Python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.version.cuda else 1)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[AIWF] Installing CUDA torch for audio engine"
    & $Python -m pip install --disable-pip-version-check --upgrade --force-reinstall `
        "torch==$TorchVersion" "torchvision==$TorchvisionVersion" "torchaudio==$TorchaudioVersion" `
        --index-url $TorchIndex
}

if (!$SkipInstall) {
    Write-Host "[AIWF] Installing MMAudio editable package"
    & $Python -m pip install --disable-pip-version-check -e $RepoDir
}

Write-Host "[AIWF] MMAudio bootstrap complete: $Python"
