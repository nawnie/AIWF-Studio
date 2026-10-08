param(
    [switch]$SkipInstall,
    [string]$DataRoot
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($DataRoot)) { $DataRoot = $Root }
$DataRoot = [System.IO.Path]::GetFullPath($DataRoot)
$EngineDir = Join-Path $DataRoot "engines\qwen_nunchaku"
$VenvDir = Join-Path $EngineDir ".venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $Root "engines\qwen_nunchaku\requirements.txt"
$RunnerSource = Join-Path $Root "engines\qwen_nunchaku\run_qwen_lightning.py"
$RunnerPath = Join-Path $EngineDir "run_qwen_lightning.py"
$WheelName = "nunchaku-1.3.0.dev20260213+cu13.0torch2.11-cp312-cp312-win_amd64.whl"
$WheelUrl = "https://github.com/nunchux-ai/nunchaku/releases/download/v1.3.0dev20260213/nunchaku-1.3.0.dev20260213%2Bcu13.0torch2.11-cp312-cp312-win_amd64.whl"
$WheelSha256 = "ff4bab58d2b26e301dbf894efabcb996799a593ff76f2c7c3d9006c7b6a7afbb"

function Invoke-Checked {
    param([Parameter(Mandatory = $true)][scriptblock]$Script)
    & $Script
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}

if ($SkipInstall) {
    if (!(Test-Path $Python)) { throw "Qwen Nunchaku Python is missing: $Python" }
} elseif (!(Test-Path $Python)) {
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if (!$Launcher) { throw "Python 3.12 is required to create the isolated Nunchaku runtime." }
    Invoke-Checked { py -3.12 -m venv $VenvDir }
}

if (!(Test-Path $RunnerSource)) { throw "Qwen Nunchaku runner source is missing: $RunnerSource" }
New-Item -ItemType Directory -Path $EngineDir -Force | Out-Null
if ([System.IO.Path]::GetFullPath($RunnerSource) -ne [System.IO.Path]::GetFullPath($RunnerPath)) {
    Copy-Item -LiteralPath $RunnerSource -Destination $RunnerPath -Force
}
if (!(Test-Path $Python)) { throw "Qwen Nunchaku Python was not created: $Python" }
$RuntimeVersion = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $RuntimeVersion.Trim() -ne "3.12") {
    throw "Qwen Nunchaku runtime requires Python 3.12 for the pinned Windows wheel; found $RuntimeVersion at $Python"
}

if (!$SkipInstall) {
    New-Item -ItemType Directory -Path $EngineDir -Force | Out-Null
    Invoke-Checked { & $Python -m pip install --upgrade pip }
    Invoke-Checked {
        & $Python -m pip install "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" --index-url "https://download.pytorch.org/whl/cu130"
    }
    Invoke-Checked { & $Python -m pip install -r $Requirements }

    $WheelPath = Join-Path $EngineDir (([guid]::NewGuid().ToString("N")) + "-" + $WheelName)
    try {
        Invoke-WebRequest -Uri $WheelUrl -OutFile $WheelPath
        $ActualHash = (Get-FileHash -LiteralPath $WheelPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ActualHash -ne $WheelSha256) {
            throw "Nunchaku wheel SHA-256 mismatch: expected $WheelSha256, got $ActualHash"
        }
        Invoke-Checked { & $Python -m pip install --no-deps $WheelPath }
    } finally {
        Remove-Item -LiteralPath $WheelPath -Force -ErrorAction SilentlyContinue
    }
}

Invoke-Checked {
    & $Python -c "import torch, diffusers, transformers, nunchaku; from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel; from diffusers import QwenImagePipeline; assert torch.version.cuda == '13.0', torch.version.cuda; print(f'Qwen Nunchaku imports ready: torch={torch.__version__}, diffusers={diffusers.__version__}, transformers={transformers.__version__}')"
}
Write-Host "[AIWF] Qwen Nunchaku isolated runtime setup completed. No model weights were loaded."
