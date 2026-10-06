param(
    [switch]$WithAiToolkit,
    [switch]$SkipInstall,
    [string]$AiToolkitRepo = "https://github.com/ostris/ai-toolkit.git"
)

# Bootstraps the Qwen Image 2.1 Studio app venv (PySide6, no torch) and optionally
# ai-toolkit with its own venv for LoRA training. ComfyUI itself is not managed here.

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EngineDir = Join-Path $Root "engines\qwen_image_2_1"
$VenvDir = Join-Path $EngineDir ".venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"

function Invoke-Checked {
    param([Parameter(Mandatory = $true)][scriptblock]$Script)
    & $Script
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}

if (!(Test-Path $Python)) {
    Write-Host "[AIWF] Creating Qwen 2.1 Studio venv: $VenvDir"
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($Launcher) { py -3.12 -m venv $VenvDir; if ($LASTEXITCODE -ne 0) { py -3 -m venv $VenvDir } }
    else { python -m venv $VenvDir }
    if (!(Test-Path $Python)) { throw "venv python was not created: $Python" }
}

if (!$SkipInstall) {
    Invoke-Checked { & $Python -m pip install --disable-pip-version-check --upgrade pip }
    Invoke-Checked { & $Python -m pip install --disable-pip-version-check -r (Join-Path $EngineDir "requirements.txt") }
}
Invoke-Checked { & $Python (Join-Path $EngineDir "app.py") --check }

if ($WithAiToolkit) {
    $TkDir = Join-Path $EngineDir "ai-toolkit"
    $TkPython = Join-Path $TkDir "venv\Scripts\python.exe"
    $TorchIndex = if ($env:AITK_TORCH_INDEX_URL) { $env:AITK_TORCH_INDEX_URL } else { "https://download.pytorch.org/whl/cu130" }
    $TorchSpec = if ($env:AITK_TORCH_SPEC) { $env:AITK_TORCH_SPEC } else { "torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0" }
    if (!(Test-Path (Join-Path $TkDir "run.py"))) {
        Write-Host "[AIWF] Cloning ai-toolkit into $TkDir"
        Invoke-Checked { git clone $AiToolkitRepo $TkDir }
    } else {
        Write-Host "[AIWF] Updating ai-toolkit"
        Push-Location $TkDir; try { git pull --ff-only } finally { Pop-Location }
    }
    if (!(Test-Path $TkPython)) {
        Write-Host "[AIWF] Creating ai-toolkit venv (Python 3.12 recommended by upstream)"
        Push-Location $TkDir
        try {
            $Launcher = Get-Command py -ErrorAction SilentlyContinue
            if ($Launcher) { py -3.12 -m venv venv; if ($LASTEXITCODE -ne 0) { python -m venv venv } } else { python -m venv venv }
        } finally { Pop-Location }
    }
    if (!$SkipInstall) {
        Write-Host "[AIWF] Installing torch ($TorchSpec) from $TorchIndex, then ai-toolkit requirements"
        Invoke-Checked { & $TkPython -m pip install --disable-pip-version-check --upgrade pip }
        Invoke-Checked { & $TkPython -m pip install --disable-pip-version-check --no-cache-dir ($TorchSpec -split " ") --index-url $TorchIndex }
        Invoke-Checked { & $TkPython -m pip install --disable-pip-version-check -r (Join-Path $TkDir "requirements.txt") }
    }
    Invoke-Checked { & $TkPython -c "import torch; print('ai-toolkit torch', torch.__version__, 'cuda', torch.cuda.is_available())" }
    Write-Host "[AIWF] ai-toolkit ready: $TkPython  (Settings tab: ai-toolkit folder = $TkDir)"
}

Write-Host "[AIWF] Qwen Image 2.1 Studio bootstrap complete. Launch: '$Root\Qwen Image 2.1 Studio.bat'"
