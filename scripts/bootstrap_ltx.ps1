param(
    [switch]$Enable,
    [switch]$SkipInstall,
    [string]$RepoUrl = "https://github.com/Lightricks/LTX-2.git"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EngineDir = Join-Path $Root "engines\ltx"
$RepoDir = Join-Path $EngineDir "LTX-2"
$VenvDir = Join-Path $EngineDir ".venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"
$EnginesJson = Join-Path $Root "engines.json"

$TorchIndex = if ($env:LTX_TORCH_INDEX_URL) { $env:LTX_TORCH_INDEX_URL } else { "https://download.pytorch.org/whl/cu128" }
$TorchVersion = if ($env:LTX_TORCH_VERSION) { $env:LTX_TORCH_VERSION } else { "2.7.1+cu128" }
$TorchvisionVersion = if ($env:LTX_TORCHVISION_VERSION) { $env:LTX_TORCHVISION_VERSION } else { "0.22.1+cu128" }
$TorchaudioVersion = if ($env:LTX_TORCHAUDIO_VERSION) { $env:LTX_TORCHAUDIO_VERSION } else { "2.7.1+cu128" }

function Invoke-Checked {
    param([Parameter(Mandatory = $true)][scriptblock]$Script)
    & $Script
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

function New-LtxVenv {
    if (Test-Path $Python) {
        return
    }
    Write-Host "[AIWF] Creating LTX 2.3 venv: $VenvDir"
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($Launcher) {
        py -3.12 -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Python 3.12 launcher failed; falling back to the default Python."
            python -m venv $VenvDir
        }
    } else {
        python -m venv $VenvDir
    }
    if (!(Test-Path $Python)) {
        throw "LTX engine python was not created: $Python"
    }
}

function Enable-LtxEngine {
    $config = @{}
    if (Test-Path $EnginesJson) {
        try {
            $loaded = Get-Content $EnginesJson -Raw | ConvertFrom-Json -AsHashtable
            if ($loaded -is [hashtable]) {
                $config = $loaded
            }
        } catch {
            Write-Warning "Could not parse engines.json; rewriting a minimal engine config."
            $config = @{}
        }
    }

    $entry = @{
        enabled = $true
        repo_dir = "engines/ltx/LTX-2"
        venv_dir = "engines/ltx/.venv"
        worker_script = "engines/ltx/worker.py"
        _comment = "Optional LTX 2.3 video worker. Installed by scripts/bootstrap_ltx.ps1."
    }
    $config["ltx"] = $entry
    if ($config.ContainsKey("engines") -and ($config["engines"] -is [hashtable])) {
        $config["engines"]["ltx"] = $entry
    }
    $config | ConvertTo-Json -Depth 8 | Set-Content -Path $EnginesJson -Encoding UTF8
    Write-Host "[AIWF] Enabled LTX engine in $EnginesJson"
}

if (!(Test-Path $EngineDir)) {
    New-Item -ItemType Directory -Path $EngineDir | Out-Null
}

if (!(Test-Path $RepoDir)) {
    Write-Host "[AIWF] Cloning LTX-2 into $RepoDir"
    Invoke-Checked { git clone $RepoUrl $RepoDir }
} else {
    Write-Host "[AIWF] LTX-2 repo already exists: $RepoDir"
}

New-LtxVenv

$Version = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($Version -ne "3.12") {
    Write-Warning "LTX upstream currently recommends Python 3.12. This venv is Python $Version."
}

if (!$SkipInstall) {
    Write-Host "[AIWF] Installing LTX 2.3 runtime packages"
    Invoke-Checked { & $Python -m pip install --upgrade pip wheel setuptools }
    Invoke-Checked {
        & $Python -m pip install --disable-pip-version-check --upgrade --force-reinstall `
            "torch==$TorchVersion" "torchvision==$TorchvisionVersion" "torchaudio==$TorchaudioVersion" `
            --index-url $TorchIndex --extra-index-url "https://pypi.org/simple"
    }
    Invoke-Checked {
        & $Python -m pip install --disable-pip-version-check --upgrade `
            -e (Join-Path $RepoDir "packages\ltx-core") `
            -e (Join-Path $RepoDir "packages\ltx-pipelines")
    }
}

if ($Enable) {
    Enable-LtxEngine
}

Write-Host "[AIWF] LTX 2.3 bootstrap complete: $Python"
