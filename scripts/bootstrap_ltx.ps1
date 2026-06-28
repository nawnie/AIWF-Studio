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

function Repair-LtxWindowsSafetensorsLoader {
    $Loader = Join-Path $RepoDir "packages\ltx-core\src\ltx_core\loader\sft_loader.py"
    if (!(Test-Path $Loader)) {
        Write-Warning "Could not find LTX safetensors loader to patch: $Loader"
        return
    }
    $Text = Get-Content $Loader -Raw
    $Patched = $Text -replace "f\.get_tensor\(name\)\.to\(device=device, non_blocking=True, copy=False\)", "f.get_tensor(name).to(device=device, non_blocking=True, copy=True)"
    $Patched = $Patched -replace 'safetensors\.safe_open\(shard_path, framework="pt", device=str\(device\)\)', 'safetensors.safe_open(shard_path, framework="pt", device="cpu")'
    $Patched = $Patched -replace "f\.get_tensor\(name\)\.to\(device=device, non_blocking=True, copy=True\)", "f.get_tensor(name).clone().contiguous().to(device=device, non_blocking=False, copy=True)"
    $Patched = $Patched -replace "f\.get_tensor\(name\)\.clone\(\)\.to\(device=device, non_blocking=True, copy=True\)", "f.get_tensor(name).clone().contiguous().to(device=device, non_blocking=False, copy=True)"
    if ($Patched -ne $Text) {
        Set-Content -Path $Loader -Value $Patched -Encoding UTF8
        Write-Host "[AIWF] Patched LTX safetensors loader for Windows CPU-first storage handling."
    } else {
        Write-Host "[AIWF] LTX safetensors loader already has Windows CPU-first storage handling."
    }
}

function Repair-LtxFp8CastScaleHandling {
    $File = Join-Path $RepoDir "packages\ltx-core\src\ltx_core\quantization\fp8_cast.py"
    if (!(Test-Path $File)) {
        Write-Warning "Could not find LTX fp8_cast policy to patch: $File"
        return
    }
    $Text = Get-Content $File -Raw
    $Patched = $Text.Replace(
        'if not k.endswith("_scale"):',
        'if not (k.endswith(".weight_scale") or k.endswith(".bias_scale")):'
    )
    $DropScaleBlock = @'
    def _drop_scale(scale_key: str, _value: torch.Tensor) -> list[KeyValueOperationResult]:
        param_key = scale_key.removesuffix("_scale")
        if param_key not in scales:
            raise ValueError(
                f"Scale key {scale_key!r} has no matching entry in the prequant scales dict; "
                f"_read_scales and the loader's rename map have drifted"
            )
        return []
'@
    $DropScaleWithInputBlock = @'
    def _drop_scale(scale_key: str, _value: torch.Tensor) -> list[KeyValueOperationResult]:
        param_key = scale_key.removesuffix("_scale")
        if param_key not in scales:
            raise ValueError(
                f"Scale key {scale_key!r} has no matching entry in the prequant scales dict; "
                f"_read_scales and the loader's rename map have drifted"
            )
        return []

    def _drop_input_scale(_scale_key: str, _value: torch.Tensor) -> list[KeyValueOperationResult]:
        return []
'@
    if ($Patched -notmatch "def _drop_input_scale") {
        $Patched = $Patched.Replace($DropScaleBlock, $DropScaleWithInputBlock)
    }
    $ScaleOpsNeedle = @'
        SDOps("FP8_CAST_PREQUANT_AWARE")
        .with_kv_operation(key_suffix=".weight_scale", operation=_drop_scale)
'@
    $ScaleOpsReplacement = @'
        SDOps("FP8_CAST_PREQUANT_AWARE")
        .with_kv_operation(key_suffix=".input_scale", operation=_drop_input_scale)
        .with_kv_operation(key_suffix=".weight_scale", operation=_drop_scale)
'@
    if ($Patched -notmatch 'key_suffix="\.input_scale"') {
        $Patched = $Patched.Replace($ScaleOpsNeedle, $ScaleOpsReplacement)
    }
    if ($Patched -ne $Text) {
        Set-Content -Path $File -Value $Patched -Encoding UTF8
        Write-Host "[AIWF] Patched LTX fp8-cast scale handling for prequant checkpoints."
    } else {
        Write-Host "[AIWF] LTX fp8-cast scale handling already patched."
    }
}

function Repair-LtxBlockStreamingScaleScan {
    $File = Join-Path $RepoDir "packages\ltx-core\src\ltx_core\block_streaming\builder.py"
    if (!(Test-Path $File)) {
        Write-Warning "Could not find LTX block streaming builder to patch: $File"
        return
    }
    $Text = Get-Content $File -Raw
    $Needle = @'
                model_key = sd_ops.apply_to_key(sft_key) if sd_ops else sft_key
                if model_key is None:
                    continue
                if model_key.startswith(prefix_dot):
'@
    $Replacement = @'
                model_key = sd_ops.apply_to_key(sft_key) if sd_ops else sft_key
                if model_key is None:
                    continue
                if sd_ops is not None and model_key.endswith((".input_scale", ".weight_scale", ".bias_scale")):
                    continue
                if model_key.startswith(prefix_dot):
'@
    $Patched = $Text.Replace($Needle, $Replacement)
    if ($Patched -ne $Text) {
        Set-Content -Path $File -Value $Patched -Encoding UTF8
        Write-Host "[AIWF] Patched LTX block streaming scale-key scan."
    } else {
        Write-Host "[AIWF] LTX block streaming scale-key scan already patched."
    }
}

function Repair-LtxWindowsPinnedMemory {
    $File = Join-Path $RepoDir "packages\ltx-core\src\ltx_core\block_streaming\utils.py"
    if (!(Test-Path $File)) {
        Write-Warning "Could not find LTX block streaming utils to patch: $File"
        return
    }
    $Text = Get-Content $File -Raw
    $Patched = $Text
    if ($Patched -notmatch "\bimport os\b") {
        $Patched = $Patched.Replace("import math`r`n", "import math`r`nimport os`r`n")
    }
    if ($Patched -match 'os\.name == "nt" and \(device is None or torch\.device\(device\)\.type == "cpu"\)') {
        if ($Patched -ne $Text) {
            Set-Content -Path $File -Value $Patched -Encoding UTF8
        }
        Write-Host "[AIWF] LTX Windows pinned-memory bypass already patched."
        return
    }
    $Needle = @'
    if pin_memory and (device is None or torch.device(device).type == "cpu"):
        if not torch.cuda.is_available():
'@
    $Replacement = @'
    if pin_memory and os.name == "nt" and (device is None or torch.device(device).type == "cpu"):
        return torch.empty(nbytes, dtype=torch.uint8, device=device)
    if pin_memory and (device is None or torch.device(device).type == "cpu"):
        if not torch.cuda.is_available():
'@
    $Patched = $Patched.Replace($Needle, $Replacement)
    if ($Patched -ne $Text) {
        Set-Content -Path $File -Value $Patched -Encoding UTF8
        Write-Host "[AIWF] Patched LTX block streaming to avoid pinned host buffers on Windows."
    } else {
        Write-Host "[AIWF] LTX Windows pinned-memory bypass already patched."
    }
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
    Invoke-Checked {
        & $Python -m pip install --disable-pip-version-check --upgrade "gguf>=0.17"
    }
}

Repair-LtxWindowsSafetensorsLoader
Repair-LtxFp8CastScaleHandling
Repair-LtxBlockStreamingScaleScan
Repair-LtxWindowsPinnedMemory

if ($Enable) {
    Enable-LtxEngine
}

Write-Host "[AIWF] LTX 2.3 bootstrap complete: $Python"
