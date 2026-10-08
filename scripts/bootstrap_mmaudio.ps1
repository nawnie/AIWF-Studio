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

function Invoke-CheckedNative {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )
    & $Command
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode."
    }
}

function Resolve-AudioEnginePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (Test-Path -LiteralPath $fullPath) {
        return (Resolve-Path -LiteralPath $fullPath).Path
    }
    $parent = Split-Path -Parent $fullPath
    $leaf = Split-Path -Leaf $fullPath
    if (!(Test-Path -LiteralPath $parent -PathType Container)) {
        throw "Installer path parent does not exist: $parent"
    }
    return Join-Path (Resolve-Path -LiteralPath $parent).Path $leaf
}

function Assert-AudioEnginePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $engineRoot = (Resolve-Path -LiteralPath $EngineDir).Path.TrimEnd('\', '/')
    $resolvedPath = Resolve-AudioEnginePath -Path $Path
    $prefix = $engineRoot + [System.IO.Path]::DirectorySeparatorChar
    if (!$resolvedPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Installer path escapes the audio engine directory: $resolvedPath"
    }
    return $resolvedPath
}

function Remove-AudioEngineAttemptDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)
    $resolvedPath = Assert-AudioEnginePath -Path $Path
    if (Test-Path -LiteralPath $resolvedPath -PathType Container) {
        Remove-Item -LiteralPath $resolvedPath -Recurse -Force
    }
}

function Test-SupportedAudioPythonCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [string[]]$PrefixArguments = @()
    )
    $probeArguments = @($PrefixArguments) + @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    $version = & $Command @probeArguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0 -or $version -notmatch '^3\.(10|11|12)$') {
        return $false
    }
    return $true
}

function Test-SupportedAudioPython {
    param([Parameter(Mandatory = $true)][string]$PythonPath)
    if (!(Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        return $false
    }
    $version = & $PythonPath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    $exitCode = $LASTEXITCODE
    return ($exitCode -eq 0 -and $version -match '^3\.(10|11|12)$')
}

function New-AudioEngineVenv {
    param(
        [Parameter(Mandatory = $true)][string]$EnvironmentPath,
        [Parameter(Mandatory = $true)][string]$PythonPath
    )
    if (Test-Path $EnvironmentPath) {
        $resolvedEnvironmentPath = Assert-AudioEnginePath -Path $EnvironmentPath
        if (Test-Path -LiteralPath $PythonPath -PathType Leaf) {
            [void](Assert-AudioEnginePath -Path $PythonPath)
            return
        }
        throw "Audio engine venv folder exists but has no Python executable: $PythonPath. Move or repair it before retrying."
    }
    $resolvedEnvironmentPath = Assert-AudioEnginePath -Path $EnvironmentPath

    $launcher = Get-Command py -ErrorAction SilentlyContinue
    $candidates = @()
    if ($launcher) {
        $candidates += [pscustomobject]@{ Command = "py"; PrefixArguments = @("-3.10") }
    }
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $candidates += [pscustomobject]@{ Command = $pythonCommand.Source; PrefixArguments = @() }
    }
    if ($candidates.Count -eq 0) {
        throw "Python 3.10-3.12 was not found. Install a supported Python or Python Launcher, then retry."
    }

    $lastFailure = "No supported Python could create the audio engine venv."
    foreach ($candidate in $candidates) {
        $commandName = [string]$candidate.Command
        $prefixArguments = @($candidate.PrefixArguments)
        if (!(Test-SupportedAudioPythonCommand -Command $commandName -PrefixArguments $prefixArguments)) {
            $lastFailure = "$commandName does not provide Python 3.10, 3.11, or 3.12."
            Write-Warning "Skipping unsupported audio engine Python: $lastFailure"
            continue
        }
        $environmentExisted = Test-Path -LiteralPath $resolvedEnvironmentPath
        try {
            $arguments = @($prefixArguments) + @("-m", "venv", $resolvedEnvironmentPath)
            Write-Host "[AIWF] Creating audio engine venv with $commandName $($arguments -join ' ')"
            & $commandName @arguments
            $exitCode = $LASTEXITCODE
            if ($exitCode -ne 0) {
                throw "venv creation exited with code $exitCode"
            }
            if (!(Test-SupportedAudioPython -PythonPath $PythonPath)) {
                throw "created Python is missing or outside the supported 3.10-3.12 range"
            }
            return
        } catch {
            $lastFailure = "${commandName}: $($_.Exception.Message)"
            if (!$environmentExisted -and (Test-Path -LiteralPath $resolvedEnvironmentPath)) {
                Remove-AudioEngineAttemptDirectory -Path $resolvedEnvironmentPath
            }
            Write-Warning "Audio engine venv attempt failed ($lastFailure)."
        }
    }
    throw "Could not create an audio engine venv with Python 3.10-3.12. $lastFailure"
}

New-Item -ItemType Directory -Force -Path $EngineDir | Out-Null

if (!(Test-Path $RepoDir)) {
    $stagingRepo = "$RepoDir.bootstrap-$([guid]::NewGuid().ToString('N'))"
    $resolvedStagingRepo = Assert-AudioEnginePath -Path $stagingRepo
    $resolvedRepoDir = Assert-AudioEnginePath -Path $RepoDir
    if (Test-Path -LiteralPath $resolvedStagingRepo) {
        throw "Unique MMAudio staging path already exists; preserving it: $resolvedStagingRepo"
    }
    $cloneAttemptStarted = $false
    Write-Host "[AIWF] Cloning MMAudio into $RepoDir"
    try {
        $cloneAttemptStarted = $true
        Invoke-CheckedNative -Label "MMAudio repository clone" -Command { git clone https://github.com/hkchengrex/MMAudio.git $resolvedStagingRepo }
        if (!(Test-Path -LiteralPath $resolvedStagingRepo -PathType Container) -or !(Get-ChildItem -LiteralPath $resolvedStagingRepo -Force | Select-Object -First 1)) {
            throw "MMAudio repository clone returned success but created no repository files."
        }
        $resolvedStagingRepo = Assert-AudioEnginePath -Path $resolvedStagingRepo
        $resolvedRepoDir = Assert-AudioEnginePath -Path $resolvedRepoDir
        Move-Item -LiteralPath $resolvedStagingRepo -Destination $resolvedRepoDir
    } catch {
        if ($cloneAttemptStarted -and (Test-Path -LiteralPath $resolvedStagingRepo)) {
            Remove-AudioEngineAttemptDirectory -Path $resolvedStagingRepo
        }
        throw
    }
}
if (!(Test-Path $RepoDir -PathType Container) -or !(Get-ChildItem -LiteralPath $RepoDir -Force | Select-Object -First 1)) {
    throw "MMAudio repository folder is missing or empty: $RepoDir. Move or repair it before retrying."
}

if (!(Test-Path $VenvDir)) {
    Write-Host "[AIWF] Creating audio engine venv: $VenvDir"
}
New-AudioEngineVenv -EnvironmentPath $VenvDir -PythonPath (Join-Path $VenvDir "Scripts\python.exe")

$Python = Assert-AudioEnginePath -Path (Join-Path $VenvDir "Scripts\python.exe")
if (!(Test-Path $Python)) {
    throw "Audio engine python was not created: $Python"
}
if (!(Test-SupportedAudioPython -PythonPath $Python)) {
    throw "Audio engine Python must be version 3.10, 3.11, or 3.12: $Python"
}

Invoke-CheckedNative -Label "Audio engine pip upgrade" -Command { & $Python -m pip install --upgrade pip }

$TorchProbe = @'
import importlib.util
if importlib.util.find_spec('torch') is None:
    raise SystemExit(1)
try:
    import torch
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if torch.cuda.is_available() and torch.version.cuda else 1)
'@
& $Python -c $TorchProbe 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[AIWF] Installing CUDA torch for audio engine"
    Invoke-CheckedNative -Label "CUDA PyTorch installation for audio engine" -Command {
        & $Python -m pip install --disable-pip-version-check --upgrade --force-reinstall `
            "torch==$TorchVersion" "torchvision==$TorchvisionVersion" "torchaudio==$TorchaudioVersion" `
            --index-url $TorchIndex
    }
}

if (!$SkipInstall) {
    Write-Host "[AIWF] Installing MMAudio editable package"
    Invoke-CheckedNative -Label "MMAudio editable package installation" -Command {
        & $Python -m pip install --disable-pip-version-check -e $RepoDir
    }
}

Write-Host "[AIWF] MMAudio bootstrap complete: $Python"
