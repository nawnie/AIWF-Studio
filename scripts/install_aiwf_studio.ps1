[CmdletBinding()]
param(
    [ValidateSet("prompt", "express", "full", "custom", "quit")]
    [string]$Mode = "prompt",
    [switch]$DryRun,
    [switch]$ShortcutsOnly,
    [switch]$SkipPrerequisites,
    [switch]$SkipFrontendBuild,
    [switch]$SkipRuntimeSetup,
    [switch]$WithDefaultModel,
    [switch]$WithNvidiaVideoFx,
    [switch]$FullImageStack,
    [switch]$TaskLocalMode,
    [string]$TaskLocalRoot,
    [switch]$UseBackendLock
)

$ErrorActionPreference = "Stop"

function Resolve-TaskLocalRoot {
    param([string]$Candidate, [string]$SourceCheckout)
    if ([string]::IsNullOrWhiteSpace($Candidate)) { throw "Task-local mode requires -TaskLocalRoot pointing to a prepared disposable AIWF Studio copy." }
    if (-not (Test-Path -LiteralPath $Candidate -PathType Container)) { throw "Task-local root must already exist; the installer never clears or creates a target project copy." }
    $resolved = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $Candidate).Path)
    $rootKey = $resolved.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    $sourceKey = [System.IO.Path]::GetFullPath($SourceCheckout).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    $sourcePrefix = $sourceKey + [System.IO.Path]::DirectorySeparatorChar
    if ([string]::Equals($rootKey, $sourceKey, [System.StringComparison]::OrdinalIgnoreCase) -or $rootKey.StartsWith($sourcePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Task-local mode cannot target the source checkout or a directory inside it."
    }
    $volumeRoot = [System.IO.Path]::GetPathRoot($rootKey)
    $current = $rootKey
    while ($current) {
        $item = Get-Item -LiteralPath $current -Force -ErrorAction Stop
        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            throw "Task-local mode cannot use a target path with a junction or symbolic-link component: $current"
        }
        if ([string]::Equals($current, $volumeRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar), [System.StringComparison]::OrdinalIgnoreCase)) { break }
        $parent = [System.IO.Directory]::GetParent($current)
        if (-not $parent -or [string]::Equals($parent.FullName, $current, [System.StringComparison]::OrdinalIgnoreCase)) { break }
        $current = $parent.FullName
    }
    return $resolved
}

function Assert-TaskLocalWritePath {
    param([string]$Path)
    if (-not $TaskLocalMode) { return }
    $rootKey = [System.IO.Path]::GetFullPath($Root).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    $target = [System.IO.Path]::GetFullPath($Path)
    if ([string]::Equals($target, $rootKey, [System.StringComparison]::OrdinalIgnoreCase)) { $relative = "" }
    elseif ($target.StartsWith($rootKey + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        $relative = $target.Substring($rootKey.Length + 1)
    } else {
        throw "Task-local mode refused a write path outside its selected root: $target"
    }
    $current = $rootKey
    if ($relative) {
        foreach ($segment in ($relative -split '[\\/]')) {
            if (-not $segment) { continue }
            $current = Join-Path $current $segment
            $item = Get-Item -LiteralPath $current -Force -ErrorAction SilentlyContinue
            if ($item -and ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
                throw "Task-local mode refused a write path with a junction or symbolic-link component: $current"
            }
        }
    }
}

$SourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Root = $SourceRoot
if ($TaskLocalMode) {
    $Root = Resolve-TaskLocalRoot -Candidate $TaskLocalRoot -SourceCheckout $SourceRoot
    foreach ($requiredPath in @(".aiwf-task-local-install", "launch.py", "frontend\package-lock.json")) {
        if (-not (Test-Path -LiteralPath (Join-Path $Root $requiredPath))) { throw "Task-local root is not a marked AIWF Studio copy; missing $requiredPath." }
    }
    if ($Mode -notin @("prompt", "express") -or $ShortcutsOnly -or $WithDefaultModel -or $WithNvidiaVideoFx -or $FullImageStack) { throw "Task-local mode supports Express only and disables shortcuts, model downloads, and NVIDIA SDK linking." }
    $Mode = "express"
    $SkipPrerequisites = $true
} elseif (-not [string]::IsNullOrWhiteSpace($TaskLocalRoot)) {
    throw "-TaskLocalRoot requires -TaskLocalMode."
}
$VenvDir = Join-Path $Root "venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$BackendLockMinimumPython = [version]"3.12.13"
$BackendLockMaximumPython = [version]"3.13.0"
$PythonVersion = if ($UseBackendLock) { "3.12.13" } else { "3.12" }

function Test-BackendLockPythonVersion {
    param([string]$Version)
    try {
        $parsed = [version]$Version
        return ($parsed -ge $BackendLockMinimumPython -and $parsed -lt $BackendLockMaximumPython)
    } catch {
        return $false
    }
}

function Assert-BackendLockHost {
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
        throw "-UseBackendLock supports Windows AMD64 only; use the existing installer path for this platform."
    }
    $architecture = $env:PROCESSOR_ARCHITEW6432
    if ([string]::IsNullOrWhiteSpace($architecture)) { $architecture = $env:PROCESSOR_ARCHITECTURE }
    if ($architecture -notin @("AMD64", "x86_64")) {
        throw "-UseBackendLock supports Windows AMD64 only; detected host architecture '$architecture'."
    }
    $lockProject = Join-Path $Root "dependencies\windows-py312"
    foreach ($required in @("pyproject.toml", "uv.lock")) {
        if (-not (Test-Path -LiteralPath (Join-Path $lockProject $required) -PathType Leaf)) {
            throw "The Windows Python 3.12 backend lock is incomplete: $lockProject\$required"
        }
    }
}

function Assert-BackendLockPython {
    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        throw "The locked backend install requires the app venv Python: $VenvPython"
    }
    $runtime = & $VenvPython -c "import platform,sys; print('|'.join((platform.python_implementation(), platform.python_version(), sys.platform, platform.machine())))"
    if ($LASTEXITCODE -ne 0) { throw "Could not inspect the app venv Python for the locked runtime." }
    $parts = $runtime.Trim().Split('|')
    if ($parts.Count -ne 4 -or $parts[0] -ne "CPython" -or $parts[2] -ne "win32" -or $parts[3] -ne "AMD64" -or -not (Test-BackendLockPythonVersion -Version $parts[1])) {
        throw "-UseBackendLock requires CPython >=3.12.13,<3.13 on Windows AMD64; found '$($runtime.Trim())'."
    }
}

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "== $Message =="
}

function Update-ProcessPath {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $existing = $env:Path -split ";"
    $extra = @(
        "$env:ProgramFiles\Git\cmd",
        "$env:ProgramFiles\nodejs",
        "$env:USERPROFILE\.local\bin",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links"
    )
    $env:Path = (($machine, $user) + $extra + $existing | Where-Object { $_ } | Select-Object -Unique) -join ";"
}

function Test-CommandsAvailable {
    param([string[]]$Names)
    foreach ($name in $Names) {
        if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
            return $false
        }
    }
    return $true
}

function Invoke-External {
    param(
        [string]$Label,
        [string]$FilePath,
        [string[]]$Arguments
    )
    $shown = "$FilePath $($Arguments -join ' ')".Trim()
    if ($DryRun) {
        Write-Host "[dry-run] $Label"
        Write-Host "          $shown"
        return
    }

    Write-Host $shown
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Ensure-WingetPackage {
    param(
        [string]$Label,
        [string]$Id,
        [string[]]$Commands
    )

    if (Test-CommandsAvailable $Commands) {
        Write-Host "$Label already available."
        return
    }
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget was not found. Install App Installer from the Microsoft Store, then run this installer again."
    }

    Invoke-External "Install $Label" "winget" @(
        "install",
        "--id", $Id,
        "--exact",
        "--source", "winget",
        "--accept-package-agreements",
        "--accept-source-agreements",
        "--disable-interactivity"
    )
    Update-ProcessPath
}

function Get-VenvPythonMinor {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        return ""
    }
    try {
        return (& $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    } catch {
        return ""
    }
}

function Get-VenvPythonVersion {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        return ""
    }
    try {
        return (& $VenvPython -c "import platform; print(platform.python_version())").Trim()
    } catch {
        return ""
    }
}

function Move-StaleVenv {
    param([string]$Reason)
    Assert-TaskLocalWritePath -Path $VenvDir
    if (-not (Test-Path -LiteralPath $VenvDir)) { return }
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backupName = "installer-venv-$stamp-$([guid]::NewGuid().ToString('N'))"
    $trash = Join-Path (Join-Path $Root "_trash") $backupName
    Assert-TaskLocalWritePath -Path $trash
    New-Item -ItemType Directory -Path (Split-Path $trash -Parent) -Force | Out-Null
    Write-Host "$Reason Moving the existing venv to $trash"
    Move-Item -LiteralPath $VenvDir -Destination $trash
}

function Get-CondaCommand {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @(
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "Miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:ProgramData "miniconda3\Scripts\conda.exe")
    )) { if (Test-Path -LiteralPath $p) { return $p } }
    return $null
}

function Install-Miniconda {
    $target = Join-Path $env:USERPROFILE "miniconda3"
    $installer = Join-Path $env:TEMP "Miniconda3-latest-Windows-x86_64.exe"
    if ($DryRun) {
        Write-Host "[dry-run] Would download + silently install Miniconda to $target"
        return (Join-Path $target "Scripts\conda.exe")
    }
    Write-Host "Downloading Miniconda (Python $PythonVersion provider)..."
    Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $installer -UseBasicParsing
    Write-Host "Installing Miniconda silently to $target ..."
    Start-Process -FilePath $installer -ArgumentList "/InstallationType=JustMe","/RegisterPython=0","/AddToPath=0","/S","/D=$target" -Wait
    Update-ProcessPath
    return (Join-Path $target "Scripts\conda.exe")
}

function Invoke-WithInstallerEnvironmentLock {
    param([scriptblock]$Action)
    $localRoot = Join-Path $Root "_local"
    if ($TaskLocalMode) { Assert-TaskLocalWritePath -Path $localRoot }
    New-Item -ItemType Directory -Path $localRoot -Force | Out-Null
    $lockPath = Join-Path $localRoot "installer-setup.lock"
    if ($TaskLocalMode) { Assert-TaskLocalWritePath -Path $lockPath }
    $lockStream = $null
    $previousUvCacheDir = [Environment]::GetEnvironmentVariable("UV_CACHE_DIR", "Process")
    $previousUvPythonInstallDir = [Environment]::GetEnvironmentVariable("UV_PYTHON_INSTALL_DIR", "Process")
    try {
        try {
            $lockStream = [System.IO.File]::Open(
                $lockPath,
                [System.IO.FileMode]::OpenOrCreate,
                [System.IO.FileAccess]::ReadWrite,
                [System.IO.FileShare]::None
            )
        } catch [System.IO.IOException] {
            $win32Error = $_.Exception.HResult -band 0xFFFF
            if ($win32Error -notin @(32, 33)) { throw }
            throw "Another AIWF Studio installer is performing setup. Wait for it to finish, then retry."
        }
        if ($TaskLocalMode) {
            $uvCacheDir = Join-Path $localRoot "uv-cache"
            $uvPythonInstallDir = Join-Path $localRoot "uv-python"
            $uvPythonBinDir = Join-Path $localRoot "uv-python-bin"
            Assert-TaskLocalWritePath -Path $uvCacheDir
            Assert-TaskLocalWritePath -Path $uvPythonInstallDir
            Assert-TaskLocalWritePath -Path $uvPythonBinDir
            New-Item -ItemType Directory -Path $uvCacheDir, $uvPythonInstallDir, $uvPythonBinDir -Force | Out-Null
            [Environment]::SetEnvironmentVariable("UV_CACHE_DIR", $uvCacheDir, "Process")
            [Environment]::SetEnvironmentVariable("UV_PYTHON_INSTALL_DIR", $uvPythonInstallDir, "Process")
            [Environment]::SetEnvironmentVariable("UV_PYTHON_BIN_DIR", $uvPythonBinDir, "Process")
        }
        & $Action
    } finally {
        [Environment]::SetEnvironmentVariable("UV_CACHE_DIR", $previousUvCacheDir, "Process")
        [Environment]::SetEnvironmentVariable("UV_PYTHON_INSTALL_DIR", $previousUvPythonInstallDir, "Process")
        if ($lockStream) { $lockStream.Dispose() }
    }
}

function Move-StaleCondaEnv {
    param([string]$PythonEnvironmentDir)
    if (-not (Test-Path -LiteralPath $PythonEnvironmentDir)) { return }
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backupName = "installer-conda-python-$stamp-$([guid]::NewGuid().ToString('N'))"
    $trash = Join-Path (Join-Path $Root "_trash") $backupName
    Assert-TaskLocalWritePath -Path $PythonEnvironmentDir
    Assert-TaskLocalWritePath -Path $trash
    New-Item -ItemType Directory -Path (Split-Path $trash -Parent) -Force | Out-Null
    Write-Host "Moving the existing conda Python environment to $trash"
    Move-Item -LiteralPath $PythonEnvironmentDir -Destination $trash
}
# Fallback provisioner: use conda to get the target Python, then build a
# STANDARD venv from it so the app still finds venv\Scripts\python.exe.
function Ensure-PythonVenv-Conda {
    if ($TaskLocalMode) {
        throw "Task-local mode requires uv for Python provisioning and will not use conda, which may register environments in the user profile. Install uv first, then retry."
    }
    Write-Host "uv could not provide Python $PythonVersion. Trying conda."
    $conda = Get-CondaCommand
    if (-not $conda) {
        if ($TaskLocalMode) {
            throw "Task-local mode does not install Miniconda globally. Install uv or conda first, then retry task-local mode."
        }
        $answer = Read-Host "Python $PythonVersion is required and was not available. Install Miniconda now to create it automatically? [Y/n]"
        if ($answer -and $answer.Trim().ToLowerInvariant().StartsWith("n")) {
            throw "Python $PythonVersion is required. Install Python $PythonVersion (or conda) and re-run the installer."
        }
        $conda = Install-Miniconda
    }
    if (-not $conda -or -not (Test-Path -LiteralPath $conda)) {
        throw "conda was not available after the install attempt; cannot provision Python $PythonVersion."
    }
    $pyenv = Join-Path $Root "_pyenv$($PythonVersion.Replace('.',''))"
    Assert-TaskLocalWritePath -Path $pyenv
    Move-StaleCondaEnv -PythonEnvironmentDir $pyenv
    Invoke-External "Create conda Python $PythonVersion" $conda @("create", "-y", "-p", $pyenv, "python=$PythonVersion")
    $condaPython = Join-Path $pyenv "python.exe"
    if (-not (Test-Path -LiteralPath $condaPython)) {
        throw "conda did not produce a Python at $condaPython"
    }
    Move-StaleVenv -Reason "Rebuilding the venv with the conda-provided Python $PythonVersion."
    Invoke-External "Create AIWF venv from conda Python" $condaPython @("-m", "venv", $VenvDir)
}

function Ensure-PythonVenv {
    Assert-TaskLocalWritePath -Path $VenvDir
    Write-Section "Python environment"
    if ($DryRun) {
        Invoke-External "Install Python $PythonVersion with uv" "uv" @("python", "install", $PythonVersion)
        Invoke-External "Create AIWF venv" "uv" @("venv", "--python", $PythonVersion, $VenvDir)
        Invoke-External "Seed pip" "uv" @("pip", "install", "--python", $VenvPython, "pip", "setuptools", "wheel")
        return
    }

    # Preferred path: uv provides a standalone Python with no system dependency.
    $uvOk = $false
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        try {
            Invoke-External "Install Python $PythonVersion with uv" "uv" @("python", "install", $PythonVersion)
            $minor = Get-VenvPythonMinor
            if ($UseBackendLock) {
                $existingPython = Get-VenvPythonVersion
                if (-not (Test-BackendLockPythonVersion -Version $existingPython)) {
                    Move-StaleVenv -Reason "Existing venv uses Python $existingPython; the locked runtime needs >=3.12.13,<3.13."
                    $minor = ""
                }
            } elseif ($minor -and $minor -ne $PythonVersion) {
                Move-StaleVenv -Reason "Existing venv uses Python $minor (need $PythonVersion)."
                $minor = ""
            }
            if (-not $minor) {
                Invoke-External "Create AIWF venv" "uv" @("venv", "--python", $PythonVersion, $VenvDir)
            } else {
                Write-Host "AIWF venv already uses Python $minor."
            }
            if ($UseBackendLock) {
                $verifiedPython = Get-VenvPythonVersion
                if (Test-BackendLockPythonVersion -Version $verifiedPython) { $uvOk = $true }
            } elseif ((Get-VenvPythonMinor) -eq $PythonVersion) {
                $uvOk = $true
            }
        } catch {
            Write-Host "uv Python provisioning failed: $($_.Exception.Message)"
        }
    } else {
        Write-Host "uv is not available; will use the conda fallback for Python $PythonVersion."
    }

    # Fallback: conda (offered to the user if not already installed).
    if (-not $uvOk) {
        Ensure-PythonVenv-Conda
    }

    if (-not (Test-Path -LiteralPath $VenvPython)) {
        throw "Expected venv Python was not created: $VenvPython"
    }
    if ($UseBackendLock) {
        Assert-BackendLockPython
    } else {
        $finalMinor = Get-VenvPythonMinor
        if ($finalMinor -ne $PythonVersion) {
            throw "venv Python is $finalMinor but $PythonVersion is required. Install Python $PythonVersion or conda and re-run."
        }
    }

    # Seed pip via uv when present, otherwise the venv's own pip (conda path).
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Invoke-External "Seed pip" "uv" @("pip", "install", "--python", $VenvPython, "pip", "setuptools", "wheel")
    } else {
        Invoke-External "Seed pip" $VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    }
}

function Prepare-AiwfRuntime {
    if ($SkipRuntimeSetup) {
        if ($UseBackendLock) { throw "-UseBackendLock cannot be combined with -SkipRuntimeSetup." }
        Write-Host "Skipping Python runtime setup. Launching generation later may install runtime packages."
        return
    }

    Write-Section "AIWF Python dependencies"
    if ($UseBackendLock) {
        Assert-BackendLockHost
        if (-not $DryRun) { Assert-BackendLockPython }
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            throw "-UseBackendLock requires uv. Install uv or use the existing installer path."
        }
        $lockProject = Join-Path $Root "dependencies\windows-py312"
        $previousProjectEnvironment = [Environment]::GetEnvironmentVariable("UV_PROJECT_ENVIRONMENT", "Process")
        try {
            [Environment]::SetEnvironmentVariable("UV_PROJECT_ENVIRONMENT", $VenvDir, "Process")
            Invoke-External "Install hash-locked Windows Python 3.12 runtime" "uv" @(
                "sync", "--project", $lockProject, "--locked", "--inexact", "--python", $VenvPython
            )
        } finally {
            [Environment]::SetEnvironmentVariable("UV_PROJECT_ENVIRONMENT", $previousProjectEnvironment, "Process")
        }
        return
    }
    Invoke-External "Prepare AIWF runtime" $VenvPython @(
        "-c",
        "import launch; launch.prepare(False, False, [])"
    )
}

function Build-ProFrontend {
    if ($SkipFrontendBuild) {
        Write-Host "Skipping frontend build."
        return
    }

    Write-Section "Pro React frontend"
    $frontend = Join-Path $Root "frontend"
    $nodeModules = Join-Path $frontend "node_modules"
    if ($TaskLocalMode) {
        Assert-TaskLocalWritePath -Path $frontend
        Assert-TaskLocalWritePath -Path $nodeModules
        Assert-TaskLocalWritePath -Path (Join-Path $frontend "dist")
    }
    $lock = Join-Path $frontend "package-lock.json"
    Push-Location $frontend
    try {
        if (Test-Path -LiteralPath $lock) {
            Invoke-External "Install frontend packages" "npm" @("ci")
        } else {
            Invoke-External "Install frontend packages" "npm" @("install")
        }
        Invoke-External "Build Pro frontend" "npm" @("run", "build")
    } finally {
        Pop-Location
    }
}

function Install-DefaultBaseModel {
    if (-not ($WithDefaultModel -or $FullImageStack)) {
        Write-Host "Skipping default SD 1.5 model download. Use -WithDefaultModel or -FullImageStack to install it."
        return
    }

    Write-Section "Default image model"
    if ($DryRun) {
        Write-Host "[dry-run] Would install Stable Diffusion 1.5 fp16 pruned base model if missing."
        return
    }
    Invoke-External "Install default SD 1.5 base model" $VenvPython @(
        "-c",
        "import runpy; runpy.run_path(r'scripts\ensure_default_sd15.py', run_name='__main__')"
    )
}

function New-DesktopShortcut {
    param(
        [string]$Name,
        [string]$TargetPath,
        [string]$Arguments = "",
        [string]$IconPath,
        [string]$Description,
        [string]$Directory = ""
    )

    if ([string]::IsNullOrWhiteSpace($Directory)) {
        $Directory = [Environment]::GetFolderPath("DesktopDirectory")
        if ([string]::IsNullOrWhiteSpace($Directory)) {
            $shell = New-Object -ComObject WScript.Shell
            $Directory = $shell.SpecialFolders("Desktop")
        }
    }
    if (-not (Test-Path $Directory)) {
        if ($DryRun) {
            Write-Host "[dry-run] Would create shortcut folder: $Directory"
        } else {
            New-Item -ItemType Directory -Force -Path $Directory | Out-Null
        }
    }
    $shortcutPath = Join-Path $Directory "$Name.lnk"
    if ($DryRun) {
        Write-Host "[dry-run] Shortcut: $shortcutPath"
        Write-Host "          Target: $TargetPath"
        if ($Arguments) {
            Write-Host "          Args:   $Arguments"
        }
        Write-Host "          Icon:   $IconPath"
        return
    }

    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $TargetPath
    $shortcut.Arguments = $Arguments
    $shortcut.WorkingDirectory = $Root
    $shortcut.IconLocation = "$IconPath,0"
    $shortcut.Description = $Description
    $shortcut.WindowStyle = 7
    $shortcut.Save()
    try {
        $shortcut.Refresh()
    } catch {
    }
    Write-Host "Created shortcut: $shortcutPath"
}

function Refresh-DesktopShell {
    param([string]$Path)

    try {
        Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class AiwfShellNotify {
    [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
    public static extern void SHChangeNotify(uint wEventId, uint uFlags, string dwItem1, IntPtr dwItem2);
}
"@ -ErrorAction SilentlyContinue | Out-Null
        [AiwfShellNotify]::SHChangeNotify(0x02000000, 0x0005, $Path, [IntPtr]::Zero)
        [AiwfShellNotify]::SHChangeNotify(0x08000000, 0x0000, $null, [IntPtr]::Zero)
    } catch {
    }
}

function Install-DesktopShortcuts {
    if ($TaskLocalMode) {
        Write-Host "Skipping Desktop and Start Menu shortcuts in task-local mode."
        return
    }
    Write-Section "Desktop shortcuts"
    New-DesktopShortcut `
        -Name "AIWF Studio Pro" `
        -TargetPath (Join-Path $Root "AIWF Studio Pro.vbs") `
        -IconPath (Join-Path $Root "static\icons\aiwf-studio-pro.ico") `
        -Description "AIWF Studio Pro production React app"
    Refresh-DesktopShell -Path (Join-Path ([Environment]::GetFolderPath("DesktopDirectory")) "AIWF Studio Pro.lnk")

    New-DesktopShortcut `
        -Name "AIWF Studio Gradio Lab" `
        -TargetPath (Join-Path $Root "AIWF Studio Gradio Lab.vbs") `
        -IconPath (Join-Path $Root "static\icons\aiwf-studio-gradio-lab.ico") `
        -Description "AIWF Studio Gradio Lab for WIP features"
    Refresh-DesktopShell -Path (Join-Path ([Environment]::GetFolderPath("DesktopDirectory")) "AIWF Studio Gradio Lab.lnk")

    Write-Section "Start Menu shortcuts"
    $startMenuDir = Join-Path ([Environment]::GetFolderPath("Programs")) "AIWF Studio"
    New-DesktopShortcut `
        -Name "AIWF Studio Pro" `
        -TargetPath (Join-Path $Root "AIWF Studio Pro.vbs") `
        -IconPath (Join-Path $Root "static\icons\aiwf-studio-pro.ico") `
        -Description "AIWF Studio Pro production React app" `
        -Directory $startMenuDir

    New-DesktopShortcut `
        -Name "AIWF Studio Gradio Lab" `
        -TargetPath (Join-Path $Root "AIWF Studio Gradio Lab.vbs") `
        -IconPath (Join-Path $Root "static\icons\aiwf-studio-gradio-lab.ico") `
        -Description "AIWF Studio Gradio Lab for WIP features" `
        -Directory $startMenuDir
}

function Install-NvidiaVideoFx {
    if (-not ($WithNvidiaVideoFx -or $FullImageStack)) {
        Write-Host "Skipping NVIDIA VideoFX SDK linking. Use -WithNvidiaVideoFx or -FullImageStack after installing the SDK locally."
        return
    }

    Write-Section "NVIDIA VideoFX (VSR) SDK"
    $enginesDir = Join-Path $Root "engines"
    $sdkLink = Join-Path $enginesDir "nvidia-vfx-sdk"
    $samplesLink = Join-Path $enginesDir "nvidia-vfx-sdk-samples"
    $anchor = (Get-Item $Root).PSDrive.Root

    $sdkCandidates = @(
        "$env:ProgramFiles\NVIDIA Corporation\NVIDIA Video Effects",
        (Join-Path $anchor "VideoFX"),
        (Join-Path $anchor "sdks\nvidia\VideoFX")
    )
    $samplesCandidates = @(
        (Join-Path $anchor "sdks\nvidia\nvidia-vfx-sdk-samples")
    )

    $sdkRoot = $sdkCandidates | Where-Object { Test-Path (Join-Path $_ "bin\NVVideoEffects.dll") } | Select-Object -First 1
    $samplesRoot = $samplesCandidates | Where-Object {
        Test-Path (Join-Path $_ "build\apps\VideoEffectsApp\Release\VideoEffectsApp.exe")
    } | Select-Object -First 1

    if (-not $sdkRoot) {
        Write-Host "NVIDIA Video Effects SDK runtime was not found."
        Write-Host "VSR upscaling stays disabled until the SDK is installed:"
        Write-Host "  1. Download the NVIDIA Video Effects SDK (Maxine VideoFX) for your GPU generation."
        Write-Host "  2. Install it, then run features\install_feature.ps1 for nvvfxvideosuperres and nvvfxupscale."
        Write-Host "  3. Re-run this installer; it links the SDK into engines\ automatically."
        return
    }

    if ($DryRun) {
        Write-Host "[dry-run] Would link $sdkLink -> $sdkRoot"
        if ($samplesRoot) { Write-Host "[dry-run] Would link $samplesLink -> $samplesRoot" }
        return
    }

    New-Item -ItemType Directory -Force $enginesDir | Out-Null
    if (-not (Test-Path $sdkLink)) {
        New-Item -ItemType Junction -Path $sdkLink -Target $sdkRoot | Out-Null
        Write-Host "Linked VideoFX SDK: $sdkLink -> $sdkRoot"
    } else {
        Write-Host "VideoFX SDK link already present: $sdkLink"
    }
    if ($samplesRoot -and -not (Test-Path $samplesLink)) {
        New-Item -ItemType Junction -Path $samplesLink -Target $samplesRoot | Out-Null
        Write-Host "Linked VideoFX sample apps: $samplesLink -> $samplesRoot"
    } elseif (-not $samplesRoot) {
        Write-Host "Built VideoFX sample apps (VideoEffectsApp.exe) were not found."
        Write-Host "Build NVIDIA-Maxine/VFX-SDK-Samples once, or set AIWF_VSR_VIDEO_EFFECTS_APP to a built binary."
    }

    $modelsDir = Join-Path $sdkRoot "bin\models"
    if (Test-Path $modelsDir) {
        $modelCount = (Get-ChildItem $modelsDir -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host "VideoFX feature models detected: $modelCount package(s)."
    } else {
        Write-Host "No VideoFX feature models found yet. Run features\install_feature.ps1 in the SDK to install VSR models."
    }
}

function Show-InstallerFailureDialog {
    param(
        [string]$Title,
        [string]$Summary,
        [string]$Details
    )

    try {
        Add-Type -AssemblyName System.Drawing, System.Windows.Forms
        [System.Windows.Forms.Application]::EnableVisualStyles()

        $form = New-Object System.Windows.Forms.Form
        $form.Text = $Title
        $form.StartPosition = "CenterScreen"
        $form.Size = New-Object System.Drawing.Size(920, 680)
        $form.MinimumSize = New-Object System.Drawing.Size(760, 520)
        $form.TopMost = $true
        $form.FormBorderStyle = "SizableToolWindow"
        $form.MaximizeBox = $false
        $form.MinimizeBox = $false

        $layout = New-Object System.Windows.Forms.TableLayoutPanel
        $layout.Dock = "Fill"
        $layout.ColumnCount = 1
        $layout.RowCount = 4
        $layout.Padding = New-Object System.Windows.Forms.Padding(18)
        $layout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::AutoSize)))
        $layout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::AutoSize)))
        $layout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::Percent, 100)))
        $layout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::AutoSize)))
        $form.Controls.Add($layout)

        $titleLabel = New-Object System.Windows.Forms.Label
        $titleLabel.AutoSize = $true
        $titleLabel.MaximumSize = New-Object System.Drawing.Size(860, 0)
        $titleLabel.Font = New-Object System.Drawing.Font("Segoe UI", 11, [System.Drawing.FontStyle]::Bold)
        $titleLabel.Text = $Summary
        $layout.Controls.Add($titleLabel, 0, 0)

        $hintLabel = New-Object System.Windows.Forms.Label
        $hintLabel.AutoSize = $true
        $hintLabel.Margin = New-Object System.Windows.Forms.Padding(0, 8, 0, 8)
        $hintLabel.MaximumSize = New-Object System.Drawing.Size(860, 0)
        $hintLabel.Text = "Copy the details below before closing this window."
        $layout.Controls.Add($hintLabel, 0, 1)

        $detailsBox = New-Object System.Windows.Forms.TextBox
        $detailsBox.Dock = "Fill"
        $detailsBox.Multiline = $true
        $detailsBox.ReadOnly = $true
        $detailsBox.ScrollBars = "Vertical"
        $detailsBox.WordWrap = $false
        $detailsBox.Font = New-Object System.Drawing.Font("Consolas", 9)
        $detailsBox.Text = $Details
        $layout.Controls.Add($detailsBox, 0, 2)

        $buttonRow = New-Object System.Windows.Forms.FlowLayoutPanel
        $buttonRow.Dock = "Fill"
        $buttonRow.AutoSize = $true
        $buttonRow.FlowDirection = "RightToLeft"
        $buttonRow.WrapContents = $false
        $buttonRow.Margin = New-Object System.Windows.Forms.Padding(0, 12, 0, 0)

        $closeButton = New-Object System.Windows.Forms.Button
        $closeButton.Text = "Close"
        $closeButton.AutoSize = $true
        $closeButton.DialogResult = [System.Windows.Forms.DialogResult]::OK
        $buttonRow.Controls.Add($closeButton)

        $copyButton = New-Object System.Windows.Forms.Button
        $copyButton.Text = "Copy details"
        $copyButton.AutoSize = $true
        $copyButton.Add_Click({
            try {
                [System.Windows.Forms.Clipboard]::SetText($detailsBox.Text)
                $copyButton.Text = "Copied"
            } catch {
                $copyButton.Text = "Copy failed"
            }
        })
        $buttonRow.Controls.Add($copyButton)

        $layout.Controls.Add($buttonRow, 0, 3)
        $form.AcceptButton = $copyButton
        $form.CancelButton = $closeButton
        [void]$form.ShowDialog()
    } catch {
        Write-Host $Title
        Write-Host $Summary
        Write-Host $Details
    }
}

function Read-InstallerMode {
    Write-Host "AIWF Studio installer"
    Write-Host ""
    Write-Host "Express installs or checks Git, uv, Python $PythonVersion, Node.js LTS, the app runtime, the Pro frontend, and Desktop shortcuts."
    Write-Host "Full does Express plus the default SD 1.5 model and optional NVIDIA VideoFX SDK link checks."
    Write-Host "Custom is the existing manual path: use the .bat files or Python launch commands yourself."
    Write-Host ""
    $choice = Read-Host "Choose [E]xpress, [F]ull image stack, [C]ustom, or [Q]uit"
    if ([string]::IsNullOrWhiteSpace($choice)) {
        return "express"
    }
    switch ($choice.Trim().Substring(0, 1).ToLowerInvariant()) {
        "e" { return "express" }
        "f" { return "full" }
        "c" { return "custom" }
        "q" { return "quit" }
        default { return "express" }
    }
}

$TaskLocalEnvironmentNames = @("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_PYTHON_BIN_DIR", "PIP_CACHE_DIR", "NPM_CONFIG_CACHE", "CONDA_PKGS_DIRS", "CONDARC")
$PreviousTaskLocalEnvironment = @{}
try {
    Set-Location -LiteralPath $Root
    if ($UseBackendLock) {
        Assert-BackendLockHost
        if ($ShortcutsOnly -or $SkipRuntimeSetup -or $Mode -eq "custom") {
            throw "-UseBackendLock requires an Express or Full install with Python runtime setup enabled."
        }
    }
    if ($TaskLocalMode) {
        $localRoot = Join-Path $Root "_local"
        $taskLocalWritePaths = @(
            $localRoot,
            (Join-Path $localRoot "uv-cache"),
            (Join-Path $localRoot "uv-python"),
            (Join-Path $localRoot "uv-python-bin"),
            (Join-Path $localRoot "pip-cache"),
            (Join-Path $localRoot "npm-cache"),
            (Join-Path $localRoot "conda-pkgs"),
            (Join-Path $localRoot "condarc"),
            $VenvDir,
            (Join-Path $Root "_pyenv$($PythonVersion.Replace('.',''))"),
            (Join-Path $Root "_trash"),
            (Join-Path $Root "frontend"),
            (Join-Path $Root "frontend\node_modules"),
            (Join-Path $Root "frontend\dist")
        )
        foreach ($writePath in $taskLocalWritePaths) { Assert-TaskLocalWritePath -Path $writePath }
        foreach ($name in $TaskLocalEnvironmentNames) { $PreviousTaskLocalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process") }
        [Environment]::SetEnvironmentVariable("UV_CACHE_DIR", (Join-Path $localRoot "uv-cache"), "Process")
        [Environment]::SetEnvironmentVariable("UV_PYTHON_INSTALL_DIR", (Join-Path $localRoot "uv-python"), "Process")
        [Environment]::SetEnvironmentVariable("UV_PYTHON_BIN_DIR", (Join-Path $localRoot "uv-python-bin"), "Process")
        [Environment]::SetEnvironmentVariable("PIP_CACHE_DIR", (Join-Path $localRoot "pip-cache"), "Process")
        [Environment]::SetEnvironmentVariable("NPM_CONFIG_CACHE", (Join-Path $localRoot "npm-cache"), "Process")
        [Environment]::SetEnvironmentVariable("CONDA_PKGS_DIRS", (Join-Path $localRoot "conda-pkgs"), "Process")
        [Environment]::SetEnvironmentVariable("CONDARC", (Join-Path $localRoot "condarc"), "Process")
    }
    if (-not $TaskLocalMode) { Update-ProcessPath }

    if ($Mode -eq "prompt" -and -not $ShortcutsOnly) {
        $Mode = Read-InstallerMode
    }

    if ($Mode -eq "quit") {
        Write-Host "Install cancelled."
        exit 0
    }

    if ($UseBackendLock -and ($Mode -in @("custom", "quit") -or $ShortcutsOnly -or $SkipRuntimeSetup)) {
        throw "-UseBackendLock requires an Express or Full install with Python runtime setup enabled."
    }

    if ($Mode -eq "custom") {
        Write-Host "Custom install is the existing manual path:"
        Write-Host "  AIWF Studio Pro.bat"
        Write-Host "  AIWF Studio Gradio Lab.bat"
        Write-Host "  python launch_pro.py"
        Write-Host "  python launch_gradio.py"
        exit 0
    }

    if ($Mode -eq "full") {
        $FullImageStack = $true
    }

    if ($ShortcutsOnly) {
        Install-DesktopShortcuts
        exit 0
    }

    Write-Section "Prerequisites"
    if (-not $SkipPrerequisites) {
        Ensure-WingetPackage -Label "Git" -Id "Git.Git" -Commands @("git")
        Ensure-WingetPackage -Label "uv Python manager" -Id "astral-sh.uv" -Commands @("uv")
        Ensure-WingetPackage -Label "Node.js LTS" -Id "OpenJS.NodeJS.LTS" -Commands @("node", "npm")
    } else {
        Write-Host "Skipping prerequisite installation."
    }

    Invoke-WithInstallerEnvironmentLock -Action {
        Ensure-PythonVenv
        Prepare-AiwfRuntime
        Install-DefaultBaseModel
        Build-ProFrontend
    }
    Install-NvidiaVideoFx
    Install-DesktopShortcuts

    Write-Section "Done"
    if ($TaskLocalMode) {
        Write-Host "Task-local installation completed at $Root. No shortcuts or model downloads were performed."
    } else {
        Write-Host "Use the Desktop shortcuts:"
        Write-Host "  AIWF Studio Pro"
        Write-Host "  AIWF Studio Gradio Lab"
    }
} catch {
    $errorRecord = $_
    $summary = if ($errorRecord.Exception -and $errorRecord.Exception.Message) {
        $errorRecord.Exception.Message
    } else {
        "The installer failed."
    }
    $details = @(
        "AIWF Studio installer failed."
        ""
        "Summary:"
        $summary
        ""
        "Details:"
        ($errorRecord | Out-String).TrimEnd()
    ) -join [Environment]::NewLine

    Show-InstallerFailureDialog -Title "AIWF Studio installer failed" -Summary $summary -Details $details
    exit 1
} finally {
    if ($TaskLocalMode) {
        foreach ($name in $TaskLocalEnvironmentNames) {
            [Environment]::SetEnvironmentVariable($name, $PreviousTaskLocalEnvironment[$name], "Process")
        }
    }
}
