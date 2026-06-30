[CmdletBinding()]
param(
    [ValidateSet("prompt", "express", "custom", "quit")]
    [string]$Mode = "prompt",
    [switch]$DryRun,
    [switch]$ShortcutsOnly,
    [switch]$SkipPrerequisites,
    [switch]$SkipFrontendBuild
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvDir = Join-Path $Root "venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$PythonVersion = "3.10"

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "== $Message =="
}

function Update-ProcessPath {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $extra = @(
        "$env:ProgramFiles\Git\cmd",
        "$env:ProgramFiles\nodejs",
        "$env:USERPROFILE\.local\bin",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links"
    )
    $env:Path = (($machine, $user) + $extra | Where-Object { $_ }) -join ";"
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

function Ensure-PythonVenv {
    Write-Section "Python environment"
    if ($DryRun) {
        Invoke-External "Install Python $PythonVersion with uv" "uv" @("python", "install", $PythonVersion)
        Invoke-External "Create AIWF venv" "uv" @("venv", "--python", $PythonVersion, $VenvDir)
        Invoke-External "Seed pip" "uv" @("pip", "install", "--python", $VenvPython, "pip", "setuptools", "wheel")
        return
    }

    Invoke-External "Install Python $PythonVersion with uv" "uv" @("python", "install", $PythonVersion)

    $minor = Get-VenvPythonMinor
    if ($minor -and $minor -ne $PythonVersion) {
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $trash = Join-Path $Root "_trash\installer-venv-$stamp"
        New-Item -ItemType Directory -Path (Split-Path $trash -Parent) -Force | Out-Null
        Write-Host "Existing venv uses Python $minor. Moving it to $trash"
        Move-Item -LiteralPath $VenvDir -Destination $trash
        $minor = ""
    }

    if (-not $minor) {
        Invoke-External "Create AIWF venv" "uv" @("venv", "--python", $PythonVersion, $VenvDir)
    } else {
        Write-Host "AIWF venv already uses Python $minor."
    }

    if (-not (Test-Path -LiteralPath $VenvPython)) {
        throw "Expected venv Python was not created: $VenvPython"
    }
    Invoke-External "Seed pip" "uv" @("pip", "install", "--python", $VenvPython, "pip", "setuptools", "wheel")
}

function Prepare-AiwfRuntime {
    Write-Section "AIWF Python dependencies"
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
    Write-Section "Default image model"
    if ($DryRun) {
        Write-Host "[dry-run] Would install Stable Diffusion 1.5 fp16 pruned base model if missing."
        return
    }
    Invoke-External "Install default SD 1.5 base model" $VenvPython @(
        "scripts\ensure_default_sd15.py"
    )
}

function New-DesktopShortcut {
    param(
        [string]$Name,
        [string]$TargetPath,
        [string]$IconPath,
        [string]$Description
    )

    $desktop = [Environment]::GetFolderPath("Desktop")
    $shortcutPath = Join-Path $desktop "$Name.lnk"
    if ($DryRun) {
        Write-Host "[dry-run] Shortcut: $shortcutPath"
        Write-Host "          Target: $TargetPath"
        Write-Host "          Icon:   $IconPath"
        return
    }

    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $TargetPath
    $shortcut.WorkingDirectory = $Root
    $shortcut.IconLocation = "$IconPath,0"
    $shortcut.Description = $Description
    $shortcut.Save()
    Write-Host "Created Desktop shortcut: $shortcutPath"
}

function Install-DesktopShortcuts {
    Write-Section "Desktop shortcuts"
    New-DesktopShortcut `
        -Name "AIWF Studio Pro" `
        -TargetPath (Join-Path $Root "AIWF Studio Pro.bat") `
        -IconPath (Join-Path $Root "static\icons\aiwf-studio-pro.ico") `
        -Description "AIWF Studio Pro production React app"

    New-DesktopShortcut `
        -Name "AIWF Studio Gradio Lab" `
        -TargetPath (Join-Path $Root "AIWF Studio Gradio Lab.bat") `
        -IconPath (Join-Path $Root "static\icons\aiwf-studio-gradio-lab.ico") `
        -Description "AIWF Studio Gradio Lab for WIP features"
}

function Read-InstallerMode {
    Write-Host "AIWF Studio installer"
    Write-Host ""
    Write-Host "Express installs or checks Git, uv, Python $PythonVersion, Node.js LTS, app dependencies, the Pro frontend, and Desktop shortcuts."
    Write-Host "Custom is the existing manual path: use the .bat files or Python launch commands yourself."
    Write-Host ""
    $choice = Read-Host "Choose [E]xpress, [C]ustom, or [Q]uit"
    if ([string]::IsNullOrWhiteSpace($choice)) {
        return "express"
    }
    switch ($choice.Trim().Substring(0, 1).ToLowerInvariant()) {
        "e" { return "express" }
        "c" { return "custom" }
        "q" { return "quit" }
        default { return "express" }
    }
}

Update-ProcessPath

if ($Mode -eq "prompt" -and -not $ShortcutsOnly) {
    $Mode = Read-InstallerMode
}

if ($Mode -eq "quit") {
    Write-Host "Install cancelled."
    exit 0
}

if ($Mode -eq "custom") {
    Write-Host "Custom install is the existing manual path:"
    Write-Host "  AIWF Studio Pro.bat"
    Write-Host "  AIWF Studio Gradio Lab.bat"
    Write-Host "  python launch_pro.py"
    Write-Host "  python launch_gradio.py"
    exit 0
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

Ensure-PythonVenv
Prepare-AiwfRuntime
Install-DefaultBaseModel
Build-ProFrontend
Install-DesktopShortcuts

Write-Section "Done"
Write-Host "Use the Desktop shortcuts:"
Write-Host "  AIWF Studio Pro"
Write-Host "  AIWF Studio Gradio Lab"
