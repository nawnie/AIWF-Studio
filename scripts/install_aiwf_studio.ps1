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
    [switch]$FullImageStack
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvDir = Join-Path $Root "venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$PythonVersion = "3.12"

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

function Move-StaleVenv {
    param([string]$Reason)
    if (-not (Test-Path -LiteralPath $VenvDir)) { return }
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $trash = Join-Path $Root "_trash\installer-venv-$stamp"
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

# Fallback provisioner: use conda to get the target Python, then build a
# STANDARD venv from it so the app still finds venv\Scripts\python.exe.
function Ensure-PythonVenv-Conda {
    Write-Host "uv could not provide Python $PythonVersion. Trying conda."
    $conda = Get-CondaCommand
    if (-not $conda) {
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
    if (Test-Path -LiteralPath $pyenv) { Remove-Item -Recurse -Force -LiteralPath $pyenv }
    Invoke-External "Create conda Python $PythonVersion" $conda @("create", "-y", "-p", $pyenv, "python=$PythonVersion")
    $condaPython = Join-Path $pyenv "python.exe"
    if (-not (Test-Path -LiteralPath $condaPython)) {
        throw "conda did not produce a Python at $condaPython"
    }
    Move-StaleVenv -Reason "Rebuilding the venv with the conda-provided Python $PythonVersion."
    Invoke-External "Create AIWF venv from conda Python" $condaPython @("-m", "venv", $VenvDir)
}

function Ensure-PythonVenv {
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
            if ($minor -and $minor -ne $PythonVersion) {
                Move-StaleVenv -Reason "Existing venv uses Python $minor (need $PythonVersion)."
                $minor = ""
            }
            if (-not $minor) {
                Invoke-External "Create AIWF venv" "uv" @("venv", "--python", $PythonVersion, $VenvDir)
            } else {
                Write-Host "AIWF venv already uses Python $minor."
            }
            if ((Get-VenvPythonMinor) -eq $PythonVersion) { $uvOk = $true }
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
    $finalMinor = Get-VenvPythonMinor
    if ($finalMinor -ne $PythonVersion) {
        throw "venv Python is $finalMinor but $PythonVersion is required. Install Python $PythonVersion or conda and re-run."
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
        Write-Host "Skipping Python runtime setup. Launching generation later may install runtime packages."
        return
    }

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

try {
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

    Ensure-PythonVenv
    Prepare-AiwfRuntime
    Install-DefaultBaseModel
    Install-NvidiaVideoFx
    Build-ProFrontend
    Install-DesktopShortcuts

    Write-Section "Done"
    Write-Host "Use the Desktop shortcuts:"
    Write-Host "  AIWF Studio Pro"
    Write-Host "  AIWF Studio Gradio Lab"
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
}
