[CmdletBinding()]
param(
    [string]$StudioRoot,
    [string]$InstallDirectory,
    [string]$BuildOutputDirectory,
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release',
    [switch]$SkipBuild,
    [switch]$NoShortcut
)

$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($StudioRoot)) { $StudioRoot = (Resolve-Path (Join-Path $scriptRoot '..')).Path }
$root = (Resolve-Path -LiteralPath $StudioRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $root 'aiwf\engine_api.py') -PathType Leaf)) {
    throw "StudioRoot must be an AIWF Studio checkout containing aiwf\engine_api.py: $root"
}
$source = if ([string]::IsNullOrWhiteSpace($BuildOutputDirectory)) { Join-Path $scriptRoot "bin\$Configuration" } else { (Resolve-Path -LiteralPath $BuildOutputDirectory).Path }
if (-not $SkipBuild) {
    & (Join-Path $scriptRoot 'build.ps1') -Configuration $Configuration
    if ($LASTEXITCODE -ne 0) { throw "Native build failed ($LASTEXITCODE)." }
}
if (-not (Test-Path -LiteralPath (Join-Path $source 'AIWFStudio.exe') -PathType Leaf)) {
    throw "Native build output is missing: $source\AIWFStudio.exe"
}

function Test-NativeVcRuntime {
    param([string]$BuildDirectory, [string]$SystemDirectory)
    $missing = @('MSVCP140.dll', 'VCRUNTIME140.dll', 'VCRUNTIME140_1.dll') | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $BuildDirectory $_) -PathType Leaf) -and
        -not (Test-Path -LiteralPath (Join-Path $SystemDirectory $_) -PathType Leaf)
    }
    if ($missing) {
        throw "Install the Microsoft Visual C++ 2015-2022 x64 Redistributable before installing AIWF Studio. Missing: $($missing -join ', ')"
    }
}
$systemDirectory = if ([Environment]::Is64BitProcess) { Join-Path $env:WINDIR 'System32' } else { Join-Path $env:WINDIR 'Sysnative' }
Test-NativeVcRuntime -BuildDirectory $source -SystemDirectory $systemDirectory

$installRoot = if ([string]::IsNullOrWhiteSpace($InstallDirectory)) { Join-Path $root 'native\installed' } else { [System.IO.Path]::GetFullPath($InstallDirectory) }
$rootPrefix = $root.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
if (-not $installRoot.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Install directory must stay inside the selected StudioRoot so the app can discover its local backend: $installRoot"
}
$marker = Join-Path $installRoot '.aiwf-native-install'
if ((Test-Path -LiteralPath $installRoot) -and -not (Test-Path -LiteralPath $marker -PathType Leaf)) {
    throw "Refusing to copy over an unmarked existing folder: $installRoot"
}
New-Item -ItemType Directory -Path $installRoot -Force | Out-Null
Copy-Item -Path (Join-Path $source '*') -Destination $installRoot -Recurse -Force
Copy-Item -LiteralPath (Join-Path $scriptRoot 'AIWF.Desktop\engines.json') -Destination (Join-Path $installRoot 'engines.json') -Force
Set-Content -LiteralPath $marker -Value "AIWF Studio for Windows install; StudioRoot=$root" -Encoding utf8

$exe = Join-Path $installRoot 'AIWFStudio.exe'
if (-not $NoShortcut) {
    $desktop = [Environment]::GetFolderPath('DesktopDirectory')
    $shortcutPath = Join-Path $desktop 'AIWF Studio for Windows.lnk'
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $exe
    $shortcut.WorkingDirectory = $installRoot
    $shortcut.Description = 'AIWF Studio for Windows'
    $shortcut.Save()
}

Write-Host "Installed: $exe"
Write-Host "Studio root: $root"
Write-Host "User data and optional engine overrides remain in %LOCALAPPDATA%\AIWF Studio."
