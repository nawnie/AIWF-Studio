# Builds AIWF Studio for Windows (native shell) with Visual Studio 2022 Build Tools or Visual Studio.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File F:\AIWF_Studio\native\build.ps1 [-Configuration Release] [-Run]
#
# Steps: find MSBuild with vswhere, restore the NuGet packages listed in packages.config into
# .\packages, then build x64. Output: .\bin\<Configuration>\AIWFStudio.exe plus the self-contained
# Windows App SDK runtime beside it. Needs the "Desktop development with C++" workload (MSVC v143)
# and a Windows 10/11 SDK; the UWP C++ component is NOT needed (see AIWF.Desktop\DesktopBuildHooks.targets).
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release',
    [switch]$Run
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# this block locates MSBuild through vswhere, which ships with every Visual Studio 2017+ installer
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found. Install Visual Studio 2022 Build Tools with the C++ workload." }
$msbuild = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -find 'MSBuild\**\Bin\MSBuild.exe' | Select-Object -First 1
if (-not $msbuild) { throw "MSBuild was not found by vswhere." }
Write-Host "MSBuild: $msbuild"

# this block restores packages.config packages (Windows App SDK, C++/WinRT, WIL, SDK build tools)
& $msbuild (Join-Path $root 'AIWF.Desktop.sln') -t:restore -p:RestorePackagesConfig=true -p:Configuration=$Configuration -p:Platform=x64 -nologo -v:minimal
if ($LASTEXITCODE -ne 0) { throw "Package restore failed ($LASTEXITCODE)." }

# this block builds the app
& $msbuild (Join-Path $root 'AIWF.Desktop.sln') -p:Configuration=$Configuration -p:Platform=x64 -nologo -v:minimal -m
if ($LASTEXITCODE -ne 0) { throw "Build failed ($LASTEXITCODE)." }

$exe = Join-Path $root "bin\$Configuration\AIWFStudio.exe"
Write-Host "Built: $exe"
if ($Run) { Start-Process -FilePath $exe }
