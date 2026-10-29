# Builds AIWF Studio for Windows (native shell) with Visual Studio 2022 Build Tools or Visual Studio.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File F:\AIWF_Studio\native\build.ps1 [-Configuration Release] [-SourceRoot F:\AIWF_Studio] [-Run]
#
# Steps: find MSBuild with vswhere, restore the locked PackageReference graph from nuget.org,
# then build x64 and write a source/executable compatibility manifest. Output:
# .\bin\<Configuration>\AESStudio.exe, native-build-info.json, and the self-contained Windows
# App SDK runtime. Needs the "Desktop development with C++" workload (MSVC v143).
# Windows SDK 10.0.26100 contracts and build tools are restored as project-local NuGet packages;
# no global Windows SDK install or UWP C++ component is needed (see AIWF.Desktop\DesktopBuildHooks.targets).
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release',
    [string]$SourceRoot,
    [string]$RestoreSource = 'https://api.nuget.org/v3/index.json',
    [switch]$Offline,
    [switch]$Run
)
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($RestoreSource)) {
    throw 'RestoreSource must name a NuGet feed or a local offline feed directory.'
}
if ($Offline -and -not (Test-Path -LiteralPath $RestoreSource -PathType Container)) {
    throw 'Offline builds require RestoreSource to be an existing local NuGet feed directory.'
}
$root = $PSScriptRoot
$sourceRoot = if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    Split-Path -Parent $root
} else {
    $SourceRoot
}
$sourceRoot = [System.IO.Path]::GetFullPath($sourceRoot)
$scriptSourceRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $root))
$sourceRootCompare = $sourceRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
$scriptRootCompare = $scriptSourceRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
if (-not [string]::Equals($sourceRootCompare, $scriptRootCompare, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to build native sources from '$root' against a different API checkout '$sourceRoot'. Run this build script from the selected Studio checkout or use a prebuilt payload that verifies against it."
}

# the build metadata binds this native binary to the AIWF Python API source tree
$nativeCompat = Join-Path $sourceRoot 'aiwf\native_compat.py'
if (-not (Test-Path -LiteralPath $nativeCompat -PathType Leaf)) {
    throw "AIWF native build metadata helper was not found at '$nativeCompat'. Pass -SourceRoot for the AIWF Studio checkout used by this build."
}

$pythonCommand = $null
$pythonPrefix = @()
$venvPython = Join-Path $sourceRoot 'venv\Scripts\python.exe'
if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $pythonCommand = $venvPython
} else {
    $pythonLauncher = Get-Command 'py.exe' -ErrorAction SilentlyContinue
    if ($null -ne $pythonLauncher) {
        $pythonCommand = $pythonLauncher.Source
        $pythonPrefix = @('-3.12')
    } else {
        throw "Python 3.12 is required to write native build metadata. Run the native engine API bootstrap or install CPython 3.12, then retry."
    }
}
$versionArgs = $pythonPrefix + @('-B', '-c', 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))')
& $pythonCommand @versionArgs
if ($LASTEXITCODE -ne 0) {
    throw "Native build metadata requires CPython 3.12. Interpreter '$pythonCommand' did not report the supported major/minor version."
}
Push-Location $sourceRoot
try {
    $sourcePreflightArgs = $pythonPrefix + @('-B', '-c', 'import sys; from aiwf.native_compat import compute_api_source_files; compute_api_source_files(sys.argv[1])', $sourceRoot)
    & $pythonCommand @sourcePreflightArgs
    if ($LASTEXITCODE -ne 0) { throw "Native API source preflight failed ($LASTEXITCODE) for '$sourceRoot'." }
} finally {
    Pop-Location
}

# this block locates MSBuild through vswhere, which ships with every Visual Studio 2017+ installer
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found. Install Visual Studio 2022 Build Tools with the C++ workload." }
$msbuild = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -find 'MSBuild\**\Bin\MSBuild.exe' | Select-Object -First 1
if (-not $msbuild) { throw "MSBuild was not found by vswhere." }
Write-Host "MSBuild: $msbuild"

# restore exact SDK package versions and enforce the committed transitive dependency graph
$restoreArgs = @((Join-Path $root 'AIWF.Desktop.sln'), '-t:restore', '-p:RestorePackagesConfig=false',
    '-p:RestorePackagesWithLockFile=true', '-p:RestoreLockedMode=true', "-p:RestoreSources=$RestoreSource",
    "-p:Configuration=$Configuration", '-p:Platform=x64', '-nologo', '-v:minimal')
if ($Offline) { $restoreArgs += '-p:NuGetAudit=false' }
& $msbuild @restoreArgs
if ($LASTEXITCODE -ne 0) { throw "Package restore failed ($LASTEXITCODE)." }

# this block builds the app
& $msbuild (Join-Path $root 'AIWF.Desktop.sln') -p:Configuration=$Configuration -p:Platform=x64 -p:PreferredToolArchitecture=x64 -nologo -v:minimal -m:1
if ($LASTEXITCODE -ne 0) { throw "Build failed ($LASTEXITCODE)." }

$exe = Join-Path $root "bin\$Configuration\AESStudio.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "MSBuild succeeded but the expected native executable is missing: $exe"
}
$buildInfo = Join-Path (Split-Path -Parent $exe) 'native-build-info.json'
Push-Location $sourceRoot
try {
    $writeArgs = $pythonPrefix + @('-B', '-m', 'aiwf.native_compat', 'write', '--source-root', $sourceRoot, '--output', $buildInfo, '--exe', $exe)
    & $pythonCommand @writeArgs
    if ($LASTEXITCODE -ne 0) { throw "Writing native build metadata failed ($LASTEXITCODE)." }
    $verifyArgs = $pythonPrefix + @('-B', '-m', 'aiwf.native_compat', 'verify', '--source-root', $sourceRoot, '--build-info', $buildInfo, '--exe', $exe)
    & $pythonCommand @verifyArgs
    if ($LASTEXITCODE -ne 0) { throw "Native build metadata verification failed ($LASTEXITCODE)." }
} finally {
    Pop-Location
}
Write-Host "Built: $exe"
Write-Host "Build metadata: $buildInfo"
if ($Run) { Start-Process -FilePath $exe }

