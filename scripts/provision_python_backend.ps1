[CmdletBinding()]
param(
    [string]$StudioRoot = '',
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
$minimumPython = $null
$maximumPython = $null
$pythonConstraint = ''
$pythonCreationVersion = ''

function Get-CanonicalPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$Directory
    )

    if (-not ('AiwfPythonBackendPathResolver' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;

public static class AiwfPythonBackendPathResolver {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern SafeFileHandle CreateFile(
        string name, uint access, uint share, IntPtr securityAttributes,
        uint creationDisposition, uint flagsAndAttributes, IntPtr templateFile);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern uint GetFinalPathNameByHandle(
        SafeFileHandle file, StringBuilder path, uint pathLength, uint flags);

    public static string Resolve(string path, bool directory) {
        const uint FILE_READ_ATTRIBUTES = 0x80;
        const uint FILE_SHARE_ALL = 0x1 | 0x2 | 0x4;
        const uint OPEN_EXISTING = 3;
        const uint FILE_FLAG_BACKUP_SEMANTICS = 0x02000000;
        uint flags = directory ? FILE_FLAG_BACKUP_SEMANTICS : 0;
        using (SafeFileHandle handle = CreateFile(path, FILE_READ_ATTRIBUTES, FILE_SHARE_ALL,
            IntPtr.Zero, OPEN_EXISTING, flags, IntPtr.Zero)) {
            if (handle.IsInvalid) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
            int capacity = 512;
            while (true) {
                StringBuilder buffer = new StringBuilder(capacity);
                uint length = GetFinalPathNameByHandle(handle, buffer, (uint)capacity, 0);
                if (length == 0) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
                if (length < capacity) return buffer.ToString();
                capacity = checked((int)length + 1);
            }
        }
    }
}
'@ | Out-Null
    }

    $resolved = [AiwfPythonBackendPathResolver]::Resolve($Path, [bool]$Directory)
    if ($resolved.StartsWith('\\?\UNC\', [System.StringComparison]::OrdinalIgnoreCase)) {
        return '\\' + $resolved.Substring(8)
    }
    if ($resolved.StartsWith('\\?\', [System.StringComparison]::OrdinalIgnoreCase)) {
        return $resolved.Substring(4)
    }
    return $resolved
}

function Get-PythonProcessSnapshot {
    try {
        return @(Get-CimInstance -ClassName Win32_Process -Filter "Name LIKE 'python%.exe'" -ErrorAction Stop)
    } catch {
        throw 'Could not inspect running Python processes. Close AIWF Studio and retry after process inspection is available.'
    }
}

function Get-PythonProcessIdentityKey {
    param([Parameter(Mandatory = $true)][object]$Process)

    if ($null -eq $Process.ProcessId -or [string]::IsNullOrWhiteSpace([string]$Process.Name) -or
        [string]::IsNullOrWhiteSpace([string]$Process.CreationDate)) {
        return ''
    }
    $created = $Process.CreationDate
    if ($created -is [datetime]) { $created = $created.ToUniversalTime().Ticks }
    return '{0}|{1}|{2}' -f ([long]$Process.ProcessId), ([string]$Process.Name).ToLowerInvariant(), ([string]$created)
}

function Get-PythonProcesses {
    param(
        [hashtable]$KnownUnidentifiableProcesses = @{},
        [switch]$FreshTarget
    )

    $processes = @(Get-PythonProcessSnapshot)

    $resolved = @()
    foreach ($process in $processes) {
        if ([string]::IsNullOrWhiteSpace($process.ExecutablePath)) {
            $identity = Get-PythonProcessIdentityKey -Process $process
            if (-not [string]::IsNullOrWhiteSpace($identity) -and $KnownUnidentifiableProcesses.ContainsKey($identity)) {
                continue
            }
            if ($FreshTarget) {
                throw "Could not verify the interpreter path for new or unidentifiable Python process PID $($process.ProcessId) ($($process.Name)); refusing to change a possibly active backend environment."
            }
            throw "Could not verify the interpreter path for running Python process PID $($process.ProcessId) ($($process.Name)); preserving the existing venv. Close or resolve this process, or provision Python in a fresh versioned StudioRoot before engine-api bootstrap with scripts\provision_python_backend.ps1 -StudioRoot <fresh-root>."
        }
        try {
            $processPath = Get-CanonicalPath -Path $process.ExecutablePath
        } catch {
            throw "Could not resolve the interpreter path for running Python process PID $($process.ProcessId) ($($process.Name)); refusing to change a possibly active backend environment."
        }
        $resolved += [pscustomobject]@{
            ProcessId = $process.ProcessId
            Name = $process.Name
            ExecutablePath = $processPath
            ScriptsDirectory = [System.IO.Path]::GetDirectoryName($processPath)
        }
    }
    return $resolved
}

function Get-SelectedPythonProcesses {
    param(
        [Parameter(Mandatory = $true)][string]$ScriptsDirectory,
        [hashtable]$KnownUnidentifiableProcesses = @{},
        [switch]$FreshTarget
    )

    return @(Get-PythonProcesses -KnownUnidentifiableProcesses $KnownUnidentifiableProcesses -FreshTarget:$FreshTarget | Where-Object {
        [string]::Equals($_.ScriptsDirectory, $ScriptsDirectory, [System.StringComparison]::OrdinalIgnoreCase)
    })
}

function New-NativeVenvDirectoryExclusive {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not ('AiwfPythonBackendVenvDirectory' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;

public static class AiwfPythonBackendVenvDirectory {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true, EntryPoint = "CreateDirectoryW")]
    private static extern bool CreateDirectoryW(string path, IntPtr securityAttributes);

    public static void CreateNew(string path) {
        if (!CreateDirectoryW(path, IntPtr.Zero)) {
            int error = Marshal.GetLastWin32Error();
            if (error == 183) throw new IOException("The target directory already exists.");
            throw new Win32Exception(error);
        }
    }
}
'@ | Out-Null
    }

    try {
        [AiwfPythonBackendVenvDirectory]::CreateNew($Path)
    } catch {
        throw "Could not claim the new venv directory without overwriting an existing target at '$Path': $($_.Exception.Message)"
    }
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'The locked AIWF Python backend helper supports Windows AMD64 only.'
}
$architecture = $env:PROCESSOR_ARCHITEW6432
if ([string]::IsNullOrWhiteSpace($architecture)) { $architecture = $env:PROCESSOR_ARCHITECTURE }
if ($architecture -notin @('AMD64', 'x86_64')) {
    throw "The locked AIWF Python backend helper supports Windows AMD64 only; detected '$architecture'."
}

$sourceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if ([string]::IsNullOrWhiteSpace($StudioRoot)) {
    $root = $sourceRoot
} else {
    if (-not (Test-Path -LiteralPath $StudioRoot -PathType Container)) {
        throw "StudioRoot must be an existing AIWF Studio checkout: $StudioRoot"
    }
    $root = (Resolve-Path -LiteralPath $StudioRoot).Path
}
if (-not (Test-Path -LiteralPath (Join-Path $root 'aiwf\engine_api.py') -PathType Leaf)) {
    throw "StudioRoot must be an AIWF Studio checkout containing aiwf\engine_api.py: $root"
}

$lockDirectory = Join-Path ([System.IO.Path]::GetTempPath()) 'AIWF-Studio'
[void][System.IO.Directory]::CreateDirectory($lockDirectory)
$hashAlgorithm = [System.Security.Cryptography.SHA256]::Create()
try {
    $rootIdentity = (Get-CanonicalPath -Path $root -Directory).TrimEnd('\', '/').ToUpperInvariant()
    $rootHash = [System.BitConverter]::ToString($hashAlgorithm.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($rootIdentity))).Replace('-', '').ToLowerInvariant()
} finally {
    $hashAlgorithm.Dispose()
}
$bootstrapLockPath = Join-Path $lockDirectory "python-backend-provision-$rootHash.lock"
$bootstrapLock = $null
try {
    try {
        # FileShare.None serializes Python-backend setup for this StudioRoot.
        $bootstrapLock = [System.IO.File]::Open(
            $bootstrapLockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    } catch {
        throw "Another Python-backend provisioner is active for StudioRoot '$root', or its per-root lock is unavailable: $($_.Exception.Message)"
    }

$lockProject = Join-Path $root 'dependencies\windows-py312'
foreach ($required in @('pyproject.toml', 'uv.lock')) {
    if (-not (Test-Path -LiteralPath (Join-Path $lockProject $required) -PathType Leaf)) {
        throw "The locked Windows Python backend project is incomplete: $lockProject\$required"
    }
}
$projectPythonRequirement = [regex]::Match((Get-Content -LiteralPath (Join-Path $lockProject 'pyproject.toml') -Raw), '(?m)^\s*requires-python\s*=\s*"([^"]+)"')
$lockPythonRequirement = [regex]::Match((Get-Content -LiteralPath (Join-Path $lockProject 'uv.lock') -Raw), '(?m)^\s*requires-python\s*=\s*"([^"]+)"')
if (-not $projectPythonRequirement.Success -or -not $lockPythonRequirement.Success) {
    throw "Could not read requires-python from the Windows backend project and lock: $lockProject"
}
$projectConstraint = $projectPythonRequirement.Groups[1].Value -replace '\s', ''
$pythonConstraint = $lockPythonRequirement.Groups[1].Value -replace '\s', ''
if (-not [string]::Equals($projectConstraint, $pythonConstraint, [System.StringComparison]::Ordinal)) {
    throw "The Python range in pyproject.toml ('$projectConstraint') does not match uv.lock ('$pythonConstraint'). Regenerate and review the lock before provisioning."
}
$rangeMatch = [regex]::Match($pythonConstraint, '^>=([0-9]+\.[0-9]+\.[0-9]+),<([0-9]+\.[0-9]+)$')
if (-not $rangeMatch.Success) {
    throw "Unsupported Python range in the locked Windows backend project: '$pythonConstraint'. Expected a lower patch bound and an upper major.minor bound."
}
try {
    $minimumPython = [version]$rangeMatch.Groups[1].Value
    $maximumPython = [version]($rangeMatch.Groups[2].Value + '.0')
} catch {
    throw "Could not parse the locked Windows backend Python range: '$pythonConstraint'."
}
if ($minimumPython.Major -ne 3 -or $minimumPython.Minor -ne 12 -or $maximumPython -le $minimumPython) {
    throw "This Windows backend helper currently supports only a Python 3.12 lock range; found '$pythonConstraint'."
}
$pythonCreationVersion = '{0}.{1}.{2}' -f $minimumPython.Major, $minimumPython.Minor, $minimumPython.Build
$uv = Get-Command 'uv.exe' -ErrorAction SilentlyContinue
if (-not $uv) { $uv = Get-Command 'uv' -ErrorAction SilentlyContinue }
if (-not $uv) { throw 'uv was not found. Install uv, close AIWF Studio, and rerun this helper.' }

$venvRoot = Join-Path $root 'venv'
$pythonPath = Join-Path $venvRoot 'Scripts\python.exe'
$venvItem = Get-Item -LiteralPath $venvRoot -Force -ErrorAction SilentlyContinue
$venvExists = $null -ne $venvItem
if ($venvExists -and -not (Test-Path -LiteralPath $venvRoot -PathType Container)) {
    throw "The selected venv path exists but is not an accessible directory: $venvRoot. Refusing to replace it."
}
$freshVenvCreated = $false
$knownUnidentifiableProcesses = @{}

if (-not $venvExists) {
    $createArguments = @('venv', '--python', $pythonCreationVersion, $venvRoot)
    $syncArguments = @('sync', '--project', $lockProject, '--locked', '--inexact', '--python', $pythonPath)
    Write-Output "Studio root: $root"
    Write-Output "Python runtime: CPython $pythonConstraint, Windows AMD64"
    Write-Output "Target environment: $venvRoot"
    Write-Output "Create command: $($uv.Source) $($createArguments -join ' ')"
    Write-Output "Sync command: $($uv.Source) $($syncArguments -join ' ')"
    # PlanOnly is observational: do not make a global process-ownership check
    # when this invocation will not create or mutate an environment.
    if ($PlanOnly) { return }

    # Only unknown process identities captured while the target did not yet
    # exist can be exempted after this invocation creates the new venv.
    $processesBeforeCreation = @(Get-PythonProcessSnapshot)
    foreach ($process in $processesBeforeCreation) {
        if ([string]::IsNullOrWhiteSpace($process.ExecutablePath)) {
            $identity = Get-PythonProcessIdentityKey -Process $process
            if ([string]::IsNullOrWhiteSpace($identity)) {
                throw "Could not identify the creation time of running Python process PID $($process.ProcessId) ($($process.Name)); refusing to create a venv while process ownership is uncertain."
            }
            $knownUnidentifiableProcesses[$identity] = $true
        }
    }

    # Atomically claim the missing path; if another actor created it after our
    # entry check, this fails without opening or overwriting that directory.
    New-NativeVenvDirectoryExclusive -Path $venvRoot
    & $uv.Source @createArguments
    if ($LASTEXITCODE -ne 0) {
        throw "uv could not create the selected Python $pythonCreationVersion venv (exit $LASTEXITCODE). No existing venv was replaced."
    }
    $venvItem = Get-Item -LiteralPath $venvRoot -Force -ErrorAction SilentlyContinue
    if (-not $venvItem -or -not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        throw "uv did not create the expected interpreter at '$pythonPath'. The partial target was preserved; inspect it before retrying."
    }
    $freshVenvCreated = $true
} elseif (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "The selected venv already exists but has no interpreter at '$pythonPath'. Refusing to replace it; repair that venv with a supported CPython $pythonConstraint first."
}

function Assert-SupportedTargetPython {
    param([Parameter(Mandatory = $true)][string]$PythonPath, [Parameter(Mandatory = $true)][string]$VenvPath)

    $runtime = & $PythonPath -B -c "import platform,sys; print('|'.join((platform.python_implementation(), '.'.join(map(str,sys.version_info[:3])), sys.platform, platform.machine(), sys.prefix)))"
    if ($LASTEXITCODE -ne 0) { throw "Could not inspect the selected Studio Python interpreter at '$PythonPath'." }
    $parts = (@($runtime) -join '').Trim().Split('|')
    $version = $null
    if ($parts.Count -eq 5) {
        try { $version = [version]$parts[1] } catch { $version = $null }
    }
    if (-not $version -or $parts[0] -ne 'CPython' -or $parts[2] -ne 'win32' -or $parts[3] -notin @('AMD64', 'x86_64') -or $version -lt $minimumPython -or $version -ge $maximumPython) {
        $found = (@($runtime) -join '').Trim()
        throw "Python backend requires CPython $pythonConstraint on Windows AMD64; found '$found' at '$PythonPath'. Close Studio and prepare a supported Python $pythonConstraint venv; this helper will not replace an existing venv."
    }

    try {
        $selectedPrefix = Get-CanonicalPath -Path $VenvPath -Directory
        $runtimePrefix = Get-CanonicalPath -Path $parts[4] -Directory
        $selectedPython = Get-CanonicalPath -Path $PythonPath
    } catch {
        throw "Could not resolve the selected Studio venv paths safely: $($_.Exception.Message)"
    }
    if (-not [string]::Equals($selectedPrefix, $runtimePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "The interpreter at '$PythonPath' reports a different venv prefix '$($parts[4])'; refusing to sync another environment."
    }
    return [pscustomobject]@{
        Version = $version
        Python = $selectedPython
        ScriptsDirectory = [System.IO.Path]::GetDirectoryName($selectedPython)
    }
}

$target = Assert-SupportedTargetPython -PythonPath $pythonPath -VenvPath $venvRoot
$syncArguments = @('sync', '--project', $lockProject, '--locked', '--inexact', '--python', $pythonPath)

if (-not $freshVenvCreated) {
    # A locked dry-run can prove an existing environment is already ready;
    # in that case no process inspection or package mutation is needed.
    $previousProjectEnvironment = [Environment]::GetEnvironmentVariable('UV_PROJECT_ENVIRONMENT', 'Process')
    $readinessOutput = @()
    $readinessExit = -1
    try {
        [Environment]::SetEnvironmentVariable('UV_PROJECT_ENVIRONMENT', $venvRoot, 'Process')
        $readinessOutput = @(& $uv.Source sync --project $lockProject --locked --inexact --python $pythonPath --dry-run --check --offline --quiet 2>&1)
        $readinessExit = $LASTEXITCODE
    } finally {
        [Environment]::SetEnvironmentVariable('UV_PROJECT_ENVIRONMENT', $previousProjectEnvironment, 'Process')
    }
    if ($readinessExit -eq 0) {
        Write-Output "The locked AIWF Python backend is already ready in '$pythonPath'; no processes or packages were changed."
        return
    }
    $readinessText = ($readinessOutput | ForEach-Object { [string]$_ }) -join "`n"
    if ($readinessText -notmatch 'The environment is outdated; run `uv sync` to update the environment') {
        throw "Could not verify locked Python-backend readiness for existing venv '$venvRoot'; refusing to mutate it. uv reported: $readinessText"
    }
}

Write-Output "Studio root: $root"
Write-Output "Python runtime: CPython $($target.Version), Windows AMD64"
Write-Output "Target environment: $venvRoot"
Write-Output "Sync command: $($uv.Source) $($syncArguments -join ' ')"
if ($PlanOnly) { return }

# Verify ownership immediately before the locked synchronization. Only blank
# paths captured before this helper created a fresh target can be exempted.
$active = @(Get-SelectedPythonProcesses -ScriptsDirectory $target.ScriptsDirectory -KnownUnidentifiableProcesses $knownUnidentifiableProcesses -FreshTarget:$freshVenvCreated)
if ($active.Count -gt 0) {
    $details = ($active | ForEach-Object { "PID $($_.ProcessId) ($($_.Name))" }) -join ', '
    throw "The selected Studio Python interpreter owns running process(es): $details. Close AIWF Studio and any jobs using this venv, then retry."
}

# Check again immediately before uv mutates the selected interpreter's packages.
$active = @(Get-SelectedPythonProcesses -ScriptsDirectory $target.ScriptsDirectory -KnownUnidentifiableProcesses $knownUnidentifiableProcesses -FreshTarget:$freshVenvCreated)
if ($active.Count -gt 0) {
    $details = ($active | ForEach-Object { "PID $($_.ProcessId) ($($_.Name))" }) -join ', '
    throw "The selected Studio Python interpreter started running process(es): $details. Close AIWF Studio and any jobs using this venv, then retry."
}

$previousProjectEnvironment = [Environment]::GetEnvironmentVariable('UV_PROJECT_ENVIRONMENT', 'Process')
try {
    [Environment]::SetEnvironmentVariable('UV_PROJECT_ENVIRONMENT', $venvRoot, 'Process')
    & $uv.Source @syncArguments
    if ($LASTEXITCODE -ne 0) { throw "Locked AIWF Python backend sync failed (exit $LASTEXITCODE). The selected venv was not replaced." }
} finally {
    [Environment]::SetEnvironmentVariable('UV_PROJECT_ENVIRONMENT', $previousProjectEnvironment, 'Process')
}

Write-Output "The locked AIWF Python backend dependencies are ready in '$pythonPath'."
} finally {
    if ($null -ne $bootstrapLock) { $bootstrapLock.Dispose() }
}
