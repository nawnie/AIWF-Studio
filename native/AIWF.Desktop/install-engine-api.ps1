param(
    [string]$StudioRoot = '',
    [string]$PythonExe = ''
)

$ErrorActionPreference = 'Stop'
$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$repoRoot = if ([string]::IsNullOrWhiteSpace($StudioRoot)) {
    $sourceRoot
} else {
    (Resolve-Path -LiteralPath $StudioRoot).Path
}
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot 'aiwf\engine_api.py') -PathType Leaf)) {
    throw "StudioRoot must be an AIWF Studio checkout containing aiwf\engine_api.py: $repoRoot"
}
$venvRoot = Join-Path $repoRoot 'venv'
$pythonPath = Join-Path $venvRoot 'Scripts\python.exe'
$lockDirectory = Join-Path ([System.IO.Path]::GetTempPath()) 'AIWF-Studio'
[void][System.IO.Directory]::CreateDirectory($lockDirectory)
$hashAlgorithm = [System.Security.Cryptography.SHA256]::Create()
try {
    $rootIdentity = $repoRoot.TrimEnd('\', '/').ToUpperInvariant()
    $rootHash = [System.BitConverter]::ToString($hashAlgorithm.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($rootIdentity))).Replace('-', '').ToLowerInvariant()
} finally {
    $hashAlgorithm.Dispose()
}
$bootstrapLockPath = Join-Path $lockDirectory "engine-api-bootstrap-$rootHash.lock"

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

function New-NativeVenvDirectoryExclusive {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not ('AiwfEngineApiVenvDirectory' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;

public static class AiwfEngineApiVenvDirectory {
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
        [AiwfEngineApiVenvDirectory]::CreateNew($Path)
    } catch {
        throw "Could not claim the new venv directory without overwriting an existing target at '$Path': $($_.Exception.Message)"
    }
}

$bootstrapLock = $null
try {
    try {
        # FileShare.None serializes bootstraps targeting this checkout without truncating the lock file.
        $bootstrapLock = [System.IO.File]::Open(
            $bootstrapLockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    } catch {
        throw "Another native engine API bootstrap is active for StudioRoot '$repoRoot', or its per-root lock is unavailable: $($_.Exception.Message)"
    }

$freshVenvCreated = $false
$preCreationUnknownProcessKeys = @{}

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    if (Test-Path -LiteralPath $venvRoot) {
        throw "The existing venv has no Python executable at '$pythonPath'; refusing to replace it. Repair the selected StudioRoot venv with the AIWF installer."
    }
    $minimumPython = [version]'3.12.13'
    $basePython = ''
    $basePythonPrefix = @()
    if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
        $basePython = (Resolve-Path -LiteralPath $PythonExe -ErrorAction Stop).Path
        $baseVersionText = (& $basePython -c "import sys; print('.'.join(map(str, sys.version_info[:3])))").Trim()
        try { $baseVersion = [version]$baseVersionText } catch { $baseVersion = [version]'0.0' }
        if ($LASTEXITCODE -ne 0 -or $baseVersion -lt $minimumPython -or $baseVersion -ge [version]'3.13') {
            throw "-PythonExe must be a working Python 3.12.13-or-newer base interpreter below 3.13; found '$baseVersionText'."
        }
    } else {
        $uv = Get-Command 'uv.exe' -ErrorAction SilentlyContinue
        if ($uv) {
            & $uv.Source python install '3.12.13'
            if ($LASTEXITCODE -ne 0) {
                throw 'uv could not provision CPython 3.12.13 for the native engine API.'
            }
            # Resolve outside any Studio checkout so uv cannot mistake a project .venv
            # for the installed base interpreter it just selected.
            $basePython = (& $uv.Source --directory ([System.IO.Path]::GetTempPath()) --no-config python find '3.12.13' --managed-python --no-python-downloads --resolve-links).Trim()
            if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($basePython) -or -not (Test-Path -LiteralPath $basePython -PathType Leaf)) {
                throw 'uv could not resolve its installed CPython 3.12.13 for the native engine API.'
            }
            $baseVersionText = (& $basePython -c "import sys; print('.'.join(map(str, sys.version_info[:3])))").Trim()
            try { $baseVersion = [version]$baseVersionText } catch { $baseVersion = [version]'0.0' }
            if ($LASTEXITCODE -ne 0 -or $baseVersion -lt $minimumPython -or $baseVersion -ge [version]'3.13') {
                throw "uv resolved an unsupported native API Python interpreter '$baseVersionText' at '$basePython'."
            }
        } else {
            $pyLauncher = Get-Command 'py.exe' -ErrorAction SilentlyContinue
            if (-not $pyLauncher) {
                throw 'CPython 3.12.13 was not found. Install Python 3.12.13 or uv, or pass -PythonExe with a Python 3.12.13 base interpreter.'
            }
            $launcherVersionText = (& $pyLauncher.Source '-3.12' -c "import sys; print('.'.join(map(str, sys.version_info[:3])))").Trim()
            try { $launcherVersion = [version]$launcherVersionText } catch { $launcherVersion = [version]'0.0' }
            if ($LASTEXITCODE -ne 0 -or $launcherVersion -lt $minimumPython -or $launcherVersion -ge [version]'3.13') {
                throw "The py launcher must resolve -3.12 to CPython 3.12.13 or newer below 3.13 before creating a native engine API venv; found '$launcherVersionText'. Install Python 3.12.13 or uv, or pass -PythonExe with a supported base interpreter."
            }
            $basePython = $pyLauncher.Source
            $basePythonPrefix = @('-3.12')
        }
    }

    # Snapshot unidentifiable processes while this target venv still cannot exist.
    # Their pre-creation identity proves they cannot already be using its interpreter.
    $processesBeforeVenvCreation = @(Get-PythonProcessSnapshot)
    foreach ($process in $processesBeforeVenvCreation) {
        if ([string]::IsNullOrWhiteSpace($process.ExecutablePath)) {
            $identity = Get-PythonProcessIdentityKey -Process $process
            if (-not [string]::IsNullOrWhiteSpace($identity)) { $preCreationUnknownProcessKeys[$identity] = $true }
        }
    }

    if (Test-Path -LiteralPath $venvRoot) {
        throw "The selected target venv appeared during bootstrap at '$venvRoot'; refusing to write into a directory this helper did not create."
    }
    New-NativeVenvDirectoryExclusive -Path $venvRoot
    $venvArguments = @()
    $venvArguments += $basePythonPrefix
    $venvArguments += @('-m', 'venv', $venvRoot)
    & $basePython @venvArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.12 could not create the native engine API venv (exit $LASTEXITCODE)."
    }
    if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        throw "Python venv creation reported success but did not create '$pythonPath'; refusing to repair or replace the partial directory automatically."
    }
    $freshVenvCreated = $true
}

$pythonVersionText = (& $pythonPath -c "import sys; print('.'.join(map(str, sys.version_info[:3])))").Trim()
if ($LASTEXITCODE -ne 0 -or $pythonVersionText -notmatch '^3\.12\.\d+$') {
    throw "The native engine API venv must use Python 3.12; found '$pythonVersionText' at '$pythonPath'."
}

$requirements = Join-Path $PSScriptRoot 'engine-api-requirements.txt'
$preflight = Join-Path $PSScriptRoot 'engine-api-preflight.py'
$probeOutput = & $pythonPath $preflight $requirements 2>$null
$apiReady = ($LASTEXITCODE -eq 0 -and $probeOutput -contains 'READY')
if ($apiReady) {
    Write-Output "Native engine API dependencies are already ready in '$pythonPath'; no packages were changed."
    return
}

$existingVersion = [version]$pythonVersionText
if ($existingVersion -lt [version]'3.12.13') {
    throw "The existing Studio venv uses CPython $pythonVersionText and is missing one or more exact native API lock packages. It was preserved. This helper will not update an older venv in place; use a separate Studio checkout with CPython 3.12.13+ or restore the exact API lock first, then rerun."
}

function Get-CanonicalPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$Directory
    )

    if (-not ('AiwfEngineApiPathResolver' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;

public static class AiwfEngineApiPathResolver {
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

    $resolved = [AiwfEngineApiPathResolver]::Resolve($Path, [bool]$Directory)
    if ($resolved.StartsWith('\\?\UNC\', [System.StringComparison]::OrdinalIgnoreCase)) {
        return '\\' + $resolved.Substring(8)
    }
    if ($resolved.StartsWith('\\?\', [System.StringComparison]::OrdinalIgnoreCase)) {
        return $resolved.Substring(4)
    }
    return $resolved
}

$pythonProcesses = @(Get-PythonProcessSnapshot)
$selectedScripts = Get-CanonicalPath -Path (Join-Path $venvRoot 'Scripts') -Directory
$runningVenvPython = @()
foreach ($process in $pythonProcesses) {
    if ([string]::IsNullOrWhiteSpace($process.ExecutablePath)) {
        $identity = Get-PythonProcessIdentityKey -Process $process
        if ($freshVenvCreated -and -not [string]::IsNullOrWhiteSpace($identity) -and $preCreationUnknownProcessKeys.ContainsKey($identity)) {
            continue
        }
        throw "Could not verify the interpreter path for running Python process PID $($process.ProcessId) ($($process.Name)); refusing to change a possibly active backend environment."
    }
    try {
        $processPath = Get-CanonicalPath -Path $process.ExecutablePath
    } catch {
        throw "Could not resolve the interpreter path for running Python process PID $($process.ProcessId) ($($process.Name)); refusing to change a possibly active backend environment."
    }
    $processScripts = [System.IO.Path]::GetDirectoryName($processPath)
    if ([string]::Equals($processScripts, $selectedScripts, [System.StringComparison]::OrdinalIgnoreCase)) {
        $runningVenvPython += $process
    }
}
if ($runningVenvPython.Count -gt 0) {
    $processIds = ($runningVenvPython | ForEach-Object { $_.ProcessId }) -join ', '
    throw "Python is running from the selected Studio venv (PID $processIds). Close AIWF Studio and other uses of this venv before changing API packages."
}

& $pythonPath -m pip install --disable-pip-version-check --requirement $requirements
if ($LASTEXITCODE -ne 0) {
    throw "The native engine API dependencies did not install (exit $LASTEXITCODE)."
}
$verification = & $pythonPath $preflight $requirements 2>$null
if ($LASTEXITCODE -ne 0 -or $verification -notcontains 'READY') {
    throw 'The native engine API dependency lock did not pass its exact-version and import preflight after installation.'
}

Write-Output "Native engine API dependencies are ready in '$pythonPath'."
} finally {
    if ($null -ne $bootstrapLock) { $bootstrapLock.Dispose() }
}
