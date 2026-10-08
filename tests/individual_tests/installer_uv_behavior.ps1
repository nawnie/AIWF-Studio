param([Parameter(Mandatory=$true)][string]$InstallerPath)
$ErrorActionPreference = 'Stop'
function uv { }
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($InstallerPath, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw "Installer parse errors: $($parseErrors -join '; ')" }
$definition = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Ensure-PythonVenv' }, $true)
if (-not $definition) { throw 'Ensure-PythonVenv definition not found.' }
Invoke-Expression $definition.Extent.Text

function Get-VenvPythonMinor { return $script:MockMinor }
function Get-VenvPythonVersion { return $script:MockVersion }
function Test-BackendLockPythonVersion { param([string]$Version); try { $v=[version]$Version; return $v -ge [version]'3.12.13' -and $v -lt [version]'3.13.0' } catch { return $false } }
function Assert-TaskLocalWritePath { param([string]$Path) }
function Write-Section { param([string]$Title) }
function Move-StaleVenv { param([string]$Reason); $script:MoveCalls++; $script:MockMinor=''; $script:MockVersion='' }
function Ensure-PythonVenv-Conda { $script:CondaCalls++; throw 'Unexpected conda fallback' }
function Assert-BackendLockPython { $script:AssertLockPythonCalls++ }
function Invoke-External {
    param([string]$Label,[string]$Executable,[string[]]$Arguments)
    $script:Calls += ,@($Label,$Executable,$Arguments)
    if ($Label -eq 'Create AIWF venv') { $script:MockMinor='3.12'; $script:MockVersion='3.12.13' }
}
function Invoke-Case {
    param([bool]$Locked,[string]$Minor,[string]$Version,[bool]$ExpectMove,[bool]$ExpectCreate)
    $script:UseBackendLock=$Locked
    $script:DryRun=$false
    $script:TaskLocalMode=$true
    $script:PythonVersion=if ($Locked) {'3.12.13'} else {'3.12'}
    $script:VenvDir=Join-Path $env:TEMP ([guid]::NewGuid().ToString('N'))
    $script:VenvPython=Join-Path $script:VenvDir 'Scripts/python.exe'
    New-Item -ItemType Directory -Path (Split-Path $script:VenvPython -Parent) -Force | Out-Null
    Set-Content -LiteralPath $script:VenvPython -Value 'mock interpreter placeholder'
    $script:MockMinor=$Minor; $script:MockVersion=$Version
    $script:Calls=@(); $script:CondaCalls=0; $script:MoveCalls=0; $script:AssertLockPythonCalls=0
    Ensure-PythonVenv
    if ($script:CondaCalls -ne 0) { throw 'Conda fallback was called.' }
    if ($script:MoveCalls -ne [int]$ExpectMove) { throw "Move count $($script:MoveCalls), expected $ExpectMove" }
    $created=@($script:Calls | Where-Object { $_[0] -eq 'Create AIWF venv' }).Count -gt 0
    if ($created -ne $ExpectCreate) { throw "Create venv=$created, expected $ExpectCreate" }
    if ($Locked -and $script:AssertLockPythonCalls -ne 1) { throw 'Locked runtime assertion was not called once.' }
    if (-not $Locked -and $script:AssertLockPythonCalls -ne 0) { throw 'Default path called the backend-lock assertion.' }
    Remove-Item -LiteralPath $script:VenvDir -Recurse -Force
}
Invoke-Case -Locked $true -Minor '3.12' -Version '3.12.13' -ExpectMove $false -ExpectCreate $false
Invoke-Case -Locked $true -Minor '3.12' -Version '3.12.1' -ExpectMove $true -ExpectCreate $true
Invoke-Case -Locked $false -Minor '3.12' -Version '3.12.13' -ExpectMove $false -ExpectCreate $false
Write-Output 'PASS: locked current, stale-v3 backup/recreate, and default uv branches avoid conda and follow expected provisioning.'
