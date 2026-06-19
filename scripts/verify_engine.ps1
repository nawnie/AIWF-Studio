param(
    [Parameter(Mandatory = $true)]
    [string]$Name
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EngineDir = Join-Path $Root "engines\$Name"
$Python = Join-Path $EngineDir ".venv\Scripts\python.exe"
$Worker = Join-Path $EngineDir "worker.py"
$Tmp = Join-Path $Root "outputs\worker-probes"
$Request = Join-Path $Tmp "$Name-probe-request.json"

if (!(Test-Path $Python)) {
    throw "Engine python missing: $Python"
}
if (!(Test-Path $Worker)) {
    throw "Engine worker missing: $Worker"
}

New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
$Payload = @{
    _job_id = "$Name-probe"
    _engine = $Name
    mode = "probe"
} | ConvertTo-Json -Depth 5
Set-Content -Path $Request -Value $Payload -Encoding UTF8

$env:PYTHONPATH = if ($env:PYTHONPATH) { "$Root;$env:PYTHONPATH" } else { $Root }
& $Python $Worker $Request
