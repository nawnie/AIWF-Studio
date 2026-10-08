[CmdletBinding()]
param([int]$Port = 8191)

$ErrorActionPreference = "Stop"
$listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
if ($listeners.Count -eq 0) {
    Write-Output "COMFY_SANA_UI_NOT_RUNNING=http://127.0.0.1:$Port"
    exit 0
}
if ($listeners.Count -ne 1 -or $listeners[0].LocalAddress -ne "127.0.0.1") {
    throw "Refusing to stop an ambiguous or non-loopback listener on port $Port."
}

$process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listeners[0].OwningProcess)"
if (
    $process.CommandLine -notlike "*C:\ComfyUI\main.py*" -or
    $process.CommandLine -notlike "*--port $Port*"
) {
    throw "Refusing to stop port $Port because its process is not the dedicated Sana ComfyUI."
}
Stop-Process -Id $process.ProcessId
Write-Output "COMFY_SANA_UI_STOPPED_PID=$($process.ProcessId)"
