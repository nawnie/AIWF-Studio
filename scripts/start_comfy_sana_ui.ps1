[CmdletBinding()]
param([int]$Port = 8191)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$nodePath = "C:\ComfyUI\custom_nodes\aiwf_sana_split"
$userDirectory = Join-Path $repoRoot "_local\comfy-sana-ui-user"
$logRoot = Join-Path $repoRoot "_local\logs"
New-Item -ItemType Directory -Force -Path $userDirectory, $logRoot | Out-Null

if (-not (Test-Path -LiteralPath $nodePath -PathType Container)) {
    throw "The AIWF Sana ComfyUI node is not installed at $nodePath."
}
$existing = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
if ($existing.Count -ne 0) {
    $info = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/object_info/AIWFSanaSplitGenerate" -TimeoutSec 5
    if ($info.AIWFSanaSplitGenerate) {
        Write-Output "COMFY_SANA_UI_ALREADY_READY=http://127.0.0.1:$Port"
        exit 0
    }
    throw "Port $Port is already occupied by another service."
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdout = Join-Path $logRoot "comfy-sana-ui-$stamp.out.log"
$stderr = Join-Path $logRoot "comfy-sana-ui-$stamp.err.log"
$process = Start-Process -FilePath "C:\ComfyUI\.venv\Scripts\python.exe" -ArgumentList @(
    "C:\ComfyUI\main.py",
    "--listen", "127.0.0.1",
    "--port", "$Port",
    "--base-directory", "C:\ComfyUI",
    "--user-directory", $userDirectory,
    "--lowvram",
    "--reserve-vram", "1",
    "--database-url", "sqlite:///:memory:",
    "--disable-auto-launch"
) -WorkingDirectory "C:\ComfyUI" -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru

$ready = $false
$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
    if ($process.HasExited) {
        break
    }
    try {
        $info = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/object_info/AIWFSanaSplitGenerate" -TimeoutSec 3
        if ($info.AIWFSanaSplitGenerate) {
            $ready = $true
            break
        }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}
if (-not $ready) {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
    }
    $errorText = if (Test-Path $stderr) { Get-Content -LiteralPath $stderr -Raw } else { "" }
    throw "The dedicated Sana ComfyUI did not become ready: $errorText"
}

Write-Output "COMFY_SANA_UI_READY=http://127.0.0.1:$Port"
Write-Output "AIWF_NODE=AIWFSanaSplitGenerate"
Write-Output "LAUNCHER_PID=$($process.Id)"
Write-Output "COMFY_STDOUT=$stdout"
Write-Output "COMFY_STDERR=$stderr"
