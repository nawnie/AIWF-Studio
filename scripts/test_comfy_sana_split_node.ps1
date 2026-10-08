[CmdletBinding()]
param(
    [int]$Port = 8190,
    [int]$TimeoutSeconds = 600
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logRoot = Join-Path $repoRoot "_local\logs"
$testUser = Join-Path $repoRoot "_local\comfy-sana-node-test-user"
New-Item -ItemType Directory -Force -Path $logRoot, $testUser | Out-Null

$existing = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
if ($existing.Count -ne 0) {
    throw "Refusing to use occupied test port $Port."
}

$receipt = Join-Path $logRoot "comfy-sana-node-desktop-vram-$stamp.json"
$stdout = Join-Path $logRoot "comfy-sana-node-$stamp.out.log"
$stderr = Join-Path $logRoot "comfy-sana-node-$stamp.err.log"
$guardPython = Join-Path $repoRoot "venv\Scripts\python.exe"
$guard = "D:\Codex-Projects\Desktop\Active Projects\Tools\Agent Skills\staged\nawnie\skills\sentinel-jarvis\scripts\run_with_vram_guard.py"
$comfyPython = "C:\ComfyUI\.venv\Scripts\python.exe"
$comfyMain = "C:\ComfyUI\main.py"

$launch = @"
& "$guardPython" "$guard" --gpu-index 0 --receipt "$receipt" -- "$comfyPython" "$comfyMain" --listen 127.0.0.1 --port $Port --base-directory C:\ComfyUI --user-directory "$testUser" --database-url sqlite:///:memory: --cpu --disable-auto-launch
exit `$LASTEXITCODE
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($launch))
$owned = Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoProfile", "-EncodedCommand", $encoded
) -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru

try {
    $ready = $false
    $deadline = (Get-Date).AddSeconds(120)
    while ((Get-Date) -lt $deadline) {
        if ($owned.HasExited) {
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
        $errorText = if (Test-Path $stderr) { Get-Content -LiteralPath $stderr -Raw } else { "" }
        throw "Temporary ComfyUI did not load the AIWF node: $errorText"
    }
    Write-Output "NODE_LOADED=AIWFSanaSplitGenerate"

    $workflow = @{
        client_id = "aiwf-sana-node-test-$stamp"
        prompt = @{
            "1" = @{
                class_type = "AIWFSanaSplitGenerate"
                inputs = @{
                    prompt = "A small copper robot reading beside a warm workshop window."
                    steps = 2
                    width = 512
                    height = 512
                    seed = 42
                }
            }
        }
    }
    $queued = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/prompt" -Method Post -ContentType "application/json" -Body ($workflow | ConvertTo-Json -Depth 8 -Compress) -TimeoutSec 30
    if (-not $queued.prompt_id) {
        throw "ComfyUI did not return a prompt id."
    }
    Write-Output "PROMPT_ID=$($queued.prompt_id)"

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $result = $null
    while ((Get-Date) -lt $deadline) {
        $history = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/history/$($queued.prompt_id)" -TimeoutSec 10
        $entry = $history.PSObject.Properties[$queued.prompt_id]
        if ($entry) {
            $result = $entry.Value
            break
        }
        Start-Sleep -Seconds 1
    }
    if ($null -eq $result) {
        throw "The ComfyUI node did not finish within $TimeoutSeconds seconds."
    }
    if ($result.status.status_str -ne "success" -or -not $result.status.completed) {
        throw "The ComfyUI node failed: $($result.status | ConvertTo-Json -Depth 8 -Compress)"
    }
    Write-Output "NODE_EXECUTION_STATUS=$($result.status.status_str)"
    Write-Output "NODE_EXECUTION_COMPLETED=$($result.status.completed)"
} finally {
    $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
    foreach ($listener in $listeners) {
        $candidate = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)" -ErrorAction SilentlyContinue
        if (
            $candidate -and
            $candidate.CommandLine -like "*C:\ComfyUI\main.py*" -and
            $candidate.CommandLine -like "*--port $Port*"
        ) {
            Stop-Process -Id $candidate.ProcessId -ErrorAction SilentlyContinue
        }
    }
    if (-not $owned.WaitForExit(30000)) {
        Stop-Process -Id $owned.Id -ErrorAction SilentlyContinue
    }
}

if (-not (Test-Path -LiteralPath $receipt -PathType Leaf)) {
    throw "The desktop VRAM receipt was not written: $receipt"
}
$guardResult = Get-Content -LiteralPath $receipt -Raw | ConvertFrom-Json
if (
    $guardResult.guard_exit_code -ne 0 -and
    $guardResult.child_exit_code -ne [uint32]::MaxValue
) {
    throw "The desktop VRAM guard exited $($guardResult.guard_exit_code)."
}
Write-Output "DESKTOP_RECEIPT=$receipt"
Write-Output "DESKTOP_MINIMUM_FREE_MIB=$($guardResult.minimum_free_mib)"
Write-Output "TEST_SERVER_CLEANUP_EXIT=$($guardResult.child_exit_code)"
Write-Output "COMFY_STDOUT=$stdout"
Write-Output "COMFY_STDERR=$stderr"
Write-Output "COMFY_NODE_TEST_EXIT=0"
