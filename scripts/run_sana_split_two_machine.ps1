[CmdletBinding()]
param(
    [string]$LaptopHost = "4070-laptop-lan",
    [int]$TunnelPort = 18794,
    [int]$LaptopPort = 8794,
    [int]$Steps = 2,
    [int]$Width = 512,
    [int]$Height = 512
)

$ErrorActionPreference = "Stop"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$repoRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $repoRoot "_local\logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$sshOut = Join-Path $logRoot "sana-split-ssh-$stamp.out.log"
$sshErr = Join-Path $logRoot "sana-split-ssh-$stamp.err.log"
$desktopReceipt = Join-Path $logRoot "sana-split-desktop-vram-$stamp.json"
$laptopReceiptRemote = "C:\Users\shawn\AppData\Local\AIWF\SanaEncoder\receipts\sana-encoder-vram-$stamp.json"
$laptopReceiptLocal = Join-Path $logRoot "sana-split-laptop-vram-$stamp.json"
$python = Join-Path $repoRoot "venv\Scripts\python.exe"
$guard = "D:\Codex-Projects\Desktop\Active Projects\Tools\Agent Skills\staged\nawnie\skills\sentinel-jarvis\scripts\run_with_vram_guard.py"
$modelRoot = Join-Path $repoRoot "models\sana\Diffusers\Sana_Sprint_0.6B_1024px_diffusers"
$token = [Convert]::ToHexString(
    [Security.Cryptography.RandomNumberGenerator]::GetBytes(32)
).ToLowerInvariant()

$remote = @"
`$ErrorActionPreference = "Stop"
`$env:AIWF_SANA_ENCODER_TOKEN = "$token"
`$root = "C:\Users\shawn\AppData\Local\AIWF\SanaEncoder"
`$python = Join-Path `$root ".venv310\Scripts\python.exe"
`$guard = Join-Path `$root "tools\run_with_vram_guard.py"
& `$python `$guard --gpu-index 0 --receipt "$laptopReceiptRemote" -- `$python (Join-Path `$root "scripts\start_sana_encoder.py") --model-root (Join-Path `$root "models\sana-sprint") --port $LaptopPort --device cuda:0 --dtype bfloat16 --max-encodes 1
exit `$LASTEXITCODE
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remote))
$ssh = Start-Process -FilePath "ssh.exe" -ArgumentList @(
    "-L", "${TunnelPort}:127.0.0.1:${LaptopPort}", $LaptopHost,
    "powershell.exe", "-NoProfile", "-EncodedCommand", $encoded
) -WindowStyle Hidden -RedirectStandardOutput $sshOut -RedirectStandardError $sshErr -PassThru

try {
    $headers = @{ Authorization = "Bearer $token" }
    $ready = $false
    $deadline = (Get-Date).AddMinutes(4)
    while ((Get-Date) -lt $deadline) {
        if ($ssh.HasExited) {
            break
        }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$TunnelPort/healthz" -Headers $headers -TimeoutSec 2
            if ($health.ready) {
                $ready = $true
                break
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $ready) {
        $errText = if (Test-Path $sshErr) { Get-Content -LiteralPath $sshErr -Raw } else { "" }
        throw "Laptop encoder did not become ready. SSH exited=$($ssh.HasExited). SSH stderr: $errText"
    }

    Write-Output "ENCODER_READY gpu=$($health.gpu_name) allocated_mib=$($health.vram_allocated_mib) ram_gib=$($health.available_ram_gib) disk_offload=$($health.disk_offload)"
    $env:AIWF_SANA_ENCODER_TOKEN = $token
    & $python $guard --gpu-index 0 --receipt $desktopReceipt -- $python (Join-Path $repoRoot "scripts\smoke_sana_split.py") --model-root $modelRoot --encoder-url "http://127.0.0.1:$TunnelPort" --steps $Steps --width $Width --height $Height --device cuda:0 --dtype bfloat16
    $desktopExit = $LASTEXITCODE
    Remove-Item Env:\AIWF_SANA_ENCODER_TOKEN -ErrorAction SilentlyContinue
    if ($desktopExit -ne 0) {
        throw "Desktop split smoke exited $desktopExit"
    }
    if (-not $ssh.WaitForExit(60000)) {
        throw "Laptop encoder SSH session did not exit after its one-request limit."
    }
    if ($ssh.ExitCode -ne 0) {
        throw "Laptop guarded encoder exited $($ssh.ExitCode)"
    }

    scp "${LaptopHost}:C:/Users/shawn/AppData/Local/AIWF/SanaEncoder/receipts/sana-encoder-vram-$stamp.json" $laptopReceiptLocal
    if ($LASTEXITCODE -ne 0) {
        throw "Could not retrieve laptop VRAM receipt."
    }
    Write-Output "DESKTOP_RECEIPT=$desktopReceipt"
    Write-Output "LAPTOP_RECEIPT=$laptopReceiptLocal"
    Write-Output "SSH_STDOUT=$sshOut"
    Write-Output "SSH_STDERR=$sshErr"
    Write-Output "TWO_MACHINE_EXIT=0"
} finally {
    Remove-Item Env:\AIWF_SANA_ENCODER_TOKEN -ErrorAction SilentlyContinue
    if (-not $ssh.HasExited) {
        Stop-Process -Id $ssh.Id -ErrorAction SilentlyContinue
    }
}
