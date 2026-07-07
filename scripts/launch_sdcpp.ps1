param(
    [string]$SdCli = "",
    [string]$Backend = "cuda0",
    [string]$ParamsBackend = "",
    [string]$MaxVram = "0",
    [switch]$OffloadToCpu,
    [switch]$StreamLayers,
    [switch]$NoDiffusionFlashAttention,
    [switch]$VaeTiling,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:AIWF_INFERENCE_BACKEND = "sdcpp"
$env:AIWF_SDCPP_BACKEND = $Backend
$env:AIWF_SDCPP_MAX_VRAM = $MaxVram
$env:AIWF_SDCPP_MMAP = "1"

if ($SdCli) {
    $env:AIWF_SDCPP_BINARY = (Resolve-Path $SdCli).Path
}
if ($ParamsBackend) {
    $env:AIWF_SDCPP_PARAMS_BACKEND = $ParamsBackend
}
if ($OffloadToCpu) {
    $env:AIWF_SDCPP_OFFLOAD_TO_CPU = "1"
}
if ($StreamLayers) {
    $env:AIWF_SDCPP_STREAM_LAYERS = "1"
}
if ($NoDiffusionFlashAttention) {
    $env:AIWF_SDCPP_DIFFUSION_FA = "0"
} else {
    $env:AIWF_SDCPP_DIFFUSION_FA = "1"
}
if ($VaeTiling) {
    $env:AIWF_SDCPP_VAE_TILING = "1"
}

$LaunchArgs = @()
if ($SkipInstall) {
    $LaunchArgs += "--skip-install"
}

Write-Host "[AIWF] Launching production UI with stable-diffusion.cpp backend" -ForegroundColor Cyan
Write-Host "[AIWF] Backend: $Backend | Max VRAM: $MaxVram | sd-cli: $($env:AIWF_SDCPP_BINARY)"
python launch.py @LaunchArgs
