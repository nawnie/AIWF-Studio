param(
    [ValidateSet("diffusers", "sdcpp", "onnx")]
    [string]$Profile = "diffusers",
    [string]$SdCli = "",
    [string]$SdcppBackend = "cuda0",
    [string]$ParamsBackend = "",
    [string]$MaxVram = "0",
    [switch]$OffloadToCpu,
    [switch]$StreamLayers,
    [switch]$NoDiffusionFlashAttention,
    [switch]$VaeTiling,
    [switch]$SetDefault,
    [switch]$SkipInstall,
    [switch]$SkipFrontendBuild,
    [switch]$Terminal
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ($Profile -eq "sdcpp") {
    $env:AIWF_SDCPP_BACKEND = $SdcppBackend
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
}

$LaunchArgs = @("--backend", $Profile)
if ($SetDefault) {
    $LaunchArgs += "--set-default"
}
if ($SkipInstall) {
    $LaunchArgs += "--skip-install"
}
if ($SkipFrontendBuild) {
    $LaunchArgs += "--skip-frontend-build"
}
if ($Terminal) {
    $LaunchArgs += "--terminal"
}

Write-Host "[AIWF] Launching Pro with backend profile: $Profile" -ForegroundColor Yellow
if ($Profile -eq "sdcpp") {
    Write-Host "[AIWF] sd.cpp backend: $SdcppBackend | Max VRAM: $MaxVram | sd-cli: $($env:AIWF_SDCPP_BINARY)"
}
python launch_backend_profile.py @LaunchArgs
