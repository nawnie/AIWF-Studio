param(
    [string]$SdCli = "",
    [string]$Backend = "cuda0",
    [string]$ParamsBackend = "",
    [string]$MaxVram = "0",
    [switch]$OffloadToCpu,
    [switch]$StreamLayers,
    [switch]$NoDiffusionFlashAttention,
    [switch]$VaeTiling,
    [switch]$SetDefault,
    [switch]$SkipInstall,
    [switch]$Terminal
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Args = @(
    "-Profile", "sdcpp",
    "-SdcppBackend", $Backend,
    "-MaxVram", $MaxVram
)
if ($SdCli) {
    $Args += @("-SdCli", $SdCli)
}
if ($ParamsBackend) {
    $Args += @("-ParamsBackend", $ParamsBackend)
}
if ($OffloadToCpu) {
    $Args += "-OffloadToCpu"
}
if ($StreamLayers) {
    $Args += "-StreamLayers"
}
if ($NoDiffusionFlashAttention) {
    $Args += "-NoDiffusionFlashAttention"
}
if ($VaeTiling) {
    $Args += "-VaeTiling"
}
if ($SetDefault) {
    $Args += "-SetDefault"
}
if ($SkipInstall) {
    $Args += "-SkipInstall"
}
if ($Terminal) {
    $Args += "-Terminal"
}

& (Join-Path $PSScriptRoot "launch_backend_profile.ps1") @Args
