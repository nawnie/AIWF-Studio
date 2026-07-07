param(
    [ValidateSet("build", "clone", "clean")]
    [string]$Mode = "build",
    [ValidateSet("cuda", "cpu")]
    [string]$Backend = "cuda",
    [string]$RepoUrl = "https://github.com/leejet/stable-diffusion.cpp.git",
    [string]$ToolRoot = "",
    [string[]]$CMakeOption = @(),
    [switch]$CleanBuild,
    [switch]$SetProfile
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
if (-not $ToolRoot) {
    $ToolRoot = Join-Path $RepoRoot "tools\stable-diffusion.cpp"
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name is required. Install it first, then re-run this helper."
    }
}

function Run {
    param([string]$FilePath, [string[]]$Arguments)
    Write-Host "$FilePath $($Arguments -join ' ')" -ForegroundColor DarkYellow
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

Require-Command git
Require-Command cmake

if ($Mode -eq "clean" -and (Test-Path -LiteralPath $ToolRoot)) {
    Remove-Item -Recurse -Force -LiteralPath $ToolRoot
    Write-Host "Removed $ToolRoot"
    exit 0
}

if (-not (Test-Path -LiteralPath $ToolRoot)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $ToolRoot -Parent) | Out-Null
    Run "git" @("clone", "--recursive", $RepoUrl, $ToolRoot)
} else {
    Push-Location $ToolRoot
    try {
        Run "git" @("submodule", "update", "--init", "--recursive")
    } finally {
        Pop-Location
    }
}

if ($Mode -eq "clone") {
    Write-Host "stable-diffusion.cpp cloned at $ToolRoot"
    exit 0
}

$BuildDir = Join-Path $ToolRoot "build"
if ($CleanBuild -and (Test-Path -LiteralPath $BuildDir)) {
    Remove-Item -Recurse -Force -LiteralPath $BuildDir
}

$flags = @()
if ($Backend -eq "cuda") {
    $flags += "-DSD_CUDA=ON"
}
$flags += $CMakeOption

Run "cmake" @("-S", $ToolRoot, "-B", $BuildDir, "-DCMAKE_BUILD_TYPE=Release") + $flags
Run "cmake" @("--build", $BuildDir, "--config", "Release", "--parallel")

$candidates = @(
    (Join-Path $ToolRoot "build\bin\Release\sd-cli.exe"),
    (Join-Path $ToolRoot "build\bin\sd-cli.exe"),
    (Join-Path $ToolRoot "bin\sd-cli.exe"),
    (Join-Path $ToolRoot "build\bin\Release\sd-cli"),
    (Join-Path $ToolRoot "build\bin\sd-cli")
)
$SdCli = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $SdCli) {
    throw "Build completed but sd-cli was not found. Check the build output under $BuildDir."
}

$BinDir = Join-Path $ToolRoot "bin"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
if ($SdCli -notlike "$BinDir*") {
    Copy-Item -Force -LiteralPath $SdCli -Destination (Join-Path $BinDir (Split-Path $SdCli -Leaf))
    $SdCli = Join-Path $BinDir (Split-Path $SdCli -Leaf)
}

Write-Host "sd-cli ready: $SdCli" -ForegroundColor Yellow

if ($SetProfile) {
    $LocalDir = Join-Path $RepoRoot "_local"
    New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null
    @{ backend = "sdcpp" } | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $LocalDir "backend_profile.json")
    @{
        sdCli = $SdCli
        backend = "cuda0"
        maxVram = "0"
        mmap = $true
        diffusionFlashAttention = $true
    } | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $LocalDir "sdcpp_profile.json")
    Write-Host "Saved AIWF backend profile for stable-diffusion.cpp." -ForegroundColor Yellow
}
