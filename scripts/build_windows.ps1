# Windows 构建脚本
# 用于构建 executor 库的静态库和动态库

param(
    [string]$BuildType = "Release",
    [string]$Generator = "Visual Studio 17 2022",
    [string]$Architecture = "x64",
    [switch]$BuildStatic = $true,
    [switch]$BuildShared = $true,
    [switch]$BuildTests = $false,
    [switch]$BuildExamples = $false,
    [string]$OutputDir = "build_windows"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Executor Windows Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Build Type: $BuildType" -ForegroundColor Yellow
Write-Host "Generator: $Generator" -ForegroundColor Yellow
Write-Host "Architecture: $Architecture" -ForegroundColor Yellow
Write-Host "Build Static: $BuildStatic" -ForegroundColor Yellow
Write-Host "Build Shared: $BuildShared" -ForegroundColor Yellow
Write-Host "Output Dir: $OutputDir" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check CMake
$cmakePath = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmakePath) {
    Write-Host "Error: CMake not found. Please ensure CMake is installed and in PATH" -ForegroundColor Red
    exit 1
}

Write-Host "Found CMake: $($cmakePath.Source)" -ForegroundColor Green
Write-Host "CMake version:" -NoNewline
& cmake --version | Select-Object -First 1

# Get project root directory
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ProjectRoot

# Build static library
if ($BuildStatic) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Building Static Library" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    $StaticBuildDir = Join-Path $OutputDir "static"
    
    # Configure
    Write-Host "Configuring static library build..." -ForegroundColor Yellow
    & cmake -B $StaticBuildDir `
        -G "$Generator" `
        -A $Architecture `
        -DCMAKE_BUILD_TYPE=$BuildType `
        -DEXECUTOR_BUILD_SHARED=OFF `
        -DEXECUTOR_BUILD_TESTS=OFF `
        -DEXECUTOR_BUILD_EXAMPLES=OFF `
        -DCMAKE_INSTALL_PREFIX="$StaticBuildDir\install"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: CMake configuration failed" -ForegroundColor Red
        exit 1
    }
    
    # Build
    Write-Host "Building static library..." -ForegroundColor Yellow
    & cmake --build $StaticBuildDir --config $BuildType
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Build failed" -ForegroundColor Red
        exit 1
    }
    
    # Install
    Write-Host "Installing static library..." -ForegroundColor Yellow
    & cmake --install $StaticBuildDir --config $BuildType
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Installation failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Static library build completed!" -ForegroundColor Green
}

# Build shared library
if ($BuildShared) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Building Shared Library" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    $SharedBuildDir = Join-Path $OutputDir "shared"
    
    # Configure
    Write-Host "Configuring shared library build..." -ForegroundColor Yellow
    & cmake -B $SharedBuildDir `
        -G "$Generator" `
        -A $Architecture `
        -DCMAKE_BUILD_TYPE=$BuildType `
        -DEXECUTOR_BUILD_SHARED=ON `
        -DEXECUTOR_BUILD_TESTS=OFF `
        -DEXECUTOR_BUILD_EXAMPLES=OFF `
        -DCMAKE_INSTALL_PREFIX="$SharedBuildDir\install"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: CMake configuration failed" -ForegroundColor Red
        exit 1
    }
    
    # Build
    Write-Host "Building shared library..." -ForegroundColor Yellow
    & cmake --build $SharedBuildDir --config $BuildType
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Build failed" -ForegroundColor Red
        exit 1
    }
    
    # Install
    Write-Host "Installing shared library..." -ForegroundColor Yellow
    & cmake --install $SharedBuildDir --config $BuildType
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Installation failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Shared library build completed!" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Build completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Build artifacts location:" -ForegroundColor Yellow
if ($BuildStatic) {
    Write-Host "  Static library: $OutputDir\static\install" -ForegroundColor Cyan
}
if ($BuildShared) {
    Write-Host "  Shared library: $OutputDir\shared\install" -ForegroundColor Cyan
}
