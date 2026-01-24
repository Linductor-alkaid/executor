# Windows 打包脚本
# 将构建好的库打包成发行版本

param(
    [string]$Version = "0.1.0",
    [string]$BuildDir = "build_windows",
    [string]$OutputDir = "dist",
    [switch]$IncludeStatic = $true,
    [switch]$IncludeShared = $true
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Executor Windows Package Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Version: $Version" -ForegroundColor Yellow
Write-Host "Build Dir: $BuildDir" -ForegroundColor Yellow
Write-Host "Output Dir: $OutputDir" -ForegroundColor Yellow
Write-Host "Include Static: $IncludeStatic" -ForegroundColor Yellow
Write-Host "Include Shared: $IncludeShared" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get project root directory
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ProjectRoot

# Create output directories
$PackageName = "executor-${Version}-windows-${env:PROCESSOR_ARCHITECTURE}"
$PackageDir = Join-Path $OutputDir $PackageName
$PackageDirStatic = Join-Path $PackageDir "static"
$PackageDirShared = Join-Path $PackageDir "shared"

if (Test-Path $PackageDir) {
    Write-Host "Cleaning old package directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $PackageDir
}

New-Item -ItemType Directory -Path $PackageDir -Force | Out-Null
New-Item -ItemType Directory -Path $PackageDirStatic -Force | Out-Null
New-Item -ItemType Directory -Path $PackageDirShared -Force | Out-Null

Write-Host "Starting packaging..." -ForegroundColor Yellow

# Copy static library
if ($IncludeStatic) {
    $StaticInstallDir = Join-Path $BuildDir "static\install"
    if (Test-Path $StaticInstallDir) {
        Write-Host "Copying static library files..." -ForegroundColor Yellow
        Copy-Item -Recurse -Path "$StaticInstallDir\*" -Destination $PackageDirStatic -Force
        
        # Verify key files
        $libFile = Get-ChildItem -Path $PackageDirStatic -Filter "executor.lib" -Recurse | Select-Object -First 1
        if (-not $libFile) {
            Write-Host "Warning: executor.lib not found" -ForegroundColor Yellow
        } else {
            Write-Host "  Found: $($libFile.FullName)" -ForegroundColor Green
        }
    } else {
        Write-Host "Warning: Static library install directory does not exist: $StaticInstallDir" -ForegroundColor Yellow
    }
}

# Copy shared library
if ($IncludeShared) {
    $SharedInstallDir = Join-Path $BuildDir "shared\install"
    if (Test-Path $SharedInstallDir) {
        Write-Host "Copying shared library files..." -ForegroundColor Yellow
        Copy-Item -Recurse -Path "$SharedInstallDir\*" -Destination $PackageDirShared -Force
        
        # Verify key files
        $dllFile = Get-ChildItem -Path $PackageDirShared -Filter "executor.dll" -Recurse | Select-Object -First 1
        $libFile = Get-ChildItem -Path $PackageDirShared -Filter "executor.lib" -Recurse | Select-Object -First 1
        
        if (-not $dllFile) {
            Write-Host "Warning: executor.dll not found" -ForegroundColor Yellow
        } else {
            Write-Host "  Found: $($dllFile.FullName)" -ForegroundColor Green
        }
        
        if (-not $libFile) {
            Write-Host "Warning: executor.lib (import library) not found" -ForegroundColor Yellow
        } else {
            Write-Host "  Found: $($libFile.FullName)" -ForegroundColor Green
        }
    } else {
        Write-Host "Warning: Shared library install directory does not exist: $SharedInstallDir" -ForegroundColor Yellow
    }
}

# Copy documentation and license
Write-Host "Copying documentation files..." -ForegroundColor Yellow
$DocsToCopy = @(
    "README.md",
    "LICENSE",
    "CHANGELOG.md"
)

foreach ($doc in $DocsToCopy) {
    $srcPath = Join-Path $ProjectRoot $doc
    if (Test-Path $srcPath) {
        Copy-Item -Path $srcPath -Destination $PackageDir -Force
        Write-Host "  Copied: $doc" -ForegroundColor Green
    }
}

# Create usage guide
$UsageGuide = @"
# Executor Windows Distribution Package Usage Guide

## Version Information
- Version: $Version
- Platform: Windows
- Architecture: $env:PROCESSOR_ARCHITECTURE

## Directory Structure

### Static Library (static/)
- \`lib/executor.lib\` - Static library file
- \`include/executor/\` - Header files directory
- \`lib/cmake/executor/\` - CMake configuration files (for find_package)

### Shared Library (shared/)
- \`bin/executor.dll\` - Shared library file (required at runtime)
- \`lib/executor.lib\` - Import library file (for linking)
- \`include/executor/\` - Header files directory
- \`lib/cmake/executor/\` - CMake configuration files (for find_package)

## Usage

### Using Static Library

\`\`\`cmake
find_package(executor REQUIRED)
target_link_libraries(your_target PRIVATE executor::executor)
\`\`\`

Make sure to set the path when configuring CMake:
\`\`\`bash
cmake -DCMAKE_PREFIX_PATH=path/to/executor-$Version-windows-$env:PROCESSOR_ARCHITECTURE/static
\`\`\`

### Using Shared Library

\`\`\`cmake
find_package(executor REQUIRED)
target_link_libraries(your_target PRIVATE executor::executor)
\`\`\`

Make sure to set the path when configuring CMake:
\`\`\`bash
cmake -DCMAKE_PREFIX_PATH=path/to/executor-$Version-windows-$env:PROCESSOR_ARCHITECTURE/shared
\`\`\`

**Note**: When using shared library, ensure \`executor.dll\` is available at runtime:
- Copy \`executor.dll\` to the executable directory
- Or add the directory containing \`executor.dll\` to PATH environment variable

## System Requirements

- Windows 10 or higher
- Visual Studio 2019 or higher (MSVC 14.0+)
- CMake 3.16 or higher
- C++20 support

## More Information

Please refer to README.md and documents in docs/ directory.
"@

$UsageGuidePath = Join-Path $PackageDir "USAGE.md"
Set-Content -Path $UsageGuidePath -Value $UsageGuide -Encoding UTF8
Write-Host "  Created: USAGE.md" -ForegroundColor Green

# Create zip package
Write-Host ""
Write-Host "Creating zip package..." -ForegroundColor Yellow
$ZipPath = Join-Path $OutputDir "${PackageName}.zip"

if (Test-Path $ZipPath) {
    Remove-Item -Force $ZipPath
}

# 使用 .NET 压缩功能
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($PackageDir, $ZipPath)

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Packaging completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Package directory: $PackageDir" -ForegroundColor Cyan
Write-Host "Zip package: $ZipPath" -ForegroundColor Cyan
Write-Host ""

# Display package content summary
Write-Host "Package content summary:" -ForegroundColor Yellow
if ($IncludeStatic) {
    $staticLibs = Get-ChildItem -Path $PackageDirStatic -Filter "*.lib" -Recurse
    $staticHeaders = Get-ChildItem -Path $PackageDirStatic -Filter "*.hpp" -Recurse
    Write-Host "  Static library: $($staticLibs.Count) .lib files, $($staticHeaders.Count) header files" -ForegroundColor Cyan
}
if ($IncludeShared) {
    $sharedDlls = Get-ChildItem -Path $PackageDirShared -Filter "*.dll" -Recurse
    $sharedLibs = Get-ChildItem -Path $PackageDirShared -Filter "*.lib" -Recurse
    $sharedHeaders = Get-ChildItem -Path $PackageDirShared -Filter "*.hpp" -Recurse
    Write-Host "  Shared library: $($sharedDlls.Count) .dll files, $($sharedLibs.Count) .lib files, $($sharedHeaders.Count) header files" -ForegroundColor Cyan
}
