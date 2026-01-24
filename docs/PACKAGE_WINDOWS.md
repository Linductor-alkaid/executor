# Windows 打包指南

本文档说明如何在 Windows 平台上将 executor 库打包成静态库和动态库，用于发行。

---

## 快速开始

### 一键构建和打包

使用提供的 PowerShell 脚本一键完成构建和打包：

```powershell
.\scripts\build_and_package_windows.ps1
```

这将：
1. 构建静态库（Release 模式）
2. 构建动态库（Release 模式）
3. 打包成发行版本（ZIP 格式）

### 自定义构建选项

```powershell
.\scripts\build_and_package_windows.ps1 `
    -Version "0.1.0" `
    -BuildType "Release" `
    -Generator "Visual Studio 17 2022" `
    -Architecture "x64" `
    -BuildStatic:$true `
    -BuildShared:$true
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-Version` | `0.1.0` | 版本号，用于打包命名 |
| `-BuildType` | `Release` | 构建类型（Release/Debug） |
| `-Generator` | `Visual Studio 17 2022` | CMake 生成器 |
| `-Architecture` | `x64` | 目标架构（x64/x86） |
| `-BuildStatic` | `$true` | 是否构建静态库 |
| `-BuildShared` | `$true` | 是否构建动态库 |
| `-BuildTests` | `$false` | 是否构建测试 |
| `-BuildExamples` | `$false` | 是否构建示例 |
| `-BuildDir` | `build_windows` | 构建目录 |
| `-OutputDir` | `dist` | 打包输出目录 |

---

## 分步操作

### 步骤 1: 构建库

使用构建脚本分别构建静态库和动态库：

```powershell
# 构建静态库和动态库
.\scripts\build_windows.ps1 -BuildType Release

# 仅构建静态库
.\scripts\build_windows.ps1 -BuildType Release -BuildShared:$false

# 仅构建动态库
.\scripts\build_windows.ps1 -BuildType Release -BuildStatic:$false
```

### 步骤 2: 打包发行版本

构建完成后，使用打包脚本创建发行包：

```powershell
.\scripts\package_windows.ps1 -Version "0.1.0"
```

打包脚本会：
- 复制静态库和动态库的安装文件
- 复制文档（README.md, LICENSE, CHANGELOG.md）
- 创建使用说明（USAGE.md）
- 生成 ZIP 压缩包

---

## 手动构建（不使用脚本）

### 构建静态库

```powershell
# 配置
cmake -B build_static `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DEXECUTOR_BUILD_SHARED=OFF `
    -DEXECUTOR_BUILD_TESTS=OFF `
    -DEXECUTOR_BUILD_EXAMPLES=OFF `
    -DCMAKE_INSTALL_PREFIX=build_static\install

# 构建
cmake --build build_static --config Release

# 安装
cmake --install build_static --config Release
```

### 构建动态库

```powershell
# 配置
cmake -B build_shared `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DEXECUTOR_BUILD_SHARED=ON `
    -DEXECUTOR_BUILD_TESTS=OFF `
    -DEXECUTOR_BUILD_EXAMPLES=OFF `
    -DCMAKE_INSTALL_PREFIX=build_shared\install

# 构建
cmake --build build_shared --config Release

# 安装
cmake --install build_shared --config Release
```

---

## 打包目录结构

打包后的目录结构如下：

```
executor-0.1.0-windows-x64/
├── static/                    # 静态库
│   ├── lib/
│   │   ├── executor.lib      # 静态库文件
│   │   └── cmake/
│   │       └── executor/     # CMake 配置文件
│   └── include/
│       └── executor/         # 头文件
├── shared/                    # 动态库
│   ├── bin/
│   │   └── executor.dll      # 动态库文件（运行时）
│   ├── lib/
│   │   ├── executor.lib      # 导入库（链接时）
│   │   └── cmake/
│   │       └── executor/     # CMake 配置文件
│   └── include/
│       └── executor/         # 头文件
├── README.md
├── LICENSE
├── CHANGELOG.md
└── USAGE.md                   # 使用说明
```

---

## 在其他项目中使用

### 使用静态库

1. 解压发行包
2. 在 CMake 配置时设置路径：

```powershell
cmake -B build -DCMAKE_PREFIX_PATH=path\to\executor-0.1.0-windows-x64\static
```

3. 在项目的 `CMakeLists.txt` 中：

```cmake
find_package(executor REQUIRED)
target_link_libraries(your_target PRIVATE executor::executor)
```

### 使用动态库

1. 解压发行包
2. 在 CMake 配置时设置路径：

```powershell
cmake -B build -DCMAKE_PREFIX_PATH=path\to\executor-0.1.0-windows-x64\shared
```

3. 在项目的 `CMakeLists.txt` 中：

```cmake
find_package(executor REQUIRED)
target_link_libraries(your_target PRIVATE executor::executor)
```

4. **重要**: 确保 `executor.dll` 在运行时可用：
   - 将 `executor.dll` 复制到可执行文件目录
   - 或将包含 `executor.dll` 的目录添加到 PATH 环境变量

---

## 系统要求

- **操作系统**: Windows 10 或更高版本
- **编译器**: Visual Studio 2019 或更高版本（MSVC 14.0+）
- **CMake**: 3.16 或更高版本
- **C++ 标准**: C++20

---

## 常见问题

### Q: 如何选择 Visual Studio 版本？

A: 使用 `-Generator` 参数指定：

```powershell
# Visual Studio 2019
.\scripts\build_windows.ps1 -Generator "Visual Studio 16 2019"

# Visual Studio 2022
.\scripts\build_windows.ps1 -Generator "Visual Studio 17 2022"
```

### Q: 如何构建 x86 版本？

A: 使用 `-Architecture` 参数：

```powershell
.\scripts\build_windows.ps1 -Architecture "Win32"
```

### Q: 构建失败，提示找不到 CMake？

A: 确保 CMake 已安装并在 PATH 环境变量中。可以运行 `cmake --version` 验证。

### Q: 使用动态库时提示找不到 executor.dll？

A: 确保 `executor.dll` 在以下位置之一：
- 可执行文件所在目录
- 系统 PATH 环境变量中的目录
- 当前工作目录

### Q: 如何同时构建 Debug 和 Release 版本？

A: 分别运行两次构建：

```powershell
# Debug 版本
.\scripts\build_windows.ps1 -BuildType Debug -BuildDir build_windows_debug

# Release 版本
.\scripts\build_windows.ps1 -BuildType Release -BuildDir build_windows_release
```

---

## 验证构建结果

构建完成后，可以验证关键文件：

### 静态库
- `build_windows/static/install/lib/executor.lib` - 静态库文件
- `build_windows/static/install/include/executor/` - 头文件目录

### 动态库
- `build_windows/shared/install/bin/executor.dll` - 动态库文件
- `build_windows/shared/install/lib/executor.lib` - 导入库文件
- `build_windows/shared/install/include/executor/` - 头文件目录

---

## 发布检查清单

在发布前，请确认：

- [ ] 版本号正确（在 `CMakeLists.txt` 和打包脚本中）
- [ ] 静态库和动态库都已成功构建
- [ ] 所有头文件都已包含在打包中
- [ ] CMake 配置文件已正确生成
- [ ] 文档文件（README.md, LICENSE, CHANGELOG.md）已包含
- [ ] 使用说明（USAGE.md）已生成
- [ ] ZIP 压缩包已创建
- [ ] 在测试环境中验证了静态库和动态库的使用

---

## 相关文档

- [BUILD.md](BUILD.md) - 通用构建说明
- [API.md](API.md) - API 使用文档
- [README.md](../README.md) - 项目说明
