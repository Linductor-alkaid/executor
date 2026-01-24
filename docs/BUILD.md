# Executor 构建与安装说明

本文档说明如何配置、构建、测试、安装 executor 库，以及如何在其他项目通过 `find_package(executor)` 集成。

---

## 1. 环境要求

- **C++20** 编译器（GCC 10+、Clang 10+ 等）
- **CMake** 3.16 或更高
- **Linux**：`pthread`、`rt`（一般系统已提供）

---

## 2. 配置选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `EXECUTOR_BUILD_TESTS` | `ON` | 是否构建测试 |
| `EXECUTOR_BUILD_EXAMPLES` | `OFF` | 是否构建示例 |
| `EXECUTOR_BUILD_SHARED` | `OFF` | 是否构建动态库（`OFF` 时构建静态库） |
| `EXECUTOR_ENABLE_COVERAGE` | `OFF` | 是否启用代码覆盖率（gcov/lcov，见 [COVERAGE.md](COVERAGE.md)） |

---

## 3. 配置与构建

### 3.1 默认构建（静态库 + 测试）

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### 3.2 构建示例

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DEXECUTOR_BUILD_EXAMPLES=ON
cmake --build build
```

### 3.3 构建动态库

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DEXECUTOR_BUILD_SHARED=ON
cmake --build build
```

### 3.4 关闭测试

```bash
cmake -B build -DEXECUTOR_BUILD_TESTS=OFF
cmake --build build
```

### 3.5 指定安装前缀（安装时使用）

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build
```

---

## 4. 运行测试

```bash
ctest --test-dir build
```

或带输出：

```bash
ctest --test-dir build --output-on-failure
```

按标签过滤（若已配置）：

```bash
ctest --test-dir build -L "unit|integration" --output-on-failure
```

---

## 5. 安装

安装到 `CMAKE_INSTALL_PREFIX`（默认一般为 `/usr/local`）：

```bash
cmake --install build
```

指定前缀：

```bash
cmake --install build --prefix /opt/executor
```

安装内容包含：

- **头文件**：`<prefix>/include/executor/`（如 `executor.hpp`、`config.hpp`、`types.hpp` 等）
- **库文件**：`<prefix>/lib/` 或 `<prefix>/lib64/`（`libexecutor.a` 或 `libexecutor.so`）
- **CMake 配置**：`<prefix>/lib/cmake/executor/`（`executorConfig.cmake`、`executorConfigVersion.cmake`、`executorTargets.cmake` 等），供 `find_package(executor)` 使用

---

## 6. 在其他项目中使用（find_package）

### 6.1 安装后使用

确保安装路径在 CMake 的搜索路径中。若安装到自定义前缀，可设置：

```bash
export CMAKE_PREFIX_PATH=/opt/executor
```

或配置时传入：

```bash
cmake -B build -DCMAKE_PREFIX_PATH=/opt/executor
```

消费者项目 `CMakeLists.txt` 示例：

```cmake
cmake_minimum_required(VERSION 3.16)
project(myapp LANGUAGES CXX)

find_package(executor REQUIRED)

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE executor::executor)
```

### 6.2 未安装：add_subdirectory

若将 executor 作为子目录加入当前项目：

```cmake
add_subdirectory(path/to/executor)
add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE executor::executor)
```

可根据需要关闭测试、示例等：

```cmake
set(EXECUTOR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(EXECUTOR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(path/to/executor)
```

---

## 7. 代码覆盖率

使用 gcov/lcov 生成覆盖率报告。详见 [COVERAGE.md](COVERAGE.md)。简要步骤：

```bash
cmake -B build -DEXECUTOR_ENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build -L "unit|integration" --output-on-failure
# 随后在 build 目录运行 lcov/genhtml，或使用 scripts/run_coverage.sh
./scripts/run_coverage.sh
```

---

## 8. 常见问题

| 问题 | 处理 |
|------|------|
| `find_package(executor)` 找不到 | 确认已 `cmake --install`，且 `CMAKE_PREFIX_PATH` 包含安装前缀；或使用 `add_subdirectory`。 |
| 链接错误（如 `pthread`） | executor 通过 `Threads::Threads` 拉取 `pthread`，确保消费者项目同样使用 `executor::executor` 而不是手动 `-lpthread` 覆盖。 |
| 头文件 `executor/executor.hpp` 找不到 | 使用 `target_link_libraries(… executor::executor)`，勿手动添加 `-I`；`executor::executor` 已携带 `INTERFACE_INCLUDE_DIRECTORIES`。 |
| 静态库与动态库混用 | 同一进程内链接的 executor 应与主程序同类型（全静态或全动态），避免符号重复或加载冲突。 |

---

## 9. 准备发布包（源码归档）

以当前工程创建源码归档，便于分发或发布：

```bash
git archive --format=tar.gz --prefix=executor-0.1.0/ -o executor-0.1.0.tar.gz HEAD
```

或仅打包 `include/`、`src/`、`cmake/`、`examples/`、`tests/`、`CMakeLists.txt`、`README.md`、`CHANGELOG.md`、`docs/` 等必要目录与文件（按需调整）。解压后按 [§3](#3-配置与构建) 配置与构建即可。

---

## 10. 构建目录结构速览

```
build/
├── libexecutor.a（或 libexecutor.so）
├── executor 可执行目标（若启用示例）
├── test_* 测试可执行文件
└── ...
```

安装后：

```
<prefix>/
├── include/executor/
│   ├── executor.hpp
│   ├── config.hpp
│   ├── types.hpp
│   ├── interfaces.hpp
│   └── executor_manager.hpp
├── lib/libexecutor.a（或 lib64/）
└── lib/cmake/executor/
    ├── executorConfig.cmake
    ├── executorConfigVersion.cmake
    └── executorTargets.cmake
```
