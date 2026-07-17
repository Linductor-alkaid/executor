---
title: 构建与安装
description: 从源码构建 Executor，并把教程示例加入构建。
---

# 构建与安装

## 学习目标

完成一个无 GPU、无实时权限要求的 Release 构建，并编译教程示例。

## 从源码构建

在仓库根目录执行：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DEXECUTOR_BUILD_EXAMPLES=ON -DEXECUTOR_ENABLE_GPU=OFF
cmake --build build
```

Linux 需要 CMake 3.16+ 和 C++20 编译器；Windows 使用 Visual Studio 2019+ / MSVC。完整安装、`add_subdirectory` 与 `find_package(executor)` 集成方式见仓库的 [`docs/BUILD.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/BUILD.md)。

## 验证教程示例

```bash
./build/examples/tutorial/01_first_task
ctest --test-dir build -L tutorial --output-on-failure
```

Windows 下可通过构建目录中的同名 `.exe` 运行。预期首先看到 `answer=42`，然后看到已捕获的示例异常。

## 常见错误

- 忘记 `-DEXECUTOR_BUILD_EXAMPLES=ON`：教程可执行文件不会生成。
- 把 GPU 依赖带入首次构建：首个教程不需要 GPU，先显式设为 `OFF` 可减少环境差异。
- 只构建不运行：下一页用输出确认 `future.get()` 的行为。

## 下一步

进入[第一个任务](/zh/quick-start/first-task)。
