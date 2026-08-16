# Android 适配实施计划

本文档把 [Android 适配方案](../design/android_port.md) 拆分为可提交、可验证的实现任务。
目标是先交付 **CPU-only、best-effort 调度、可交叉编译、可在真机 smoke test** 的 executor Android 支持；
GPU、OpenCL、硬实时和弱内存序性能调优不进入一期完成定义。

所有 [ ] 均为待办；合入顺序遵循“先构建系统与平台层，再公开 API 兼容层，最后设备验证与分发”。

---

## 当前基线

- [x] 核心库为 C++20 / CMake 3.16+，无第三方必需依赖。
- [x] 桌面平台已有 Linux + Windows 实现与 CI。
- [x] `ThreadPoolExecutor`、`RealtimeThreadExecutor`、`BlockingIoExecutor`、`ExecutorManager`、`executor::comm` 和监控面均已存在。
- [x] 已用 NDK r26c / r28b 完成可行性交叉编译实验。
- [ ] 当前 `src/executor/util/thread_utils.cpp` 使用 `pthread_setaffinity_np` / `pthread_getaffinity_np`，Android 无法编译。
- [ ] 当前 `include/executor/blocking_io.hpp` 暴露 `std::stop_token`，NDK r26c 无此头文件。
- [ ] 当前 `src/CMakeLists.txt` 在 Android 上链接不存在的 `librt` 并导出 `atomic`。
- [ ] 当前无 Android CI、打包脚本、Gradle/Prefab 集成和设备测试路径。

---

## 实施原则与不可变约束

1. **公开 API 桌面兼容**：`executor::StopToken` 在桌面必须是 `std::stop_token` 别名，不破坏现有 override 和 ABI。
2. **不依赖实验性标准库**：Android 不得要求消费者手动开启 `-fexperimental-library`。
3. **平台调优 best-effort**：Android 上 priority / affinity / mlock / timer slack 失败不改变任务接受结果，必须可通过状态字段观察。
4. **编译通过不等于并发正确**：ARM 弱内存序路径必须通过设备侧压力与功能测试。
5. **一期 GPU 关闭**：CUDA/OpenCL 默认不在 Android 构建中启用；API 保持 `BackendUnavailable` 语义。
6. **生命周期显式化**：JNI/App 集成文档必须要求显式 `shutdown()`，不依赖静态析构。
7. **桌面不回归**：每个阶段合入后，Linux / Windows 现有测试必须保持通过。
8. **文档以可复现命令为准**：所有 Android 命令必须在文档中给出，且 CI 与文档使用同一脚本。

---

## 阶段 A0：Android 构建系统与平台层

> 进度说明：A0 已随 A1 的 `executor::StopToken` 兼容层一起完成代码任务；
> CI workflow 已新增，默认 Android 构建不再依赖 `-fexperimental-library`。

### 任务

- [x] 在 `src/CMakeLists.txt` 增加 `if(ANDROID)` 分支：
  - [x] `find_package(Threads REQUIRED)` 保留。
  - [x] 不链接 `rt`。
  - [x] 不把 `atomic` 作为 Android 公共依赖导出。
- [x] 在根 `CMakeLists.txt` 中，Android 构建默认关闭 GPU：
  - [x] `EXECUTOR_ENABLE_CUDA` 默认 OFF。
  - [x] `EXECUTOR_ENABLE_OPENCL` 保持现有默认 OFF。
  - [x] 保留用户显式覆盖能力。
- [x] 在 `src/executor/util/thread_utils.cpp` 增加 `__ANDROID__` affinity 路径：
  - [x] `set_cpu_affinity()` 使用 `sched_setaffinity(gettid(), ...)`。
  - [x] `get_current_thread_affinity()` 使用 `sched_getaffinity(gettid(), ...)`。
  - [x] 保持返回值语义：空列表 / false 表示未生效。
- [x] 处理 NDK clang 不支持的 warning 选项：
  - [x] `-Wlogical-op`、`-Wnoexcept`、`-Wstrict-null-sentinel` 不用于 Android/Clang。
  - [x] 保持 GCC 与 MSVC 现有 warning 行为不变。
- [x] 新增 `scripts/build_android.sh`：
  - [x] 参数：NDK 路径、ABI、API level、static/shared、examples 开关。
  - [x] 默认 arm64-v8a + x86_64，API 21，CPU-only。
  - [x] 输出物统一到 `build-android/<abi>/...`。
- [x] 新增 `.github/workflows/android.yml`：
  - [x] NDK r26c 与 r28b 双版本交叉编译。
  - [x] arm64-v8a / x86_64，static + shared。
  - [x] 缓存 NDK，不上传 GB 级 SDK 到 artifacts。
  - [x] 当前只做构建门禁，不运行设备测试。

### 验收

- [x] 默认配置（无 `-fexperimental-library`）在 NDK r26c / r28b 上产出 CPU-only 静态库与共享库。
- [x] arm64-v8a / x86_64，API 21，static + shared 已本地交叉编译验证。
- [x] `basic_submit` 和全部 CPU examples/tutorial 已交叉编译通过。
- [x] Linux 本机构建通过；Windows CI 待远端 workflow 运行后确认，本轮改动未触及 MSVC 分支。
- [x] Android 共享库 `llvm-readelf -d` 不含对 `librt` 的依赖。

### 合并粒度

- [x] CMake 链接分支 + GPU 默认值：独立提交。
- [x] thread_utils Android affinity：独立提交。
- [x] warning 裁剪：独立提交。
- [x] 构建脚本 + CI workflow：已提交。

---

## 阶段 A1：`executor::StopToken` 兼容层与 Blocking I/O 生命周期

### 任务

- [x] 新增 `include/executor/stop_token.hpp`：
  - [x] 桌面：`using StopToken = std::stop_token;`。
  - [x] Android 且 `__cpp_lib_jthread` 不可用时启用自有实现。
  - [x] 提供 `StopToken::stop_requested()`。
  - [x] 提供 `StopSource::request_stop()` / `get_token()`。
  - [x] 提供 `detail::JThread`：可移动、析构 request_stop + join。
  - [x] 不在 `std` 命名空间注入符号。
  - [x] 头文件可独立包含。
- [x] `include/executor/blocking_io.hpp`：
  - [x] `IBlockingIoWorker::run()` 参数改为 `executor::StopToken`。
  - [x] 文档注释继续说明 stop token 只表达请求，wakeup 才能解除底层阻塞。
- [x] `src/executor/blocking_io_executor.hpp/.cpp`：
  - [x] 桌面继续使用 `std::jthread`（通过 `detail::JThread` 别名）。
  - [x] Android fallback 使用 `executor::detail::JThread` 与 `StopSource`。
  - [x] `request_stop_locked()` 的平台分支保持“标记 -> request_stop -> wakeup”顺序。
  - [x] `stop()` 的 join 与自停止语义不改变。
- [x] 项目内示例与测试迁移：
  - [x] `examples/tutorial/12_blocking_io_worker.cpp` 使用 `executor::StopToken`。
  - [x] `tests/test_blocking_io_executor.cpp` 与 facade 测试使用 `executor::StopToken`。
  - [x] 确保桌面代码使用 `std::stop_token` 仍可 override（alias 兼容性测试）。
- [x] 新增编译期测试：
  - [x] 桌面 `static_assert(std::is_same_v<executor::StopToken, std::stop_token>)`。
  - [x] Android fallback 的复制/移动和 request_stop 单测（`tests/test_stop_token_compat.cpp`；主机 forced fallback 已运行，真机执行归 A3）。
- [x] 文档：
  - [x] `docs/API.md` 增加 `executor::StopToken` 说明。
  - [x] `docs/MIGRATION.md` 增加 Android Blocking I/O worker 接入说明。

### 验收

- [x] NDK r26c 与 r28b（不传 `-fexperimental-library`）均能编译 `blocking_io.hpp` 并构建完整库。
- [x] `tutorial_12_blocking_io_worker` 交叉编译通过。
- [x] 桌面 Blocking I/O 测试全部通过，ABI/API 不回归。
- [x] 重复 stop、ready 超时、worker 异常、自停止路径在桌面继续通过。
- [x] Android fallback 的 `JThread` 析构/移动不产生 double join（`EXECUTOR_STOP_TOKEN_FORCE_FALLBACK` 已在本机运行时验证；真机执行仍归 A3）。

### 合并粒度

- [x] 公开兼容层头文件：独立提交。
- [x] BlockingIoExecutor 实现迁移：独立提交。
- [x] 示例/测试迁移：独立提交。
- [x] API/MIGRATION 文档：独立提交。

---

## 阶段 A2：Android 运行时语义调整

### 任务

- [ ] `ExecutorManager` 默认线程数：
  - [ ] Android 下默认 `max_threads` 上限设为 4（待评审后固化）。
  - [ ] `hardware_concurrency()` 失败时继续走现有安全默认。
- [ ] 自动 CPU affinity：
  - [ ] Android 下不使用 `0..hw-1` 伪列表。
  - [ ] 优先从 `util::get_current_thread_affinity()` 获得允许 cpuset。
  - [ ] 空 mask 时保持 OS 自由调度。
- [ ] 实时线程执行器：
  - [ ] 保持 `priority_applied` / `cpu_affinity_applied` / `process_memory_lock_applied` / `timer_slack_applied` 状态语义。
  - [ ] 确认 `SCHED_FIFO` EPERM 时只记录失败，不影响周期任务。
  - [ ] 评审是否在 Android 上默认关闭自动 `SCHED_FIFO` 建议值。
- [ ] 测试代码平台守卫：
  - [ ] `/proc/self/task/.../comm` 测试增加 `__ANDROID__` 跳过或改用其他验证。
  - [ ] `sched_getcpu()` 测试增加 Android 分支注释，确认 API 21 行为。
  - [ ] 所有仅适合 desktop Linux 的测试用 `__linux__ && !defined(__ANDROID__)` 区分。
- [ ] 更新状态字段注释：
  - [ ] Android 上 priority / affinity / mlock / timer slack 均标注 best-effort。

### 验收

- [ ] Android 上默认 async executor 初始化成功，线程数不超过 4。
- [ ] 显式配置 affinity 在无权限设备上失败时，任务提交和 shutdown 不异常。
- [ ] 相关桌面测试通过。
- [ ] 现有 Linux 行为（线程数、affinity 列表）不变。

### 合并粒度

- [ ] 线程数上限：独立提交。
- [ ] cpuset 自适应：独立提交。
- [ ] 测试守卫：独立提交。

---

## 阶段 A3：设备侧验证与 ARM 弱内存序

### 任务

- [ ] 增加 native smoke test 目标或复用现有 examples：
  - [ ] `basic_submit`：线程池 submit / future / shutdown。
  - [ ] `tutorial_12_blocking_io_worker`：注册、启动、request_stop、join。
  - [ ] 至少一个 `comm` 示例：channel/mailbox/phase gate。
- [ ] 建立设备测试脚本：
  - [ ] `adb push` 测试二进制。
  - [ ] `adb shell` 运行并收集退出码与输出。
  - [ ] 支持 `ANDROID_SERIAL` 指定设备。
- [ ] 获取可在 Android 运行的 GTest：
  - [ ] 优先使用与 NDK 一致的 prebuilt 或自行交叉编译 GTest。
  - [ ] 若一期不引入 GTest，则用轻量 assert main 覆盖核心 smoke。
- [ ] ARM 并发专项：
  - [ ] `test_lockfree_queue_size` / `test_lockfree_queue_status`。
  - [ ] `test_lockfree_worker_queue_concurrent_steal`。
  - [ ] `test_multithread_mpsc`。
  - [ ] batch push/pop 与 reservation cancellation 路径。
  - [ ] 单核、4 核、big.LITTLE 设备各至少一轮。
- [ ] 长稳测试：
  - [ ] MPSC 多生产者 + 单消费者 soak ≥ 10 分钟。
  - [ ] 监控 dropped / queue_full / peak size 无异常翻转。
- [ ] 收集结果：
  - [ ] 设备型号、Android API、内核版本、ABI。
  - [ ] 单轮吞吐与失败计数。
  - [ ] 如失败，先关闭 `EXECUTOR_LOCKFREE_QUEUE` 复测以定位队列实现问题。

### 验收

- [ ] arm64 真机或官方模拟器 smoke test 全部通过。
- [ ] ARM 并发专项无失败、无 TSAN/ASAN 报告的数据竞争。
- [ ] 长稳 MPSC 无队列状态破坏。
- [ ] 若存在弱内存序缺陷，按缺陷单独建 PR，不阻塞 Android 库禁用 lockfree 选项交付。

### 合并粒度

- [ ] 测试脚本与 smoke 目标：独立提交。
- [ ] 设备测试文档：独立提交。
- [ ] 并发缺陷修复：按问题独立提交。

---

## 阶段 A4：Android 分发与集成文档

### 任务

- [ ] 新增 `docs/PACKAGE_ANDROID.md`：
  - [ ] NDK 路径、ABI、API level、STL 选择。
  - [ ] 静态库 / 共享库差异。
  - [ ] `CMAKE_FIND_ROOT_PATH` / `executor_DIR` 交叉查找说明。
  - [ ] AGP `externalNativeBuild` + CMake 接入示例。
  - [ ] `c++_shared` 时打包 `libc++_shared.so`。
  - [ ] JNI 生命周期中显式 `shutdown()` 示例。
- [ ] 更新 `docs/BUILD.md`：
  - [ ] 增加 Android 一节与 `scripts/build_android.sh` 入口。
- [ ] 更新 README / README_zh 平台表述：
  - [ ] 在完成 A0-A2 后再更新 badge / 支持平台。
  - [ ] 不提前宣称硬实时或 GPU。
- [ ] 评估 Prefab/AAR：
  - [ ] 若接入 AGP，提供 `prefab` module 配置。
  - [ ] 产出物：`libexecutor.so` + 公开头 + prefab metadata。
  - [ ] 不把测试或 examples 打入 AAR。
- [ ] 更新 `docs/RELEASE_CHECKLIST.md`：
  - [ ] 增加 Android 构建产物与设备测试项。

### 验收

- [ ] 一个最小 AGP/CMake 消费者工程可按文档集成 executor。
- [ ] `find_package(executor)` 路径按文档可复现。
- [ ] 共享库模式下 APK 包含 `libc++_shared.so`（如适用）。
- [ ] 文档命令在 CI 或本地实际执行过。

### 合并粒度

- [ ] PACKAGE_ANDROID 文档：独立提交。
- [ ] BUILD/README 交叉更新：独立提交。
- [ ] Prefab/AAR 支持：独立提交。

---

## 阶段 A5（可选，二期）：OpenCL 后端评估

### 任务

- [ ] 调研目标设备 OpenCL 可用性：`/vendor/lib64/libOpenCL.so`、`/system/vendor/lib64`。
- [ ] 在 `opencl_loader.cpp` 增加 Android 搜索路径。
- [ ] 处理 App 对 vendor 库的 `dlopen` namespace / SELinux 限制。
- [ ] 建立设备白名单与运行时探测报告。
- [ ] 在未检测到 OpenCL 时保持 `BackendUnavailable` 语义。

### 验收

- [ ] 至少一个白名单设备完成 OpenCL kernel 提交 smoke test。
- [ ] 无 OpenCL 设备行为与一期相同。
- [ ] 文档明确 Android OpenCL 是可选能力，不是默认能力。

---

## 文档与维护

- [ ] 本计划完成后，将完成状态回写到 [Android 适配方案](../design/android_port.md) 的待决项。
- [x] 在 [项目任务清单](todolist.md) 增加 Android 阶段并链接本计划。
- [ ] 所有 Android 相关公开文档必须有对应可执行命令或已运行记录。
- [ ] 涉及公开 API 的文档不得早于实现合并。

---

## 建议合并顺序

1. A0：平台构建基础，先让 CI 能交叉编译失败变少。
2. A1：公开 API 兼容层与 Blocking I/O，解除标准库阻塞。
3. A2：运行时语义，处理 Android 默认值。
4. A3：设备验证，ARM 弱内存序专项。
5. A4：分发与集成文档，最后更新对外平台宣称。
6. A5：可选 OpenCL，独立评审。

每个阶段必须以独立可回滚提交合入；A4 不得早于 A3 的 smoke test 结果，README 平台宣称不得早于 A3 完成。

---

## 风险与待决项

- [ ] NDK 双版本 CI 是否长期保留，或只保留推荐版本。
- [ ] Android 默认线程数上限 4 是否写入公开默认。
- [ ] Android 上自动 `SCHED_FIFO` 建议值是否关闭。
- [ ] 无锁队列弱内存序缺陷的降级策略：修复 / 默认关闭 / 要求用户显式启用。
- [ ] OpenCL 是否进入版本路线图。
- [ ] 是否需要自托管 Android runner；如无设备，至少保留 API 21/24 x86_64 模拟器 job。

---

## 完成定义（Definition of Done）

一期 Android 支持只有在以下条件全部满足后才可对外宣布：

- [ ] A0、A1、A2、A3、A4 全部验收项通过。
- [ ] Linux / Windows 既有 CI 不回归。
- [ ] Android CI 稳定构建 arm64-v8a + x86_64，static + shared，API 21。
- [ ] 至少一台 arm64 设备完成 smoke、Blocking I/O、MPSC 压力与长稳测试。
- [ ] 打包与集成文档可复现。
- [ ] 公开 README / BUILD / API 文档与实际行为一致。
