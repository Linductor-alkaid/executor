# Executor 项目实现任务清单

本文档基于 [Executor 工具项目架构设计](executor.md) 和 [C++ 项目设计方案](cpp-project-design.md)，列出项目实现的任务清单。

---

## 阶段 0：项目初始化

- [x] 创建项目目录结构
- [x] 创建根 `CMakeLists.txt`
- [x] 创建 `cmake/CompilerWarnings.cmake`
- [x] 创建 `cmake/Sanitizers.cmake`（可选）
- [x] 创建 `cmake/ExecutorConfig.cmake`（可选，用于 find_package 支持）
- [x] 创建 `src/CMakeLists.txt`
- [x] 创建 `tests/CMakeLists.txt`
- [x] 创建 `examples/CMakeLists.txt`

---

## 阶段 1：接口与类型定义

- [x] 实现 `include/executor/config.hpp`（ExecutorConfig, ThreadPoolConfig, RealtimeThreadConfig）
- [x] 实现 `include/executor/types.hpp`（AsyncExecutorStatus, RealtimeExecutorStatus, Task, TaskPriority）
- [x] 实现 `include/executor/interfaces.hpp`（IAsyncExecutor, IRealtimeExecutor, ICycleManager）
- [x] 实现 `include/executor/executor_manager.hpp`（ExecutorManager 类声明）

---

## 阶段 2：工具模块（util）

- [x] 实现 `src/executor/util/lockfree_queue.hpp`（无锁队列）
- [x] 实现 `src/executor/util/exception_handler.cpp`
- [x] 实现 `src/executor/util/thread_utils.cpp`（线程优先级、CPU 亲和性，支持 Linux/Windows）
- [x] 编写 util 模块单元测试

---

## 阶段 3：任务模块（task）

- [ ] 实现 `src/executor/task/task.cpp`
- [ ] 实现 `src/executor/task/task_dependency_manager.cpp`
- [ ] 编写 task 模块单元测试

---

## 阶段 4：线程池模块（thread_pool）

- [ ] 实现 `src/executor/thread_pool/priority_scheduler.cpp`
- [ ] 实现 `src/executor/thread_pool/thread_pool.cpp`（基础版本，暂不实现动态扩缩容）
- [ ] 编写 thread_pool 模块单元测试

---

## 阶段 5：线程池执行器

- [ ] 实现 `src/executor/thread_pool_executor.cpp`（实现 IAsyncExecutor 接口）
- [ ] 编写 ThreadPoolExecutor 集成测试

---

## 阶段 6：执行器管理器

- [ ] 实现 `src/executor/executor_manager.cpp`（单例模式 + 实例化模式）
- [ ] 实现 `initialize_async_executor` 和 `get_default_async_executor`
- [ ] 实现 `register_realtime_executor` 和 `get_realtime_executor`
- [ ] 实现 `create_realtime_executor`（便捷方法）
- [ ] 实现 `get_realtime_executor_names` 和 `shutdown`
- [ ] 实现 RAII 生命周期管理（析构时自动释放所有执行器）
- [ ] 编写 ExecutorManager 集成测试

---

## 阶段 7：实时线程执行器

- [ ] 实现 `src/executor/realtime_thread_executor.cpp`（实现 IRealtimeExecutor 接口）
- [ ] 实现内置 `simple_cycle_loop`（使用 `std::this_thread::sleep_until`）
- [ ] 编写 RealtimeThreadExecutor 集成测试

---

## 阶段 8：Executor Facade

- [ ] 实现 `include/executor/executor.hpp`（Facade 模式）
- [ ] 实现 `src/executor/executor.cpp`（单例模式 + 实例化模式）
- [ ] 实现 `submit`、`submit_priority`、`submit_delayed` 等任务提交 API
- [ ] 实现 `register_realtime_task`、`start_realtime_task`、`stop_realtime_task` 等实时任务 API
- [ ] 实现监控查询 API（`get_async_executor_status`、`get_realtime_executor_status` 等）
- [ ] 编写 Executor Facade 集成测试
- [ ] 编写示例 `examples/basic_submit.cpp`

---

## 阶段 9：可选功能 - 线程池增强

- [ ] 实现 `src/executor/thread_pool/task_dispatcher.cpp`
- [ ] 实现 `src/executor/thread_pool/load_balancer.cpp`
- [ ] 在 `thread_pool.cpp` 中实现动态扩缩容功能
- [ ] 实现工作窃取（Work Stealing）机制
- [ ] 编写 LoadBalancer 和动态扩缩容单元测试

---

## 阶段 10：监控模块（可选）

- [ ] 实现 `src/executor/monitor/task_monitor.cpp`
- [ ] 实现 `src/executor/monitor/statistics_collector.cpp`
- [ ] 在 `Executor` 中实现 `enable_monitoring`、`get_task_statistics` 等 API
- [ ] 编写监控模块单元测试

---

## 阶段 11：ICycleManager 集成（可选）

- [ ] 在 `RealtimeThreadExecutor` 中支持注入 `ICycleManager`
- [ ] 实现 `cycle_loop` 方法（使用外部周期管理器）
- [ ] 编写 ICycleManager 集成测试
- [ ] 编写示例 `examples/realtime_can.cpp`（展示周期管理器使用）

---

## 阶段 12：测试与示例完善

- [ ] 完善所有模块的单元测试覆盖
- [ ] 编写端到端集成测试（完整工作流测试）
- [ ] 编写性能测试和压力测试
- [ ] 编写示例 `examples/realtime_can.cpp`（CAN 通信实时线程示例）
- [ ] 编写示例 `examples/multi_project.cpp`（多项目/多模块使用示例）
- [ ] 配置 CTest 测试框架
- [ ] 配置代码覆盖率工具（如 gcov/lcov）

---

## 阶段 13：文档与发布准备

- [ ] 编写 API 使用文档（README.md 或 API.md）
- [ ] 编写构建说明文档
- [ ] 编写迁移指南（如有）
- [ ] 创建 `executorConfig.cmake` 以支持 `find_package(executor)`
- [ ] 配置安装规则（头文件、库文件）
- [ ] 添加版本号管理
- [ ] 编写 CHANGELOG.md
- [ ] 代码审查和重构优化
- [ ] 性能测试和优化
- [ ] 准备发布包