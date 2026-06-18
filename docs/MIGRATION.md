# 迁移指南

本文档说明不同 executor 版本之间的迁移方式。若你从旧版本升级，请按对应版本节的说明操作。

---

## 从 0.2.1 升级到 0.2.2

0.2.2 是向后兼容版本，**没有破坏性变更**。已有 0.2.1 代码可以直接重新编译使用；需要注意的是，部分 facade 默认值改为"默认即最优"，零配置用户会自动获得更积极的线程池与实时线程配置。

### 默认值变化：默认即最优 Facade

- `RealtimeThreadConfig.enable_memory_lock` 默认 `true`：Linux 下尝试 `mlockall`，降低分页导致的实时抖动；失败静默。
- `RealtimeThreadConfig.timer_slack_ns` 默认 `1`：Linux 下将 timer slack 调到 1 ns；设置为 `0` 表示显式 opt-out。
- `ThreadPoolConfig.min_threads` / `max_threads` 默认 `0`：作为 sentinel，初始化时自动探测 `hardware_concurrency()`；探测失败退到安全默认。
- `ThreadPoolConfig.enable_work_stealing` 默认 `true`：`max_threads == 1` 时自动关闭。
- `cpu_affinity` 为空时自动分配：线程池使用 [0..hw-1]；实时线程空 affinity 时自动分配核心，显式配置始终保留。

### 新增 API

- `IRealtimeExecutor::push_task_ex(std::function<void()>) -> bool`：背压可见版本的实时任务推送 API。返回 `true` 表示成功入队，返回 `false` 表示任务因空任务、队列满或对象池耗尽被丢弃；`push_task()` 的 `void` 签名保留以保证兼容。
- `RealtimeExecutorStatus` 新增背压字段：`dropped_task_count`、`failed_pushes`、`peak_queue_size`、`queue_capacity`，用于观察实时任务队列是否出现丢任务。
- `task_timeout_ms` 软超时：当任务开始执行前发现排队时间 `elapsed >= timeout` 时跳过任务并增加 `timeout_count`。执行中的任务不会被强制中断。

### 破坏性变更

**无。** 0.2.2 保持 0.2.1 公开 API 兼容；新增字段、默认值和 API 均为向后兼容扩展。

### 升级检查清单

- [ ] 如果业务不希望库自动锁内存或调整 timer slack，显式设置 `enable_memory_lock = false` 或 `timer_slack_ns = 0`。
- [ ] 如果线程池线程数或 CPU 亲和性必须固定，显式设置 `min_threads`、`max_threads` 与 `cpu_affinity`，不要依赖默认 sentinel。
- [ ] 实时任务推送路径建议从 `push_task()` 迁移到 `push_task_ex()`，并监控 `dropped_task_count`。
- [ ] 使用 `task_timeout_ms` 时确认它是软超时：长任务需要在任务内部自行检查取消条件。
- [ ] 打包或安装 GPU 版本时确认 CUDA/OpenCL 为可选运行时依赖；无 GPU 或无 CUDA 驱动时会运行时降级。

---

## 从无到有（首次使用）

**0.1.0** 为首个发布版本，无需迁移。直接参考 [README.md](../README.md)、[docs/API.md](API.md) 与 [docs/BUILD.md](BUILD.md) 集成即可。

---

变更摘要见 [CHANGELOG.md](../CHANGELOG.md)。
