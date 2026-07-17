---
title: 版本与迁移
description: 当前开发快照、发布版本和 API 迁移的入口。
---

# 版本与迁移

## 当前口径

项目 CMake 和最新发布记录的源码基线是 `v0.2.3`。本站描述当前 `master` 的开发快照：它也包含迁移文档中面向 `0.3.0` 规划的通信/任务图能力。除非对应稳定 tag 已发布，不要把这些能力标记为“已在 `v0.2.3` 稳定版提供”。首发不维护历史版本站点；发布时应以 tag 重新核对页面。

| 需要确认什么 | 入口 |
| --- | --- |
| 已发布版本与破坏性变更 | [CHANGELOG.md](https://github.com/Linductor-alkaid/executor/blob/master/CHANGELOG.md) |
| 从旧 API 的推荐迁移路径 | [MIGRATION.md](https://github.com/Linductor-alkaid/executor/blob/master/docs/MIGRATION.md) |
| 选项、编译器与后端前置 | [BUILD.md](https://github.com/Linductor-alkaid/executor/blob/master/docs/BUILD.md) |
| 当前完整签名 | [API.md](https://github.com/Linductor-alkaid/executor/blob/master/docs/API.md) |

## `bool` 到 `_ex` 的迁移

旧入口保持兼容，适合调用方只需成功/失败的场景；新代码在需要诊断、日志或可靠回退时优先使用 `_ex`，读取 `ExecutorResult::error_code` 与 `message`。

<div class="migration-table">

| 迁移 | 适用情况 |
| --- | --- |
| `initialize(config)` → `initialize_ex(config)` | 配置错误、重复初始化或 shutdown 后调用需要区分原因。 |
| `register_realtime_task(name, config)` → `register_realtime_task_ex(name, config)` | 需要区分非法配置、重名、权限/启动问题。 |
| `start_realtime_task(name)` → `start_realtime_task_ex(name)` | 需要区分不存在、重复启动与平台启动失败。 |
| `register_gpu_executor(name, config)` → `register_gpu_executor_ex(name, config)` | 需要区分无效配置与 `BackendUnavailable`。 |
| `wait_for_completion()` → `wait_for_completion_for()` / `_ex()` | 不可无限等待，或超时后需要状态快照。 |
| `IRealtimeExecutor::push_task()` → `Executor::try_push_realtime_task()` | 希望得到拒绝返回、failure event 和背压计数。 |

</div>

`_ex` 不是“总是更好”的第二套业务 API：若调用方只需布尔结果，兼容入口仍有效。迁移的价值在于把失败原因接到业务日志、告警或降级策略，而非改变任务执行模型。

## 升级检查

1. 阅读目标版本 CHANGELOG，并确认本页所述能力已经在目标 tag 中存在。
2. 用目标编译器、操作系统与 GPU/实时权限重新配置并构建。
3. 将初始化、实时/GPU 注册等关键边界换为 `_ex`；为 `future`、返回值和状态计数保留观察路径。
4. 对实时配置复查亲和性、内存锁与 timer slack 的实际应用状态；对 GPU 复查后端、驱动和设备。
5. 运行测试和教程 smoke tests，再在目标负载下复测超时、背压与性能。

## 术语约定

- **稳定公开 API**：`include/executor/` 下安装并受兼容约束的声明。
- **兼容入口**：为保留既有调用而存在的 `bool` / `void` API；不等于废弃。
- **开发快照能力**：`master` 中已有但尚未标记到稳定发布版本的内容。
- **测试钩子和内部实现**：测试注入 API、`src/` 类型和实现细节，不作为普通集成依赖。

## 发布前核对

发布维护者应更新 CMake 项目版本、CHANGELOG、MIGRATION、README 和本站版本文本；然后对照[API 覆盖索引](/zh/reference/api)检查 Facade 分组，确保新增公开入口至少有教程、专题、选型或参考说明。
