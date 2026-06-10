# 迁移指南

本文档说明不同 executor 版本之间的迁移方式。若你从旧版本升级，请按对应版本节的说明操作。

---

## 从无到有（首次使用）

**0.1.0** 为首个发布版本，无需迁移。直接参考 [README.md](../README.md)、[docs/API.md](API.md) 与 [docs/BUILD.md](BUILD.md) 集成即可。

---

## 后续版本

若未来发布包含**破坏性变更**的版本，将在此添加迁移步骤与兼容性说明。

### P016：RealtimeThreadConfig 新增 opt-in 字段

`RealtimeThreadConfig` 新增 `enable_memory_lock`（`bool`，默认 `false`）和 `timer_slack_ns`（`uint64_t`，默认 `0`）字段。两字段均为 opt-in，默认值与旧版行为完全一致，**对已有代码无需任何修改**。

变更摘要见 [CHANGELOG.md](../CHANGELOG.md)。
