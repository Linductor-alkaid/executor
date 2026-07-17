---
title: 监控与采样
description: 使用任务统计与采样率，在可接受开销内观察 Executor 的长期趋势。
---

# 监控与采样

## 学习目标

按任务类型查询成功、失败、超时和执行时间统计，并在高频路径上选择合适的监控开关与采样率。

## 推荐方案

初始化时可通过 `ExecutorConfig::enable_monitoring` 选择初始状态；运行期间用 `enable_monitoring()` 切换，并查询统计：

```cpp
executor.enable_monitoring(true);
executor.set_monitoring_sampling_rate(0.1);  // 采样约 10% 的任务

const auto default_stats = executor.get_task_statistics("default");
const auto all_stats = executor.get_all_task_statistics();
```

`TaskStatistics` 包含总数、成功、失败、超时、总执行时间及最大/最小执行时间。未知任务类型返回零值统计；关闭监控期间新完成的任务不会增加计数。

完整示例：[`examples/monitoring_sampling_example.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/monitoring_sampling_example.cpp)。

## 选择采样率

| 目标 | 设置 | 取舍 |
| --- | --- | --- |
| 调试、低频控制路径、精确计数 | `1.0` | 每个任务都记录，观测最完整。 |
| 高吞吐服务的趋势和回归监测 | `0.01`–`0.1` | 降低监控成本，统计是样本而非逐任务账本。 |
| 极端敏感路径或暂不需要统计 | `enable_monitoring(false)` | 没有新增统计，失去任务趋势可见性。 |

`set_monitoring_sampling_rate(rate)` 的合法范围是 `0.0` 到 `1.0`。采样适合比较趋势、发现异常和容量规划；若需要某个任务的确定结果，仍应保留 `future` 或业务层状态。

## 不要混淆三类数据

- `TaskStatistics`：按 task type 聚合的任务执行统计，适合监控趋势。
- `get_async_executor_status()`：执行器此刻的队列、活跃、完成和失败快照，适合诊断拥塞与生命周期。
- `ExecutorFailureStatus`：Facade 记录的失败类别累计数，适合失败告警与根因分流。

先从一个明确问题选择数据：想知道“是否堆积”看状态快照；想知道“失败是否增长”看 failure status；想知道“长期耗时分布是否恶化”看任务统计。

## 下一步阅读

[如何选择提交接口](/zh/guides/choosing-submit-api)从任务语义选择默认 API；通信数据语义请看[通信组件选型](/zh/guides/choosing-communication)。
