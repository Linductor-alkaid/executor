---
title: 如何选择提交接口
description: 先按业务语义选择 Executor Facade，再考虑性能和底层接口。
---

# 如何选择提交接口

| 你遇到的问题 | 默认接口 | 返回与失败观察 |
| --- | --- | --- |
| 立即执行一次后台工作 | `submit()` | 保存 `future`，通过 `get()` 取得结果和异常。 |
| 少数控制工作需要先排队 | `submit_priority()` | 同样检查 future；只影响排队顺序。 |
| 过一段时间重试 | `submit_delayed()` | future 仍承载结果与异常。 |
| 普通后台健康检查 | `submit_periodic()` | 保存任务 ID，检查状态、失败事件并取消。 |
| 大量相似工作且需逐项结果 | `submit_batch()` | 对全部 future 调用 `get()`。 |
| 大量相似工作、不需逐项结果 | `submit_batch_no_future()` | 使用 failure status/callback 与有界等待。 |
| 一整批控制工作更紧急 | `submit_batch_priority()` | 逐项 future 仍是结果边界。 |
| 后续工作依赖前置结果 | `submit_after()` / `when_all()` | 依赖失败会阻止后续任务执行。 |

```text
是否需要严格周期？
├─ 否：选择普通 Facade 提交接口
│  ├─ 立即：submit / submit_priority
│  ├─ 稍后：submit_delayed
│  └─ 软周期：submit_periodic
└─ 是：使用专用实时执行器（后续教程）
```

默认选项优先解决语义和可观察性；性能接口不是无条件更优。

## 循环提交还是批量提交

先以清晰性为准：数量少、任务异构或彼此依赖时，循环 `submit()` 通常最直白；同一生产者一次产生大量独立、同类任务时，`submit_batch()` 更准确地表达批次语义。批量提交可能减少提交路径开销，但不承诺固定性能提升；在目标硬件和任务规模上运行本地 benchmark 后再决定。

## 软周期还是实时线程

`submit_periodic()` 适合刷新、重试和健康检查等软周期后台工作。若需求包含固定周期、jitter 预算、CPU 亲和性、队列单周期预算或平台实时权限，选择专用实时执行器；优先级和普通周期接口不能替代它。

## future 还是 no-future

要返回值、需要在请求边界处理异常，使用 `future` 路径。只有调用方明确不需要逐项完成结果，并且已设置 callback、失败状态或业务级事件来观察失败时，才使用 no-future 路径。fire-and-forget 不等于 failure-and-forget。

通信不是任务提交的变体：需要跨线程传递数据时，请按数据语义阅读[通信组件选型](/zh/guides/choosing-communication)。完整行为、错误码和状态字段以[API 参考](/zh/reference/api)为准。
