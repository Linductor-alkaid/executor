---
title: 如何选择提交接口
description: 先按业务语义选择 Executor Facade，再考虑性能和底层接口。
---

# 如何选择提交接口

| 你遇到的问题 | 默认接口 | 需要注意 |
| --- | --- | --- |
| 立即执行一次后台工作 | `submit()` | 用 future 获取结果和异常。 |
| 少数控制工作需要先执行 | `submit_priority()` | 只影响排队顺序，不保证实时性。 |
| 过一段时间重试 | `submit_delayed()` | 属于 Facade 的延迟调度。 |
| 普通后台健康检查 | `submit_periodic()` | 是软周期，不是专用实时线程。 |
| 大量相似工作 | `submit_batch()` | 先以本地 benchmark 验证收益。 |
| 后续工作依赖前置结果 | `submit_after()` / `when_all()` | 需要明确依赖失败传播。 |

```text
是否需要严格周期？
├─ 否：选择普通 Facade 提交接口
│  ├─ 立即：submit / submit_priority
│  ├─ 稍后：submit_delayed
│  └─ 软周期：submit_periodic
└─ 是：使用专用实时执行器（后续教程）
```

默认选项优先解决语义和可观察性；性能接口不是无条件更优。
