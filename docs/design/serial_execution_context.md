# SerialExecutionContext 设计

## 范围

S2 提供一个不依赖第三方事件循环的 FIFO 串行上下文，以及
`Executor::submit_on`/`submit_on_with_handle` 纳管入口。提交首先经过默认
executor 的 admission、任务统计、失败记录与取消 registry，再由上下文单线程
按提交顺序执行 callable。

## 生命周期

`SerialExecutionContext` 析构或显式 `shutdown()` 后拒绝新任务，并排空已经接收
的排队回调。已开始的回调运行至返回；其异常通过返回的 future 传播。上下文不
拥有 `Executor`，两者可独立关闭。调用方应在关闭 executor 前停止上下文提交，
并在需要时先 `context.shutdown()` 收敛外部状态。

## 取消与监控

返回的 `TaskHandle` 是 facade 纳管任务句柄。排队阶段取消保证 callable 不执行，
future 以既有 `TaskCancelled` 异常完成；运行阶段取消仍是协作请求。上下文本身
不提供抢占，也不改变外部事件循环的线程或销毁语义。快照中的 admission、执行、
异常和取消计数来自 facade；上下文队列深度不作为独立 backend 状态承诺。

该 API 不绑定 asio strand。需要对象始终在同一外部 strand 上访问和销毁的 timer
仍属于 T2，继续使用应用侧 timer。
