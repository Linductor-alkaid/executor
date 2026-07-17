---
title: GPU
description: 从可选后端诊断开始，再按需使用 GPU 执行器和自动调度。
---

# GPU

GPU 是可选能力，不应阻塞首次任务。CUDA/OpenCL 运行时不可用或没有设备时，基础 CPU 路径仍可工作；注册 GPU 执行器时应读取诊断结果并提供回退。

构建选项、设备查询和现有示例见 [`docs/BUILD.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/BUILD.md)、[`examples/gpu_device_query.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/gpu_device_query.cpp) 与 [`docs/design/gpu_executor.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/design/gpu_executor.md)。性能结论必须注明设备、驱动、数据规模和构建类型。
