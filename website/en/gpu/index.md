---
title: GPU
description: Start with optional-backend diagnostics, then use GPU executors and automatic scheduling only when needed.
---

# GPU

GPU is optional and must not block a first task. When CUDA/OpenCL runtime is unavailable or no device exists, the basic CPU path still works; GPU registration must inspect diagnostics and provide a fallback.

1. [Diagnose backend and fall back safely](/en/gpu/diagnostics): verify registration failure and CPU fallback even on a machine without GPU.
2. [Register and submit GPU work](/en/gpu/register-and-submit): register, submit, and query status on a real device.
3. [CPU/GPU automatic selection](/en/gpu/automatic-scheduling): understand `submit_auto()` thresholds, preferences, and failure boundaries.

Build options, device query, and examples are in [`docs/BUILD.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/BUILD.md), [`examples/gpu_device_query.cpp`](https://github.com/Linductor-alkaid/executor/blob/master/examples/gpu_device_query.cpp), and [`docs/design/gpu_executor.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/design/gpu_executor.md). Any performance conclusion records device, driver, data size, and build type.
