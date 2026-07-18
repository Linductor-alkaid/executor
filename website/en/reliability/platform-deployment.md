---
title: Linux and Windows Deployment
description: Verify build artifacts, CPU availability, real-time scheduling, memory locking, timers, and runtime status to distinguish expected platform differences from deployment failure.
---

# Linux and Windows Deployment

## Define deployment success first

“The program started” proves only that the process can run—not that requested priority, CPU affinity, memory locking, or timer precision applied. Deployment acceptance has three layers:

1. **Basic correctness:** ordinary tasks, future exceptions, communication, and bounded shutdown work.
2. **Platform capability:** target machine offers the needed backend, CPU set, and permission.
3. **Runtime outcome:** Executor status confirms requests applied, and load testing meets latency/jitter targets.

For an ordinary thread pool, the first layer is usually enough. Do not request real-time privilege by default “for speed”; enter the latter layers only for an explicit control-period or tail-latency target.

## Build and runtime differences

| Item | Linux | Windows |
| --- | --- | --- |
| Toolchain | CMake 3.16+, GCC/Clang, C++20 | CMake 3.16+, Visual Studio 2019+/MSVC, C++20 |
| Thread implementation | pthread | Windows thread API |
| CMake generator | Usually single-config | Visual Studio usually multi-config; build/CTest specify `Release` |
| Real-time priority | `SCHED_FIFO` 1–99 with authorization | `SetThreadPriority` level; not equivalent to `SCHED_FIFO` |
| CPU affinity | Select within current allowed cpuset | `SetThreadAffinityMask`; current implementation uses one 64-bit mask |
| Memory locking | `mlockall(MCL_CURRENT | MCL_FUTURE)` | No equivalent; `memory_locked` is expected false |
| Timer slack | `PR_SET_TIMERSLACK` | No per-thread equivalent; `timer_slack_applied` expected false |
| Short-period timing | Monotonic clock and timer slack | Requests 1 ms timer period for thread lifetime when period <20 ms |

Linux Release build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTOR_BUILD_TESTS=ON \
  -DEXECUTOR_BUILD_EXAMPLES=ON \
  -DEXECUTOR_ENABLE_GPU=OFF
cmake --build build -j
ctest --test-dir build -L tutorial --output-on-failure
```

Windows PowerShell Release build:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" `
  -DEXECUTOR_BUILD_TESTS=ON `
  -DEXECUTOR_BUILD_EXAMPLES=ON `
  -DEXECUTOR_ENABLE_GPU=OFF
cmake --build build --config Release
ctest --test-dir build -C Release -L tutorial --output-on-failure
```

Start deployments with GPU disabled and basic smoke tests. Validate GPU driver/runtime/device visibility separately from ordinary-task health.

## Linux: check final-environment permissions

Run under the final shell/process identity:

```bash
uname -srmo
cmake --version
c++ --version
nproc
taskset -pc $$
ulimit -r
ulimit -l
grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status
```

`taskset` and `Cpus_allowed_list` identify CPUs actually available to the process; container numbering may not begin at zero. `ulimit -r` bounds requested real-time priority: insufficient `RLIMIT_RTPRIO`/`CAP_SYS_NICE` leaves the thread running but `priority_applied=false`. `ulimit -l` bounds lockable memory: insufficient value/`CAP_IPC_LOCK` leaves `memory_locked=false`. `Mems_allowed_list` is not Executor configuration but can affect locality and jitter.

If file capabilities grant authority, inspect `getcap ./your-service`. Do not use running the entire service under `sudo` as a long-term fix. Give minimal capability/resource limits through systemd, container runtime, or security policy. Check `LimitRTPRIO=`, `LimitMEMLOCK=`, `CPUAffinity=`, final user, host cpuset, capability/limits, and orchestration overrides in the final container/service—not only a host shell.

## Windows: check the final service account

```powershell
[Environment]::OSVersion.VersionString
cmake --version
Get-CimInstance Win32_OperatingSystem |
  Select-Object Caption, Version, OSArchitecture
Get-CimInstance Win32_ComputerSystem |
  Select-Object NumberOfLogicalProcessors, TotalPhysicalMemory
Get-Process -Id $PID |
  Select-Object ProcessName, PriorityClass, ProcessorAffinity
```

These record environment but do not replace thread-level Executor state. Windows priority semantics differ from Linux; one 64-bit affinity mask does not cover all processor-group cases on hosts above 64 logical CPUs; false `memory_locked` and `timer_slack_applied` are expected in this implementation; short-period threads request 1 ms timer precision (with power cost) during life; thread naming requires Windows 10 1607+ and is diagnostic only. A service account, interactive shell, and CI runner can have different limits.

## Confirm requests through runtime status

Platform inspection shows possible capability. Executor status shows this run's result:

```cpp
const auto status = executor.get_realtime_executor_status("control-loop");
std::cout
    << "running=" << status.is_running
    << ", period_ns=" << status.cycle_period_ns
    << ", priority=" << status.priority_applied
    << ", affinity=" << status.cpu_affinity_applied
    << ", memory=" << status.memory_locked
    << ", timer_slack=" << status.timer_slack_applied
    << ", cycles=" << status.cycle_count
    << ", cycle_timeouts=" << status.cycle_timeout_count
    << ", dropped=" << status.dropped_task_count << '\n';
```

Wait until `cycle_count` grows, then compare status deltas under steady and overloaded load.

| Requirement | Minimum passing condition |
| --- | --- |
| Cross-platform background period only | `is_running=true`, growing `cycle_count`, exit in budget; tuning fields may be false |
| Linux fixed CPU | `cpu_affinity_applied=true`, configured CPU allowed by process cpuset, verified with system tool |
| Linux real-time scheduling | `priority_applied=true`, plus tail latency/jitter validation under target load |
| Linux paging-jitter mitigation | `memory_locked=true`, plus memory peak inside deployment budget |
| Windows short-period control | Running and period statistics meet target; Linux-specific memory/timer-slack fields not required |

If tuning falls back, the library runs safely and records it. Business requirements decide whether to keep accepting traffic: background refresh may degrade; hard control budget should fail health checks.

## Affinity, GPU, and deployment record

With empty `RealtimeThreadConfig::cpu_affinity`, Executor round-robins among CPUs allowed to the current thread and auto-binds only if at least two are allowed. For explicit affinity: read the final allowed set, reserve capacity for OS/interrupts/ordinary workers, verify `cpu_affinity_applied`, inspect actual affinity with system tools, and measure under full load. Never copy development-machine CPU numbering into another SKU, VM, or container.

For GPU, record three layers: CMake CUDA/OpenCL enablement and headers/libraries; driver/runtime/device visibility to the final account; then `register_gpu_executor_ex()` result and post-registration `GpuExecutorStatus::last_error_message`. A real kernel future is still required; validate CPU fallback independently.

Save build version/commit, OS/architecture, compiler/CMake, service identity/start method, allowed/explicit CPUs, real-time and memlock limits, GPU backend/driver/device, realtime status, smoke tests, steady/overload/shutdown results, and accepted tuning fallbacks. Re-run after base-image, CPU SKU, service-account, or security-policy changes.

For a live symptom return to [troubleshoot by symptom](/en/reliability/troubleshooting); see [dedicated real-time control](/en/realtime-and-communication/realtime-control) for status/queue semantics.
