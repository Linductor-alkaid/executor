---
layout: home
hero:
  name: Executor Guide
  text: Reliable task execution and thread management for C++20
  tagline: Build, submit work, retrieve results, and observe failures in ten minutes.
  actions:
    - theme: brand
      text: Start in ten minutes
      link: /en/quick-start/build
    - theme: alt
      text: What is Executor?
      link: /en/getting-started/what-is-executor
features:
  - title: Finish one task first
    details: Start with submit() and future.get(); you do not need to understand thread pools, GPUs, or real-time scheduling first.
  - title: Choose APIs by scenario
    details: Introduce priority, delayed, periodic, batch, and dependency APIs only when the workload requires them.
  - title: Examples stay verified
    details: Core snippets point to tutorial sources compiled and smoke-tested by the root CMake project.
---

## At a glance

```cpp
auto& executor = executor::Executor::instance();
auto answer = executor.submit([] { return 42; });
std::cout << answer.get() << '\n';
executor.shutdown();
```

`get()` both retrieves the result and rethrows an exception from the task. See [your first task](/en/quick-start/first-task) for complete code and expected output.

## Continue from here

- First use: [build and install](/en/quick-start/build), then [run your first task](/en/quick-start/first-task).
- Learn the library boundary: [what is Executor?](/en/getting-started/what-is-executor).
- Upgrade existing code: [versions and migration](/en/reference/version-and-migration).
- The complete API signatures, options, and compatibility notes remain in [`docs/API.md`](https://github.com/Linductor-alkaid/executor/blob/master/docs/API.md).

## Release information

| Item | Current support |
| --- | --- |
| Platform | Linux, Windows |
| Language | C++20 |
| Build system | CMake 3.16+ |
| Version | Development snapshot based on `v0.2.3` |
| License | [MIT](https://github.com/Linductor-alkaid/executor/blob/master/LICENSE) |

<div class="version-note">This guide follows the current development snapshot. Features not yet released in a stable tag are not promises for `v0.2.3`.</div>
