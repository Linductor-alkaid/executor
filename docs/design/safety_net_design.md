# 兜底策略设计：懒初始化与退出时自动关闭

## 概述

本文档描述 executor 项目中两项**兜底策略**的设计：**懒初始化（Lazy Initialization）**与**退出时自动关闭（Exit-time Shutdown）**。目的是在开发者忘记显式调用 `initialize()` 或 `shutdown()` 时，仍能保证程序行为合理、避免崩溃与资源泄漏。

**设计目标**：
- 首次向线程池提交任务时若未初始化，则使用默认配置自动初始化，不抛异常
- 进程退出时若尚未关闭执行器（尤其单例模式），则自动执行关闭，避免线程未 join 导致泄漏或未定义行为
- 与现有显式初始化/关闭语义兼容，不改变“最佳实践”下的使用方式
- 实现成本低、行为可预测、便于文档与测试

**设计原则**：
- **兜底而非替代**：显式 `initialize()` / `shutdown()` 仍是推荐用法，兜底仅在“忘记调用”时生效
- **单例与实例区分**：懒初始化对单例和实例模式均适用；退出时自动关闭仅针对单例
- **线程安全**：懒初始化需保证多线程首次提交时只初始化一次
- **幂等与顺序**：`shutdown()` 已幂等，atexit 中调用安全；注意 atexit 与静态析构顺序

---

## 背景与动机

### 当前行为简述

| 场景 | 当前行为 |
|------|----------|
| 未调用 `initialize()` 就 `submit()` | 抛出 `std::runtime_error("Async executor not initialized. Call initialize() first.")` |
| 单例模式且未调用 `shutdown()` 就退出 | `ExecutorManager` 单例永不析构，工作线程未被 join，进程退出时存在资源泄漏/未定义行为风险 |
| 实例模式且未调用 `shutdown()` | `Executor` 析构时 `owned_manager_` 析构会调用 `ExecutorManager::~ExecutorManager()` → `shutdown(true)`，自动清理 |

### 兜底要解决的问题

1. **懒初始化**：用户忘记或不知道需要先 `initialize()`，直接 `submit()` 导致程序崩溃；希望首次提交时“静默”使用默认配置完成初始化。
2. **退出时自动关闭**：用户使用单例 `Executor::instance()` 却忘记在退出前 `shutdown()`，导致后台线程未正确结束；希望在进程退出时自动执行一次关闭。

---

## 策略一：懒初始化（Lazy Initialization）

### 功能描述

- **触发条件**：任意会使用默认异步执行器（线程池）的 API 被首次调用时，若当前尚未调用过 `initialize(config)`，则自动使用**默认 `ExecutorConfig`** 调用一次 `initialize_async_executor(default_config)`。
- **适用 API**：`submit`、`submit_priority`、`submit_delayed`、`submit_periodic`，以及依赖 `get_default_async_executor()` 的其它路径（如定时器线程中分发延迟/周期任务）。
- **不适用**：已显式调用过 `initialize()` 后，不会再次初始化（现有 `initialize_async_executor` 已规定“已初始化则返回 false”）；实时任务、GPU 执行器注册与提交不触发线程池懒初始化。

### 默认配置

采用 `ExecutorConfig` 的默认值（与 `include/executor/config.hpp` 一致）：

| 字段 | 默认值 |
|------|--------|
| `min_threads` | 4 |
| `max_threads` | 16 |
| `queue_capacity` | 1000 |
| `thread_priority` | 0 |
| `cpu_affinity` | 空 |
| `task_timeout_ms` | 0 |
| `enable_work_stealing` | false |
| `enable_monitoring` | true |

### 线程安全

- 多线程同时首次调用 `submit` 等时，必须保证只执行一次“默认配置初始化”。
- 建议在 `ExecutorManager` 内使用 `std::call_once` + 一个“默认初始化用”的 `std::once_flag`，在 `get_default_async_executor()` 或对外入口处：若当前 `default_async_executor_` 为空，则调用 `call_once(init_once_flag_, [this]{ initialize_async_executor(default_config); })`，再返回 `get_default_async_executor()`。
- 若在 `Executor` Facade 层做懒初始化，需通过 `ExecutorManager` 提供的“确保默认异步执行器存在（必要时用默认配置初始化）”的接口，避免 Facade 与 Manager 各做一次导致重复初始化逻辑。

### 与显式初始化的关系

- 用户先调用 `initialize(custom_config)`：行为与现在一致，使用自定义配置，后续提交使用该配置。
- 用户从未调用 `initialize()`：首次需要线程池时自动用默认配置初始化；若之后用户再调用 `initialize(custom_config)`，现有语义为“已初始化则返回 false”，即**不允许重复初始化**，因此懒初始化后用户无法再通过 `initialize()` 更换配置，这一点需在文档中说明（推荐需要自定义配置时显式先 `initialize()`）。

### 实现位置建议

- **方案 A**：在 `ExecutorManager::get_default_async_executor()` 中，若 `default_async_executor_` 为空则先执行一次“默认配置初始化”（用 `call_once`），再返回指针。这样所有通过 Manager 获取默认执行器的路径（包括 Executor 的 submit、定时器线程）都会自动受益。
- **方案 B**：在 `Executor` 的 `submit` / `submit_priority` / `submit_delayed` / `submit_periodic` 中，在调用 `manager_->get_default_async_executor()` 之前，若得到 nullptr，则先调用 `manager_->ensure_default_async_executor()`（或等价接口）再用默认配置初始化一次，再重试获取。此时初始化逻辑可仍在 Manager 内用 `call_once` 保证只执行一次。

推荐 **方案 A**：逻辑集中在 Manager，Facade 无需改动多处，且定时器线程内部通过 `get_default_async_executor()` 也能自动获得懒初始化效果。

### 文档与 API 说明

- 在 [API.md](../API.md) 的“初始化与关闭”一节中说明：**若不调用 `initialize(config)`，首次提交任务时会使用默认配置自动初始化；**需要自定义线程数、队列容量等时，请在首次提交前显式调用 `initialize(config)`。

---

## 策略二：退出时自动关闭（Exit-time Shutdown）

### 功能描述

- **触发条件**：进程通过 `std::exit()` 或 `main` 正常返回即将退出时，若使用了**单例** `ExecutorManager::instance()` 且尚未调用过 `shutdown()`，则自动执行一次关闭。
- **适用对象**：仅针对**单例模式**（通过 `Executor::instance()` 或直接使用 `ExecutorManager::instance()` 触发的全局 Manager）。实例模式下每个 `Executor` 拥有独立的 `ExecutorManager`，析构时已自动 `shutdown()`，无需 atexit。
- **不适用**：`std::quick_exit()` 不调用 atexit，本策略不保证在 quick_exit 下执行；若需支持可另行考虑 `at_quick_exit`。

### 实现方式

- **注册时机**：在 `ExecutorManager::instance()` 首次被调用（即单例首次创建）时，注册一次 `std::atexit(atexit_shutdown_callback)`。在回调中调用 `ExecutorManager::instance().shutdown(wait_for_tasks)`。
- **wait_for_tasks 策略**：atexit 阶段不宜长时间阻塞进程退出。建议 atexit 中使用 `shutdown(false)`（不等待队列中未完成任务，尽快结束工作线程）。若用户希望“退出前等待未完成任务”，应在业务逻辑中显式在退出前调用 `shutdown(true)`。
- **幂等性**：现有 `ExecutorManager::shutdown()` 已幂等（多次调用安全，执行器清空后再次调用无副作用），atexit 中重复调用无问题。
- **顺序**：atexit 回调在静态对象析构之前执行。执行 atexit 中的 shutdown 后，Manager 内执行器已被清空，后续若有静态析构误用 `Executor::instance()`，将得到已关闭状态（如 `get_default_async_executor()` 为 nullptr）；若此时再触发懒初始化，可视为“atexit 之后不应再提交”的边界情况，可按“已关闭后不再接受新任务”处理或保持懒初始化但文档说明 atexit 后行为未定义。

### 实现位置建议

- 在 `ExecutorManager::instance()` 的 `std::call_once` 初始化块中，在 `new ExecutorManager()` 之后调用 `std::atexit(&ExecutorManager::atexit_shutdown)`（或等价静态成员/自由函数），在 `atexit_shutdown` 中调用 `ExecutorManager::instance().shutdown(false)`。
- 注意：`atexit` 注册的回调不能抛异常，需确保 `shutdown(false)` 在正常使用下不抛，或在回调内 try-catch 吞掉异常，避免 `std::terminate`。

### 文档与 API 说明

- 在 [API.md](../API.md) 的“初始化与关闭”一节中说明：**使用单例时，若未显式调用 `shutdown()`，进程退出时会自动关闭所有执行器；**若需在退出前等待未完成任务完成，请在业务逻辑中显式调用 `shutdown(true)`。

---

## 与现有架构的关系

- **Executor**：Facade 层无需为“兜底”增加大量分支；懒初始化在 Manager 的 `get_default_async_executor()` 或等价入口完成，Facade 仍按“获取执行器 → 提交”流程调用。
- **ExecutorManager**：承担“默认异步执行器是否存在”的检查与懒初始化（策略一）、以及单例创建时注册 atexit（策略二）。
- **实例模式**：每个 `Executor` 持有一个独立的 `ExecutorManager`，该 Manager 不是单例，不会注册 atexit；其析构时照常 `shutdown(true)`，行为与现有一致。

---

## 测试建议

| 场景 | 建议用例 |
|------|----------|
| 懒初始化 | 不调用 `initialize()`，直接 `Executor::instance().submit(...)`，断言任务执行成功且不抛异常 |
| 懒初始化线程安全 | 多线程同时首次调用 `submit`，断言只初始化一次、任务均执行成功 |
| 显式初始化优先 | 先 `initialize(custom_config)` 再 `submit`，断言使用的线程数/配置与 custom_config 一致 |
| 退出时关闭（单例） | 使用单例、不调用 `shutdown()`，在 atexit 或退出前通过测试钩子/ mock 断言 shutdown 被调用；或进程退出后无泄漏（如 valgrind /  sanitizer） |
| 实例模式不注册 atexit | 仅使用实例模式 `Executor ex; ex.submit(...);`，不触发单例，断言 atexit 未注册或进程行为正常 |

---

## 注意事项与边界情况

1. **懒初始化后不再支持“换配置”**：一旦因首次提交触发默认配置初始化，后续再调用 `initialize(other_config)` 会返回 false，当前设计不支持运行时替换线程池配置，需在文档中明确。
2. **atexit 中不等待任务**：默认 atexit 使用 `shutdown(false)`，未完成的任务可能被丢弃；需要“优雅退出”的应用应显式在合适时机调用 `shutdown(true)`。
3. **quick_exit**：若用户调用 `std::quick_exit()`，atexit 不会执行，本方案不保证此时自动关闭；可文档说明或后续扩展 `at_quick_exit`。
4. **静态析构顺序**：若某静态对象析构时仍调用 `Executor::instance().submit(...)`，且此时 atexit 已执行过 shutdown，则懒初始化会再次创建线程池，而进程即将退出，可能造成混乱；建议文档建议“避免在静态析构中使用 Executor”。