---
title: 首发决策记录
---

# 首发决策记录

| 事项 | 决策 | 理由 |
| --- | --- | --- |
| 站点名称 | Executor 使用手册 | 与仓库名称一致，清楚表明教学用途。 |
| 受众承诺 | 十分钟内完成第一个可观察的异步任务 | MVP 的最短成功路径。 |
| 源码目录 | `website/` | 与 C++ 源码、示例和参考文档隔离。 |
| 工具链 | VitePress 1.6.3、Node.js 20 LTS、npm | 静态生成、内置搜索和低维护成本。 |
| URL | GitHub Pages 项目站点，`/executor/` | 与当前仓库 `Linductor-alkaid/executor` 对应。 |
| 语言策略 | 中文默认；`zh/`、`en/` 对称目录 | 首发不受翻译阻塞，同时避免未来迁移路由。 |
| 版本策略 | 当前开发版本 `v0.2.3-dev` | 页面以当前仓库 API 为准；稳定版发布后再增加历史版本站点。 |
| 图表策略 | 首发使用 Markdown 文本图 | 避免 Mermaid 额外依赖和构建风险。 |
| 代码事实源 | `examples/tutorial/` | 网页核心代码引用可编译的完整示例，避免双重事实源。 |

## 首批页面追踪

| 页面 | 事实源 | 完整示例 | 状态 |
| --- | --- | --- | --- |
| 首页 | `README_zh.md`、公开 Facade | `examples/tutorial/01_first_task.cpp` | 已发布 MVP |
| Executor 是什么 | `docs/design/user_guide_website.md` | — | 已发布 MVP |
| 安装与构建 | `docs/BUILD.md`、根 `CMakeLists.txt` | `01_first_task.cpp` | 已发布 MVP |
| 第一个任务 | `include/executor/executor.hpp` | `01_first_task.cpp` | 已发布 MVP |
| 返回值与异常 | `Executor::submit()`、future | `01_first_task.cpp` | 已发布 MVP |
| 初始化与关闭 | `Executor::initialize_ex()`、`shutdown()` | `01_first_task.cpp` | 已发布 MVP |

后续页面在加入导航前必须补充事实源、示例和验收状态。
