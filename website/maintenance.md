---
layout: page
title: 内容维护
---

# 内容维护

网站按责任域维护，而不是为每页复制版本说明。当前核对基线为 `v0.2.3` 源码与 `master` 开发快照；稳定 tag 发布时按[发布文档同步清单](https://github.com/Linductor-alkaid/executor/blob/master/docs/RELEASE_CHECKLIST.md)重新核对。

| 责任域 | 覆盖页面 | 事实源 / 维护者 |
| --- | --- | --- |
| 快速开始与教程 | 首页、`zh/quick-start/`、`zh/tutorial/` | `examples/tutorial/` 与核心维护者。 |
| 可靠性、实时、通信、GPU | `zh/reliability/`、`zh/realtime-and-communication/`、`zh/gpu/` | 对应 Facade 与平台维护者。 |
| 高级与参考 | `zh/advanced/`、`zh/reference/` | API、运行时与发布维护者。 |
| 站点交付 | 配置、主题、导航、404、部署 | 文档维护者。 |

发现失效内容、链接或缺少场景说明时，请在 [GitHub Issues](https://github.com/Linductor-alkaid/executor/issues/new/choose) 提交复现步骤、页面 URL 和期望行为。维护者定期查看搜索词、404、失效链接和高频跳出页面，并将修订纳入后续发布。
