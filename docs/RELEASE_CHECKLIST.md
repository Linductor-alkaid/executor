# 发布文档同步清单

每次发布 tag 前，由发布维护者逐项确认：

- [ ] CMake 项目版本、`CHANGELOG.md` 和 `docs/MIGRATION.md` 已更新并相互一致。
- [ ] `README.md`、`README_zh.md`、`docs/API.md` 与网站版本标识已核对；开发快照能力不会标为既有稳定版能力。
- [ ] 新增或变更的公开 Facade 已通过网站 [API 覆盖索引](../website/zh/reference/api.md) 找到教程、专题、选型或参考入口。
- [ ] 受影响的 `examples/tutorial/` 已构建，并通过 `ctest --test-dir build -L tutorial --output-on-failure`。
- [ ] 已执行 `npm ci --prefix website`、`npm run docs:check --prefix website` 和 `npm run docs:build --prefix website`。
- [ ] 已检查 GitHub Pages 预览：首页、快速开始、API 参考和 404 页面可访问。
- [ ] 已审阅用户反馈、404 和失效链接；需要修复的内容已建立 issue。

稳定版以发布 tag 触发正式发布；`master` 推送部署当前开发快照。若需要变更这一策略，先更新网站版本说明和 Pages workflow，再发布内容。
