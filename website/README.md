# Executor 使用手册网站

本目录包含 Executor 的 VitePress 使用手册站点。首发以中文内容为主，路由结构预留 `zh/` 和 `en/` 对称目录。

## 要求

- Node.js 24 LTS 或更新版本
- npm（使用提交的 `package-lock.json` 进行严格安装）

## 本地开发

```bash
npm ci --prefix website
npm run docs:dev --prefix website
```

## 构建与预览

```bash
npm run docs:build --prefix website
npm run docs:preview --prefix website
```

静态文件输出到 `website/.vitepress/dist/`。页面通过 VitePress 引用或仓库链接关联完整教程示例，示例仍由根 CMake 工程构建和测试。

## 本地检查

```bash
npm run docs:check --prefix website
```

该检查验证站内路由、相对文件链接及教程嵌入源文件；外部链接不在 PR 阶段访问，以避免网络波动阻塞构建。GitHub Actions 对 PR 执行严格安装、站点构建和教程 smoke tests；仅 `master` 与 `v*` tag 会部署 GitHub Pages。

## 首次启用 GitHub Pages

仓库管理员需要在 GitHub 的 **Settings → Pages → Build and deployment** 中将 **Source** 设为 **GitHub Actions**，保存后再重新运行文档 workflow。这个一次性仓库设置不在 workflow 的权限范围内；未启用时 `actions/deploy-pages` 会在创建部署时返回 404。工作流使用 Node.js 24；`actions/deploy-pages` 自身依赖产生的 `punycode` deprecation warning 不影响 artifact 或部署结果。

## 目录约定

- `.vitepress/`：站点配置、主题和样式。
- `zh/`：中文页面；首发默认语言。
- `en/`：后续英文镜像，必须和中文页面共享示例源文件。
- `decisions.md`：URL、版本和内容事实源的已冻结决策。
