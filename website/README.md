# Executor 使用手册网站

本目录包含 Executor 的 VitePress 使用手册站点。首发以中文内容为主，路由结构预留 `zh/` 和 `en/` 对称目录。

## 要求

- Node.js 20 LTS 或更新版本
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

## 目录约定

- `.vitepress/`：站点配置、主题和样式。
- `zh/`：中文页面；首发默认语言。
- `en/`：后续英文镜像，必须和中文页面共享示例源文件。
- `decisions.md`：URL、版本和内容事实源的已冻结决策。
