import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'Executor 使用手册',
  description: '从第一个任务开始构建可靠的 C++ 并发程序。',
  base: '/executor/',
  cleanUrls: true,
  lastUpdated: true,
  themeConfig: {
    siteTitle: 'Executor 使用手册',
    nav: [
      { text: '快速开始', link: '/zh/quick-start/build' },
      { text: '场景指南', link: '/zh/guides/choosing-submit-api' },
      { text: '规划', link: '/decisions' },
      { text: 'GitHub', link: 'https://github.com/Linductor-alkaid/executor' }
    ],
    sidebar: {
      '/zh/quick-start/': [
        {
          text: '快速开始',
          items: [
            { text: 'Executor 是什么', link: '/zh/getting-started/what-is-executor' },
            { text: '构建与安装', link: '/zh/quick-start/build' },
            { text: '第一个任务', link: '/zh/quick-start/first-task' },
            { text: '返回值与异常', link: '/zh/quick-start/return-values-and-errors' },
            { text: '初始化与关闭', link: '/zh/quick-start/lifecycle' }
          ]
        }
      ],
      '/zh/getting-started/': [
        {
          text: '开始之前',
          items: [
            { text: 'Executor 是什么', link: '/zh/getting-started/what-is-executor' },
            { text: '构建与安装', link: '/zh/quick-start/build' }
          ]
        }
      ],
      '/zh/guides/': [
        {
          text: '场景指南',
          items: [
            { text: '如何选择提交接口', link: '/zh/guides/choosing-submit-api' }
          ]
        }
      ]
    },
    search: { provider: 'local' },
    outline: { level: [2, 3], label: '本页内容' },
    docFooter: { prev: '上一页', next: '下一页' },
    footer: {
      message: 'MIT License',
      copyright: 'Executor contributors'
    }
  }
})
