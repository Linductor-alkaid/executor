import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'Executor 使用手册',
  description: '从第一个任务开始构建可靠的 C++ 并发程序。',
  base: '/executor/',
  cleanUrls: true,
  lastUpdated: true,
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/executor-mark.svg' }],
    ['meta', { name: 'theme-color', content: '#181d26' }]
  ],
  themeConfig: {
    siteTitle: 'Executor 使用手册',
    nav: [
      { text: '快速开始', link: '/zh/quick-start/build' },
      { text: '循序教程', link: '/zh/tutorial/' },
      { text: '场景指南', link: '/zh/guides/choosing-submit-api' },
      { text: 'API 参考', link: '/zh/reference/api' },
      {
        text: '专题',
        items: [
          { text: '可靠性', link: '/zh/reliability/' },
          { text: '实时与通信', link: '/zh/realtime-and-communication/' },
          { text: 'GPU', link: '/zh/gpu/' },
          { text: '高级与原理', link: '/zh/advanced/' },
          { text: '版本与迁移', link: '/zh/reference/version-and-migration' }
        ]
      },
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
      ],
      '/zh/tutorial/': [
        {
          text: '循序教程',
          items: [
            { text: '机器人数据流水线', link: '/zh/tutorial/' },
            { text: '让控制命令优先', link: '/zh/tutorial/priority' },
            { text: '延迟重试与健康检查', link: '/zh/tutorial/delayed-and-periodic' },
            { text: '批量处理传感器帧', link: '/zh/tutorial/batch' },
            { text: '加载、感知与规划依赖', link: '/zh/tutorial/dependencies' },
            { text: '有界等待与状态快照', link: '/zh/tutorial/waiting-and-status' }
          ]
        }
      ],
      '/zh/reliability/': [
        {
          text: '可靠性',
          items: [{ text: '失败可观察性', link: '/zh/reliability/' }]
        }
      ],
      '/zh/realtime-and-communication/': [
        {
          text: '实时与通信',
          items: [{ text: '组件选择', link: '/zh/realtime-and-communication/' }]
        }
      ],
      '/zh/gpu/': [
        {
          text: 'GPU',
          items: [{ text: 'GPU 与降级', link: '/zh/gpu/' }]
        }
      ],
      '/zh/advanced/': [
        {
          text: '高级与原理',
          items: [{ text: 'Facade 与逃生口', link: '/zh/advanced/' }]
        }
      ],
      '/zh/reference/': [
        {
          text: '参考',
          items: [
            { text: 'API 参考', link: '/zh/reference/api' },
            { text: '版本与迁移', link: '/zh/reference/version-and-migration' }
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
