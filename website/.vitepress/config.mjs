import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'Executor 使用手册',
  description: '从第一个任务开始构建可靠的 C++ 并发程序。',
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/'
    },
    en: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      title: 'Executor Guide',
      description: 'Build reliable C++ concurrent programs from your first task.',
      themeConfig: {
        siteTitle: 'Executor Guide',
        nav: [
          { text: 'Quick Start', link: '/en/quick-start/build' },
          { text: 'Tutorials', link: '/en/tutorial/' },
          { text: 'Guides', link: '/en/guides/choosing-submit-api' },
          { text: 'API Reference', link: 'https://github.com/Linductor-alkaid/executor/blob/master/docs/API.md' },
          { text: 'Development Snapshot v0.2.3', link: '/en/reference/version-and-migration' },
          {
            text: 'Topics',
            items: [
              { text: 'Reliability', link: '/en/reliability/' },
              { text: 'Real-Time & Communication', link: '/en/realtime-and-communication/' },
              { text: 'GPU', link: '/en/gpu/' },
              { text: 'Advanced', link: '/en/advanced/' },
              { text: 'Versions and Migration', link: '/en/reference/version-and-migration' }
            ]
          },
          { text: 'GitHub', link: 'https://github.com/Linductor-alkaid/executor' }
        ],
        sidebar: {
          '/en/quick-start/': [
            {
              text: 'Quick Start',
              items: [
                { text: 'What is Executor?', link: '/en/getting-started/what-is-executor' },
                { text: 'Build and Install', link: '/en/quick-start/build' },
                { text: 'Your First Task', link: '/en/quick-start/first-task' },
                { text: 'Submit Functions and Data', link: '/en/quick-start/task-inputs-and-ownership' },
                { text: 'Return Values and Errors', link: '/en/quick-start/return-values-and-errors' },
                { text: 'Initialization and Shutdown', link: '/en/quick-start/lifecycle' }
              ]
            }
          ],
          '/en/getting-started/': [
            {
              text: 'Getting Started',
              items: [
                { text: 'What is Executor?', link: '/en/getting-started/what-is-executor' },
                { text: 'Build and Install', link: '/en/quick-start/build' }
              ]
            }
          ],
          '/en/reference/': [
            {
              text: 'Reference',
              items: [
                { text: 'Versions and Migration', link: '/en/reference/version-and-migration' },
                { text: 'Complete API Reference', link: 'https://github.com/Linductor-alkaid/executor/blob/master/docs/API.md' }
              ]
            }
          ],
          '/en/tutorial/': [
            {
              text: 'Tutorials',
              items: [
                { text: 'Robot Data Pipeline', link: '/en/tutorial/' },
                { text: 'Prioritize Control Commands', link: '/en/tutorial/priority' },
                { text: 'Delayed Retry and Health Checks', link: '/en/tutorial/delayed-and-periodic' },
                { text: 'Batch Sensor Frames', link: '/en/tutorial/batch' },
                { text: 'Load, Sense, Then Plan', link: '/en/tutorial/dependencies' },
                { text: 'Bounded Waiting and Status', link: '/en/tutorial/waiting-and-status' },
                { text: 'Complete Robot Pipeline', link: '/en/tutorial/complete-robot-pipeline' },
                { text: 'Service Data Import', link: '/en/tutorial/service-data-import' }
              ]
            }
          ],
          '/en/guides/': [
            {
              text: 'Guides',
              items: [
                { text: 'Choose a Submission API', link: '/en/guides/choosing-submit-api' },
                { text: 'Choose a Communication Component', link: '/en/guides/choosing-communication' },
                { text: 'Migrate Existing Thread Code', link: '/en/guides/migrating-existing-threads' },
                { text: 'Concurrency Architecture Antipatterns', link: '/en/guides/concurrency-antipatterns' },
                { text: 'Production Readiness Checklist', link: '/en/guides/production-readiness' }
              ]
            }
          ],
          '/en/realtime-and-communication/': [
            {
              text: 'Real-Time & Communication',
              items: [
                { text: 'Overview and Boundaries', link: '/en/realtime-and-communication/' },
                { text: 'Dedicated Real-Time Control Loop', link: '/en/realtime-and-communication/realtime-control' },
                { text: 'Deliver Every Message', link: '/en/realtime-and-communication/channels' },
                { text: 'Latest Values, Snapshots, and Phases', link: '/en/realtime-and-communication/state-and-phases' },
                { text: 'Communication Observability', link: '/en/realtime-and-communication/observability' },
                { text: 'Capacity and Alerts', link: '/en/realtime-and-communication/capacity-and-alerting' }
              ]
            }
          ],
          '/en/reliability/': [
            {
              text: 'Reliability',
              items: [
                { text: 'Reliability Overview', link: '/en/reliability/' },
                { text: 'Troubleshoot by Symptom', link: '/en/reliability/troubleshooting' },
                { text: 'Linux and Windows Deployment', link: '/en/reliability/platform-deployment' },
                { text: 'Failure Observability', link: '/en/reliability/failure-observability' },
                { text: 'Monitoring and Sampling', link: '/en/reliability/monitoring' }
              ]
            }
          ],
          '/en/advanced/': [
            {
              text: 'Advanced',
              items: [
                { text: 'Overview and Boundaries', link: '/en/advanced/' },
                { text: 'Source Architecture Map', link: '/en/advanced/source-architecture' },
                { text: 'Advanced Escape Hatches', link: '/en/advanced/escape-hatches' },
                { text: 'Custom Cycle Source', link: '/en/advanced/custom-cycle-manager' },
                { text: 'How Tasks Travel Through Executor', link: '/en/advanced/execution-paths' },
                { text: 'Lock-Free and Performance Experiments', link: '/en/advanced/lockfree-and-performance' },
                { text: 'Performance Measurement and Regression Gates', link: '/en/advanced/performance-measurement' }
              ]
            }
          ],
          '/en/gpu/': [
            {
              text: 'GPU',
              items: [
                { text: 'GPU and Fallback', link: '/en/gpu/' },
                { text: 'Diagnose Backend and Fall Back Safely', link: '/en/gpu/diagnostics' },
                { text: 'Register and Submit GPU Work', link: '/en/gpu/register-and-submit' },
                { text: 'CPU/GPU Automatic Selection', link: '/en/gpu/automatic-scheduling' }
              ]
            }
          ]
        },
        outline: { level: [2, 3], label: 'On this page' },
        docFooter: { prev: 'Previous page', next: 'Next page' },
        footer: {
          message: 'MIT License · <a href="https://github.com/Linductor-alkaid/executor/issues/new/choose">Report a documentation issue</a> · <a href="/executor/en/maintenance">Content maintenance</a>',
          copyright: 'Executor contributors'
        }
      }
    }
  },
  base: '/executor/',
  cleanUrls: true,
  lastUpdated: true,
  markdown: {
    config(md) {
      const defaultFence = md.renderer.rules.fence

      md.renderer.rules.fence = (tokens, index, options, env, self) => {
        const token = tokens[index]
        if (token.info.trim() === 'mermaid') {
          const source = md.utils.escapeHtml(token.content.trim())
          return `<div class="mermaid-diagram" data-mermaid-source="${source.replaceAll('"', '&quot;')}">${source}</div>`
        }
        return defaultFence(tokens, index, options, env, self)
      }
    }
  },
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
      { text: '开发快照 v0.2.3', link: '/zh/reference/version-and-migration' },
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
            { text: '提交自己的函数与数据', link: '/zh/quick-start/task-inputs-and-ownership' },
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
            { text: '如何选择提交接口', link: '/zh/guides/choosing-submit-api' },
            { text: '如何选择通信组件', link: '/zh/guides/choosing-communication' },
            { text: '从现有线程代码迁移', link: '/zh/guides/migrating-existing-threads' },
            { text: '并发架构反模式', link: '/zh/guides/concurrency-antipatterns' },
            { text: '生产接入检查清单', link: '/zh/guides/production-readiness' }
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
            { text: '有界等待与状态快照', link: '/zh/tutorial/waiting-and-status' },
            { text: '完整机器人数据流水线', link: '/zh/tutorial/complete-robot-pipeline' },
            { text: '服务端数据导入案例', link: '/zh/tutorial/service-data-import' }
          ]
        }
      ],
      '/zh/reliability/': [
        {
          text: '可靠性',
          items: [
            { text: '可靠性概览', link: '/zh/reliability/' },
            { text: '按症状排查运行故障', link: '/zh/reliability/troubleshooting' },
            { text: 'Linux 与 Windows 部署核对', link: '/zh/reliability/platform-deployment' },
            { text: '失败可观察性', link: '/zh/reliability/failure-observability' },
            { text: '监控与采样', link: '/zh/reliability/monitoring' }
          ]
        }
      ],
      '/zh/realtime-and-communication/': [
        {
          text: '实时与通信',
          items: [
            { text: '概览与边界', link: '/zh/realtime-and-communication/' },
            { text: '启动专用实时控制循环', link: '/zh/realtime-and-communication/realtime-control' },
            { text: '传递每一条消息', link: '/zh/realtime-and-communication/channels' },
            { text: '传递最新值、快照和阶段', link: '/zh/realtime-and-communication/state-and-phases' },
            { text: '通信可观察性', link: '/zh/realtime-and-communication/observability' },
            { text: '容量判断与告警落地', link: '/zh/realtime-and-communication/capacity-and-alerting' }
          ]
        }
      ],
      '/zh/gpu/': [
        {
          text: 'GPU',
          items: [
            { text: 'GPU 与降级', link: '/zh/gpu/' },
            { text: '诊断后端并安全降级', link: '/zh/gpu/diagnostics' },
            { text: '注册并提交 GPU 工作', link: '/zh/gpu/register-and-submit' },
            { text: 'CPU/GPU 自动选择', link: '/zh/gpu/automatic-scheduling' }
          ]
        }
      ],
      '/zh/advanced/': [
        {
          text: '高级与原理',
          items: [
            { text: '概览与边界', link: '/zh/advanced/' },
            { text: '源码架构与阅读地图', link: '/zh/advanced/source-architecture' },
            { text: '何时使用高级逃生口', link: '/zh/advanced/escape-hatches' },
            { text: '接入自定义周期源', link: '/zh/advanced/custom-cycle-manager' },
            { text: '任务如何穿过执行器', link: '/zh/advanced/execution-paths' },
            { text: '无锁与性能实验', link: '/zh/advanced/lockfree-and-performance' },
            { text: '性能测量与回归门禁', link: '/zh/advanced/performance-measurement' }
          ]
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
      message: 'MIT License · <a href="https://github.com/Linductor-alkaid/executor/issues/new/choose">反馈文档问题</a> · <a href="/executor/maintenance">内容维护</a>',
      copyright: 'Executor contributors'
    }
  }
})
