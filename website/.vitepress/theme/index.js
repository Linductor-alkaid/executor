import DefaultTheme from 'vitepress/theme'
import { computed, h, nextTick, onMounted, watch } from 'vue'
import { useRoute, withBase } from 'vitepress'
import './custom.css'

const zoomStep = 0.25
const minZoom = 0.5
const maxZoom = 3

const localizedRoutes = new Map([
  ['/', '/en/'],
  ['/decisions', '/en/decisions'],
  ['/maintenance', '/en/maintenance'],
  ['/zh/getting-started/what-is-executor', '/en/getting-started/what-is-executor'],
  ['/zh/quick-start/build', '/en/quick-start/build'],
  ['/zh/quick-start/first-task', '/en/quick-start/first-task'],
  ['/zh/quick-start/task-inputs-and-ownership', '/en/quick-start/task-inputs-and-ownership'],
  ['/zh/quick-start/return-values-and-errors', '/en/quick-start/return-values-and-errors'],
  ['/zh/quick-start/lifecycle', '/en/quick-start/lifecycle'],
  ['/zh/tutorial/', '/en/tutorial/'],
  ['/zh/tutorial/priority', '/en/tutorial/priority'],
  ['/zh/tutorial/delayed-and-periodic', '/en/tutorial/delayed-and-periodic'],
  ['/zh/tutorial/batch', '/en/tutorial/batch'],
  ['/zh/tutorial/dependencies', '/en/tutorial/dependencies'],
  ['/zh/tutorial/waiting-and-status', '/en/tutorial/waiting-and-status'],
  ['/zh/tutorial/complete-robot-pipeline', '/en/tutorial/complete-robot-pipeline'],
  ['/zh/tutorial/service-data-import', '/en/tutorial/service-data-import'],
  ['/zh/guides/choosing-submit-api', '/en/guides/choosing-submit-api'],
  ['/zh/guides/choosing-communication', '/en/guides/choosing-communication'],
  ['/zh/guides/migrating-existing-threads', '/en/guides/migrating-existing-threads'],
  ['/zh/guides/concurrency-antipatterns', '/en/guides/concurrency-antipatterns'],
  ['/zh/guides/production-readiness', '/en/guides/production-readiness'],
  ['/zh/realtime-and-communication/', '/en/realtime-and-communication/'],
  ['/zh/realtime-and-communication/realtime-control', '/en/realtime-and-communication/realtime-control'],
  ['/zh/realtime-and-communication/channels', '/en/realtime-and-communication/channels'],
  ['/zh/realtime-and-communication/state-and-phases', '/en/realtime-and-communication/state-and-phases'],
  ['/zh/realtime-and-communication/observability', '/en/realtime-and-communication/observability'],
  ['/zh/realtime-and-communication/capacity-and-alerting', '/en/realtime-and-communication/capacity-and-alerting'],
  ['/zh/reliability/', '/en/reliability/'],
  ['/zh/reliability/troubleshooting', '/en/reliability/troubleshooting'],
  ['/zh/reliability/platform-deployment', '/en/reliability/platform-deployment'],
  ['/zh/reliability/failure-observability', '/en/reliability/failure-observability'],
  ['/zh/reliability/monitoring', '/en/reliability/monitoring'],
  ['/zh/advanced/', '/en/advanced/'],
  ['/zh/advanced/source-architecture', '/en/advanced/source-architecture'],
  ['/zh/advanced/escape-hatches', '/en/advanced/escape-hatches'],
  ['/zh/advanced/custom-cycle-manager', '/en/advanced/custom-cycle-manager'],
  ['/zh/advanced/execution-paths', '/en/advanced/execution-paths'],
  ['/zh/advanced/lockfree-and-performance', '/en/advanced/lockfree-and-performance'],
  ['/zh/advanced/performance-measurement', '/en/advanced/performance-measurement'],
  ['/zh/gpu/', '/en/gpu/'],
  ['/zh/gpu/diagnostics', '/en/gpu/diagnostics'],
  ['/zh/gpu/register-and-submit', '/en/gpu/register-and-submit'],
  ['/zh/gpu/automatic-scheduling', '/en/gpu/automatic-scheduling'],
  ['/zh/reference/version-and-migration', '/en/reference/version-and-migration']
])

const englishToChineseRoutes = new Map(
  [...localizedRoutes].map(([chinesePath, englishPath]) => [englishPath, chinesePath])
)

function normalizeRoutePath(path) {
  const cleanPath = path.replace(/\.html$/, '').replace(/^\/executor(?=\/|$)/, '') || '/'
  const localePath = cleanPath.replace(/^\/en\/zh(?=\/|$)/, '/zh')
  return localePath.endsWith('/') || localePath === '/' ? localePath : localePath.replace(/\/$/, '')
}

function findLocalizedRoute(routes, path) {
  return routes.get(path) ?? (path === '/' ? undefined : routes.get(`${path}/`))
}

const LanguageSwitch = {
  setup() {
    const route = useRoute()
    const isEnglish = computed(() => normalizeRoutePath(route.path).startsWith('/en/'))
    const target = computed(() => {
      const path = normalizeRoutePath(route.path)
      if (isEnglish.value) return findLocalizedRoute(englishToChineseRoutes, path) ?? '/'
      return findLocalizedRoute(localizedRoutes, path) ?? '/en/'
    })
    const label = computed(() => isEnglish.value ? '简体中文' : 'English')
    const fallback = computed(() => !isEnglish.value && target.value === '/en/')

    return () => h('a', {
      class: 'language-switch',
      href: withBase(target.value),
      title: fallback.value ? 'This page is not yet available in English' : undefined
    }, fallback.value ? `${label.value} · Home` : label.value)
  }
}

const NotFound = {
  setup() {
    const route = useRoute()
    const isEnglish = computed(() => normalizeRoutePath(route.path).startsWith('/en/'))

    return () => {
      const english = isEnglish.value
      return h('main', { class: 'language-not-found VPContent' }, [
        h('div', { class: 'container' }, [
          h('div', { class: 'content' }, [
            h('h1', english ? 'Page not found' : '页面未找到'),
            h('p', english
              ? 'This link may have moved, or the requested page is not published.'
              : '该链接可能已经移动，或对应内容尚未发布。'),
            h('ul', english
              ? [
                  h('li', [h('a', { href: withBase('/en/') }, 'Return to the English home page')]),
                  h('li', [h('a', { href: withBase('/en/quick-start/first-task') }, 'Start the ten-minute quick start')]),
                  h('li', [h('a', { href: withBase('/en/reference/version-and-migration') }, 'Check versions and migration')])
                ]
              : [
                  h('li', [h('a', { href: withBase('/') }, '从首页重新开始')]),
                  h('li', [h('a', { href: withBase('/zh/quick-start/first-task') }, '进入十分钟快速开始')]),
                  h('li', [h('a', { href: withBase('/zh/reference/version-and-migration') }, '查看版本与迁移')])
                ])
          ])
        ])
      ])
    }
  }
}

function createDiagramViewer() {
  const viewer = document.createElement('div')
  viewer.className = 'mermaid-viewer'
  viewer.hidden = true
  viewer.innerHTML = `
    <div class="mermaid-viewer__panel" role="dialog" aria-modal="true" aria-label="流程图放大查看">
      <div class="mermaid-viewer__toolbar">
        <span class="mermaid-viewer__title">流程图</span>
        <div class="mermaid-viewer__actions">
          <button type="button" data-action="zoom-out" aria-label="缩小流程图">−</button>
          <output aria-live="polite">100%</output>
          <button type="button" data-action="zoom-in" aria-label="放大流程图">＋</button>
          <button type="button" data-action="reset">重置</button>
          <button type="button" data-action="close" aria-label="关闭流程图查看器">关闭</button>
        </div>
      </div>
      <div class="mermaid-viewer__canvas"></div>
    </div>`
  document.body.appendChild(viewer)

  const canvas = viewer.querySelector('.mermaid-viewer__canvas')
  const output = viewer.querySelector('output')
  const closeButton = viewer.querySelector('[data-action="close"]')
  let zoom = 1
  let trigger = null

  const applyZoom = () => {
    const svg = canvas.querySelector('svg')
    if (svg) {
      const viewBox = svg.viewBox.baseVal
      svg.style.width = `${viewBox.width * zoom}px`
      svg.style.height = `${viewBox.height * zoom}px`
    }
    output.value = `${Math.round(zoom * 100)}%`
  }

  const close = () => {
    viewer.hidden = true
    canvas.replaceChildren()
    document.body.classList.remove('mermaid-viewer-open')
    trigger?.focus()
    trigger = null
  }

  viewer.addEventListener('click', (event) => {
    if (event.target === viewer) close()
    const action = event.target.closest('button')?.dataset.action
    if (action === 'zoom-in') zoom = Math.min(maxZoom, zoom + zoomStep)
    if (action === 'zoom-out') zoom = Math.max(minZoom, zoom - zoomStep)
    if (action === 'reset') zoom = 1
    if (action === 'close') return close()
    if (action) applyZoom()
  })
  document.addEventListener('keydown', (event) => {
    if (viewer.hidden) return
    if (event.key === 'Escape') close()
    if (event.key === '+' || event.key === '=') {
      zoom = Math.min(maxZoom, zoom + zoomStep)
      applyZoom()
    }
    if (event.key === '-') {
      zoom = Math.max(minZoom, zoom - zoomStep)
      applyZoom()
    }
  })

  return {
    open(diagram, button) {
      const svg = diagram.querySelector('svg')
      if (!svg) return
      trigger = button
      zoom = 1
      canvas.replaceChildren(svg.cloneNode(true))
      viewer.hidden = false
      document.body.classList.add('mermaid-viewer-open')
      applyZoom()
      closeButton.focus()
    }
  }
}

function addDiagramControls(diagram) {
  if (diagram.querySelector('.mermaid-diagram__expand')) return
  const button = document.createElement('button')
  button.type = 'button'
  button.className = 'mermaid-diagram__expand'
  button.setAttribute('aria-label', '放大查看流程图')
  button.textContent = '放大查看'
  button.addEventListener('click', () => {
    window.executorMermaidViewer ??= createDiagramViewer()
    window.executorMermaidViewer.open(diagram, button)
  })
  diagram.appendChild(button)
}

async function renderMermaidDiagrams() {
  if (typeof window === 'undefined') return

  const diagrams = [...document.querySelectorAll('.mermaid-diagram:not([data-mermaid-rendered])')]
  if (diagrams.length === 0) return

  const { default: mermaid } = await import('mermaid')
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',
    theme: document.documentElement.classList.contains('dark') ? 'dark' : 'base'
  })

  for (const diagram of diagrams) {
    const source = diagram.getAttribute('data-mermaid-source') || ''
    try {
      const id = `executor-mermaid-${Math.random().toString(36).slice(2)}`
      const { svg, bindFunctions } = await mermaid.render(id, source)
      diagram.innerHTML = svg
      diagram.setAttribute('data-mermaid-rendered', 'true')
      bindFunctions?.(diagram)
      addDiagramControls(diagram)
    } catch (error) {
      diagram.setAttribute('data-mermaid-error', 'true')
      console.error('Failed to render Mermaid diagram', error)
    }
  }
}

const Layout = {
  setup() {
    const route = useRoute()
    onMounted(() => renderMermaidDiagrams())
    watch(() => route.path, () => nextTick(() => renderMermaidDiagrams()))
    return () => h(DefaultTheme.Layout, null, {
      'nav-bar-content-after': () => h(LanguageSwitch)
    })
  }
}

export default {
  ...DefaultTheme,
  Layout,
  NotFound
}
