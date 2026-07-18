import DefaultTheme from 'vitepress/theme'
import { computed, h, nextTick, onMounted, watch } from 'vue'
import { useRoute, withBase } from 'vitepress'
import './custom.css'

const zoomStep = 0.25
const minZoom = 0.5
const maxZoom = 3

const translatedPaths = new Set([
  '/en/',
  '/en/getting-started/what-is-executor',
  '/en/quick-start/build',
  '/en/quick-start/first-task',
  '/en/quick-start/task-inputs-and-ownership',
  '/en/quick-start/return-values-and-errors',
  '/en/quick-start/lifecycle',
  '/en/tutorial/',
  '/en/tutorial/priority',
  '/en/tutorial/delayed-and-periodic',
  '/en/tutorial/batch',
  '/en/tutorial/dependencies',
  '/en/tutorial/waiting-and-status',
  '/en/tutorial/complete-robot-pipeline',
  '/en/tutorial/service-data-import',
  '/en/guides/choosing-submit-api',
  '/en/guides/choosing-communication',
  '/en/guides/migrating-existing-threads',
  '/en/guides/concurrency-antipatterns',
  '/en/guides/production-readiness',
  '/en/realtime-and-communication/',
  '/en/realtime-and-communication/realtime-control',
  '/en/realtime-and-communication/channels',
  '/en/realtime-and-communication/state-and-phases',
  '/en/realtime-and-communication/observability',
  '/en/realtime-and-communication/capacity-and-alerting',
  '/en/reliability/',
  '/en/reliability/troubleshooting',
  '/en/reliability/platform-deployment',
  '/en/reliability/failure-observability',
  '/en/reliability/monitoring',
  '/en/advanced/',
  '/en/advanced/source-architecture',
  '/en/advanced/escape-hatches',
  '/en/advanced/custom-cycle-manager',
  '/en/advanced/execution-paths',
  '/en/advanced/lockfree-and-performance',
  '/en/advanced/performance-measurement',
  '/en/gpu/',
  '/en/gpu/diagnostics',
  '/en/gpu/register-and-submit',
  '/en/gpu/automatic-scheduling',
  '/en/reference/version-and-migration'
])

const LanguageSwitch = {
  setup() {
    const route = useRoute()
    const target = computed(() => {
      const path = route.path.replace(/\.html$/, '')
      if (path === '/en/') return '/'
      if (path.startsWith('/en/')) return path.replace(/^\/en/, '/zh')
      const englishPath = path === '/' ? '/en/' : path.replace(/^\/zh/, '/en')
      return translatedPaths.has(englishPath) ? englishPath : '/en/'
    })
    const label = computed(() => route.path.startsWith('/en/') ? '简体中文' : 'English')
    const fallback = computed(() => !route.path.startsWith('/en/') && target.value === '/en/')

    return () => h('a', {
      class: 'language-switch',
      href: withBase(target.value),
      title: fallback.value ? 'This page is not yet available in English' : undefined
    }, fallback.value ? `${label.value} · 首页` : label.value)
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
  Layout
}
