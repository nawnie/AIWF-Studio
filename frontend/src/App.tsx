import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type {
  CSSProperties,
  ChangeEvent as ReactChangeEvent,
  KeyboardEvent as ReactKeyboardEvent,
  MouseEvent as ReactMouseEvent,
  ReactNode,
} from 'react'
import {
  Boxes,
  Brush,
  CircleHelp,
  Database,
  Cpu,
  FileImage,
  Hand,
  HardDrive,
  Highlighter,
  Image,
  Layers2,
  Maximize2,
  Monitor,
  PanelLeft,
  Rows3,
  ScanSearch,
  RefreshCcw,
  Settings,
  SlidersHorizontal,
  Sparkles,
  Video,
  Wand2,
  X,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import {
  fetchProData,
  fetchProBootstrap,
  fetchProCapabilities,
  fetchProDownloads,
  fetchProLogs,
  fetchProRuntime,
  fetchProSettings,
  formatApiError,
  generateProOutput,
  getFallbackBootstrap,
  getFallbackRuntime,
  ProApiError,
  reportProClientError,
  reportProClientEvent,
  requestProRestart,
  streamProRuntime,
  stopProGeneration,
} from './api'
import type {
  AspectRatioOption,
  CreationMode,
  EngineId,
  EngineSummary,
  GenerationSettings,
  GenerationProgressEvent,
  ProCapabilitiesStatus,
  ProDataStatus,
  ProDownloadsStatus,
  ProLogStatus,
  ProModelOption,
  ProBootstrap,
  ProMode,
  ProReadinessItem,
  ProReadinessStatus,
  PromptInsight,
  ProRuntimeStatus,
  ProSettingsStatus,
  ResourceMetric,
  RecentOutput,
} from './types'
import './styles.css'

interface IconItem<T extends string> {
  id: T
  label: string
  icon: LucideIcon
}

type ToolModalId = 'segmentation' | 'hires' | 'reactor' | 'about' | null
type MenuBarId = 'file' | 'edit' | 'view' | 'options' | 'help' | null
type DragTarget = 'left' | 'right' | 'bottom'

interface DragState {
  target: DragTarget
  origin: number
  size: number
}

interface ToolLaneAction {
  label: string
  onClick?: () => void
  disabled?: boolean
}

interface ToolLaneCard {
  id: string
  title: string
  summary: string
  stats: string[]
  actions: ToolLaneAction[]
  note?: string
}

interface LayoutPreferences {
  leftPanelWidth: number
  rightPanelWidth: number
  bottomDockHeight: number
  bottomDockVisible: boolean
}

const DEFAULT_LAYOUT_PREFERENCES: LayoutPreferences = {
  leftPanelWidth: 380,
  rightPanelWidth: 320,
  bottomDockHeight: 196,
  bottomDockVisible: true,
}

const PRO_APP_ICON = '/app-icon.png'

const LAYOUT_STORAGE_KEY = 'aiwf.pro.layout.v1'

const MODE_TABS: IconItem<ProMode>[] = [
  { id: 'image', label: 'Image', icon: Image },
  { id: 'video', label: 'Video', icon: Video },
  { id: 'inpaint', label: 'Inpaint', icon: Brush },
  { id: 'models', label: 'Models', icon: Boxes },
  { id: 'data', label: 'Data', icon: Database },
]

const RAIL_ITEMS: IconItem<string>[] = [
  { id: 'create', label: 'Create', icon: Sparkles },
  { id: 'models', label: 'Models', icon: Boxes },
  { id: 'tools', label: 'Tools', icon: Wand2 },
  { id: 'data', label: 'Data', icon: Database },
  { id: 'monitor', label: 'Monitor', icon: Monitor },
  { id: 'logs', label: 'Logs', icon: FileImage },
  { id: 'settings', label: 'Settings', icon: Settings },
]

const RAIL_IDS = new Set(RAIL_ITEMS.map((item) => item.id))

const SANA_QUANTIZATION_OPTIONS = [
  { value: 'auto', label: 'Auto' },
  { value: 'fp8_layerwise', label: 'FP8 layerwise' },
  { value: 'bnb_int8', label: 'BNB 8-bit' },
  { value: 'bnb_nf4', label: 'BNB NF4' },
  { value: 'bnb_fp4', label: 'BNB FP4' },
  { value: 'bf16', label: 'BF16' },
]

const SANA_VAE_TILING_OPTIONS = [
  { value: 'auto', label: 'Auto' },
  { value: 'off', label: 'Off' },
  { value: 'always', label: 'Always' },
]

const EMPTY_READINESS: ProReadinessStatus = {
  counts: {
    working: 0,
    'metadata-only': 0,
    'blocked-cleanly': 0,
    'broken-runtime': 0,
    'unsupported-no-route': 0,
  },
  families: [],
  working: [],
  needsWork: [],
  metadataOnlyCount: 0,
  total: 0,
  error: '',
}

const EMPTY_CAPABILITIES: ProCapabilitiesStatus = {
  gradioTabs: [],
  tools: [],
  counts: {
    gradioTabs: 0,
    reactRails: RAIL_ITEMS.length,
    checkpoints: 0,
    blockedCheckpoints: 0,
    loras: 0,
    controlnet: 0,
    sam: 0,
    reactor: 0,
    enhance: 0,
    sanaVideo: 0,
    wan: 0,
  },
  readiness: EMPTY_READINESS,
  notes: ['Capabilities are loading.'],
}

function App() {
  const fallbackBootstrap = useMemo(() => getFallbackBootstrap(), [])
  const fallbackRuntime = useMemo(() => getFallbackRuntime(), [])
  const initialLayout = useMemo(() => readLayoutPreferences(), [])
  const [bootstrap, setBootstrap] = useState<ProBootstrap>(fallbackBootstrap)
  const [runtime, setRuntime] = useState<ProRuntimeStatus>(fallbackRuntime)
  const [runtimeStreamConnected, setRuntimeStreamConnected] = useState(false)
  const [settings, setSettings] = useState<GenerationSettings>(fallbackBootstrap.defaults)
  const [dataStatus, setDataStatus] = useState<ProDataStatus | null>(null)
  const [downloadsStatus, setDownloadsStatus] = useState<ProDownloadsStatus | null>(null)
  const [capabilitiesStatus, setCapabilitiesStatus] = useState<ProCapabilitiesStatus | null>(null)
  const [logStatus, setLogStatus] = useState<ProLogStatus | null>(null)
  const [settingsStatus, setSettingsStatus] = useState<ProSettingsStatus | null>(null)
  const [promptInsight, setPromptInsight] = useState<PromptInsight>({
    status: 'idle',
    summary: 'Run browser-side analysis to check prompt structure before generating.',
    modelLabel: 'Not run',
    modelScore: 0,
    modelId: 'Transformers.js lazy load',
    progress: 0,
    signals: [],
    suggestions: ['Use this for lightweight prompt help; it does not start backend inference.'],
  })
  const [promptInsightBusy, setPromptInsightBusy] = useState(false)
  const [generationProgress, setGenerationProgress] = useState<GenerationProgressEvent[]>([])
  const [generationTimings, setGenerationTimings] = useState<Record<string, number>>({})
  const [generationReceiptPath, setGenerationReceiptPath] = useState('')
  const [generationError, setGenerationError] = useState('')
  const [activeMode, setActiveMode] = useState<ProMode>('image')
  const [activeRail, setActiveRail] = useState(readInitialRail)
  const [preview, setPreview] = useState<RecentOutput | null>(
    fallbackBootstrap.recentOutputs[0] ?? null,
  )
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [statusMessage, setStatusMessage] = useState('Ready.')
  const [isGenerating, setIsGenerating] = useState(false)
  const [backendConnected, setBackendConnected] = useState(false)
  const [backendRecovering, setBackendRecovering] = useState(false)
  const [engineFilter, setEngineFilter] = useState<EngineId>('all')
  const [leftPanelWidth, setLeftPanelWidth] = useState(initialLayout.leftPanelWidth)
  const [rightPanelWidth, setRightPanelWidth] = useState(initialLayout.rightPanelWidth)
  const [bottomDockVisible, setBottomDockVisible] = useState(initialLayout.bottomDockVisible)
  const [bottomDockHeight, setBottomDockHeight] = useState(initialLayout.bottomDockHeight)
  const [activeModal, setActiveModal] = useState<ToolModalId>(null)
  const [openMenu, setOpenMenu] = useState<MenuBarId>(null)
  const [dragState, setDragState] = useState<DragState | null>(null)
  const [hiresEnabled, setHiresEnabled] = useState(false)
  const [hiresScale, setHiresScale] = useState(1.75)
  const [hiresDenoise, setHiresDenoise] = useState(0.3)
  const [segmentationMode, setSegmentationMode] = useState('Auto mask')
  const [reactorEnabled, setReactorEnabled] = useState(false)
  const [reactorStrength, setReactorStrength] = useState(0.8)
  const generationAbortRef = useRef<AbortController | null>(null)
  const runtimeErrorLoggedRef = useRef(false)
  const auxiliaryErrorLoggedRef = useRef<Record<string, boolean>>({})
  const runtimeJobActive = isRuntimeJobActive(runtime.job)
  const generationActive = isGenerating || runtime.state.toLowerCase() === 'running' || runtimeJobActive

  const setDisconnectedRuntime = useCallback((message: string) => {
    setRuntime({
      ...fallbackRuntime,
      state: backendRecovering ? 'Recovering' : 'Disconnected',
      backend: 'Backend unreachable',
      device: 'Waiting for runtime',
      job: { ...fallbackRuntime.job, message },
      resources: fallbackRuntime.resources.map((metric) => ({ ...metric })),
      loadedModel: { ...fallbackRuntime.loadedModel },
    })
  }, [backendRecovering, fallbackRuntime])

  const markBackendDisconnected = useCallback((message: string) => {
    setBackendConnected(false)
    setRuntimeStreamConnected(false)
    setDisconnectedRuntime(message)
    setCapabilitiesStatus(null)
    setDataStatus(null)
    setDownloadsStatus(null)
    setLogStatus(null)
    setSettingsStatus(null)
    setStatusMessage(message)
  }, [setDisconnectedRuntime])

  useEffect(() => {
    const controller = new AbortController()
    fetchProBootstrap(controller.signal)
      .then((nextBootstrap) => {
        setBootstrap(nextBootstrap)
        setPreview((currentPreview) => currentPreview ?? nextBootstrap.recentOutputs[0] ?? null)
        setSettings((current) =>
          settingsMatch(current, fallbackBootstrap.defaults)
            ? nextBootstrap.defaults
            : mergeBootstrapDefaults(current, nextBootstrap),
        )
        setBackendConnected(true)
        setBackendRecovering(false)
        setStatusMessage('Connected to /api/pro/bootstrap.')
      })
      .catch((error: unknown) => {
        if (isAbortError(error)) {
          return
        }
        reportProClientError({
          kind: 'api',
          message: formatApiError(error),
          source: 'bootstrap',
          context: { route: '/api/pro/bootstrap' },
        })
        setStatusMessage('Using the local workspace view while the backend finishes starting.')
      })

    return () => controller.abort()
  }, [fallbackBootstrap.defaults])

  useEffect(() => {
    return streamProRuntime(
      (nextRuntime) => {
        setRuntime(nextRuntime)
        setBackendConnected(true)
        if (backendRecovering) {
          setBackendRecovering(false)
          setStatusMessage('Backend reconnected.')
        }
      },
      (connected) => {
        setRuntimeStreamConnected(connected)
      },
    )
  }, [backendRecovering])

  useEffect(() => {
    let disposed = false
    let requestController: AbortController | null = null
    let requestTimeoutId: number | null = null
    let inFlight = false
    const intervalMs = generationActive ? 1000 : runtimeStreamConnected ? 7000 : 5000
    const refreshRuntime = () => {
      if (disposed || inFlight) {
        return
      }
      inFlight = true
      const activeController = new AbortController()
      let timedOut = false
      requestController = activeController
      requestTimeoutId = window.setTimeout(() => {
        timedOut = true
        activeController.abort()
      }, 4000)
      fetchProRuntime(activeController.signal)
        .then((nextRuntime) => {
          if (!disposed) {
            runtimeErrorLoggedRef.current = false
            setBackendConnected(true)
            setBackendRecovering(false)
            setRuntime(nextRuntime)
          }
        })
        .catch((error: unknown) => {
          if (!disposed && (timedOut || !isAbortError(error))) {
            if (!runtimeErrorLoggedRef.current) {
              runtimeErrorLoggedRef.current = true
              reportProClientError({
                kind: 'api',
                message: timedOut ? 'Runtime refresh timed out after 4 seconds.' : formatApiError(error),
                source: 'runtime-refresh',
                context: { route: '/api/pro/runtime', timedOut },
              })
            }
            markBackendDisconnected(
              timedOut
                ? 'Backend communication timed out. Pro is holding your place and waiting to reconnect.'
                : 'Backend communication broke. Pro is waiting for the local runtime to come back.',
            )
          }
        })
        .finally(() => {
          if (requestTimeoutId !== null) {
            window.clearTimeout(requestTimeoutId)
            requestTimeoutId = null
          }
          inFlight = false
          if (requestController === activeController) {
            requestController = null
          }
        })
    }

    refreshRuntime()
    const intervalId = window.setInterval(refreshRuntime, intervalMs)
    return () => {
      disposed = true
      requestController?.abort()
      if (requestTimeoutId !== null) {
        window.clearTimeout(requestTimeoutId)
      }
      window.clearInterval(intervalId)
    }
  }, [generationActive, markBackendDisconnected, runtimeStreamConnected])

  useEffect(() => {
    const onError = (event: ErrorEvent) => {
      reportProClientError({
        kind: 'window-error',
        message: event.message || 'Unhandled browser error',
        stack: event.error instanceof Error ? event.error.stack : undefined,
        source: event.filename ? `${event.filename}:${event.lineno}:${event.colno}` : 'window',
      })
    }
    const onUnhandledRejection = (event: PromiseRejectionEvent) => {
      reportProClientError({
        kind: 'unhandledrejection',
        message: formatApiError(event.reason),
        stack: event.reason instanceof Error ? event.reason.stack : undefined,
        source: 'promise',
      })
    }
    window.addEventListener('error', onError)
    window.addEventListener('unhandledrejection', onUnhandledRejection)
    return () => {
      window.removeEventListener('error', onError)
      window.removeEventListener('unhandledrejection', onUnhandledRejection)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    const runWorkspaceFetch = <T,>(
      key: string,
      fetcher: (signal?: AbortSignal) => Promise<T>,
      setter: (value: T | null) => void,
      timeoutMs: number,
      route: string,
      emptyMessage: string,
    ) => {
      const requestController = new AbortController()
      const timeoutId = window.setTimeout(() => requestController.abort(), timeoutMs)
      const abortRelay = () => requestController.abort()
      controller.signal.addEventListener('abort', abortRelay, { once: true })
      void fetcher(requestController.signal)
        .then((value) => {
          auxiliaryErrorLoggedRef.current[key] = false
          setter(value)
        })
        .catch((error: unknown) => {
          if (!isAbortError(error)) {
            setter(null)
          } else if (!controller.signal.aborted) {
            setter(null)
          }
          if (controller.signal.aborted) {
            return
          }
          if (!auxiliaryErrorLoggedRef.current[key]) {
            auxiliaryErrorLoggedRef.current[key] = true
            reportProClientError({
              kind: 'api',
              message: isAbortError(error) ? `${route} timed out after ${timeoutMs} ms.` : formatApiError(error),
              source: key,
              context: { route, timeoutMs },
            })
          }
          setStatusMessage(emptyMessage)
        })
        .finally(() => {
          window.clearTimeout(timeoutId)
          controller.signal.removeEventListener('abort', abortRelay)
        })
    }
    const refreshWorkspaceData = () => {
      runWorkspaceFetch('data', fetchProData, setDataStatus, 6000, '/api/pro/data', 'Workspace data is temporarily unavailable.')
      runWorkspaceFetch(
        'downloads',
        fetchProDownloads,
        setDownloadsStatus,
        6000,
        '/api/pro/downloads',
        'Download catalog refresh failed. Pro is keeping the current session alive.',
      )
      runWorkspaceFetch(
        'capabilities',
        fetchProCapabilities,
        setCapabilitiesStatus,
        12000,
        '/api/pro/capabilities',
        'Capability scan is taking too long. Pro will retry without blocking the rest of the workspace.',
      )
      runWorkspaceFetch('logs', fetchProLogs, setLogStatus, 6000, '/api/pro/logs', 'Runtime logs are temporarily unavailable.')
      runWorkspaceFetch(
        'settings',
        fetchProSettings,
        setSettingsStatus,
        6000,
        '/api/pro/settings',
        'Settings data is temporarily unavailable until the backend replies again.',
      )
    }

    refreshWorkspaceData()
    const intervalId = window.setInterval(refreshWorkspaceData, 15000)
    return () => {
      controller.abort()
      window.clearInterval(intervalId)
    }
  }, [])

  const handleRecoverBackend = useCallback(async () => {
    setBackendRecovering(true)
    setStatusMessage('Restart requested. Waiting for the Pro backend to come back on the same port.')
    markBackendDisconnected('Restart requested. Waiting for the Pro backend to come back on the same port.')
    try {
      await requestProRestart()
    } catch (error) {
      setBackendRecovering(false)
      setStatusMessage(`Backend restart request failed: ${formatApiError(error)}`)
      reportProClientError({
        kind: 'api',
        message: formatApiError(error),
        source: 'restart',
        context: { route: '/api/pro/restart' },
      })
    }
  }, [markBackendDisconnected])

  useEffect(() => {
    if (!dragState) {
      return
    }
    const onMove = (event: MouseEvent) => {
      if (dragState.target === 'left') {
        setLeftPanelWidth(clamp(event.clientX, 300, 520))
      } else if (dragState.target === 'right') {
        setRightPanelWidth(clamp(window.innerWidth - event.clientX, 260, 420))
      } else {
        setBottomDockHeight(clamp(dragState.size + (dragState.origin - event.clientY), 120, 360))
      }
    }
    const onUp = () => setDragState(null)
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [dragState])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpenMenu(null)
        setActiveModal(null)
      }
    }
    const onPointerDown = () => setOpenMenu(null)
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('pointerdown', onPointerDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('pointerdown', onPointerDown)
    }
  }, [])

  useEffect(() => {
    const onHashChange = () => {
      const nextRail = readInitialRail()
      setActiveRail(nextRail)
      if (nextRail === 'models') {
        setActiveMode('models')
      } else if (nextRail === 'data') {
        setActiveMode('data')
      } else if (nextRail === 'create') {
        setActiveMode('image')
      }
    }
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  useEffect(() => {
    const nextPreferences: LayoutPreferences = {
      leftPanelWidth,
      rightPanelWidth,
      bottomDockHeight,
      bottomDockVisible,
    }
    try {
      window.localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(nextPreferences))
    } catch {
      // Layout persistence is a convenience; the app should stay usable if storage is blocked.
    }
  }, [bottomDockHeight, bottomDockVisible, leftPanelWidth, rightPanelWidth])

  useEffect(() => {
    return () => {
      void import('./ml/promptInsight').then(({ disposePromptInsightModel }) => {
        void disposePromptInsightModel()
      })
    }
  }, [])

  const filteredModels = useMemo(
    () => bootstrap.models.filter((model) => matchesEngineFilter(model, engineFilter)),
    [bootstrap.models, engineFilter],
  )

  const selectedModel = useMemo(() => {
    return (
      filteredModels.find((model) => model.id === settings.modelId) ??
      bootstrap.models.find((model) => model.id === settings.modelId) ??
      filteredModels[0] ??
      bootstrap.models[0]
    )
  }, [bootstrap.models, filteredModels, settings.modelId])

  const recentOutputs = useMemo(() => {
    const source = dataStatus?.recentOutputs.length ? dataStatus.recentOutputs : bootstrap.recentOutputs
    return source.slice(0, 8)
  }, [bootstrap.recentOutputs, dataStatus])

  const activeRatio = useMemo(
    () =>
      bootstrap.aspectRatios.find((ratio) => ratio.id === settings.aspectRatioId) ??
      bootstrap.aspectRatios[0],
    [bootstrap.aspectRatios, settings.aspectRatioId],
  )

  const handleModeSelect = useCallback((mode: ProMode) => {
    setActiveMode(mode)
    if (isCreationMode(mode)) {
      if (mode === 'video') {
        const sanaModel = bootstrap.models.find((model) => model.engineId === 'sana_video')
        setEngineFilter('sana_video')
        setSettings((current) => ({
          ...current,
          mode,
          modelId: sanaModel?.id ?? current.modelId,
          aspectRatioId: 'sana-480p',
          width: 832,
          height: 480,
          batchSize: 1,
        }))
      } else {
        setSettings((current) => ({ ...current, mode }))
      }
      setActiveRail('create')
    } else {
      setActiveRail(mode)
    }
  }, [bootstrap.models])

  const handleRailSelect = useCallback((id: string) => {
    setActiveRail(id)
    if (window.location.hash !== `#${id}`) {
      window.history.replaceState(null, '', `#${id}`)
    }
    if (id === 'models') {
      setActiveMode('models')
    } else if (id === 'data') {
      setActiveMode('data')
    } else if (id === 'tools') {
      setActiveMode('image')
    } else if (id === 'create') {
      setActiveMode('image')
      setSettings((current) => ({ ...current, mode: 'image' }))
    }
  }, [])

  const handleRatioSelect = useCallback((ratio: AspectRatioOption) => {
    setSettings((current) => ({
      ...current,
      aspectRatioId: ratio.id,
      width: ratio.width,
      height: ratio.height,
    }))
  }, [])

  const handleEngineFilterChange = useCallback(
    (nextFilter: EngineId) => {
      setEngineFilter(nextFilter)
      const nextModels = bootstrap.models.filter((model) => matchesEngineFilter(model, nextFilter))
      if (nextModels.length === 0) {
        return
      }
      setSettings((current) =>
        nextModels.some((model) => model.id === current.modelId)
          ? current
          : { ...current, modelId: nextModels[0].id },
      )
    },
    [bootstrap.models],
  )

  const handleModelSelect = useCallback(
    (modelId: string) => {
      const model = bootstrap.models.find((item) => item.id === modelId)
      setSettings((current) => ({
        ...current,
        modelId,
      }))
      if (model?.engineId) {
        setEngineFilter(model.engineId)
      }
      setStatusMessage(
        model
          ? `${model.name} selected. Runtime loads it when generation starts.`
          : `${modelId} selected. Runtime loads it when generation starts.`,
      )
    },
    [bootstrap.models],
  )

  const handleGenerate = useCallback(async () => {
    if (generationAbortRef.current) {
      if (generationActive) {
        setStatusMessage('Generation is already running.')
        return
      }
      generationAbortRef.current = null
    }
    if (generationActive) {
      setStatusMessage('Generation is already running in the backend.')
      return
    }
    if (!settings.prompt.trim()) {
      setStatusMessage('Enter a prompt before generating.')
      return
    }

    const controller = new AbortController()
    generationAbortRef.current = controller
    setIsGenerating(true)
    setGenerationProgress([])
    setGenerationTimings({})
    setGenerationReceiptPath('')
    setGenerationError('')
    setStatusMessage('Submitting to /api/pro/generate...')
    reportProClientEvent({
      action: 'pro-generate-submit',
      detail: `${settings.mode} ${settings.width}x${settings.height}`,
      context: {
        mode: settings.mode,
        modelId: settings.modelId,
        steps: settings.steps,
      },
    })
    try {
      const result = await generateProOutput(settings, controller.signal)
      setGenerationProgress(result.progress)
      setGenerationTimings(result.timings)
      setGenerationReceiptPath(result.receiptPath ?? '')
      const sessionOutputs =
        result.recentOutputs.length > 0
          ? result.recentOutputs
          : result.output
            ? [result.output]
            : []
      const stampedOutputs = sessionOutputs.map((item) => ({
        ...item,
        modelName: item.modelName || selectedModel?.name || settings.modelId,
      }))
      if (stampedOutputs.length > 0) {
        setPreview(stampedOutputs[stampedOutputs.length - 1])
        setBootstrap((current) => ({
          ...current,
          recentOutputs: mergeRecentOutputs(stampedOutputs, current.recentOutputs),
        }))
        setDataStatus((current) =>
          current
            ? {
                ...current,
                counts: {
                  ...current.counts,
                  recentOutputs: Math.max(
                    current.counts.recentOutputs,
                    mergeRecentOutputs(stampedOutputs, current.recentOutputs).length,
                  ),
                },
                recentOutputs: mergeRecentOutputs(stampedOutputs, current.recentOutputs),
              }
            : current,
        )
      }
      setStatusMessage(result.message || `Generation ${result.status}.`)
      setGenerationError('')
    } catch (error: unknown) {
      if (isGenerationCancelResult(error)) {
        setGenerationError('')
        setStatusMessage('Generation stop requested.')
      } else {
        const nextMessage = `Generation failed: ${formatApiError(error)}`
        setGenerationError(nextMessage)
        setStatusMessage(nextMessage)
        reportProClientError({
          kind: 'generation',
          message: nextMessage,
          stack: error instanceof Error ? error.stack : undefined,
          source: 'handleGenerate',
          context: {
            mode: settings.mode,
            modelId: settings.modelId,
            route: '/api/pro/generate',
          },
        })
      }
    } finally {
      if (generationAbortRef.current === controller) {
        generationAbortRef.current = null
      }
      setIsGenerating(false)
      void fetchProRuntime().then(setRuntime).catch(() => undefined)
      void fetchProLogs().then(setLogStatus).catch(() => undefined)
    }
  }, [generationActive, selectedModel?.name, settings])

  const handleStopGenerate = useCallback(() => {
    const controller = generationAbortRef.current
    if (!controller && !generationActive) {
      setStatusMessage('No active generation to stop.')
      return
    }
    setStatusMessage('Stopping generation...')
    reportProClientEvent({
      action: 'pro-generate-stop',
      detail: 'Stop requested',
      context: { runtimeState: runtime.state, jobId: runtime.job.id },
    })
    void stopProGeneration()
      .then((result) => {
        setStatusMessage(result.videoJobId ? 'Stop requested for active video job.' : 'Stop requested for active generation.')
        void fetchProRuntime().then(setRuntime).catch(() => undefined)
        void fetchProLogs().then(setLogStatus).catch(() => undefined)
      })
      .catch((error: unknown) => {
        const nextMessage = `Stop requested locally; backend interrupt failed: ${formatApiError(error)}`
        setStatusMessage(nextMessage)
        reportProClientError({
          kind: 'generation-stop',
          message: nextMessage,
          source: 'handleStopGenerate',
          context: { route: '/api/pro/interrupt' },
        })
      })
    controller?.abort()
  }, [generationActive, runtime.job.id, runtime.state])

  const handlePromptAnalyze = useCallback(async () => {
    setPromptInsightBusy(true)
    setPromptInsight((current) => ({
      ...current,
      status: 'loading',
      summary: 'Loading browser-side prompt helper...',
      progress: 5,
    }))
    try {
      const { analyzePromptWithTransformers } = await import('./ml/promptInsight')
      const nextInsight = await analyzePromptWithTransformers(
        settings.prompt,
        settings.negativePrompt,
        (message, progress) => {
          setPromptInsight((current) => ({
            ...current,
            status: 'loading',
            summary: message,
            progress,
          }))
        },
      )
      setPromptInsight(nextInsight)
      setStatusMessage('Prompt helper analysis complete.')
    } finally {
      setPromptInsightBusy(false)
    }
  }, [settings.negativePrompt, settings.prompt])

  const handleLayoutReset = useCallback(() => {
    setLeftPanelWidth(380)
    setRightPanelWidth(320)
    setBottomDockHeight(196)
    setBottomDockVisible(true)
  }, [])

  const startHorizontalDrag = useCallback(
    (target: 'left' | 'right') => {
      setDragState({
        target,
        origin: 0,
        size: target === 'left' ? leftPanelWidth : rightPanelWidth,
      })
    },
    [leftPanelWidth, rightPanelWidth],
  )

  const startBottomDrag = useCallback(
    (event: ReactMouseEvent<HTMLButtonElement>) => {
      event.preventDefault()
      setDragState({
        target: 'bottom',
        origin: event.clientY,
        size: bottomDockHeight,
      })
    },
    [bottomDockHeight],
  )

  const workspaceStyle = activeRail === 'models'
    ? undefined
    : ({
        '--left-panel-width': `${leftPanelWidth}px`,
        '--right-panel-width': `${rightPanelWidth}px`,
      } as CSSProperties)

  return (
    <div className="aiwf-pro-shell theme-preset-1" data-mode={activeMode}>
      <aside className="pro-rail" aria-label="Primary navigation">
        <button
          type="button"
          className="pro-logo-button"
          aria-label="AIWF Studio home"
          onClick={() => handleRailSelect('create')}
        >
          <img className="pro-logo-image" src={PRO_APP_ICON} alt="" />
        </button>
        <nav className="pro-rail-nav">
          {RAIL_ITEMS.map((item) => (
            <RailButton
              key={item.id}
              item={item}
              active={activeRail === item.id}
              onSelect={handleRailSelect}
            />
          ))}
        </nav>
        <div className="pro-rail-footer">
          <span className="pro-local-dot" aria-hidden="true" />
          <span>Local First</span>
          <small>All systems go</small>
          <small>{bootstrap.version}</small>
        </div>
      </aside>

      <main className="pro-main">
        <MenuBar
          openMenu={openMenu}
          onMenuChange={setOpenMenu}
          onAction={(action) => {
            setOpenMenu(null)
            if (action === 'toggle-dock') {
              setBottomDockVisible((value) => !value)
            } else if (action === 'reset-layout') {
              handleLayoutReset()
            } else if (action === 'open-hires') {
              setActiveModal('hires')
            } else if (action === 'open-segmentation') {
              setActiveModal('segmentation')
            } else if (action === 'open-reactor') {
              setActiveModal('reactor')
            } else if (action === 'copy-last') {
              void navigator.clipboard?.writeText(settings.prompt)
              setStatusMessage('Prompt copied to clipboard.')
            } else if (action === 'new-prompt') {
              setSettings((current) => ({ ...current, prompt: '', negativePrompt: '' }))
              setStatusMessage('Prompt cleared.')
            } else if (action === 'open-models') {
              handleRailSelect('models')
            } else if (action === 'open-data') {
              handleRailSelect('data')
            } else if (action === 'open-tools') {
              handleRailSelect('tools')
            } else if (action === 'open-monitor') {
              handleRailSelect('monitor')
            } else if (action === 'open-settings') {
              handleRailSelect('settings')
            } else if (action === 'open-help') {
              setActiveModal('about')
            }
          }}
        />
        <TopBar
          bootstrap={bootstrap}
          runtime={runtime}
          isGenerating={generationActive}
          statusMessage={statusMessage}
          generationError={generationError}
          selectedModelName={selectedModel?.name ?? settings.modelId}
          generationProgress={generationProgress}
          backendConnected={backendConnected}
          backendRecovering={backendRecovering}
          onRecoverBackend={handleRecoverBackend}
          onOpenSettings={() => handleRailSelect('settings')}
        />
        <ModeTabs activeMode={activeMode} onSelect={handleModeSelect} />

        <section
          className="pro-workspace"
          aria-label="AIWF Pro workspace"
          style={workspaceStyle}
        >
          {activeRail === 'models' ? (
            <ModelsWorkspace
              engineFilter={engineFilter}
              engines={bootstrap.engines}
              models={bootstrap.models}
              downloadsStatus={downloadsStatus}
              selectedModelId={settings.modelId}
              onEngineFilterChange={handleEngineFilterChange}
              onModelSelect={handleModelSelect}
            />
          ) : activeRail === 'data' ? (
            <>
              <DataControlPanel
                bootstrap={bootstrap}
                runtime={runtime}
                dataStatus={dataStatus}
                recentOutputs={recentOutputs}
                selectedModelName={selectedModel?.name ?? settings.modelId}
                onOpenModels={() => handleRailSelect('models')}
              />
              <ResizeHandle
                axis="vertical"
                label="Resize left panel"
                onMouseDown={() => startHorizontalDrag('left')}
              />
              <DataWorkspace
                bootstrap={bootstrap}
                runtime={runtime}
                dataStatus={dataStatus}
                recentOutputs={recentOutputs}
                selectedModelName={selectedModel?.name ?? settings.modelId}
              />
              <ResizeHandle
                axis="vertical"
                label="Resize right panel"
                onMouseDown={() => startHorizontalDrag('right')}
              />
            </>
          ) : activeRail === 'tools' ? (
            <>
              <ToolsControlPanel
                capabilitiesStatus={capabilitiesStatus}
                runtime={runtime}
                onOpenCreate={() => handleRailSelect('create')}
                onOpenVideo={() => {
                  handleRailSelect('create')
                  handleModeSelect('video')
                }}
                onOpenData={() => handleRailSelect('data')}
                onOpenSegmentation={() => setActiveModal('segmentation')}
                onOpenReactor={() => setActiveModal('reactor')}
              />
              <ResizeHandle
                axis="vertical"
                label="Resize left panel"
                onMouseDown={() => startHorizontalDrag('left')}
              />
              <ToolsWorkspace
                capabilitiesStatus={capabilitiesStatus}
                runtime={runtime}
                onOpenCreate={() => handleRailSelect('create')}
                onOpenVideo={() => {
                  handleRailSelect('create')
                  handleModeSelect('video')
                }}
                onOpenData={() => handleRailSelect('data')}
                onOpenSegmentation={() => setActiveModal('segmentation')}
                onOpenReactor={() => setActiveModal('reactor')}
              />
              <ResizeHandle
                axis="vertical"
                label="Resize right panel"
                onMouseDown={() => startHorizontalDrag('right')}
              />
            </>
          ) : activeRail === 'monitor' ? (
            <>
              <MonitorControlPanel
                runtime={runtime}
                logStatus={logStatus}
                statusMessage={statusMessage}
                recentOutputs={recentOutputs}
              />
              <ResizeHandle
                axis="vertical"
                label="Resize left panel"
                onMouseDown={() => startHorizontalDrag('left')}
              />
              <MonitorWorkspace
                runtime={runtime}
                logStatus={logStatus}
                statusMessage={statusMessage}
                recentOutputs={recentOutputs}
              />
              <ResizeHandle
                axis="vertical"
                label="Resize right panel"
                onMouseDown={() => startHorizontalDrag('right')}
              />
            </>
          ) : activeRail === 'logs' ? (
            <>
              <LogsControlPanel
                runtime={runtime}
                logStatus={logStatus}
                statusMessage={statusMessage}
                recentOutputs={recentOutputs}
              />
              <ResizeHandle
                axis="vertical"
                label="Resize left panel"
                onMouseDown={() => startHorizontalDrag('left')}
              />
              <LogsWorkspace
                runtime={runtime}
                logStatus={logStatus}
                statusMessage={statusMessage}
                generationError={generationError}
                recentOutputs={recentOutputs}
                selectedModelName={selectedModel?.name ?? settings.modelId}
                generationProgress={generationProgress}
              />
              <ResizeHandle
                axis="vertical"
                label="Resize right panel"
                onMouseDown={() => startHorizontalDrag('right')}
              />
            </>
          ) : activeRail === 'settings' ? (
            <>
              <SettingsControlPanel
                bootstrap={bootstrap}
                runtime={runtime}
                settings={settings}
                settingsStatus={settingsStatus}
                recentOutputs={recentOutputs}
                leftPanelWidth={leftPanelWidth}
                rightPanelWidth={rightPanelWidth}
                bottomDockHeight={bottomDockHeight}
                bottomDockVisible={bottomDockVisible}
                showAdvanced={showAdvanced}
                onBottomDockVisibleChange={setBottomDockVisible}
                onShowAdvancedChange={setShowAdvanced}
                onLayoutReset={handleLayoutReset}
              />
              <ResizeHandle
                axis="vertical"
                label="Resize left panel"
                onMouseDown={() => startHorizontalDrag('left')}
              />
              <SettingsWorkspace
                bootstrap={bootstrap}
                runtime={runtime}
                settings={settings}
                settingsStatus={settingsStatus}
                recentOutputs={recentOutputs}
                leftPanelWidth={leftPanelWidth}
                rightPanelWidth={rightPanelWidth}
                bottomDockHeight={bottomDockHeight}
                bottomDockVisible={bottomDockVisible}
                showAdvanced={showAdvanced}
              />
              <ResizeHandle
                axis="vertical"
                label="Resize right panel"
                onMouseDown={() => startHorizontalDrag('right')}
              />
            </>
          ) : (
            <>
              <PromptPanel
                settings={settings}
                bootstrap={bootstrap}
                filteredModels={filteredModels}
                engineFilter={engineFilter}
                engines={bootstrap.engines}
                selectedModelName={selectedModel?.name ?? settings.modelId}
                activeRatio={activeRatio}
                showAdvanced={showAdvanced}
                isGenerating={generationActive}
                recentOutputs={recentOutputs}
                promptInsight={promptInsight}
                promptInsightBusy={promptInsightBusy}
                generationProgress={generationProgress}
                generationTimings={generationTimings}
                generationReceiptPath={generationReceiptPath}
                onSettingsChange={setSettings}
                onEngineFilterChange={handleEngineFilterChange}
                onModelSelect={handleModelSelect}
                onRatioSelect={handleRatioSelect}
                onPreviewSelect={setPreview}
                onGenerate={handleGenerate}
                onStopGenerate={handleStopGenerate}
                onToggleAdvanced={() => setShowAdvanced((value) => !value)}
                onOpenSegmentation={() => setActiveModal('segmentation')}
                onOpenHires={() => setActiveModal('hires')}
                onOpenReactor={() => setActiveModal('reactor')}
                onPromptAnalyze={handlePromptAnalyze}
                bottomDockVisible={bottomDockVisible}
              />
              <ResizeHandle
                axis="vertical"
                label="Resize left panel"
                onMouseDown={() => startHorizontalDrag('left')}
              />
              <div className="pro-center-column">
                <CanvasPreview
                  activeMode={activeMode}
                  preview={preview}
                  statusMessage={statusMessage}
                  width={settings.width}
                  height={settings.height}
                  onOpenSegmentation={() => setActiveModal('segmentation')}
                  onOpenHires={() => setActiveModal('hires')}
                  onOpenReactor={() => setActiveModal('reactor')}
                  bottomDockVisible={bottomDockVisible}
                  onToggleBottomDock={() => setBottomDockVisible((value) => !value)}
                />
                <BottomDock
                  visible={bottomDockVisible}
                  height={bottomDockVisible ? bottomDockHeight : 0}
                  recentOutputs={recentOutputs}
                  statusMessage={statusMessage}
                  generationError={generationError}
                  selectedModelName={selectedModel?.name ?? settings.modelId}
                  onPreviewSelect={setPreview}
                  onResizeStart={startBottomDrag}
                  onToggleVisible={() => setBottomDockVisible((value) => !value)}
                />
              </div>
              <ResizeHandle
                axis="vertical"
                label="Resize right panel"
                onMouseDown={() => startHorizontalDrag('right')}
              />
            </>
          )}
          <RuntimePanel runtime={runtime} selectedModelName={selectedModel?.name ?? settings.modelId} />
        </section>

      </main>

      <ToolModal open={activeModal === 'segmentation'} title="Segmentation" onClose={() => setActiveModal(null)}>
        <div className="pro-modal-form">
          <label className="pro-field">
            <FieldLabel
              label="Mask route"
              tooltip="Choose the route first. Use quick masking for ordinary subject isolation and keep the full segmentation stack for controlled edits."
            />
            <select value={segmentationMode} onChange={(event) => setSegmentationMode(event.target.value)}>
              <option>Auto mask</option>
              <option>Paint and refine</option>
              <option>Box then segment</option>
            </select>
          </label>
        </div>
      </ToolModal>

      <ToolModal open={activeModal === 'hires'} title="High-res fix" onClose={() => setActiveModal(null)}>
        <div className="pro-modal-form">
          <label className="pro-toggle">
            <input type="checkbox" checked={hiresEnabled} onChange={(event) => setHiresEnabled(event.target.checked)} />
            <span>Enable high-res pass</span>
          </label>
          <RangeField label="Scale" min={1} max={3} step={0.05} value={hiresScale} onChange={setHiresScale} />
          <RangeField label="Denoise" min={0} max={1} step={0.05} value={hiresDenoise} onChange={setHiresDenoise} />
        </div>
      </ToolModal>

      <ToolModal open={activeModal === 'reactor'} title="ReActor" onClose={() => setActiveModal(null)}>
        <div className="pro-modal-form">
          <label className="pro-toggle">
            <input type="checkbox" checked={reactorEnabled} onChange={(event) => setReactorEnabled(event.target.checked)} />
            <span>Enable face swap pass</span>
          </label>
          <RangeField label="Blend strength" min={0} max={1} step={0.05} value={reactorStrength} onChange={setReactorStrength} />
        </div>
      </ToolModal>

      <ToolModal open={activeModal === 'about'} title="AIWF Studio" onClose={() => setActiveModal(null)}>
        <div className="pro-about-panel">
          <strong>Local creative control for open image and video models.</strong>
          <span>{bootstrap.version}</span>
        </div>
      </ToolModal>

    </div>
  )
}

function MenuBar({
  openMenu,
  onMenuChange,
  onAction,
}: {
  openMenu: MenuBarId
  onMenuChange: (value: MenuBarId) => void
  onAction: (value: string) => void
}) {
  return (
    <div
      className="pro-menu-bar"
      role="menubar"
      aria-label="Application menu"
      onPointerDown={(event) => event.stopPropagation()}
    >
      {MENU_BAR_ITEMS.map((item) => (
        <div key={item.id} className="pro-menu-group">
          <button
            type="button"
            className={openMenu === item.id ? 'pro-menu-button pro-menu-button-active' : 'pro-menu-button'}
            onClick={() => onMenuChange(openMenu === item.id ? null : item.id)}
          >
            {item.label}
          </button>
          {openMenu === item.id ? (
            <div className="pro-menu-dropdown">
              {item.items.map((entry) => (
                <button key={entry.id} type="button" className="pro-menu-item" onClick={() => onAction(entry.id)}>
                  <span>{entry.label}</span>
                  {entry.hint ? <small>{entry.hint}</small> : null}
                </button>
              ))}
            </div>
          ) : null}
        </div>
      ))}
    </div>
  )
}

function TopBar({
  bootstrap,
  runtime,
  isGenerating,
  statusMessage,
  generationError,
  selectedModelName,
  generationProgress,
  backendConnected,
  backendRecovering,
  onRecoverBackend,
  onOpenSettings,
}: {
  bootstrap: ProBootstrap
  runtime: ProRuntimeStatus
  isGenerating: boolean
  statusMessage: string
  generationError: string
  selectedModelName: string
  generationProgress: GenerationProgressEvent[]
  backendConnected: boolean
  backendRecovering: boolean
  onRecoverBackend: () => void
  onOpenSettings: () => void
}) {
  const latestProgress = generationProgress[generationProgress.length - 1]
  const runtimeJob = runtime.job
  const activeError = generationError || runtimeJob.error
  const runtimeJobActive = isRuntimeJobActive(runtimeJob)
  const progressMessage = activeError || latestProgress?.message || runtimeJob.message || statusMessage
  const progressPercent = runtimeJobActive
    ? clampPercent(runtimeJob.progress)
    : latestProgress
      ? clampPercent(Math.round(latestProgress.progress * 100))
      : clampPercent(runtimeJob.progress)
  const progressStep = runtimeJobActive && runtimeJob.totalSteps
    ? `${runtimeJob.step}/${runtimeJob.totalSteps}`
    : latestProgress?.total
      ? `${latestProgress.step}/${latestProgress.total}`
      : ''
  const active = isGenerating || runtime.state.toLowerCase() === 'running' || runtimeJobActive

  return (
    <header className="pro-topbar">
      <div className="pro-titlebar">
        <h1>{bootstrap.workspaceName || 'AIWF Studio'}</h1>
        <span>Local generation workspace</span>
      </div>
      <div className="pro-generation-strip" data-active={active} data-error={Boolean(activeError)}>
        <div className="pro-generation-copy">
          <span>{activeError ? 'Generation error' : active ? 'Generating' : 'Generation info'}</span>
          <strong>{progressMessage}</strong>
        </div>
        <div className="pro-generation-meter" aria-label={`Generation progress ${progressPercent}%`}>
          <span style={{ width: `${progressPercent}%` }} />
        </div>
        <div className="pro-generation-meta">
          {progressStep ? <span>{progressStep}</span> : null}
          <span>{selectedModelName}</span>
        </div>
      </div>
      <div className="pro-topbar-status">
        <div className="pro-engine-status" data-state={runtime.state.toLowerCase()}>
          <span className="pro-status-dot" aria-hidden="true" />
          <span>Local Engine</span>
          <strong>{runtime.state}</strong>
        </div>
        {!backendConnected || backendRecovering ? (
          <button
            type="button"
            className="pro-icon-button"
            aria-label="Restart backend"
            onClick={onRecoverBackend}
          >
            <RefreshCcw size={18} aria-hidden="true" />
          </button>
        ) : null}
        <button
          type="button"
          className="pro-icon-button"
          aria-label="Open settings"
          onClick={onOpenSettings}
        >
          <Settings size={18} aria-hidden="true" />
        </button>
        <div className="pro-window-controls" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
      </div>
    </header>
  )
}

function ModeTabs({
  activeMode,
  onSelect,
}: {
  activeMode: ProMode
  onSelect: (mode: ProMode) => void
}) {
  return (
    <nav className="pro-tabs" aria-label="Mode tabs">
      {MODE_TABS.map((item) => {
        const Icon = item.icon
        const active = activeMode === item.id
        return (
          <button
            key={item.id}
            type="button"
            className={active ? 'pro-tab pro-tab-active' : 'pro-tab'}
            aria-pressed={active}
            onClick={() => onSelect(item.id)}
          >
            <Icon size={16} aria-hidden="true" />
            <span>{item.label}</span>
          </button>
        )
      })}
    </nav>
  )
}

function PromptPanel({
  settings,
  bootstrap,
  filteredModels,
  engineFilter,
  engines,
  selectedModelName,
  activeRatio,
  showAdvanced,
  isGenerating,
  recentOutputs,
  promptInsight,
  promptInsightBusy,
  generationProgress,
  generationTimings,
  generationReceiptPath,
  onSettingsChange,
  onEngineFilterChange,
  onModelSelect,
  onRatioSelect,
  onPreviewSelect,
  onGenerate,
  onStopGenerate,
  onToggleAdvanced,
  onOpenSegmentation,
  onOpenHires,
  onOpenReactor,
  onPromptAnalyze,
  bottomDockVisible,
}: {
  settings: GenerationSettings
  bootstrap: ProBootstrap
  filteredModels: ProModelOption[]
  engineFilter: EngineId
  engines: EngineSummary[]
  selectedModelName: string
  activeRatio: AspectRatioOption | undefined
  showAdvanced: boolean
  isGenerating: boolean
  recentOutputs: RecentOutput[]
  promptInsight: PromptInsight
  promptInsightBusy: boolean
  generationProgress: GenerationProgressEvent[]
  generationTimings: Record<string, number>
  generationReceiptPath: string
  onSettingsChange: (value: GenerationSettings | ((current: GenerationSettings) => GenerationSettings)) => void
  onEngineFilterChange: (value: EngineId) => void
  onModelSelect: (modelId: string) => void
  onRatioSelect: (ratio: AspectRatioOption) => void
  onPreviewSelect: (value: RecentOutput) => void
  onGenerate: () => void
  onStopGenerate: () => void
  onToggleAdvanced: () => void
  onOpenSegmentation: () => void
  onOpenHires: () => void
  onOpenReactor: () => void
  onPromptAnalyze: () => void
  bottomDockVisible: boolean
}) {
  const handlePromptKeyDown = (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || !event.shiftKey || event.nativeEvent.isComposing) {
      return
    }
    event.preventDefault()
    onGenerate()
  }
  const handleVideoSourceChange = (event: ReactChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) {
      return
    }
    const reader = new FileReader()
    reader.onload = () => {
      const value = typeof reader.result === 'string' ? reader.result : ''
      if (!value) {
        return
      }
      onSettingsChange((current) => ({
        ...current,
        sourceImageDataUrl: value,
        sourceImageName: file.name,
      }))
    }
    reader.readAsDataURL(file)
  }

  return (
    <aside className="pro-prompt-panel" aria-label="Prompt and generation settings">
      <PanelHeader title="Prompt" actionLabel="Prompt tools" icon={PanelLeft} />
      <label className="pro-field pro-prompt-field">
        <FieldLabel
          label="Prompt"
          tooltip="Describe the scene, subject, or shot intent first. Start with the route goal, then add detail only when it changes the result you need."
        />
        <textarea
          value={settings.prompt}
          maxLength={1500}
          rows={5}
          aria-keyshortcuts="Shift+Enter"
          onKeyDown={handlePromptKeyDown}
          onChange={(event) =>
            onSettingsChange((current) => ({ ...current, prompt: event.target.value }))
          }
        />
        <small>{settings.prompt.length} / 1500</small>
      </label>

      <label className="pro-field">
        <FieldLabel
          label="Negative prompt"
          tooltip="Use this to remove failure patterns, not to rewrite the whole image. Keep it short and only exclude things you consistently do not want."
        />
        <textarea
          value={settings.negativePrompt}
          maxLength={1500}
          rows={3}
          onChange={(event) =>
            onSettingsChange((current) => ({
              ...current,
              negativePrompt: event.target.value,
            }))
          }
        />
        <small>{settings.negativePrompt.length} / 1500</small>
      </label>

      {settings.mode === 'video' ? (
        <section className="pro-video-source-card" aria-label="Video source image">
          <div className="pro-video-source-copy">
            <FileImage size={17} aria-hidden="true" />
            <div>
              <strong>Source image</strong>
              <span>{settings.sourceImageName || 'Optional first frame for image-to-video.'}</span>
            </div>
          </div>
          {settings.sourceImageDataUrl ? (
            <img className="pro-video-source-preview" src={settings.sourceImageDataUrl} alt="" />
          ) : (
            <div className="pro-video-source-empty">No image selected</div>
          )}
          <div className="pro-video-source-actions">
            <label className="pro-secondary-button" htmlFor="pro-video-source-input">
              <FileImage size={15} aria-hidden="true" />
              <span>Upload image</span>
            </label>
            <input
              id="pro-video-source-input"
              className="pro-file-input-hidden"
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={handleVideoSourceChange}
            />
            {settings.sourceImageDataUrl ? (
              <button
                type="button"
                className="pro-secondary-button ghost"
                onClick={() =>
                  onSettingsChange((current) => ({
                    ...current,
                    sourceImageDataUrl: '',
                    sourceImageName: '',
                  }))
                }
              >
                Clear
              </button>
            ) : null}
          </div>
        </section>
      ) : null}

      <div className="pro-prompt-actions" aria-label="Prompt actions">
        <button
          type="button"
          className={isGenerating ? 'pro-generate-button pro-generate-button-stop' : 'pro-generate-button'}
          onClick={isGenerating ? onStopGenerate : onGenerate}
        >
          {isGenerating ? <X size={18} aria-hidden="true" /> : <Sparkles size={18} aria-hidden="true" />}
          <span>{isGenerating ? 'Stop' : settings.mode === 'video' ? 'Generate video' : 'Generate image'}</span>
        </button>
        <button
          type="button"
          className="pro-secondary-button"
          disabled={promptInsightBusy}
          onClick={onPromptAnalyze}
        >
          {promptInsightBusy ? 'Analyzing...' : 'Analyze prompt'}
        </button>
      </div>

      <section className="pro-prompt-insight-card" aria-label="Prompt helper">
        <div className="pro-prompt-insight-header">
          <div>
            <strong>Prompt helper</strong>
            <span>Small Transformers.js model, lazy browser load</span>
          </div>
        </div>
        <div className="pro-prompt-insight-meter" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={promptInsight.progress}>
          <span style={{ width: `${promptInsight.progress}%` }} />
        </div>
        <p>{promptInsight.summary}</p>
        <div className="pro-prompt-insight-meta">
          <span>{promptInsight.modelLabel}</span>
          <span>{Math.round(promptInsight.modelScore * 100)}%</span>
        </div>
        {promptInsight.signals.length > 0 ? (
          <div className="pro-prompt-signal-grid">
            {promptInsight.signals.map((signal) => (
              <div key={signal.label} className={`pro-prompt-signal pro-prompt-signal-${signal.tone}`}>
                <span>{signal.label}</span>
                <strong>{signal.value}</strong>
              </div>
            ))}
          </div>
        ) : null}
        <ul className="pro-prompt-suggestions">
          {promptInsight.suggestions.map((suggestion) => (
            <li key={suggestion}>{suggestion}</li>
          ))}
        </ul>
      </section>

      <label className="pro-field">
        <FieldLabel
          label="Engine"
          tooltip="Choose the route first. Engines filter the model list down to the families that actually fit that workflow."
        />
        <select
          value={engineFilter}
          onChange={(event) => onEngineFilterChange(event.target.value as EngineId)}
        >
          {buildEngineFilterOptions(engines).map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>

      <label className="pro-field">
        <FieldLabel
          label="Model"
          tooltip="Pick the model after the route. Use one model long enough to learn its defaults before you start comparing families."
        />
        <div className="pro-select-row">
          <select
            value={settings.modelId}
            onChange={(event) => onModelSelect(event.target.value)}
          >
            {filteredModels.map((model) => (
              <option key={model.id} value={model.id}>
                {formatModelOptionLabel(model)}
              </option>
            ))}
          </select>
          <button type="button" className="pro-icon-button" aria-label="Refresh models">
            <RefreshCcw size={16} aria-hidden="true" />
          </button>
        </div>
      </label>

      <fieldset className="pro-aspect-group">
        <legend>
          <FieldLabel
            label="Aspect ratio"
            tooltip="Set working shape early. Changing ratio late can hide composition problems by turning a prompt problem into a crop problem."
          />
        </legend>
        <div className="pro-aspect-chips">
          {bootstrap.aspectRatios.map((ratio) => {
            const active = activeRatio?.id === ratio.id
            return (
              <button
                key={ratio.id}
                type="button"
                className={active ? 'pro-chip pro-chip-active' : 'pro-chip'}
                aria-pressed={active}
                onClick={() => onRatioSelect(ratio)}
              >
                {ratio.label}
              </button>
            )
          })}
        </div>
      </fieldset>

      <div className="pro-settings-block">
        <div className="pro-section-label">{settings.mode === 'video' ? 'Sana video settings' : 'Image settings'}</div>
        <RangeField
          label="Steps"
          tooltip="Steps control how long the model refines the image. Raise this slowly and only when the current model clearly benefits."
          min={1}
          max={settings.mode === 'video' ? 100 : 80}
          step={1}
          value={settings.steps}
          onChange={(value) => onSettingsChange((current) => ({ ...current, steps: value }))}
        />
        <RangeField
          label="CFG scale"
          tooltip="Guidance strength pushes the model harder toward the prompt. More is not automatically better, especially on distilled or speed-focused models."
          min={0}
          max={20}
          step={0.5}
          value={settings.cfgScale}
          onChange={(value) => onSettingsChange((current) => ({ ...current, cfgScale: value }))}
        />
        {settings.mode === 'video' ? (
          <>
            <RangeField
              label="Frames"
              tooltip="Frame count controls video duration and denoise work. Keep smoke tests short, then increase after timing receipts look sane."
              min={1}
              max={257}
              step={1}
              value={settings.frames}
              onChange={(value) => onSettingsChange((current) => ({ ...current, frames: value }))}
            />
            <RangeField
              label="FPS"
              tooltip="FPS changes playback duration without changing denoise frame count. Use it as a delivery setting after the motion looks right."
              min={1}
              max={60}
              step={1}
              value={settings.fps}
              onChange={(value) => onSettingsChange((current) => ({ ...current, fps: value }))}
            />
            <label className="pro-field pro-compact-field">
              <FieldLabel
                label="Quantization"
                tooltip="Auto uses the fastest safe Sana path AIWF can load. FP8 and BNB modes are explicit override paths for VRAM pressure."
              />
              <select
                value={settings.sanaQuantization}
                onChange={(event) =>
                  onSettingsChange((current) => ({ ...current, sanaQuantization: event.target.value }))
                }
              >
                {SANA_QUANTIZATION_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="pro-field pro-compact-field">
              <FieldLabel
                label="VAE tiling"
                tooltip="Auto keeps decode fast, then retries with tiling only after an out-of-memory error. Always is slower but safer."
              />
              <select
                value={settings.sanaVaeTiling}
                onChange={(event) =>
                  onSettingsChange((current) => ({ ...current, sanaVaeTiling: event.target.value }))
                }
              >
                {SANA_VAE_TILING_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          </>
        ) : (
          <label className="pro-field pro-compact-field">
            <FieldLabel
              label="Sampler"
              tooltip="Sampler changes the route the denoiser takes through the same request. Change this one variable at a time so you can see what it actually did."
            />
            <select
              value={settings.sampler}
              onChange={(event) =>
                onSettingsChange((current) => ({ ...current, sampler: event.target.value }))
              }
            >
              {bootstrap.samplers.map((sampler) => (
                <option key={sampler} value={sampler}>
                  {sampler}
                </option>
              ))}
            </select>
          </label>
        )}
        <label className="pro-field pro-compact-field">
          <FieldLabel
            label="Seed"
            tooltip="Seed is your receipt for repeatability. Keep a good seed when testing one control at a time so the comparison stays honest."
          />
          <input
            type="number"
            value={settings.seed}
            onChange={(event) =>
              onSettingsChange((current) => ({
                ...current,
                seed: Number(event.target.value),
              }))
            }
          />
        </label>
        <label className="pro-field pro-compact-field">
          <FieldLabel
            label="Width"
            tooltip="Resolution belongs up front because it changes often. Set working size before you judge detail, speed, or memory use."
          />
          <input
            type="number"
            min={256}
            step={8}
            value={settings.width}
            onChange={(event) =>
              onSettingsChange((current) => ({
                ...current,
                width: Number(event.target.value),
              }))
            }
          />
        </label>
        <label className="pro-field pro-compact-field">
          <FieldLabel
            label="Height"
            tooltip="Keep working dimensions visible. If the output feels wrong, confirm size and ratio before you keep changing prompts."
          />
          <input
            type="number"
            min={256}
            step={8}
            value={settings.height}
            onChange={(event) =>
              onSettingsChange((current) => ({
                ...current,
                height: Number(event.target.value),
              }))
            }
          />
        </label>
        {settings.mode === 'video' ? (
          <div className="pro-video-toggle-grid">
            <label className="pro-toggle">
              <input
                type="checkbox"
                checked={settings.useSageAttention}
                onChange={(event) =>
                  onSettingsChange((current) => ({ ...current, useSageAttention: event.target.checked }))
                }
              />
              <span>Sage attention</span>
            </label>
            <label className="pro-toggle">
              <input
                type="checkbox"
                checked={settings.offloadTextEncoderAfterEncode}
                onChange={(event) =>
                  onSettingsChange((current) => ({
                    ...current,
                    offloadTextEncoderAfterEncode: event.target.checked,
                  }))
                }
              />
              <span>Free text encoder</span>
            </label>
            <label className="pro-toggle">
              <input
                type="checkbox"
                checked={settings.generateAudio}
                onChange={(event) =>
                  onSettingsChange((current) => ({ ...current, generateAudio: event.target.checked }))
                }
              />
              <span>Add audio</span>
            </label>
          </div>
        ) : (
          <label className="pro-field pro-compact-field">
            <FieldLabel
              label="Batch"
              tooltip="Batch controls are front-and-center because they can change every run. Use them deliberately and keep receipts when you compare outputs."
            />
            <input
              type="number"
              min={1}
              max={8}
              value={settings.batchSize}
              onChange={(event) =>
                onSettingsChange((current) => ({
                  ...current,
                  batchSize: Number(event.target.value),
                }))
              }
            />
          </label>
        )}
      </div>

      {settings.mode === 'video' ? (
        <SanaStageReceipt
          events={generationProgress}
          timings={generationTimings}
          receiptPath={generationReceiptPath}
        />
      ) : null}

      <div className="pro-tool-launchers">
        <button type="button" className="pro-tool-button" onClick={onOpenSegmentation}>
          <ScanSearch size={15} aria-hidden="true" />
          <span>Segmentation</span>
        </button>
        <button type="button" className="pro-tool-button" onClick={onOpenHires}>
          <Highlighter size={15} aria-hidden="true" />
          <span>High-res fix</span>
        </button>
        <button type="button" className="pro-tool-button" onClick={onOpenReactor}>
          <Wand2 size={15} aria-hidden="true" />
          <span>ReActor</span>
        </button>
      </div>

      <div className="pro-panel-actions">
        <button
          type="button"
          className="pro-icon-button pro-sliders-button"
          aria-label="Toggle secondary settings"
          aria-pressed={showAdvanced}
          onClick={onToggleAdvanced}
        >
          <SlidersHorizontal size={18} aria-hidden="true" />
        </button>
      </div>

      {showAdvanced ? (
        <div className="pro-advanced-panel">
          <div className="pro-advanced-note">
            Secondary controls stay nearby, but the frequent run-to-run variables remain on the main surface.
          </div>
          <label className="pro-toggle">
            <input type="checkbox" checked={bottomDockVisible} onChange={() => undefined} readOnly />
            <span>Bottom dock visible</span>
          </label>
        </div>
      ) : null}

      <div className="pro-selected-model" title={selectedModelName}>
        <HardDrive size={14} aria-hidden="true" />
        <span>{selectedModelName}</span>
      </div>

      <div className="pro-recent-strip pro-recent-strip-hidden-desktop">
        <div className="pro-recent-strip-header">
          <FieldLabel
            label="Recent outputs"
            tooltip="Keep the native result visible. Compare from saved outputs instead of trusting memory after several prompt or setting changes."
          />
          <small>{recentOutputs.length} loaded</small>
        </div>
        <div className="pro-recent-grid">
          {recentOutputs.map((item) => (
            <button
              key={item.id}
              type="button"
              className="pro-recent-thumb"
              onClick={() => onPreviewSelect(item)}
              title={item.prompt}
            >
              <OutputMedia item={item} />
              <span>{item.modelName ?? item.mode}</span>
            </button>
          ))}
        </div>
      </div>
    </aside>
  )
}

function OutputMedia({ item }: { item: RecentOutput }) {
  const mediaUrl = item.thumbnailUrl || item.url
  const outputUrl = item.url.split('?', 1)[0].toLowerCase()
  const thumbnailUrl = item.thumbnailUrl.split('?', 1)[0].toLowerCase()
  const isVideo = item.mode === 'video' || isVideoUrl(outputUrl)
  const poster = item.thumbnailUrl && item.thumbnailUrl !== item.url && !isVideoUrl(thumbnailUrl) ? item.thumbnailUrl : undefined
  if (isVideo) {
    return (
      <video
        src={item.url}
        poster={poster}
        muted
        playsInline
        preload="metadata"
        aria-label={item.prompt}
      />
    )
  }
  return <img src={mediaUrl} alt={item.prompt} />
}

function isVideoUrl(value: string): boolean {
  return value.endsWith('.mp4') || value.endsWith('.webm') || value.endsWith('.mov')
}

function SanaStageReceipt({
  events,
  timings,
  receiptPath,
}: {
  events: GenerationProgressEvent[]
  timings: Record<string, number>
  receiptPath: string
}) {
  const latest = events.length > 0 ? events[events.length - 1] : null
  const timingRows = Object.entries(timings).filter(([, value]) => Number.isFinite(value))
  return (
    <section className="pro-sana-receipt" aria-label="Sana stage receipt">
      <div className="pro-sana-receipt-head">
        <div>
          <strong>Sana stages</strong>
          <span>{latest ? latest.message : 'No stage receipt yet.'}</span>
        </div>
        <small>{latest ? `${Math.round(latest.progress * 100)}%` : 'idle'}</small>
      </div>
      <div className="pro-sana-meter" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={latest ? Math.round(latest.progress * 100) : 0}>
        <span style={{ width: `${latest ? Math.round(latest.progress * 100) : 0}%` }} />
      </div>
      {events.length > 0 ? (
        <div className="pro-sana-stage-list">
          {events.slice(-8).map((event, index) => (
            <div key={`${event.stage}-${index}-${event.seconds}`} className="pro-sana-stage-row">
              <span>{event.stage}</span>
              <strong>{event.total ? `${event.step}/${event.total}` : `${Math.round(event.progress * 100)}%`}</strong>
              <small>{event.seconds.toFixed(2)}s</small>
            </div>
          ))}
        </div>
      ) : null}
      {timingRows.length > 0 ? (
        <div className="pro-sana-timing-grid">
          {timingRows.map(([key, value]) => (
            <span key={key}>
              <strong>{key}</strong>
              <small>{value.toFixed(2)}s</small>
            </span>
          ))}
        </div>
      ) : null}
      {receiptPath ? <div className="pro-sana-receipt-path" title={receiptPath}>{receiptPath}</div> : null}
    </section>
  )
}

function ModelsWorkspace({
  engineFilter,
  engines,
  models,
  downloadsStatus,
  selectedModelId,
  onEngineFilterChange,
  onModelSelect,
}: {
  engineFilter: EngineId
  engines: EngineSummary[]
  models: ProModelOption[]
  downloadsStatus: ProDownloadsStatus | null
  selectedModelId: string
  onEngineFilterChange: (value: EngineId) => void
  onModelSelect: (modelId: string) => void
}) {
  const visibleModels = models.filter((model) => matchesEngineFilter(model, engineFilter))
  const groupedModels = groupModelsByEngine(visibleModels, engines)
  const downloadSummary = summarizeDownloads(downloadsStatus, engineFilter)

  return (
    <section className="pro-models-workspace" aria-label="Model inventory">
      <div className="pro-models-header">
        <div>
          <strong>Model inventory</strong>
          <span>Choose an engine to see only the models that route supports.</span>
        </div>
        <label className="pro-models-filter">
          <FieldLabel
            label="Engine"
            tooltip="This inventory is grouped by route, not by hype. Use the engine filter to avoid comparing models that do not belong to the same workflow."
          />
          <select
            value={engineFilter}
            onChange={(event) => onEngineFilterChange(event.target.value as EngineId)}
          >
            {buildEngineFilterOptions(engines).map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <section className="pro-download-status-card" aria-label="Download catalog status">
        <div>
          <strong>Download catalog</strong>
          <span>{downloadSummary.subtitle}</span>
        </div>
        <div className="pro-download-stat-row">
          <StatTile label="Catalog" value={`${downloadSummary.total}`} hint="known entries" />
          <StatTile label="Installed" value={`${downloadSummary.installed}`} hint="ready locally" />
          <StatTile label="Route" value={`${downloadSummary.routeTotal}`} hint={downloadSummary.routeLabel} />
        </div>
        <div className="pro-download-chip-row">
          {downloadSummary.items.map((item) => (
            <span
              key={item.key}
              className={item.installed ? 'pro-download-chip pro-download-chip-ready' : 'pro-download-chip'}
              title={item.destination}
            >
              <strong>{item.title}</strong>
              <small>{item.installed ? 'Installed' : item.category}</small>
            </span>
          ))}
        </div>
      </section>

      <div className="pro-model-groups">
        {groupedModels.length > 0 ? (
          groupedModels.map((group) => (
            <section key={group.id} className="pro-model-group">
              <div className="pro-model-group-header">
                <strong>{group.label}</strong>
                <small>{group.models.length} models</small>
              </div>
              <div className="pro-model-card-grid">
                {group.models.map((model) => {
                  const active = model.id === selectedModelId
                  return (
                    <button
                      key={model.id}
                      type="button"
                      className={active ? 'pro-model-card pro-model-card-active' : 'pro-model-card'}
                      onClick={() => onModelSelect(model.id)}
                    >
                      <div className="pro-model-card-top">
                        <span>{model.engineLabel ?? group.label}</span>
                        <small>{model.status ?? 'Available'}</small>
                      </div>
                      <strong>{model.name}</strong>
                      <div className="pro-model-card-meta">
                        <span>{model.architecture ?? 'Unknown architecture'}</span>
                        <small>{model.assetSummary && !model.name.includes(model.assetSummary) ? model.assetSummary : 'Local asset'}</small>
                      </div>
                    </button>
                  )
                })}
              </div>
            </section>
          ))
        ) : (
          <div className="pro-empty-preview">
            <Boxes size={42} aria-hidden="true" />
            <span>No models found for this engine.</span>
          </div>
        )}
      </div>
    </section>
  )
}

function DataControlPanel({
  bootstrap,
  runtime,
  dataStatus,
  recentOutputs,
  selectedModelName,
  onOpenModels,
}: {
  bootstrap: ProBootstrap
  runtime: ProRuntimeStatus
  dataStatus: ProDataStatus | null
  recentOutputs: RecentOutput[]
  selectedModelName: string
  onOpenModels: () => void
}) {
  const outputs = dataStatus?.recentOutputs.length ? dataStatus.recentOutputs : recentOutputs
  const summary = summarizeRecentOutputs(outputs)
  return (
    <aside className="pro-prompt-panel" aria-label="Data controls">
      <PanelHeader title="Data" actionLabel="Data actions" icon={Database} />
      <div className="pro-workspace-stack">
        <InfoCard title="Library state" subtitle="The shell now treats data as a real workspace instead of a dead rail button.">
          <div className="pro-signal-grid">
            <div className="pro-signal-card">
              <span>Ratios</span>
              <strong>{bootstrap.aspectRatios.length}</strong>
            </div>
            <div className="pro-signal-card">
              <span>Outputs</span>
              <strong>{dataStatus?.counts.recentOutputs ?? outputs.length}</strong>
            </div>
          </div>
        </InfoCard>
        <InfoCard title="Receipt health" subtitle="Use this page to judge whether the shell has enough local receipts to support later dataset work.">
          <dl className="pro-runtime-list">
            <MetricRow label="Selected model" value={selectedModelName} />
            <MetricRow label="Unique models" value={`${summary.uniqueModels}`} />
            <MetricRow label="Latest receipt" value={summary.latestCreatedAt} />
            <MetricRow label="Queue state" value={`${runtime.queueCount} waiting`} />
            <MetricRow label="Output root" value={dataStatus?.outputRoot || 'Backend not connected'} />
          </dl>
        </InfoCard>
        <InfoCard title="Model linkage" subtitle="Keep data views close to model families and route context.">
          <button type="button" className="pro-secondary-button" onClick={onOpenModels}>
            Open model inventory
          </button>
        </InfoCard>
      </div>
    </aside>
  )
}

function DataWorkspace({
  bootstrap,
  runtime,
  dataStatus,
  recentOutputs,
  selectedModelName,
}: {
  bootstrap: ProBootstrap
  runtime: ProRuntimeStatus
  dataStatus: ProDataStatus | null
  recentOutputs: RecentOutput[]
  selectedModelName: string
}) {
  const outputs = dataStatus?.recentOutputs.length ? dataStatus.recentOutputs : recentOutputs
  const modeCounts = countOutputsByMode(outputs)
  const summary = summarizeRecentOutputs(outputs)
  const modelBuckets = buildOutputModelBuckets(outputs, selectedModelName)
  const aspectBuckets = buildAspectBuckets(outputs)
  return (
    <section className="pro-workspace-surface" aria-label="Data workspace">
      <WorkspaceHeader
        eyebrow="Data"
        title="Dataset and artifact staging"
        description="This page now has a dedicated surface for samples, routing buckets, and future manifests."
      />
      <div className="pro-workspace-grid">
        <InfoCard title="Artifact buckets" subtitle="Recent output receipts grouped by route for later monitor and eval hooks.">
          <div className="pro-stat-grid">
            <StatTile label="Image" value={`${modeCounts.image}`} hint="recent items" />
            <StatTile label="Video" value={`${modeCounts.video}`} hint="recent items" />
            <StatTile label="Inpaint" value={`${modeCounts.inpaint}`} hint="recent items" />
            <StatTile label="Unique models" value={`${summary.uniqueModels}`} hint="in receipts" />
          </div>
        </InfoCard>
        <InfoCard title="Output families" subtitle="Recent artifacts grouped by model so you can see whether receipts are broad or overfit to one route.">
          <div className="pro-token-grid">
            {modelBuckets.map((bucket) => (
              <div key={bucket.label} className="pro-token-card">
                <strong>{bucket.label}</strong>
                <span>{bucket.count} receipts</span>
              </div>
            ))}
          </div>
        </InfoCard>
        <InfoCard title="Aspect presets" subtitle="Ratios stay visible here so export and curation flows share the same vocabulary.">
          <div className="pro-chip-grid">
            {bootstrap.aspectRatios.map((ratio) => (
              <div key={ratio.id} className="pro-static-chip">
                <strong>{ratio.label}</strong>
                <span>{ratio.width}x{ratio.height}</span>
              </div>
            ))}
          </div>
        </InfoCard>
        <InfoCard title="Observed resolutions" subtitle="The shell now shows the actual output shapes it has seen, not only the configured presets.">
          <div className="pro-token-grid">
            {aspectBuckets.map((bucket) => (
              <div key={bucket.label} className="pro-token-card">
                <strong>{bucket.label}</strong>
                <span>{bucket.count} outputs</span>
              </div>
            ))}
          </div>
        </InfoCard>
        <InfoCard title="Artifact routing" subtitle="Runtime context and local receipt state stay visible alongside dataset prep work.">
          <div className="pro-stat-grid">
            <StatTile label="Queue" value={`${runtime.queueCount}`} hint="tasks waiting" />
            <StatTile label="Backend" value={runtime.backend} hint="active runtime" />
            <StatTile label="Selected model" value={selectedModelName} hint="current route" />
            <StatTile label="Latest receipt" value={summary.latestCreatedAt} hint="recent artifact" />
          </div>
        </InfoCard>
        <InfoCard title="Recent files" subtitle="Scroll-safe artifact list for samples, checks, and dataset handoff.">
          <div className="pro-output-list">
            {outputs.map((item) => (
              <article key={item.id} className="pro-output-row">
                <OutputMedia item={item} />
                <div>
                  <strong>{item.modelName ?? item.mode}</strong>
                  <span>{truncateText(item.prompt, 140)}</span>
                </div>
                <small>{formatDisplayDate(item.createdAt)} / {item.width}x{item.height}</small>
              </article>
            ))}
          </div>
        </InfoCard>
      </div>
    </section>
  )
}

function ToolsControlPanel({
  capabilitiesStatus,
  runtime,
  onOpenCreate,
  onOpenVideo,
  onOpenData,
  onOpenSegmentation,
  onOpenReactor,
}: {
  capabilitiesStatus: ProCapabilitiesStatus | null
  runtime: ProRuntimeStatus
  onOpenCreate: () => void
  onOpenVideo: () => void
  onOpenData: () => void
  onOpenSegmentation: () => void
  onOpenReactor: () => void
}) {
  const status = capabilitiesStatus ?? EMPTY_CAPABILITIES
  const readyCount = status.readiness.counts.working ?? 0
  const pendingCount = status.readiness.counts['metadata-only'] ?? 0
  const blockedCount =
    (status.readiness.counts['blocked-cleanly'] ?? 0) +
    (status.readiness.counts['broken-runtime'] ?? 0) +
    (status.readiness.counts['unsupported-no-route'] ?? 0)
  return (
    <aside className="pro-prompt-panel" aria-label="Tool controls">
      <PanelHeader title="Tools" actionLabel="Tool bench" icon={Wand2} />
      <div className="pro-workspace-stack">
        <InfoCard title="Tonight QA" subtitle="Open the main lanes first. The detailed coverage check is in the workspace drawer.">
          <div className="pro-stat-grid">
            <StatTile label="Models" value={`${status.counts.checkpoints}`} hint="base checkpoints" />
            <StatTile label="LoRAs" value={`${status.counts.loras}`} hint="local adapters" />
            <StatTile label="Ready" value={`${readyCount}`} hint="smoked routes" />
            <StatTile label="Blocked" value={`${blockedCount}`} hint="needs wiring" />
          </div>
          <div className="pro-inline-controls pro-wrap-controls">
            <button type="button" className="pro-primary-button" onClick={onOpenCreate}>Create</button>
            <button type="button" className="pro-secondary-button" onClick={onOpenVideo}>Sana Video</button>
            <button type="button" className="pro-secondary-button" onClick={onOpenData}>Data</button>
            <button type="button" className="pro-secondary-button" onClick={onOpenSegmentation}>Segment</button>
            <button type="button" className="pro-secondary-button" onClick={onOpenReactor}>ReActor</button>
          </div>
        </InfoCard>
        <InfoCard title="Asset counts" subtitle="Read-only checks. This panel does not load a model.">
          <dl className="pro-runtime-list">
            <MetricRow label="ControlNet" value={`${status.counts.controlnet}`} />
            <MetricRow label="SAM" value={`${status.counts.sam}`} />
            <MetricRow label="ReActor" value={`${status.counts.reactor}`} />
            <MetricRow label="Enhance" value={`${status.counts.enhance}`} />
            <MetricRow label="Sana Video" value={`${status.counts.sanaVideo}`} />
            <MetricRow label="Wan" value={`${status.counts.wan}`} />
            <MetricRow label="Pending smoke" value={`${pendingCount}`} />
            <MetricRow label="Runtime" value={runtime.backend} />
          </dl>
        </InfoCard>
      </div>
    </aside>
  )
}

function ToolsWorkspace({
  capabilitiesStatus,
  runtime,
  onOpenCreate,
  onOpenVideo,
  onOpenData,
  onOpenSegmentation,
  onOpenReactor,
}: {
  capabilitiesStatus: ProCapabilitiesStatus | null
  runtime: ProRuntimeStatus
  onOpenCreate: () => void
  onOpenVideo: () => void
  onOpenData: () => void
  onOpenSegmentation: () => void
  onOpenReactor: () => void
}) {
  const status = capabilitiesStatus ?? EMPTY_CAPABILITIES
  const readiness = status.readiness
  const readyCount = readiness.counts.working ?? 0
  const metadataOnlyCount = readiness.counts['metadata-only'] ?? readiness.metadataOnlyCount
  const blockedCount = countBlockedReadiness(readiness.counts)
  const needsWorkCount = Math.max(0, readiness.total - readyCount)
  const readinessFamilies = readiness.families.slice(0, 6)
  const readinessIssues = readiness.needsWork.slice(0, 5)
  const lanes: ToolLaneCard[] = [
    {
      id: 'create',
      title: 'Create',
      summary: 'Image generation, inpaint, prompt help, and model selection.',
      stats: [`${status.counts.checkpoints} checkpoints`, `${status.counts.loras} LoRAs`],
      actions: [{ label: 'Open Create', onClick: onOpenCreate }],
    },
    {
      id: 'image',
      title: 'Image Tools',
      summary: 'ControlNet, SAM, Enhance, and ReActor checks grouped together.',
      stats: [
        `${status.counts.controlnet} ControlNet`,
        `${status.counts.sam} SAM`,
        `${status.counts.enhance} enhance`,
        `${status.counts.reactor} ReActor`,
      ],
      actions: [
        { label: 'Segment', onClick: onOpenSegmentation },
        { label: 'ReActor', onClick: onOpenReactor },
      ],
    },
    {
      id: 'video',
      title: 'Video Tools',
      summary: 'Sana generation is wired in React; Wan/LTX and post stages remain visible for Gradio routes.',
      stats: [`${status.counts.sanaVideo} Sana`, `${status.counts.wan} Wan models`],
      note: status.counts.sanaVideo > 0 ? 'Sana snapshot detected.' : 'Sana snapshot not detected yet.',
      actions: [{ label: 'Open Sana', onClick: onOpenVideo }],
    },
    {
      id: 'data',
      title: 'Data',
      summary: 'Library, PNG Info, history, receipts, and logs for QA work.',
      stats: [`${status.counts.gradioTabs} existing tabs`, `${status.counts.reactRails} React rails`],
      actions: [{ label: 'Open Data', onClick: onOpenData }],
    },
  ]

  return (
    <section className="pro-workspace-surface" aria-label="Tools workspace">
      <WorkspaceHeader
        eyebrow="Tools"
        title="Tool bench"
        description="Four QA lanes for the current Studio surface. React controls stay up front; raw coverage stays in one drawer."
      />
      <div className="pro-workspace-grid">
        <InfoCard title="Current surface" subtitle="Inventory only. This view does not load models or start generation.">
          <div className="pro-stat-grid">
            <StatTile label="Backend" value={runtime.backend} hint="active route" />
            <StatTile label="Device" value={runtime.device} hint="local machine" />
            <StatTile label="Models" value={`${status.counts.checkpoints}`} hint="base checkpoints" />
            <StatTile label="Tool paths" value={`${status.tools.length}`} hint="mapped checks" />
          </div>
        </InfoCard>
        <InfoCard title="Release readiness" subtitle="Pipeline ledger from local metadata and existing smoke receipts.">
          <div className="pro-stat-grid">
            <StatTile label="Ready" value={`${readyCount}`} hint="smoked routes" />
            <StatTile label="Pending" value={`${metadataOnlyCount}`} hint="metadata only" />
            <StatTile label="Blocked" value={`${blockedCount}`} hint="route gaps" />
            <StatTile label="Total" value={`${readiness.total}`} hint="ledger rows" />
          </div>
          {readiness.error ? (
            <div className="pro-readiness-alert">{readiness.error}</div>
          ) : null}
          <div className="pro-readiness-layout">
            <div className="pro-readiness-section">
              <strong>Families</strong>
              <div className="pro-readiness-family-list">
                {readinessFamilies.length > 0 ? (
                  readinessFamilies.map((family) => (
                    <div key={family.family} className="pro-readiness-family-row">
                      <span>{formatReadinessLabel(family.family)}</span>
                      <small>{family.total} assets</small>
                      <em>
                        {family.counts.working ?? 0} ready / {family.counts['metadata-only'] ?? 0} pending /{' '}
                        {countBlockedReadiness(family.counts)} blocked
                      </em>
                    </div>
                  ))
                ) : (
                  <span className="pro-readiness-empty">No readiness rows loaded.</span>
                )}
              </div>
            </div>
            <div className="pro-readiness-section">
              <strong>Needs work</strong>
              <div className="pro-readiness-issue-list">
                {readinessIssues.length > 0 ? (
                  readinessIssues.map((item) => (
                    <div key={`${item.status}-${item.id}`} className="pro-readiness-issue">
                      <div>
                        <strong>{item.label || item.id}</strong>
                        <span>{formatReadinessDetail(item)}</span>
                      </div>
                      <small className={`pro-readiness-status pro-readiness-status-${readinessStatusTone(item.status)}`}>
                        {formatReadinessLabel(item.status)}
                      </small>
                    </div>
                  ))
                ) : (
                  <span className="pro-readiness-empty">
                    {needsWorkCount > 0 ? `${needsWorkCount} rows need work.` : 'No needs-work rows loaded.'}
                  </span>
                )}
              </div>
            </div>
          </div>
        </InfoCard>
        <div className="pro-simple-tool-grid">
          {lanes.map((lane) => (
            <article key={lane.id} className="pro-simple-tool-card">
              <div className="pro-simple-tool-head">
                <span>{lane.id}</span>
                <strong>{lane.title}</strong>
              </div>
              <p>{lane.summary}</p>
              <div className="pro-tool-mini-grid">
                {lane.stats.map((stat) => (
                  <span key={stat}>{stat}</span>
                ))}
              </div>
              {lane.note ? <small className="pro-tool-note">{lane.note}</small> : null}
              <div className="pro-simple-tool-actions">
                {lane.actions.map((action, index) => (
                  <button
                    key={action.label}
                    type="button"
                    className={index === 0 ? 'pro-primary-button' : 'pro-secondary-button'}
                    onClick={action.onClick}
                    disabled={action.disabled}
                  >
                    {action.label}
                  </button>
                ))}
              </div>
            </article>
          ))}
        </div>
        <details className="pro-tool-detail-toggle">
          <summary>
            <span>Show coverage comparison</span>
            <small>{status.counts.gradioTabs} existing tabs / {status.tools.length} mapped paths</small>
          </summary>
          <div className="pro-tab-list">
            {status.gradioTabs.map((tab) => (
              <div key={tab.id} className="pro-tab-row">
                <span>{tab.group}</span>
                <strong>{tab.label}</strong>
                <small>{tab.tab ?? tab.summary}</small>
              </div>
            ))}
          </div>
        </details>
      </div>
    </section>
  )
}

function MonitorControlPanel({
  runtime,
  logStatus,
  statusMessage,
  recentOutputs,
}: {
  runtime: ProRuntimeStatus
  logStatus: ProLogStatus | null
  statusMessage: string
  recentOutputs: RecentOutput[]
}) {
  const summary = summarizeRecentOutputs(recentOutputs)
  return (
    <aside className="pro-prompt-panel" aria-label="Monitor controls">
      <PanelHeader title="Monitor" actionLabel="Monitor actions" icon={Monitor} />
      <div className="pro-workspace-stack">
        <InfoCard title="Runtime watch" subtitle="Studio status without starting training or external viewers.">
          <div className="pro-log-summary">
            <strong>{runtime.state}</strong>
            <span>{statusMessage}</span>
            <small>{logStatus?.events.length ?? 0} events loaded</small>
          </div>
        </InfoCard>
        <InfoCard title="Local scope" subtitle="Studio monitoring stays focused on generation and UI runtime health.">
          <dl className="pro-runtime-list">
            <MetricRow label="Runtime" value={runtime.backend} />
            <MetricRow label="Device" value={runtime.device} />
            <MetricRow label="Queue" value={`${runtime.queueCount} tasks`} />
            <MetricRow label="Latest output" value={summary.latestCreatedAt} />
          </dl>
        </InfoCard>
        <InfoCard title="Log coverage" subtitle="Backend-discovered files and event rows feed this view.">
          <dl className="pro-runtime-list">
            <MetricRow label="Files" value={`${logStatus?.files.length ?? 0}`} />
            <MetricRow label="Events" value={`${logStatus?.events.length ?? 0}`} />
            <MetricRow label="Resources" value={`${runtime.resources.length}`} />
          </dl>
        </InfoCard>
      </div>
    </aside>
  )
}

function MonitorWorkspace({
  runtime,
  logStatus,
  statusMessage,
  recentOutputs,
}: {
  runtime: ProRuntimeStatus
  logStatus: ProLogStatus | null
  statusMessage: string
  recentOutputs: RecentOutput[]
}) {
  const summary = summarizeRecentOutputs(recentOutputs)
  const recentEvents = (logStatus?.events ?? []).slice(0, 8)
  return (
    <section className="pro-workspace-surface" aria-label="Runtime monitor">
      <WorkspaceHeader
        eyebrow="Monitor"
        title="Studio runtime monitor"
        description="Live app status, resource meters, queue state, and recent receipts for QA without opening another tool."
      />
      <div className="pro-workspace-grid">
        <InfoCard title="System pulse" subtitle="Fast read on whether the Studio shell and generation backend are healthy.">
          <div className="pro-stat-grid">
            <StatTile label="State" value={runtime.state} hint="current runtime" />
            <StatTile label="Backend" value={runtime.backend} hint="app runtime" />
            <StatTile label="Queue" value={`${runtime.queueCount}`} hint="tasks waiting" />
            <StatTile label="Events" value={`${logStatus?.events.length ?? 0}`} hint="loaded rows" />
          </div>
        </InfoCard>
        <InfoCard title="Resource meters" subtitle="Display-only local machine state. GPU stays reserved for generation.">
          <div className="pro-resource-stack">
            {runtime.resources.map((metric) => (
              <ResourceBar key={metric.label} metric={metric} />
            ))}
          </div>
        </InfoCard>
        <InfoCard title="Operator notes" subtitle="The monitor is intentionally scoped to Studio QA and generation readiness.">
          <div className="pro-settings-columns">
            <div className="pro-settings-note">
              <strong>Current status</strong>
              <span>{statusMessage}</span>
            </div>
            <div className="pro-settings-note">
              <strong>Loaded model</strong>
              <span>{runtime.loadedModel.name || 'Unavailable'}</span>
            </div>
            <div className="pro-settings-note">
              <strong>Latest receipt</strong>
              <span>{summary.latestCreatedAt}</span>
            </div>
          </div>
        </InfoCard>
        <InfoCard title="Recent monitor events" subtitle="Latest runtime rows from backend logs and generation receipts.">
          {recentEvents.length > 0 ? (
            <div className="pro-log-table">
              {recentEvents.map((event) => (
                <article key={event.id} className="pro-log-row">
                  <span>{event.source}</span>
                  <strong>{event.title}</strong>
                  <p>{event.detail}</p>
                </article>
              ))}
            </div>
          ) : (
            <div className="pro-empty-preview">
              <Monitor size={42} aria-hidden="true" />
              <strong>No monitor events loaded yet</strong>
              <span>Run a generation or open Logs to inspect backend file discovery.</span>
            </div>
          )}
        </InfoCard>
      </div>
    </section>
  )
}

function LogsControlPanel({
  runtime,
  logStatus,
  statusMessage,
  recentOutputs,
}: {
  runtime: ProRuntimeStatus
  logStatus: ProLogStatus | null
  statusMessage: string
  recentOutputs: RecentOutput[]
}) {
  const summary = summarizeRecentOutputs(recentOutputs)
  return (
    <aside className="pro-prompt-panel" aria-label="Log controls">
      <PanelHeader title="Logs" actionLabel="Log view" icon={FileImage} />
      <div className="pro-workspace-stack">
        <InfoCard title="Stream status" subtitle="Operational feedback now has a dedicated home instead of living only in the canvas footer.">
          <div className="pro-log-summary">
            <strong>{runtime.state}</strong>
            <span>{statusMessage}</span>
            <small>{logStatus?.events.length ?? recentOutputs.length} events loaded</small>
          </div>
        </InfoCard>
        <InfoCard title="Operator summary" subtitle="Enough live state to judge whether the shell is healthy before deeper logging endpoints exist.">
          <dl className="pro-runtime-list">
            <MetricRow label="Backend" value={runtime.backend} />
            <MetricRow label="Queue" value={`${runtime.queueCount} tasks`} />
            <MetricRow label="Loaded model" value={runtime.loadedModel.name || 'Unavailable'} />
            <MetricRow label="Latest receipt" value={summary.latestCreatedAt} />
            <MetricRow label="Log files" value={`${logStatus?.files.length ?? 0}`} />
          </dl>
        </InfoCard>
      </div>
    </aside>
  )
}

function LogsWorkspace({
  runtime,
  logStatus,
  statusMessage,
  generationError,
  recentOutputs,
  selectedModelName,
  generationProgress,
}: {
  runtime: ProRuntimeStatus
  logStatus: ProLogStatus | null
  statusMessage: string
  generationError: string
  recentOutputs: RecentOutput[]
  selectedModelName: string
  generationProgress: GenerationProgressEvent[]
}) {
  const currentErrorRows = generationError
    ? [{
        id: 'current-generation-error',
        title: 'Current generation error',
        detail: generationError,
        meta: runtime.state,
      }]
    : []
  const backendRows = logStatus?.events.length
    ? logStatus.events.map((event) => ({
        id: event.id,
        title: event.title,
        detail: event.detail,
        meta: event.time || event.source,
      }))
    : buildLogRows(runtime, statusMessage, selectedModelName, recentOutputs, generationProgress)
  const logRows = [...currentErrorRows, ...backendRows]
  const summary = summarizeRecentOutputs(recentOutputs)
  return (
    <section className="pro-workspace-surface" aria-label="Runtime logs">
      <WorkspaceHeader
        eyebrow="Logs"
        title="Operational receipts"
        description="The rail now opens a page built for scrolling and later backend log wiring."
      />
      <div className="pro-workspace-grid">
        <InfoCard title="Runtime receipt" subtitle="Keep the current backend state visible at the top of the log surface.">
          <div className="pro-stat-grid">
            <StatTile label="State" value={runtime.state} hint="current status" />
            <StatTile label="Queue" value={`${runtime.queueCount}`} hint="tasks waiting" />
            <StatTile label="Device" value={runtime.device} hint="execution target" />
            <StatTile label="Latest receipt" value={summary.latestCreatedAt} hint="artifact time" />
          </div>
        </InfoCard>
        <InfoCard title="Resource snapshot" subtitle="These are the same runtime metrics, but presented in the log workspace where operators expect them.">
          <div className="pro-resource-stack">
            {runtime.resources.map((metric) => (
              <ResourceBar key={metric.label} metric={metric} />
            ))}
          </div>
        </InfoCard>
        <InfoCard title="Log files" subtitle="Backend-discovered log and JSONL files from the local output directory.">
          <div className="pro-token-grid">
            {(logStatus?.files ?? []).map((file) => (
              <div key={file.path || file.name} className="pro-token-card">
                <strong>{file.name}</strong>
                <span>{formatBytes(file.sizeBytes)} / {formatDisplayDate(file.modifiedAt)}</span>
              </div>
            ))}
            {logStatus?.files.length === 0 ? (
              <div className="pro-token-card">
                <strong>No log files yet</strong>
                <span>Client event and generation logs will appear here after activity.</span>
              </div>
            ) : null}
          </div>
        </InfoCard>
        <InfoCard title="Event stream" subtitle="Synthetic rows for now, shaped like the real monitor table this shell needs.">
          <div className="pro-log-table">
            {logRows.map((row) => (
              <article key={row.id} className="pro-log-row">
                <div>
                  <strong>{row.title}</strong>
                  <span>{row.detail}</span>
                </div>
                <small>{row.meta}</small>
              </article>
            ))}
          </div>
        </InfoCard>
      </div>
    </section>
  )
}

function SettingsControlPanel({
  bootstrap,
  runtime,
  settings,
  settingsStatus,
  recentOutputs,
  leftPanelWidth,
  rightPanelWidth,
  bottomDockHeight,
  bottomDockVisible,
  showAdvanced,
  onBottomDockVisibleChange,
  onShowAdvancedChange,
  onLayoutReset,
}: {
  bootstrap: ProBootstrap
  runtime: ProRuntimeStatus
  settings: GenerationSettings
  settingsStatus: ProSettingsStatus | null
  recentOutputs: RecentOutput[]
  leftPanelWidth: number
  rightPanelWidth: number
  bottomDockHeight: number
  bottomDockVisible: boolean
  showAdvanced: boolean
  onBottomDockVisibleChange: (value: boolean) => void
  onShowAdvancedChange: (value: boolean) => void
  onLayoutReset: () => void
}) {
  const summary = summarizeRecentOutputs(recentOutputs)
  return (
    <aside className="pro-prompt-panel" aria-label="Workspace settings controls">
      <PanelHeader title="Settings" actionLabel="Workspace settings" icon={Settings} />
      <div className="pro-workspace-stack">
        <InfoCard title="Shell preferences" subtitle="Frequent visibility toggles stay reachable without clipping.">
          <label className="pro-toggle">
            <input
              type="checkbox"
              checked={bottomDockVisible}
              onChange={(event) => onBottomDockVisibleChange(event.target.checked)}
            />
            <span>Bottom dock visible</span>
          </label>
          <label className="pro-toggle">
            <input
              type="checkbox"
              checked={showAdvanced}
              onChange={(event) => onShowAdvancedChange(event.target.checked)}
            />
            <span>Secondary controls expanded</span>
          </label>
          <button type="button" className="pro-secondary-button" onClick={onLayoutReset}>
            Reset saved layout
          </button>
        </InfoCard>
        <InfoCard title="Current working set" subtitle="The settings page reflects live shell state instead of a placeholder.">
          <dl className="pro-runtime-list">
            <MetricRow label="Workspace" value={bootstrap.workspaceName} />
            <MetricRow label="Mode" value={settings.mode} />
            <MetricRow label="Backend" value={runtime.backend} />
            <MetricRow label="Left panel" value={`${leftPanelWidth}px`} />
            <MetricRow label="Right panel" value={`${rightPanelWidth}px`} />
            <MetricRow label="Bottom dock" value={`${bottomDockHeight}px`} />
            <MetricRow label="Settings file" value={settingsStatus?.paths.settings || 'Backend not connected'} />
          </dl>
        </InfoCard>
        <InfoCard title="Session scope" subtitle="Show what the shell is actually carrying right now before persistence is expanded.">
          <dl className="pro-runtime-list">
            <MetricRow label="Loaded model" value={runtime.loadedModel.name || settings.modelId} />
            <MetricRow label="Recent receipts" value={`${recentOutputs.length}`} />
            <MetricRow label="Unique models" value={`${summary.uniqueModels}`} />
            <MetricRow label="Persistence key" value={LAYOUT_STORAGE_KEY} />
            <MetricRow label="Output path" value={settingsStatus?.paths.outputs || 'Unknown'} />
          </dl>
        </InfoCard>
      </div>
    </aside>
  )
}

function SettingsWorkspace({
  bootstrap,
  runtime,
  settings,
  settingsStatus,
  recentOutputs,
  leftPanelWidth,
  rightPanelWidth,
  bottomDockHeight,
  bottomDockVisible,
  showAdvanced,
}: {
  bootstrap: ProBootstrap
  runtime: ProRuntimeStatus
  settings: GenerationSettings
  settingsStatus: ProSettingsStatus | null
  recentOutputs: RecentOutput[]
  leftPanelWidth: number
  rightPanelWidth: number
  bottomDockHeight: number
  bottomDockVisible: boolean
  showAdvanced: boolean
}) {
  const summary = summarizeRecentOutputs(recentOutputs)
  return (
    <section className="pro-workspace-surface" aria-label="Workspace settings">
      <WorkspaceHeader
        eyebrow="Settings"
        title="Planned shell preferences"
        description="This turns the settings rail into a real control surface with scrollable long-form sections."
      />
      <div className="pro-workspace-grid">
        <InfoCard title="Workspace defaults" subtitle="Read-only summary until these controls are fully wired to backend persistence.">
          <div className="pro-stat-grid">
            <StatTile label="Default model" value={settings.modelId} hint="current selection" />
            <StatTile label="Sampler" value={settingsStatus?.generationDefaults.sampler ?? settings.sampler} hint="saved default" />
            <StatTile label="Resolution" value={`${settingsStatus?.generationDefaults.width ?? settings.width}x${settingsStatus?.generationDefaults.height ?? settings.height}`} hint="saved default" />
            <StatTile label="Version" value={bootstrap.version} hint="shell build" />
          </div>
        </InfoCard>
        <InfoCard title="Backend paths" subtitle="Real paths reported by the Pro API for tonight's QA pass.">
          <dl className="pro-runtime-list">
            <MetricRow label="Config" value={settingsStatus?.paths.settings || 'Unavailable'} />
            <MetricRow label="Launch profile" value={settingsStatus?.paths.launch || 'Unavailable'} />
            <MetricRow label="Models" value={settingsStatus?.paths.models || 'Unavailable'} />
            <MetricRow label="Checkpoints" value={settingsStatus?.paths.checkpoints || 'Unavailable'} />
            <MetricRow label="Outputs" value={settingsStatus?.paths.outputs || 'Unavailable'} />
          </dl>
        </InfoCard>
        <InfoCard title="Layout memory" subtitle="Local shell layout is already persisted; this page makes those values obvious.">
          <dl className="pro-runtime-list">
            <MetricRow label="Left panel width" value={`${leftPanelWidth}px`} />
            <MetricRow label="Right panel width" value={`${rightPanelWidth}px`} />
            <MetricRow label="Bottom dock height" value={`${bottomDockHeight}px`} />
            <MetricRow label="Bottom dock visible" value={bottomDockVisible ? 'Yes' : 'No'} />
            <MetricRow label="Advanced panel open" value={showAdvanced ? 'Yes' : 'No'} />
          </dl>
        </InfoCard>
        <InfoCard title="Local persistence boundary" subtitle="Distinguish between what is saved in the browser and what is just current runtime/bootstrap state.">
          <div className="pro-settings-columns">
            <div className="pro-settings-column">
              <strong>Saved locally</strong>
              <ul className="pro-bullet-list">
                <li>Left and right panel widths.</li>
                <li>Bottom dock visibility and saved dock height.</li>
                <li>The shell layout record under `{LAYOUT_STORAGE_KEY}`.</li>
              </ul>
            </div>
            <div className="pro-settings-column">
              <strong>Live session/runtime</strong>
              <ul className="pro-bullet-list">
                <li>Current backend status, queue count, and loaded model.</li>
                <li>Current prompt route defaults from bootstrap/runtime.</li>
                <li>Recent local receipts shown in the data and logs pages.</li>
              </ul>
            </div>
          </div>
        </InfoCard>
        <InfoCard title="Workspace inventory" subtitle="The settings page now doubles as a concise shell inventory.">
          <div className="pro-stat-grid">
            <StatTile label="Models" value={`${bootstrap.models.length}`} hint="available routes" />
            <StatTile label="Samplers" value={`${bootstrap.samplers.length}`} hint="loaded options" />
            <StatTile label="Receipts" value={`${recentOutputs.length}`} hint="local artifacts" />
            <StatTile label="Latest receipt" value={summary.latestCreatedAt} hint="recent output" />
          </div>
        </InfoCard>
        <InfoCard title="Runtime policy" subtitle="CPU-side surface effects are fine here; GPU cycles stay reserved for model execution.">
          <ul className="pro-bullet-list">
            <li>Use CSS glow and pulse on threshold classes rather than shader effects.</li>
            <li>Keep timers and loading bars driven from sampled runtime state updates.</li>
            <li>Bound graph refresh cadence so telemetry stays light during runs.</li>
            <li>Runtime currently reports {runtime.device} with {runtime.precision} execution.</li>
          </ul>
        </InfoCard>
      </div>
    </section>
  )
}

function CanvasPreview({
  activeMode,
  preview,
  statusMessage,
  width,
  height,
  onOpenSegmentation,
  onOpenHires,
  onOpenReactor,
  bottomDockVisible,
  onToggleBottomDock,
}: {
  activeMode: ProMode
  preview: RecentOutput | null
  statusMessage: string
  width: number
  height: number
  onOpenSegmentation: () => void
  onOpenHires: () => void
  onOpenReactor: () => void
  bottomDockVisible: boolean
  onToggleBottomDock: () => void
}) {
  const aspectRatio = `${Math.max(1, width)} / ${Math.max(1, height)}`
  const previewStageRef = useRef<HTMLDivElement>(null)
  const [previewFrameSize, setPreviewFrameSize] = useState({ width: 0, height: 0 })
  const previewIsVideo = preview?.mode === 'video'
  useEffect(() => {
    const stage = previewStageRef.current
    if (!stage) {
      return
    }

    const updateFrameSize = () => {
      const style = getComputedStyle(stage)
      const horizontalPadding = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight)
      const verticalPadding = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom)
      const availableWidth = Math.max(1, stage.clientWidth - horizontalPadding)
      const availableHeight = Math.max(1, stage.clientHeight - verticalPadding)
      const targetAspect = Math.max(1, width) / Math.max(1, height)
      let nextWidth = Math.min(availableWidth, 1040)
      let nextHeight = nextWidth / targetAspect

      if (nextHeight > availableHeight) {
        nextHeight = availableHeight
        nextWidth = nextHeight * targetAspect
      }

      setPreviewFrameSize((current) =>
        Math.abs(current.width - nextWidth) < 1 && Math.abs(current.height - nextHeight) < 1
          ? current
          : { width: nextWidth, height: nextHeight },
      )
    }

    updateFrameSize()

    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', updateFrameSize)
      return () => window.removeEventListener('resize', updateFrameSize)
    }

    const observer = new ResizeObserver(updateFrameSize)
    observer.observe(stage)
    return () => observer.disconnect()
  }, [height, width])

  const outputFrameStyle = useMemo<CSSProperties>(() => {
    if (previewFrameSize.width <= 0 || previewFrameSize.height <= 0) {
      return { aspectRatio }
    }
    return {
      aspectRatio,
      width: `${previewFrameSize.width}px`,
      height: `${previewFrameSize.height}px`,
    }
  }, [aspectRatio, previewFrameSize.height, previewFrameSize.width])

  return (
    <section className="pro-canvas" aria-label="Canvas and output preview">
      <div className="pro-canvas-header">
        <div className="pro-canvas-title">
          <strong>Canvas</strong>
          <small>{width}x{height}</small>
        </div>
        <div className="pro-canvas-tools" aria-label="Canvas tools">
          <button type="button" className="pro-tool-chip" onClick={onOpenSegmentation}>
            <ScanSearch size={14} aria-hidden="true" />
            <span>Segment</span>
          </button>
          <button type="button" className="pro-tool-chip" onClick={onOpenHires}>
            <Highlighter size={14} aria-hidden="true" />
            <span>Hi-res</span>
          </button>
          <button type="button" className="pro-tool-chip" onClick={onOpenReactor}>
            <Wand2 size={14} aria-hidden="true" />
            <span>ReActor</span>
          </button>
          <button type="button" className="pro-icon-button" aria-label="Pan preview">
            <Hand size={16} aria-hidden="true" />
          </button>
          <button type="button" className="pro-icon-button" aria-label="Fit preview">
            <Maximize2 size={16} aria-hidden="true" />
          </button>
          <button type="button" className="pro-zoom-button">100%</button>
          <button type="button" className="pro-tool-chip" onClick={onToggleBottomDock}>
            {bottomDockVisible ? <Rows3 size={14} aria-hidden="true" /> : <Layers2 size={14} aria-hidden="true" />}
            <span>{bottomDockVisible ? 'Hide dock' : 'Show dock'}</span>
          </button>
        </div>
      </div>

      <div className="pro-preview-stage" ref={previewStageRef}>
        <div className="pro-output-frame" style={outputFrameStyle}>
          {previewIsVideo ? (
            <video src={preview.url} controls playsInline />
          ) : preview ? (
            <img src={preview.url} alt={preview.prompt} />
          ) : (
            <div className="pro-empty-preview pro-stage-empty">
              <Monitor size={42} aria-hidden="true" />
              <strong>{activeMode === 'video' ? 'Video preview' : 'Image preview'}</strong>
              <span>Ready for the next render.</span>
            </div>
          )}
        </div>
      </div>

      <div className="pro-canvas-footer">
        <span>{width}x{height}</span>
        <span>{preview?.modelName ?? 'Local model'}</span>
        <span>{statusMessage}</span>
      </div>
    </section>
  )
}

function BottomDock({
  visible,
  height,
  recentOutputs,
  statusMessage,
  generationError,
  selectedModelName,
  onPreviewSelect,
  onResizeStart,
  onToggleVisible,
}: {
  visible: boolean
  height: number
  recentOutputs: RecentOutput[]
  statusMessage: string
  generationError: string
  selectedModelName: string
  onPreviewSelect: (value: RecentOutput) => void
  onResizeStart: (event: ReactMouseEvent<HTMLButtonElement>) => void
  onToggleVisible: () => void
}) {
  return (
    <div className={visible ? 'pro-bottom-dock' : 'pro-bottom-dock pro-bottom-dock-hidden'} style={{ height }}>
      <button
        type="button"
        className="pro-bottom-resize"
        onMouseDown={onResizeStart}
        aria-label="Resize bottom dock"
      />
      <div className="pro-bottom-header">
        <div>
          <strong>Output dock</strong>
          <span>{selectedModelName}</span>
        </div>
        <button type="button" className="pro-icon-button" onClick={onToggleVisible} aria-label="Hide bottom dock">
          <X size={16} aria-hidden="true" />
        </button>
      </div>
      <div className="pro-bottom-body">
        <div className="pro-bottom-status-card" data-error={Boolean(generationError)}>
          <span>{generationError ? 'Generation error' : 'Status'}</span>
          <strong>{generationError || statusMessage}</strong>
        </div>
        <div className="pro-bottom-gallery">
          {recentOutputs.map((item) => (
            <button
              key={item.id}
              type="button"
              className="pro-bottom-thumb"
              onClick={() => onPreviewSelect(item)}
              title={item.prompt}
            >
              <OutputMedia item={item} />
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

function RuntimePanel({
  runtime,
  selectedModelName,
}: {
  runtime: ProRuntimeStatus
  selectedModelName: string
}) {
  const loadedModelName = runtime.loadedModel.loaded ? runtime.loadedModel.name : 'No model loaded'
  return (
    <aside className="pro-status-panel" aria-label="Runtime status">
      <div className="pro-status-heading">
        <span>System</span>
        <strong>
          <span className="pro-status-dot" aria-hidden="true" />
          {runtime.state}
        </strong>
      </div>

      <div className="pro-signal-grid">
        <div className="pro-signal-card">
          <span>Workspace</span>
          <strong>Local</strong>
        </div>
        <div className="pro-signal-card">
          <span>Control</span>
          <strong>Manual</strong>
        </div>
        <div className="pro-signal-card">
          <span>History</span>
          <strong>On</strong>
        </div>
      </div>

      <dl className="pro-runtime-list">
        <MetricRow label="Backend" value={runtime.backend} />
        <MetricRow label="Device" value={runtime.device} />
        <MetricRow label="Precision" value={runtime.precision} />
        <MetricRow label="Attention" value={runtime.attention} />
        <MetricRow label="Max resolution" value={runtime.maxResolution} />
      </dl>

      <div className="pro-resource-section">
        <span className="pro-section-label">Resource usage</span>
        {runtime.resources.map((metric) => (
          <ResourceBar key={metric.label} metric={metric} />
        ))}
      </div>

      <div className="pro-loaded-model">
        <span className="pro-section-label">Loaded model</span>
        <div className="pro-loaded-model-title">
          <strong>{loadedModelName}</strong>
          <span>{runtime.loadedModel.loaded ? 'Loaded' : 'Loads on generate'}</span>
        </div>
        <div className="pro-loaded-model-banner">
          <Cpu size={14} aria-hidden="true" />
          <span>{runtime.backend}</span>
          <small>{runtime.attention}</small>
        </div>
        <dl className="pro-runtime-list">
          <MetricRow label="Selected model" value={selectedModelName} />
          <MetricRow label="Type" value={runtime.loadedModel.type} />
          <MetricRow label="Base model" value={runtime.loadedModel.baseModel} />
          <MetricRow label="Size on disk" value={runtime.loadedModel.sizeOnDisk} />
          <MetricRow label="Precision" value={runtime.loadedModel.precision} />
          <MetricRow label="VAE" value={runtime.loadedModel.vae} />
          <MetricRow label="Text encoder" value={runtime.loadedModel.textEncoder} />
          <MetricRow label="UNet" value={runtime.loadedModel.unet} />
        </dl>
        <button type="button" className="pro-unload-button" disabled>Unload model</button>
      </div>

      <div className="pro-queue-row">
        <span>Queue</span>
        <strong>{runtime.queueCount} tasks</strong>
      </div>
    </aside>
  )
}

function buildEngineFilterOptions(engines: EngineSummary[]): Array<{ value: EngineId; label: string }> {
  return [
    { value: 'all', label: 'All engines' },
    ...engines.map((engine) => ({
      value: engine.id,
      label: `${engine.label} (${engine.count})`,
    })),
  ]
}

function matchesEngineFilter(model: ProModelOption, filter: EngineId): boolean {
  if (filter === 'all') {
    return true
  }
  if (model.engineId) {
    return model.engineId === filter
  }
  const architecture = `${model.architecture ?? ''} ${model.name ?? ''} ${model.id}`.toLowerCase()
  switch (filter) {
    case 'flux':
      return architecture.includes('flux') && !architecture.includes('flux2') && !architecture.includes('klein')
    case 'flux2':
      return architecture.includes('flux2') || architecture.includes('flux.2') || architecture.includes('klein')
    case 'sana_video':
      return architecture.includes('sana') && architecture.includes('video')
    case 'sd15':
      return architecture.includes('sd15') || architecture.includes('sd1.5') || architecture.includes('stable diffusion 1.5')
    case 'sdxl':
      return architecture.includes('sdxl') || architecture.includes('stable diffusion xl')
    case 'sd35':
      return architecture.includes('sd35') || architecture.includes('sd3.5') || architecture.includes('stable diffusion 3.5')
    case 'zimage':
      return architecture.includes('z-image') || architecture.includes('z image') || architecture.includes('zimage')
    default:
      return true
  }
}

function formatModelOptionLabel(model: ProModelOption): string {
  if (model.assetSummary && !model.name.includes(model.assetSummary)) {
    return `${model.name} (${model.assetSummary})`
  }
  return model.name
}

function groupModelsByEngine(models: ProModelOption[], engines: EngineSummary[]) {
  const labels = new Map<EngineId, string>()
  labels.set('unknown', 'Other')
  for (const engine of engines) {
    labels.set(engine.id, engine.label)
  }
  const groups = new Map<string, { id: EngineId; label: string; models: ProModelOption[] }>()
  for (const model of models) {
    const id = (model.engineId ?? 'unknown') as EngineId
    const existing = groups.get(id)
    if (existing) {
      existing.models.push(model)
    } else {
      groups.set(id, {
        id,
        label: model.engineLabel ?? labels.get(id) ?? 'Other',
        models: [model],
      })
    }
  }
  return Array.from(groups.values()).sort((left, right) => left.label.localeCompare(right.label))
}

function summarizeDownloads(downloadsStatus: ProDownloadsStatus | null, engineFilter: EngineId) {
  const routeLabel = engineFilter === 'all' ? 'all engines' : engineFilter
  if (!downloadsStatus) {
    return {
      subtitle: 'Waiting for /api/pro/downloads.',
      total: 0,
      installed: 0,
      routeTotal: 0,
      routeLabel,
      items: [] as ProDownloadsStatus['catalog'],
    }
  }
  const routeItems = downloadsStatus.catalog.filter(
    (item) => engineFilter === 'all' || item.engineId === engineFilter,
  )
  const items = routeItems
    .slice()
    .sort((left, right) => Number(right.installed) - Number(left.installed) || left.title.localeCompare(right.title))
    .slice(0, 8)

  return {
    subtitle: 'Local install state from the guarded model download service.',
    total: downloadsStatus.counts.catalog,
    installed: downloadsStatus.counts.installed,
    routeTotal: routeItems.length,
    routeLabel,
    items,
  }
}

function PanelHeader({
  title,
  actionLabel,
  icon: Icon,
}: {
  title: string
  actionLabel: string
  icon: LucideIcon
}) {
  return (
    <div className="pro-panel-header">
      <span>{title}</span>
      <button type="button" className="pro-icon-button" aria-label={actionLabel}>
        <Icon size={15} aria-hidden="true" />
      </button>
    </div>
  )
}

function ResizeHandle({
  axis,
  label,
  onMouseDown,
}: {
  axis: 'vertical' | 'horizontal'
  label: string
  onMouseDown: () => void
}) {
  return (
    <button
      type="button"
      className={axis === 'vertical' ? 'pro-resize-handle pro-resize-handle-vertical' : 'pro-resize-handle'}
      aria-label={label}
      onMouseDown={onMouseDown}
    />
  )
}

function RailButton({
  item,
  active,
  onSelect,
}: {
  item: IconItem<string>
  active: boolean
  onSelect: (id: string) => void
}) {
  const Icon = item.icon
  return (
    <button
      type="button"
      className={active ? 'pro-rail-button pro-rail-button-active' : 'pro-rail-button'}
      aria-pressed={active}
      onClick={() => onSelect(item.id)}
    >
      <Icon size={18} aria-hidden="true" />
      <span>{item.label}</span>
    </button>
  )
}

function RangeField({
  label,
  tooltip,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string
  tooltip?: string
  min: number
  max: number
  step: number
  value: number
  onChange: (value: number) => void
}) {
  return (
    <label className="pro-range-field">
      <FieldLabel label={label} tooltip={tooltip} compact />
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <output>{value}</output>
    </label>
  )
}

function FieldLabel({
  label,
  tooltip,
  compact = false,
}: {
  label: string
  tooltip?: string
  compact?: boolean
}) {
  return (
    <span className={compact ? 'pro-field-label pro-field-label-compact' : 'pro-field-label'}>
      <span>{label}</span>
      {tooltip ? (
        <TooltipBadge label={label} text={tooltip} />
      ) : null}
    </span>
  )
}

function TooltipBadge({ label, text }: { label: string; text: string }) {
  return (
    <span className="pro-tooltip">
      <button type="button" className="pro-tooltip-button" aria-label={`${label} help`}>
        <CircleHelp size={13} aria-hidden="true" />
      </button>
      <span className="pro-tooltip-bubble" role="tooltip">
        {text}
      </span>
    </span>
  )
}

function ToolModal({
  open,
  title,
  children,
  onClose,
}: {
  open: boolean
  title: string
  children: ReactNode
  onClose: () => void
}) {
  if (!open) {
    return null
  }
  return (
    <div className="pro-modal-backdrop" onClick={onClose}>
      <div className="pro-modal" onClick={(event) => event.stopPropagation()}>
        <div className="pro-modal-header">
          <h2 className="pro-modal-title">{title}</h2>
          <button type="button" className="pro-icon-button" onClick={onClose} aria-label={`Close ${title}`}>
            <X size={16} aria-hidden="true" />
          </button>
        </div>
        <div className="pro-modal-body">{children}</div>
      </div>
    </div>
  )
}

function ResourceBar({ metric }: { metric: ResourceMetric }) {
  return (
    <div className="pro-resource-meter">
      <div className="pro-resource-labels">
        <span>{metric.label}</span>
        <strong>{metric.value}</strong>
        <small>{metric.percent}%</small>
      </div>
      <div
        className={`pro-meter pro-meter-${metric.tone}`}
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={metric.percent}
      >
        <span style={{ width: `${metric.percent}%` }} />
      </div>
    </div>
  )
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="pro-metric-row">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}

function countBlockedReadiness(counts: ProReadinessStatus['counts']): number {
  return (
    (counts['blocked-cleanly'] ?? 0) +
    (counts['broken-runtime'] ?? 0) +
    (counts['unsupported-no-route'] ?? 0)
  )
}

function formatReadinessLabel(value: string): string {
  return value
    .split(/[-_./]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function readinessStatusTone(status: string): string {
  if (status === 'working') {
    return 'ready'
  }
  if (status === 'metadata-only') {
    return 'pending'
  }
  return 'blocked'
}

function formatReadinessDetail(item: ProReadinessItem): string {
  return truncateText(
    item.suggestedAction || item.reason || item.smokeCommand || item.route || 'No readiness note.',
    116,
  )
}

function WorkspaceHeader({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string
  title: string
  description: string
}) {
  return (
    <header className="pro-workspace-header">
      <span>{eyebrow}</span>
      <div>
        <strong>{title}</strong>
        <p>{description}</p>
      </div>
    </header>
  )
}

function InfoCard({
  title,
  subtitle,
  children,
}: {
  title: string
  subtitle: string
  children: ReactNode
}) {
  return (
    <section className="pro-info-card">
      <div className="pro-info-card-header">
        <strong>{title}</strong>
        <span>{subtitle}</span>
      </div>
      <div className="pro-info-card-body">{children}</div>
    </section>
  )
}

function StatTile({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="pro-stat-tile">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{hint}</small>
    </div>
  )
}

function summarizeRecentOutputs(recentOutputs: RecentOutput[]) {
  const uniqueModels = new Set(
    recentOutputs.map((item) => item.modelName || item.mode).filter((value) => value.length > 0),
  ).size
  return {
    uniqueModels,
    latestCreatedAt: formatDisplayDate(recentOutputs[0]?.createdAt || ''),
  }
}

function mergeRecentOutputs(nextOutputs: RecentOutput[], currentOutputs: RecentOutput[]) {
  const seen = new Set<string>()
  const merged: RecentOutput[] = []
  for (const item of [...nextOutputs, ...currentOutputs]) {
    const key = item.path || item.id || item.url
    if (seen.has(key)) {
      continue
    }
    seen.add(key)
    merged.push(item)
    if (merged.length >= 8) {
      break
    }
  }
  return merged
}

function buildOutputModelBuckets(recentOutputs: RecentOutput[], fallbackModelName: string) {
  const counts = new Map<string, number>()
  for (const item of recentOutputs) {
    const label = item.modelName || fallbackModelName
    counts.set(label, (counts.get(label) ?? 0) + 1)
  }
  return Array.from(counts.entries())
    .map(([label, count]) => ({ label, count }))
    .sort((left, right) => right.count - left.count || left.label.localeCompare(right.label))
}

function buildAspectBuckets(recentOutputs: RecentOutput[]) {
  const counts = new Map<string, number>()
  for (const item of recentOutputs) {
    const label = `${item.width}x${item.height}`
    counts.set(label, (counts.get(label) ?? 0) + 1)
  }
  return Array.from(counts.entries())
    .map(([label, count]) => ({ label, count }))
    .sort((left, right) => right.count - left.count || left.label.localeCompare(right.label))
}

function truncateText(value: string, maxLength: number) {
  if (value.length <= maxLength) {
    return value
  }
  return `${value.slice(0, Math.max(0, maxLength - 3)).trimEnd()}...`
}

function formatDisplayDate(value: string) {
  if (!value) {
    return 'No receipts'
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

function formatBytes(value: number) {
  if (!Number.isFinite(value) || value <= 0) {
    return '0 B'
  }
  const units = ['B', 'KB', 'MB', 'GB']
  let nextValue = value
  let unitIndex = 0
  while (nextValue >= 1024 && unitIndex < units.length - 1) {
    nextValue /= 1024
    unitIndex += 1
  }
  return `${nextValue >= 10 || unitIndex === 0 ? nextValue.toFixed(0) : nextValue.toFixed(1)} ${units[unitIndex]}`
}

function countOutputsByMode(recentOutputs: RecentOutput[]) {
  return recentOutputs.reduce(
    (counts, item) => {
      counts[item.mode] += 1
      return counts
    },
    { image: 0, video: 0, inpaint: 0 } as Record<CreationMode, number>,
  )
}

function buildLogRows(
  runtime: ProRuntimeStatus,
  statusMessage: string,
  selectedModelName: string,
  recentOutputs: RecentOutput[],
  generationProgress: GenerationProgressEvent[],
) {
  const progressRows = generationProgress.slice(-6).reverse().map((event, index) => ({
    id: `generation-progress-${index}-${event.stage}-${event.step}`,
    title: event.stage || 'Generation progress',
    detail: event.message || `${Math.round(event.progress * 100)}%`,
    meta: event.total ? `${event.step}/${event.total} steps` : `${Math.round(event.progress * 100)}%`,
  }))
  const runtimeRows = runtime.resources.map((metric, index) => ({
    id: `metric-${metric.label}-${index}`,
    title: `${metric.label} monitor`,
    detail: `${metric.value} at ${metric.percent}%`,
    meta: runtime.state,
  }))
  const outputRows = recentOutputs.map((item) => ({
    id: item.id,
    title: item.modelName ?? selectedModelName,
    detail: item.prompt,
    meta: item.createdAt,
  }))

  return [
    {
      id: 'status',
      title: 'Workspace status',
      detail: statusMessage,
      meta: runtime.backend,
    },
    {
      id: 'runtime-job',
      title: runtime.job.state === 'idle' ? 'Generation job' : `Generation ${runtime.job.state}`,
      detail: runtime.job.message || 'No active generation job.',
      meta: runtime.job.totalSteps ? `${runtime.job.step}/${runtime.job.totalSteps} steps` : `${runtime.job.progress}%`,
    },
    {
      id: 'queue',
      title: 'Queue depth',
      detail: `${runtime.queueCount} tasks waiting`,
      meta: runtime.device,
    },
    ...progressRows,
    ...runtimeRows,
    ...outputRows,
  ]
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) {
    return 0
  }
  return Math.max(0, Math.min(100, Math.round(value)))
}

function isRuntimeJobActive(job: ProRuntimeStatus['job']): boolean {
  const state = job.state.toLowerCase()
  return state !== 'idle' && state !== 'completed' && state !== 'failed' && state !== 'cancelled' && state !== 'canceled'
}

function mergeBootstrapDefaults(
  current: GenerationSettings,
  nextBootstrap: ProBootstrap,
): GenerationSettings {
  const modelStillExists = nextBootstrap.models.some((model) => model.id === current.modelId)
  const samplerStillExists = nextBootstrap.samplers.includes(current.sampler)
  const ratioStillExists = nextBootstrap.aspectRatios.some((ratio) => ratio.id === current.aspectRatioId)
  const ratio = ratioStillExists
    ? nextBootstrap.aspectRatios.find((item) => item.id === current.aspectRatioId)
    : nextBootstrap.aspectRatios.find((item) => item.id === nextBootstrap.defaults.aspectRatioId)

  return {
    ...current,
    modelId: modelStillExists ? current.modelId : nextBootstrap.defaults.modelId,
    sampler: samplerStillExists ? current.sampler : nextBootstrap.defaults.sampler,
    aspectRatioId: ratio?.id ?? nextBootstrap.defaults.aspectRatioId,
    width: ratio?.width ?? current.width,
    height: ratio?.height ?? current.height,
  }
}

function settingsMatch(current: GenerationSettings, expected: GenerationSettings): boolean {
  return (
    current.mode === expected.mode &&
    current.prompt === expected.prompt &&
    current.negativePrompt === expected.negativePrompt &&
    current.modelId === expected.modelId &&
    current.aspectRatioId === expected.aspectRatioId &&
    current.width === expected.width &&
    current.height === expected.height &&
    current.steps === expected.steps &&
    current.cfgScale === expected.cfgScale &&
    current.sampler === expected.sampler &&
    current.scheduler === expected.scheduler &&
    current.seed === expected.seed &&
    current.batchSize === expected.batchSize &&
    current.batchCount === expected.batchCount &&
    current.sourceImageDataUrl === expected.sourceImageDataUrl &&
    current.sourceImageName === expected.sourceImageName
  )
}

function isCreationMode(mode: ProMode): mode is CreationMode {
  return mode === 'image' || mode === 'video' || mode === 'inpaint'
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

function isGenerationCancelResult(error: unknown): boolean {
  return isAbortError(error) || (error instanceof ProApiError && error.status === 499)
}

function readInitialRail(): string {
  const hash = window.location.hash.replace(/^#/, '').trim()
  return RAIL_IDS.has(hash) ? hash : 'create'
}

function readLayoutPreferences(): LayoutPreferences {
  try {
    const raw = window.localStorage.getItem(LAYOUT_STORAGE_KEY)
    if (!raw) {
      return DEFAULT_LAYOUT_PREFERENCES
    }
    const parsed = JSON.parse(raw) as Partial<LayoutPreferences>
    return {
      leftPanelWidth: clamp(Number(parsed.leftPanelWidth) || 380, 300, 520),
      rightPanelWidth: clamp(Number(parsed.rightPanelWidth) || 320, 260, 420),
      bottomDockHeight: clamp(Number(parsed.bottomDockHeight) || 196, 120, 360),
      bottomDockVisible:
        typeof parsed.bottomDockVisible === 'boolean'
          ? parsed.bottomDockVisible
          : DEFAULT_LAYOUT_PREFERENCES.bottomDockVisible,
    }
  } catch {
    return DEFAULT_LAYOUT_PREFERENCES
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

const MENU_BAR_ITEMS: Array<{
  id: Exclude<MenuBarId, null>
  label: string
  items: Array<{ id: string; label: string; hint?: string }>
}> = [
  {
    id: 'file',
    label: 'File',
    items: [
      { id: 'new-prompt', label: 'New prompt' },
      { id: 'copy-last', label: 'Copy prompt' },
    ],
  },
  {
    id: 'edit',
    label: 'Edit',
    items: [
      { id: 'copy-last', label: 'Copy prompt' },
      { id: 'reset-layout', label: 'Reset workspace' },
    ],
  },
  {
    id: 'view',
    label: 'View',
    items: [
      { id: 'toggle-dock', label: 'Output dock' },
      { id: 'open-models', label: 'Model inventory' },
      { id: 'open-tools', label: 'Tools' },
      { id: 'open-data', label: 'Data view' },
      { id: 'open-monitor', label: 'Monitor' },
    ],
  },
  {
    id: 'options',
    label: 'Options',
    items: [
      { id: 'open-settings', label: 'Settings' },
      { id: 'open-segmentation', label: 'Segmentation' },
      { id: 'open-hires', label: 'High-res fix' },
      { id: 'open-reactor', label: 'ReActor' },
    ],
  },
  {
    id: 'help',
    label: 'Help',
    items: [
      { id: 'open-help', label: 'About AIWF Studio' },
    ],
  },
]

export default App
