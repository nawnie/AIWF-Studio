import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type {
  CSSProperties,
  ChangeEvent as ReactChangeEvent,
  Dispatch,
  DragEvent as ReactDragEvent,
  KeyboardEvent as ReactKeyboardEvent,
  MouseEvent as ReactMouseEvent,
  PointerEvent as ReactPointerEvent,
  ReactNode,
  SetStateAction,
} from 'react'
import {
  ArrowLeftRight,
  Boxes,
  Brush,
  CircleHelp,
  Database,
  Clipboard,
  Cpu,
  Eye,
  EyeOff,
  FileImage,
  Hand,
  HardDrive,
  Highlighter,
  Image,
  Layers2,
  Maximize2,
  Monitor,
  PanelLeft,
  PanelRight,
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
import { Workflow as WorkflowIcon } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { WorkflowPanel } from './workflow/WorkflowPanel'
import {
  createWorkflowBlocksFromSettings,
  loadWorkflowBlocksFromStorage,
  renumberWorkflowBlocks,
  saveWorkflowBlocksToStorage,
} from './workflow/workflowBlocks'
import type { WorkflowCodeBlock } from './types'
import { ModelFamilyMatrixLayout } from './layouts/studio/ModelFamilyMatrixLayout'
import { ProjectCenterLayout } from './layouts/studio/ProjectCenterLayout'
import { AgenticChatLayout } from './layouts/studio/AgenticChatLayout'
import { PipelineAtlasLayout } from './layouts/studio/PipelineAtlasLayout'
import { MediaFoundryImageLayout } from './layouts/studio/MediaFoundryImageLayout'
import { AudioStudioLayout } from './layouts/studio/AudioStudioLayout'
import type { LayoutProps } from './layouts/studio/LayoutTypes'
import {
  fetchProData,
  fetchProBootstrap,
  fetchProCapabilities,
  fetchProDownloads,
  downloadCatalogModel,
  fetchProLogs,
  fetchProRuntime,
  fetchProStartup,
  fetchProExtensions,
  fetchProSettings,
  fetchVideoLabStatus,
  toggleProExtension,
  formatApiError,
  generateAutoMask,
  generateProOutput,
  importGenerationMetadataFromImage,
  getProApiLatencySamples,
  runFaceSwap,
  runVideoLab,
  reorganizeModels,
  uploadModelFile,
  uploadVideoLabFile,
  getFallbackBootstrap,
  getFallbackRuntime,
  ProApiError,
  notifyProWindowReady,
  reportProClientError,
  reportProClientEvent,
  requestProRestart,
  runEnhanceImage,
  saveProSettings,
  setGerrorEnabled,
  streamProRuntime,
  stopProGeneration,
  unloadProModel,
  runVsrImage,
} from './api'
import type { ProExtensionsStatus, ProModelSortResult, ProStartupStatus, VideoLabProbe, VideoLabStatus } from './api'
import type {
  AspectRatioOption,
  CreationMode,
  EngineId,
  EngineSummary,
  GenerationSettings,
  GenerationProgressEvent,
  ImportedGenerationMetadata,
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
  ProGenerateResult,
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

type ToolModalId = 'segmentation' | 'hires' | 'enhance' | 'reactor' | 'controlnet' | 'xyPlot' | 'about' | null
type MenuBarId = 'file' | 'edit' | 'view' | 'options' | 'help' | null
type DragTarget = 'left' | 'right' | 'bottom'

interface XyPlotCell {
  id: string
  modelId: string
  steps: number
}

type GenerationSettingsPatch = Partial<GenerationSettings> & { modelName?: string }

interface SupportIssue {
  title: string
  message: string
  source: string
  createdAt: string
  detail?: unknown
  context?: Record<string, unknown>
}

interface ControlNetCompatibility {
  supported: boolean
  message: string
  modelFamily: 'sd15' | 'sdxl' | null
  controlNetFamily: 'sd15' | 'sdxl' | null
}

const IMAGE_FILE_ACCEPT = 'image/*,.png,.jpg,.jpeg,.webp,.bmp,.gif,.tif,.tiff,.avif'
const VIDEO_FILE_ACCEPT = 'video/*,.mp4,.mov,.mkv,.webm,.avi,.m4v,.wmv,.flv,.mpeg,.mpg,.ts,.mts,.m2ts,.3gp,.ogv'
const MODEL_FILE_ACCEPT = '.safetensors,.ckpt,.pt,.pth,.bin,.gguf,.onnx'
const IMAGE_FILE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff', '.avif'])
const VIDEO_FILE_EXTENSIONS = new Set(['.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v', '.wmv', '.flv', '.mpeg', '.mpg', '.ts', '.mts', '.m2ts', '.3gp', '.ogv'])
const MODEL_FILE_EXTENSIONS = new Set(['.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf', '.onnx'])
const XY_PLOT_DEFAULT_CELLS = 4
const XY_PLOT_MAX_CELLS = 12

async function copyTextToClipboard(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text)
    return
  }

  const textarea = document.createElement('textarea')
  textarea.value = text
  textarea.readOnly = true
  textarea.setAttribute('aria-hidden', 'true')
  textarea.style.position = 'fixed'
  textarea.style.top = '0'
  textarea.style.left = '-9999px'
  textarea.style.opacity = '0'
  document.body.appendChild(textarea)
  try {
    textarea.focus()
    textarea.select()
    textarea.setSelectionRange(0, text.length)

    const copied = document.execCommand('copy')
    if (!copied) {
      throw new Error('Clipboard copy failed.')
    }
  } finally {
    document.body.removeChild(textarea)
  }
}

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
  outputPreviewVisible: boolean
}

const DEFAULT_LAYOUT_PREFERENCES: LayoutPreferences = {
  leftPanelWidth: 380,
  rightPanelWidth: 320,
  bottomDockHeight: 196,
  bottomDockVisible: true,
  outputPreviewVisible: true,
}

const PRO_APP_ICON = '/app-icon.png'
const FALLBACK_STARTUP_STATUS: ProStartupStatus = {
  status: 'server-ready',
  serverReady: false,
  windowReady: false,
  startedAt: '',
  serverReadyAt: '',
  windowReadyAt: '',
  minSplashMs: 1800,
  readyHoldMs: 1200,
}
const STARTUP_SPLASH_FAILSAFE_MS = 14000

const LAYOUT_STORAGE_KEY = 'aiwf.pro.layout.v1'

const MODE_TABS: IconItem<ProMode>[] = [
  { id: 'image', label: 'Image', icon: Image },
  { id: 'inpaint', label: 'Inpaint', icon: Brush },
  { id: 'video', label: 'Video', icon: Video },
  { id: 'audio', label: 'Audio', icon: Wand2 },
  { id: 'models', label: 'Models', icon: Boxes },
  { id: 'data', label: 'Data', icon: Database },
  { id: 'settings', label: 'Settings', icon: Settings },
]

const RAIL_ITEMS: IconItem<string>[] = [
  { id: 'create', label: 'Create', icon: Sparkles },
  { id: 'workflow', label: 'Workflow', icon: WorkflowIcon },
  { id: 'models', label: 'Models', icon: Boxes },
  { id: 'families', label: 'Families', icon: Database },
  { id: 'foundry', label: 'Foundry', icon: Image },
  { id: 'pipeline', label: 'Pipeline', icon: WorkflowIcon },
  { id: 'projects', label: 'Projects', icon: Boxes },
  { id: 'assistant', label: 'Assistant', icon: Sparkles },
  { id: 'audiolab', label: 'Audio', icon: Wand2 },
  { id: 'tools', label: 'Tools', icon: Wand2 },
  { id: 'data', label: 'Data', icon: Database },
  { id: 'monitor', label: 'Monitor', icon: Monitor },
  { id: 'logs', label: 'Logs', icon: FileImage },
  { id: 'settings', label: 'Settings', icon: Settings },
]

const RAIL_IDS = new Set(RAIL_ITEMS.map((item) => item.id))
const RAIL_ITEM_BY_ID = new Map(RAIL_ITEMS.map((item) => [item.id, item]))

const RAILS_BY_MODE: Record<string, string[]> = {
  image: ['create', 'workflow', 'models', 'families', 'foundry', 'pipeline', 'projects', 'data', 'monitor', 'logs'],
  inpaint: ['create', 'workflow', 'models', 'families', 'foundry', 'pipeline', 'data', 'monitor', 'logs'],
  video: ['create', 'workflow', 'models', 'pipeline', 'projects', 'data', 'monitor', 'logs'],
  audio: ['audiolab', 'projects', 'data', 'monitor', 'logs'],
  settings: ['settings', 'models', 'data', 'monitor', 'logs', 'assistant'],
  models: ['create', 'workflow', 'models', 'families', 'foundry', 'pipeline', 'projects', 'data', 'monitor', 'logs'],
  data: ['create', 'workflow', 'models', 'families', 'foundry', 'pipeline', 'projects', 'data', 'monitor', 'logs'],
}

const STUDIO_RAIL_ATTRIBUTE: Record<string, string> = {
  projects: 'project',
  assistant: 'agent',
  audiolab: 'audio',
}

const FULL_SURFACE_RAILS = new Set(['workflow', 'families', 'foundry', 'pipeline', 'projects', 'assistant', 'audiolab'])

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

const CONTROLNET_MODULE_OPTIONS = [
  { value: 'none', label: 'None / passthrough' },
  { value: 'canny', label: 'Canny' },
  { value: 'depth', label: 'Depth' },
  { value: 'openpose', label: 'OpenPose' },
  { value: 'lineart', label: 'Lineart' },
  { value: 'scribble', label: 'Scribble' },
  { value: 'softedge', label: 'SoftEdge' },
  { value: 'normal', label: 'Normal' },
  { value: 'segmentation', label: 'Segmentation' },
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

const SCHEDULER_OPTIONS = [
  { id: 'automatic', label: 'Automatic' },
  { id: 'uniform', label: 'Uniform' },
  { id: 'karras', label: 'Karras' },
  { id: 'exponential', label: 'Exponential' },
  { id: 'sgm_uniform', label: 'SGM Uniform' },
  { id: 'beta', label: 'Beta' },
]

const OUTPUT_FORMAT_OPTIONS = [
  { id: 'png', label: 'PNG' },
  { id: 'jpg', label: 'JPG' },
  { id: 'webp', label: 'WebP' },
]

const WAN_OFFLOAD_OPTIONS = [
  { id: 'balanced', label: 'Balanced (recommended on 16 GB)' },
  { id: 'model', label: 'Model swap' },
  { id: 'group', label: 'Group (block-level)' },
  { id: 'streamed', label: 'Streamed group' },
  { id: 'sequential', label: 'Sequential (slowest, least VRAM)' },
  { id: 'resident', label: 'Resident (24 GB+)' },
  { id: 'none', label: 'None' },
]

const LTX_DTYPE_OPTIONS = [
  { id: 'bf16', label: 'bfloat16 (recommended)' },
  { id: 'fp16', label: 'float16 (pre-Ampere fallback)' },
]

const LTX_OFFLOAD_OPTIONS = [
  { id: 'auto', label: 'Auto (resident when it fits)' },
  { id: 'model', label: 'Always model offload' },
  { id: 'none', label: 'Never offload (keep on GPU)' },
]

type SettingsSectionId = 'generation' | 'interface' | 'output' | 'video' | 'system' | 'about'

const SETTINGS_SECTIONS: Array<{ id: SettingsSectionId; label: string; hint: string }> = [
  { id: 'generation', label: 'Generation', hint: 'Default model, sampler, and quality values' },
  { id: 'interface', label: 'Interface', hint: 'Previews, gallery, and layout memory' },
  { id: 'output', label: 'Output & Metadata', hint: 'File formats, filenames, and infotext' },
  { id: 'video', label: 'Video & Performance', hint: 'Wan/LTX precision, offload, and VRAM strategy' },
  { id: 'system', label: 'System & Launch', hint: 'Paths, runtime flags, and API policy' },
  { id: 'about', label: 'About', hint: 'Build and credits' },
]

const WAN_SAMPLER_OPTIONS = [
  { id: 'unipc', label: 'UniPC' },
  { id: 'euler', label: 'Euler' },
  { id: 'heun', label: 'Heun' },
]

const WAN_RUNTIME_MODE_OPTIONS = [
  { id: 'fast_5b', label: 'Fast 5B' },
  { id: 'high_low', label: 'High / low split' },
]

const RUNTIME_BACKEND_OPTIONS = [
  { id: 'diffusers', label: 'Diffusers' },
  { id: 'dual', label: 'Dual (Diffusers + C++)' },
  { id: 'sdcpp', label: 'stable-diffusion.cpp' }, // AIWF-SDCPP-BACKEND-OPTION
  { id: 'onnx', label: 'ONNX' },
]

const RUNTIME_ATTENTION_OPTIONS = [
  { id: 'sage_sdpa', label: 'Sage SDPA' },
  { id: 'sdpa', label: 'SDPA' },
  { id: 'xformers', label: 'xFormers' },
  { id: 'none', label: 'None' },
]

const ONNX_PROVIDER_OPTIONS = [
  { id: 'auto', label: 'Auto' },
  { id: 'cuda', label: 'CUDA' },
  { id: 'directml', label: 'DirectML' },
  { id: 'cpu', label: 'CPU' },
]

const VRAM_PROFILE_OPTIONS = [
  { id: 'normal', label: 'Normal VRAM' },
  { id: 'cpu', label: 'CPU only' },
  { id: 'low', label: 'Low VRAM (4-8 GB)' },
  { id: 'mid', label: 'Mid VRAM (8-16 GB)' },
  { id: 'high', label: 'High VRAM (16+ GB)' },
]

const RESOLUTION_PRESETS = [
  { id: '480', label: '480', shortEdge: 480 },
  { id: '512', label: '512', shortEdge: 512 },
  { id: '720', label: '720', shortEdge: 720 },
  { id: '1024', label: '1024', shortEdge: 1024 },
]

function StartupSplash({ ready }: { ready: boolean }) {
  return (
    <div className={`pro-startup-splash${ready ? ' is-ready' : ''}`} role="status" aria-live="polite">
      <div className="pro-startup-logo" aria-hidden="true">
        <span className="pro-startup-logo-layer pro-startup-logo-frame" />
        <span className="pro-startup-logo-layer pro-startup-logo-bolt" />
        <span className="pro-startup-logo-layer pro-startup-logo-core" />
      </div>
      <div className="pro-startup-copy">
        <span>{ready ? 'Ready' : 'Loading AIWF Studio'}</span>
        <small>{ready ? 'Opening workspace' : 'Preparing local runtime'}</small>
      </div>
    </div>
  )
}

function App() {
  const fallbackBootstrap = useMemo(() => getFallbackBootstrap(), [])
  const fallbackRuntime = useMemo(() => getFallbackRuntime(), [])
  const initialLayout = useMemo(() => readLayoutPreferences(), [])
  const [bootstrap, setBootstrap] = useState<ProBootstrap>(fallbackBootstrap)
  const [runtime, setRuntime] = useState<ProRuntimeStatus>(fallbackRuntime)
  const [runtimeStreamConnected, setRuntimeStreamConnected] = useState(false)
  const [startupStatus, setStartupStatus] = useState<ProStartupStatus>(FALLBACK_STARTUP_STATUS)
  const [startupSplashVisible, setStartupSplashVisible] = useState(true)
  const [settings, setSettings] = useState<GenerationSettings>(fallbackBootstrap.defaults)
  const [dataStatus, setDataStatus] = useState<ProDataStatus | null>(null)
  const [downloadsStatus, setDownloadsStatus] = useState<ProDownloadsStatus | null>(null)
  const [capabilitiesStatus, setCapabilitiesStatus] = useState<ProCapabilitiesStatus | null>(null)
  const [logStatus, setLogStatus] = useState<ProLogStatus | null>(null)
  const [settingsStatus, setSettingsStatus] = useState<ProSettingsStatus | null>(null)
  const [settingsSaveStatus, setSettingsSaveStatus] = useState('')
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
  const [supportIssue, setSupportIssue] = useState<SupportIssue | null>(null)
  const [activeMode, setActiveMode] = useState<ProMode>(readInitialMode)
  const [activeRail, setActiveRail] = useState(readInitialRail)
  const [workflowBlocks, setWorkflowBlocks] = useState<WorkflowCodeBlock[]>(() => loadWorkflowBlocksFromStorage())
  const [workflowStatus, setWorkflowStatus] = useState('')
  const [previews, setPreviews] = useState<Partial<Record<CreationMode, RecentOutput | null>>>({
    image: fallbackBootstrap.recentOutputs[0] ?? null,
  })
  const activeCreationMode: CreationMode = isCreationMode(activeMode) ? activeMode : settings.mode
  const preview = previews[activeCreationMode] ?? null
  // Each creation mode keeps its own canvas/preview; outputs land in the canvas
  // that matches their mode instead of overwriting whatever tab is open.
  const setPreview = useCallback(
    (value: RecentOutput | null | ((current: RecentOutput | null) => RecentOutput | null)) => {
      setPreviews((current) => {
        if (typeof value === 'function') {
          return { ...current, image: value(current.image ?? null) }
        }
        const targetMode: CreationMode =
          value && (value.mode === 'image' || value.mode === 'video' || value.mode === 'inpaint')
            ? value.mode
            : 'image'
        return { ...current, [targetMode]: value }
      })
    },
    [],
  )
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [statusMessage, setStatusMessage] = useState('Ready.')
  const [isGenerating, setIsGenerating] = useState(false)
  const [continuousGenerating, setContinuousGenerating] = useState(false)
  const [backendConnected, setBackendConnected] = useState(false)
  const [backendRecovering, setBackendRecovering] = useState(false)
  const [engineFilter, setEngineFilter] = useState<EngineId>('all')
  const [downloadingCatalogKey, setDownloadingCatalogKey] = useState('')
  const [leftPanelWidth, setLeftPanelWidth] = useState(initialLayout.leftPanelWidth)
  const [rightPanelWidth, setRightPanelWidth] = useState(initialLayout.rightPanelWidth)
  const [leftPanelCollapsed, setLeftPanelCollapsed] = useState(false)
  const [rightPanelCollapsed, setRightPanelCollapsed] = useState(false)
  const [bottomDockVisible, setBottomDockVisible] = useState(initialLayout.bottomDockVisible)
  const [outputPreviewVisible, setOutputPreviewVisible] = useState(initialLayout.outputPreviewVisible)
  const [bottomDockHeight, setBottomDockHeight] = useState(initialLayout.bottomDockHeight)
  const [activeModal, setActiveModal] = useState<ToolModalId>(null)
  const [openMenu, setOpenMenu] = useState<MenuBarId>(null)
  const [dragState, setDragState] = useState<DragState | null>(null)
  const [enhanceSourceDataUrl, setEnhanceSourceDataUrl] = useState('')
  const [enhanceSourceName, setEnhanceSourceName] = useState('')
  const [enhanceMode, setEnhanceMode] = useState<'restore' | 'upscale' | 'restore-upscale' | 'vsr'>('restore')
  const [enhanceRestoreModel, setEnhanceRestoreModel] = useState('gfpgan-v1.4')
  const [enhanceUpscaleModel, setEnhanceUpscaleModel] = useState('realesrgan-x4plus')
  const [enhanceRestoreVisibility, setEnhanceRestoreVisibility] = useState(1)
  const [enhanceCodeformerWeight, setEnhanceCodeformerWeight] = useState(0.5)
  const [enhanceUpscaleScale, setEnhanceUpscaleScale] = useState(2)
  const [enhanceTileSize, setEnhanceTileSize] = useState(256)
  const [enhanceTileOverlap, setEnhanceTileOverlap] = useState(32)
  const [enhanceVsrScale, setEnhanceVsrScale] = useState(2)
  const [enhanceVsrMode, setEnhanceVsrMode] = useState(0)
  const [enhanceVsrStrength, setEnhanceVsrStrength] = useState(0.4)
  const [enhanceBusy, setEnhanceBusy] = useState(false)
  const [enhanceMessage, setEnhanceMessage] = useState('')
  const [segmentationMode, setSegmentationMode] = useState('Auto mask')
  const [reactorSourceDataUrl, setReactorSourceDataUrl] = useState('')
  const [reactorBusy, setReactorBusy] = useState(false)
  const [reactorMessage, setReactorMessage] = useState('')
  const [xyPlotCells, setXyPlotCells] = useState<XyPlotCell[]>([])
  const [xyPlotStatus, setXyPlotStatus] = useState('')
  const generationAbortRef = useRef<AbortController | null>(null)
  const startupSplashStartedAtRef = useRef(Date.now())
  const startupWindowReadyReportedRef = useRef(false)
  const runtimeErrorLoggedRef = useRef(false)
  const auxiliaryErrorLoggedRef = useRef<Record<string, boolean>>({})
  const auxiliaryFingerprintRef = useRef<Record<string, string>>({})
  const auxiliaryFetchInFlightRef = useRef<Record<string, boolean>>({})
  const fileDropDepthRef = useRef(0)
  const [fileDropActive, setFileDropActive] = useState(false)
  const continuousGenerateRef = useRef(false)
  const runtimeJobActive = isRuntimeJobActive(runtime.job)
  const generationActive = isGenerating || runtime.state.toLowerCase() === 'running' || runtimeJobActive
  const sdcppRuntimeAvailable = isSdcppRuntime(runtime)
  const dualRuntimeAvailable = isDualRuntime(runtime)

  useEffect(() => {
    setGerrorEnabled(runtime.gerror)
  }, [runtime.gerror])

  useEffect(() => {
    if (settings.pipelineBackend === 'dual' && !dualRuntimeAvailable) {
      setSettings((current) => (
        current.pipelineBackend === 'dual'
          ? { ...current, pipelineBackend: sdcppRuntimeAvailable ? 'sdcpp' : 'aiwf' }
          : current
      ))
      return
    }
    if (settings.pipelineBackend === 'sdcpp' && !sdcppRuntimeAvailable) {
      setSettings((current) => (
        current.pipelineBackend === 'sdcpp'
          ? { ...current, pipelineBackend: 'aiwf' }
          : current
      ))
    }
  }, [dualRuntimeAvailable, sdcppRuntimeAvailable, settings.pipelineBackend])

  const setDisconnectedRuntime = useCallback((message: string) => {
    setRuntime((current) => ({
      ...fallbackRuntime,
      state: backendRecovering ? 'Recovering' : 'Disconnected',
      backend: 'Backend unreachable',
      device: 'Waiting for runtime',
      job: { ...fallbackRuntime.job, message },
      resources: fallbackRuntime.resources.map((metric) => ({ ...metric })),
      loadedModel: { ...fallbackRuntime.loadedModel },
      gerror: current.gerror,
    }))
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

  const showSupportIssue = useCallback(
    (
      title: string,
      errorOrMessage: unknown,
      source: string,
      context: Record<string, unknown> = {},
    ) => {
      const message = typeof errorOrMessage === 'string' ? errorOrMessage : formatApiError(errorOrMessage)
      const detail = errorOrMessage instanceof ProApiError ? errorOrMessage.detail : undefined
      const apiLatency = getProApiLatencySamples().slice(-10)
      const issue: SupportIssue = {
        title,
        message,
        source,
        createdAt: new Date().toISOString(),
        detail,
        context: {
          ...context,
          activeRail,
          activeMode,
          selectedModelId: settings.modelId,
          selectedModelName: bootstrap.models.find((model) => model.id === settings.modelId)?.name ?? settings.modelId,
          runtimeState: runtime.state,
          backend: runtime.backend,
          device: runtime.device,
          apiLatency,
          appVersion: bootstrap.version,
          userAgent: window.navigator.userAgent,
          url: window.location.href,
        },
      }
      setSupportIssue(issue)
      reportProClientError({
        kind: 'support-popup',
        message,
        source,
        context: issue.context,
      })
    },
    [activeMode, activeRail, bootstrap.models, bootstrap.version, runtime.backend, runtime.device, runtime.state, settings.modelId],
  )

  useEffect(() => {
    const controller = new AbortController()
    fetchProStartup(controller.signal)
      .then(setStartupStatus)
      .catch(() => undefined)
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!startupSplashVisible) {
      return undefined
    }
    const elapsedMs = Date.now() - startupSplashStartedAtRef.current
    const minSplashMs = Math.max(0, startupStatus.minSplashMs || FALLBACK_STARTUP_STATUS.minSplashMs)
    const readyHoldMs = Math.max(0, startupStatus.readyHoldMs || FALLBACK_STARTUP_STATUS.readyHoldMs)
    const delayMs = backendConnected
      ? Math.max(0, minSplashMs - elapsedMs) + readyHoldMs
      : Math.max(1000, STARTUP_SPLASH_FAILSAFE_MS - elapsedMs)
    const timeoutId = window.setTimeout(() => {
      setStartupSplashVisible(false)
    }, delayMs)
    return () => window.clearTimeout(timeoutId)
  }, [backendConnected, startupSplashVisible, startupStatus.minSplashMs, startupStatus.readyHoldMs])

  useEffect(() => {
    if (startupSplashVisible || startupWindowReadyReportedRef.current) {
      return
    }
    startupWindowReadyReportedRef.current = true
    notifyProWindowReady()
      .then(setStartupStatus)
      .catch(() => undefined)
  }, [startupSplashVisible])

  const handleSaveProSettings = useCallback(async () => {
    setSettingsSaveStatus('Saving settings...')
    try {
      const nextStatus = await saveProSettings(
        settings,
        settingsStatus?.ui,
        settingsStatus?.output,
        settingsStatus?.video,
        settingsStatus?.runtime,
      )
      setSettingsStatus(nextStatus)
      setBootstrap((current) => ({
        ...current,
        defaults: {
          ...current.defaults,
          ...nextStatus.generationDefaults,
        },
      }))
      setStatusMessage('Settings saved.')
      setSettingsSaveStatus('Saved.')
    } catch (error: unknown) {
      const message = formatApiError(error)
      setStatusMessage(message)
      setSettingsSaveStatus(message)
      reportProClientError({
        kind: 'api',
        message,
        source: 'settings-save',
        context: { route: '/api/pro/settings' },
      })
    }
  }, [settings, settingsStatus])

  const refreshWorkspaceDataNow = useCallback(async () => {
    const [bootstrapResult, dataResult, downloadsResult, capabilitiesResult, settingsResult] = await Promise.allSettled([
      fetchProBootstrap(),
      fetchProData(),
      fetchProDownloads(),
      fetchProCapabilities(),
      fetchProSettings(),
    ])
    if (bootstrapResult.status === 'fulfilled') {
      setBootstrap(bootstrapResult.value)
    }
    if (dataResult.status === 'fulfilled') {
      setDataStatus(dataResult.value)
    }
    if (downloadsResult.status === 'fulfilled') {
      setDownloadsStatus(downloadsResult.value)
    }
    if (capabilitiesResult.status === 'fulfilled') {
      setCapabilitiesStatus(capabilitiesResult.value)
    }
    if (settingsResult.status === 'fulfilled') {
      setSettingsStatus(settingsResult.value)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    fetchProBootstrap(controller.signal)
      .then((nextBootstrap) => {
        setBootstrap(nextBootstrap)
        setPreview((currentPreview) => currentPreview ?? nextBootstrap.recentOutputs[0] ?? null)
        setSettings((current) => {
          const merged = settingsMatch(current, fallbackBootstrap.defaults)
            ? nextBootstrap.defaults
            : mergeBootstrapDefaults(current, nextBootstrap)
          const model = nextBootstrap.models.find((item) => item.id === merged.modelId)
          return applyModelPresetSettings(merged, model, nextBootstrap.aspectRatios)
        })
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

  // Read via ref so the stream effect does not tear down and rebuild the
  // EventSource every time the recovery flag flips.
  const backendRecoveringRef = useRef(backendRecovering)
  useEffect(() => {
    backendRecoveringRef.current = backendRecovering
  }, [backendRecovering])

  useEffect(() => {
    return streamProRuntime(
      (nextRuntime) => {
        setRuntime(nextRuntime)
        setBackendConnected(true)
        if (backendRecoveringRef.current) {
          setBackendRecovering(false)
          setStatusMessage('Backend reconnected.')
        }
      },
      (connected) => {
        setRuntimeStreamConnected(connected)
      },
    )
  }, [])

  useEffect(() => {
    let disposed = false
    let requestController: AbortController | null = null
    let requestTimeoutId: number | null = null
    let inFlight = false
    // While the SSE stream is healthy it is the single source of runtime
    // state; the GET poll only checks liveness. Applying poll responses while
    // streaming caused stale data (older step / older preview) to overwrite
    // newer stream ticks — visible as progress jumping backwards and live
    // previews flickering.
    const applyPollResults = !runtimeStreamConnected
    const intervalMs = runtimeStreamConnected ? 30000 : backendRecovering ? 500 : generationActive ? 250 : 750
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
      }, 1200)
      fetchProRuntime(activeController.signal)
        .then((nextRuntime) => {
          if (!disposed) {
            runtimeErrorLoggedRef.current = false
            setBackendConnected(true)
            setBackendRecovering(false)
            if (applyPollResults) {
              setRuntime(nextRuntime)
            }
          }
        })
        .catch((error: unknown) => {
          if (!disposed && (timedOut || !isAbortError(error))) {
            if (!runtimeErrorLoggedRef.current) {
              runtimeErrorLoggedRef.current = true
              reportProClientError({
                kind: 'api',
                message: timedOut ? 'Runtime refresh timed out after 1200 ms.' : formatApiError(error),
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
  }, [backendRecovering, generationActive, markBackendDisconnected, runtimeStreamConnected])

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
      if (auxiliaryFetchInFlightRef.current[key]) {
        return
      }
      auxiliaryFetchInFlightRef.current[key] = true
      const requestController = new AbortController()
      const timeoutId = window.setTimeout(() => requestController.abort(), timeoutMs)
      const abortRelay = () => requestController.abort()
      controller.signal.addEventListener('abort', abortRelay, { once: true })
      void fetcher(requestController.signal)
        .then((value) => {
          auxiliaryErrorLoggedRef.current[key] = false
          // Skip the state update (and the full-shell re-render it causes)
          // when the payload is byte-identical to the previous refresh.
          const fingerprint = JSON.stringify(value)
          if (auxiliaryFingerprintRef.current[key] === fingerprint) {
            return
          }
          auxiliaryFingerprintRef.current[key] = fingerprint
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
          auxiliaryFetchInFlightRef.current[key] = false
          window.clearTimeout(timeoutId)
          controller.signal.removeEventListener('abort', abortRelay)
        })
    }
    const refreshWorkspaceData = () => {
      if (generationActive) {
        return
      }
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
        45000,
        '/api/pro/capabilities',
        'Capability inventory is still scanning. Pro will keep the workspace available.',
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
    // Inventory data moves slowly; a fast poll here caused repeated multi-MB
    // payload parses and full-shell re-renders that made the UI feel laggy.
    const intervalId = window.setInterval(refreshWorkspaceData, 45000)
    return () => {
      controller.abort()
      window.clearInterval(intervalId)
    }
  }, [generationActive])

  useEffect(() => {
    const job = runtime.job
    if (!isRuntimeJobActive(job) || !job.previewUrl || settings.mode !== 'image') {
      return
    }
    setPreview({
      id: `runtime-preview-${job.id}-${job.step}`,
      url: job.previewUrl,
      thumbnailUrl: job.previewUrl,
      prompt: settings.prompt || job.message || 'Live preview',
      width: settings.width,
      height: settings.height,
      createdAt: new Date().toISOString(),
      mode: 'image',
      modelName: settings.modelId,
      status: 'preview',
      source: 'runtime-preview',
    })
  }, [
    runtime.job.id,
    runtime.job.message,
    runtime.job.previewUrl,
    runtime.job.step,
    settings.height,
    settings.mode,
    settings.modelId,
    settings.prompt,
    settings.width,
  ])

  const handleRecoverBackend = useCallback(async () => {
    setBackendRecovering(true)
    setStatusMessage('Restart requested. Waiting for the Pro backend to come back on the same port.')
    markBackendDisconnected('Restart requested. Waiting for the Pro backend to come back on the same port.')
    try {
      await requestProRestart()
    } catch (error) {
      setBackendRecovering(false)
      const message = `Backend restart request failed: ${formatApiError(error)}`
      setStatusMessage(message)
      showSupportIssue('Backend restart failed', error, 'restart', { route: '/api/pro/restart' })
      reportProClientError({
        kind: 'api',
        message,
        source: 'restart',
        context: { route: '/api/pro/restart' },
      })
    }
  }, [markBackendDisconnected, showSupportIssue])

  const handleCopyGenerationError = useCallback(async () => {
    const text = generationError || runtime.job.error
    if (!text) {
      return
    }
    try {
      await copyTextToClipboard(text)
      setStatusMessage('Error copied.')
    } catch {
      setStatusMessage('Could not copy the error text.')
    }
  }, [generationError, runtime.job.error])

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
      } else if (nextRail === 'settings') {
        setActiveMode('settings')
      } else if (nextRail === 'audiolab') {
        setActiveMode('audio')
      } else if (nextRail === 'create') {
        setActiveMode('image')
      } else {
        setActiveMode((current) => modeContainingRail(nextRail, current))
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
      outputPreviewVisible,
    }
    try {
      window.localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(nextPreferences))
    } catch {
      // Layout persistence is a convenience; the app should stay usable if storage is blocked.
    }
  }, [bottomDockHeight, bottomDockVisible, leftPanelWidth, outputPreviewVisible, rightPanelWidth])

  useEffect(() => {
    return () => {
      void import('./ml/promptInsight').then(({ disposePromptInsightModel }) => {
        void disposePromptInsightModel()
      })
    }
  }, [])

  const creationModels = useMemo(
    () => modelsForCreationMode(bootstrap.models, activeCreationMode),
    [activeCreationMode, bootstrap.models],
  )

  const creationEngines = useMemo(
    () => summarizeEnginesForModels(bootstrap.engines, creationModels),
    [bootstrap.engines, creationModels],
  )

  const filteredModels = useMemo(
    () => creationModels.filter((model) => matchesEngineFilter(model, engineFilter)),
    [creationModels, engineFilter],
  )

  const xyPlotModels = useMemo(
    () => modelsForCreationMode(bootstrap.models, 'image').filter((model) => !isModelBlocked(model)),
    [bootstrap.models],
  )

  const selectedModel = useMemo(() => {
    return (
      filteredModels.find((model) => model.id === settings.modelId) ??
      filteredModels[0] ??
      creationModels[0]
    )
  }, [creationModels, filteredModels, settings.modelId])

  const selectedModelWarning = useMemo(() => {
    const requestedModel = bootstrap.models.find((model) => model.id === settings.modelId)
    if (requestedModel && !modelFitsCreationMode(requestedModel, settings.mode)) {
      return settings.mode === 'video'
        ? 'Pick a Wan or Sana Video model before generating video.'
        : 'Video models are only available from the Video tab.'
    }
    if (isModelBlocked(selectedModel)) {
      return modelBlockedMessage(selectedModel)
    }
    if (!bootstrap.models.some((model) => model.id === settings.modelId)) {
      return 'Selected model is not available in the current Pro model list.'
    }
    return ''
  }, [bootstrap.models, selectedModel, settings.mode, settings.modelId])

  const controlNetCompatibility = useMemo(
    () => getControlNetCompatibility(selectedModel, settings.controlNetModel),
    [selectedModel, settings.controlNetModel],
  )

  useEffect(() => {
    saveWorkflowBlocksToStorage(workflowBlocks)
  }, [workflowBlocks])

  const handleSendToWorkflow = useCallback(
    (source = 'Create panel') => {
      setWorkflowBlocks((current) => {
        const created = createWorkflowBlocksFromSettings(
          {
            settings,
            bootstrap,
            runtime,
            selectedModel,
            selectedModelName: selectedModel?.name ?? settings.modelId,
            source,
          },
          current.length,
        )
        return renumberWorkflowBlocks([...current, ...created])
      })
      setWorkflowStatus('Captured current settings as a workflow node. Open the Workflow tab to reorder.')
    },
    [bootstrap, runtime, selectedModel, settings],
  )

  useEffect(() => {
    if (creationModels.length === 0) {
      return
    }
    const filterHasModels = creationModels.some((model) => matchesEngineFilter(model, engineFilter))
    if (!filterHasModels && engineFilter !== 'all') {
      setEngineFilter('all')
    }
    if (creationModels.some((model) => model.id === settings.modelId)) {
      return
    }
    const replacement = creationModels[0]
    setSettings((current) =>
      applyModelPresetSettings(
        {
          ...current,
          modelId: replacement.id,
        },
        replacement,
        bootstrap.aspectRatios,
      ),
    )
  }, [bootstrap.aspectRatios, creationModels, engineFilter, settings.modelId])

  const recentOutputs = useMemo(() => {
    const source = dataStatus?.recentOutputs.length ? dataStatus.recentOutputs : bootstrap.recentOutputs
    return source.slice(0, 8)
  }, [bootstrap.recentOutputs, dataStatus])

  const commitRecentOutputs = useCallback((outputs: RecentOutput[]) => {
    if (outputs.length === 0) {
      return
    }
    setBootstrap((current) => ({
      ...current,
      recentOutputs: mergeRecentOutputs(outputs, current.recentOutputs),
    }))
    setDataStatus((current) =>
      current
        ? {
            ...current,
            counts: {
              ...current.counts,
              recentOutputs: Math.max(
                current.counts.recentOutputs,
                mergeRecentOutputs(outputs, current.recentOutputs).length,
              ),
            },
            recentOutputs: mergeRecentOutputs(outputs, current.recentOutputs),
          }
        : current,
    )
  }, [])

  // Shared props bundle fed to every migrated studio layout screen. Keeps the
  // ex-paid layouts as pure presentation over my real state + handlers.
  const buildLayoutProps = useCallback(
    (): LayoutProps => ({
      settings,
      bootstrap,
      runtime,
      recentOutputs,
      preview,
      selectedModel,
      selectedModelName: selectedModel?.name ?? settings.modelId,
      statusMessage,
      isGenerating,
      onSettingsChange: setSettings,
      onGenerate: handleGenerate,
      onSendToWorkflow: handleSendToWorkflow,
      workflowBlocks,
      onWorkflowBlocksChange: setWorkflowBlocks,
      onPreviewSelect: setPreview,
      onOpenModels: () => handleRailSelect('models'),
      onOpenSettings: () => handleRailSelect('settings'),
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [bootstrap, isGenerating, preview, recentOutputs, runtime, selectedModel, settings, statusMessage, workflowBlocks],
  )

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
        const currentModel = bootstrap.models.find((model) => model.id === settings.modelId)
        const videoModels = modelsForCreationMode(bootstrap.models, 'video')
        const currentIsVideo = currentModel ? modelFitsCreationMode(currentModel, 'video') : false
        const videoModel = currentIsVideo
          ? currentModel
          : videoModels.find((model) => model.engineId === 'wan') ??
            videoModels.find((model) => model.engineId === 'sana_video') ??
            videoModels[0]
        setEngineFilter(videoModel?.engineId ?? 'sana_video')
        setSettings((current) => ({
          ...current,
          mode,
          modelId: videoModel?.id ?? current.modelId,
          aspectRatioId: '16:9',
          width: 832,
          height: 480,
          batchSize: 1,
        }))
      } else {
        const currentModel = bootstrap.models.find((model) => model.id === settings.modelId)
        const routeModels = modelsForCreationMode(bootstrap.models, mode)
        const routeModel = currentModel && modelFitsCreationMode(currentModel, mode) ? currentModel : routeModels[0]
        setEngineFilter(routeModel?.engineId ?? 'all')
        setSettings((current) => {
          const next = {
            ...current,
            mode,
            modelId: routeModel?.id ?? current.modelId,
          }
          return routeModel ? applyModelPresetSettings(next, routeModel, bootstrap.aspectRatios) : next
        })
      }
      setActiveRail('create')
      if (window.location.hash !== `#${mode}`) {
        window.history.replaceState(null, '', `#${mode}`)
      }
    } else if (mode === 'audio') {
      setActiveRail('audiolab')
      if (window.location.hash !== '#audiolab') {
        window.history.replaceState(null, '', '#audiolab')
      }
    } else if (mode === 'settings') {
      setActiveRail('settings')
      if (window.location.hash !== '#settings') {
        window.history.replaceState(null, '', '#settings')
      }
    } else {
      setActiveRail(mode)
      if (window.location.hash !== `#${mode}`) {
        window.history.replaceState(null, '', `#${mode}`)
      }
    }
  }, [bootstrap.aspectRatios, bootstrap.models, settings.modelId])

  useEffect(() => {
    if (!isCreationMode(activeMode) || settings.mode === activeMode) {
      return
    }
    if (activeMode === 'video') {
      const currentModel = bootstrap.models.find((model) => model.id === settings.modelId)
      const videoModels = modelsForCreationMode(bootstrap.models, 'video')
      const currentIsVideo = currentModel ? modelFitsCreationMode(currentModel, 'video') : false
      const videoModel = currentIsVideo
        ? currentModel
        : videoModels.find((model) => model.engineId === 'wan') ??
          videoModels.find((model) => model.engineId === 'sana_video') ??
          videoModels[0]
      setEngineFilter(videoModel?.engineId ?? 'sana_video')
      setSettings((current) => ({
        ...current,
        mode: activeMode,
        modelId: videoModel?.id ?? current.modelId,
        aspectRatioId: '16:9',
        width: 832,
        height: 480,
        batchSize: 1,
      }))
      return
    }
    const currentModel = bootstrap.models.find((model) => model.id === settings.modelId)
    const routeModels = modelsForCreationMode(bootstrap.models, activeMode)
    const routeModel = currentModel && modelFitsCreationMode(currentModel, activeMode) ? currentModel : routeModels[0]
    setEngineFilter(routeModel?.engineId ?? 'all')
    setSettings((current) => {
      const next = {
        ...current,
        mode: activeMode,
        modelId: routeModel?.id ?? current.modelId,
      }
      return routeModel ? applyModelPresetSettings(next, routeModel, bootstrap.aspectRatios) : next
    })
  }, [activeMode, bootstrap.aspectRatios, bootstrap.models, settings.mode, settings.modelId])

  const handleRailSelect = useCallback((id: string) => {
    setActiveRail(id)
    if (window.location.hash !== `#${id}`) {
      window.history.replaceState(null, '', `#${id}`)
    }
    if (id === 'models') {
      setActiveMode('models')
    } else if (id === 'data') {
      setActiveMode('data')
    } else if (id === 'settings') {
      setActiveMode('settings')
    } else if (id === 'audiolab') {
      setActiveMode('audio')
    } else if (id === 'create') {
      const nextMode: CreationMode = isCreationMode(activeMode) ? activeMode : settings.mode
      setActiveMode(nextMode)
      const imageModels = modelsForCreationMode(bootstrap.models, nextMode)
      const currentModel = bootstrap.models.find((model) => model.id === settings.modelId)
      const imageModel = currentModel && modelFitsCreationMode(currentModel, nextMode) ? currentModel : imageModels[0]
      setEngineFilter(imageModel?.engineId ?? 'all')
      setSettings((current) => {
        const next = {
          ...current,
          mode: nextMode,
          modelId: imageModel?.id ?? current.modelId,
        }
        return imageModel ? applyModelPresetSettings(next, imageModel, bootstrap.aspectRatios) : next
      })
    } else {
      // Rails like projects/families/tools can be reached from any mode via
      // menu or hash; keep the mode consistent so the rail list contains them.
      setActiveMode((current) => modeContainingRail(id, current))
    }
  }, [activeMode, bootstrap.aspectRatios, bootstrap.models, settings.mode, settings.modelId])

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
      const nextModels = creationModels.filter((model) => matchesEngineFilter(model, nextFilter))
      if (nextModels.length === 0) {
        return
      }
      setSettings((current) => {
        const selected = nextModels.find((model) => model.id === current.modelId) ?? nextModels[0]
        if (nextFilter === 'all' && selected.id === current.modelId) {
          return current
        }
        return applyModelPresetSettings(
          selected.id === current.modelId ? current : { ...current, modelId: selected.id },
          selected,
          bootstrap.aspectRatios,
        )
      })
    },
    [bootstrap.aspectRatios, creationModels],
  )

  const handleModelSelect = useCallback(
    (modelId: string) => {
      const model = creationModels.find((item) => item.id === modelId) ?? bootstrap.models.find((item) => item.id === modelId)
      setSettings((current) => applyModelPresetSettings({ ...current, modelId }, model, bootstrap.aspectRatios))
      if (model?.engineId) {
        setEngineFilter(model.engineId)
      }
      setStatusMessage(
        model
          ? `${model.name} selected. Runtime loads it when generation starts.`
          : `${modelId} selected. Runtime loads it when generation starts.`,
      )
    },
    [bootstrap.aspectRatios, bootstrap.models, creationModels],
  )

  const uploadModelFilesAndRefresh = useCallback(
    async (files: File[]) => {
      const modelFiles = files.filter(isModelFile)
      if (modelFiles.length === 0) {
        throw new Error(`Choose a model file: ${MODEL_FILE_ACCEPT}.`)
      }
      setStatusMessage(`Sorting ${modelFiles.length} model file${modelFiles.length === 1 ? '' : 's'}...`)
      const results: ProModelSortResult[] = []
      for (const file of modelFiles) {
        results.push(await uploadModelFile(file))
      }
      await refreshWorkspaceDataNow()
      const totalMoved = results.reduce((sum, result) => sum + result.counts.moved, 0)
      const totalLeft = results.reduce((sum, result) => sum + result.counts.left, 0)
      const inventory = results[results.length - 1]?.counts.inventoryCount ?? 0
      const message = totalMoved > 0
        ? `Sorted ${totalMoved} model file${totalMoved === 1 ? '' : 's'}. Inventory: ${inventory}.`
        : `${totalLeft || modelFiles.length} model file${modelFiles.length === 1 ? '' : 's'} need review in models to sort. Inventory: ${inventory}.`
      setStatusMessage(message)
      return message
    },
    [refreshWorkspaceDataNow],
  )

  const reorganizeModelFilesNow = useCallback(async () => {
    setStatusMessage('Re-reading model headers...')
    const result = await reorganizeModels()
    await refreshWorkspaceDataNow()
    const message = summarizeModelSort(result)
    setStatusMessage(message)
    return message
  }, [refreshWorkspaceDataNow])

  const handleCatalogDownload = useCallback(
    async (key: string) => {
      setDownloadingCatalogKey(key)
      setStatusMessage('Downloading model...')
      try {
        const nextDownloads = await downloadCatalogModel(key)
        setDownloadsStatus(nextDownloads)
        await refreshWorkspaceDataNow()
        setStatusMessage('Model downloaded and inventory refreshed.')
      } catch (error: unknown) {
        const message = formatApiError(error)
        setStatusMessage(message)
        showSupportIssue('Catalog download failed', error, 'catalog-download', {
          route: `/api/pro/downloads/catalog/${key}`,
          catalogKey: key,
        })
        reportProClientError({
          kind: 'api',
          message,
          source: 'catalog-download',
          context: { route: `/api/pro/downloads/catalog/${key}` },
        })
      } finally {
        setDownloadingCatalogKey('')
      }
    },
    [refreshWorkspaceDataNow, showSupportIssue],
  )

  const handleUnloadModel = useCallback(async () => {
    setStatusMessage('Unloading current model...')
    try {
      const nextRuntime = await unloadProModel()
      setRuntime(nextRuntime)
      setStatusMessage('Current model unloaded.')
    } catch (error: unknown) {
      const message = formatApiError(error)
      setStatusMessage(message)
      showSupportIssue('Unload model failed', error, 'model-unload', { route: '/api/pro/models/unload' })
    }
  }, [showSupportIssue])

  const handleReloadFrontend = useCallback(() => {
    window.location.reload()
  }, [])

  const handleDroppedFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) {
        return
      }
      const modelFiles = files.filter(isModelFile)
      const firstImage = files.find((file) => !isModelFile(file) && isImageFile(file))
      const firstVideo = files.find((file) => !isModelFile(file) && isVideoFile(file))
      if (modelFiles.length > 0) {
        await uploadModelFilesAndRefresh(modelFiles)
      }
      if (firstImage) {
        const dataUrl = await readFileAsDataUrl(firstImage)
        const metadataImport = await importGenerationMetadataFromImage(dataUrl, firstImage.name).catch((error: unknown) => {
          reportProClientError({
            kind: 'api',
            message: formatApiError(error),
            source: 'metadata-import',
            context: { filename: firstImage.name },
          })
          return null
        })
        const importedSettings = metadataImport?.settings ?? {}
        if (metadataImport && Object.keys(importedSettings).length > 0) {
          const dimensions = await imageDimensionsFromDataUrl(dataUrl)
          setSettings((current) => applyGenerationSettingsPatch(current, importedSettings, bootstrap.models, bootstrap.aspectRatios))
          setActiveMode(importedSettings.mode === 'inpaint' ? 'inpaint' : 'image')
          setActiveRail('create')
          setPreview({
            id: `metadata-${Date.now()}`,
            url: dataUrl,
            thumbnailUrl: dataUrl,
            prompt: importedSettings.prompt || firstImage.name,
            negativePrompt: importedSettings.negativePrompt,
            infotext: metadataImport.infotext,
            width: dimensions.width || importedSettings.width || settings.width,
            height: dimensions.height || importedSettings.height || settings.height,
            createdAt: new Date().toISOString(),
            mode: importedSettings.mode === 'inpaint' ? 'inpaint' : 'image',
            seed: importedSettings.seed,
            steps: importedSettings.steps,
            cfgScale: importedSettings.cfgScale,
            clipSkip: importedSettings.clipSkip,
            sampler: importedSettings.sampler,
            scheduler: importedSettings.scheduler,
            durationSeconds: readReceiptNumber(metadataImport.receipt, 'elapsed_seconds'),
            speed: readReceiptSpeed(metadataImport.receipt),
            generationSettings: importedSettings,
            generationReceipt: metadataImport.receipt,
            metadata: metadataImport.metadata,
            metadataSchema: typeof metadataImport.metadata.metadata_schema === 'string' ? metadataImport.metadata.metadata_schema : undefined,
            modelName: readImportedModelName(metadataImport, importedSettings),
            source: 'metadata-import',
          })
          setStatusMessage(`Applied generation settings from ${firstImage.name}.`)
        } else if (activeMode === 'video') {
          setSettings((current) => ({
            ...current,
            mode: 'video',
            sourceImageDataUrl: dataUrl,
            sourceImageName: firstImage.name,
          }))
          setStatusMessage(`Loaded ${firstImage.name} as the video first frame.`)
        } else if (activeMode === 'inpaint') {
          setSettings((current) => ({
            ...current,
            mode: 'inpaint',
            initImageDataUrl: dataUrl,
            maskImageDataUrl: '',
          }))
          setStatusMessage(`Loaded ${firstImage.name} into the inpaint canvas.`)
        } else {
          const dimensions = await imageDimensionsFromDataUrl(dataUrl)
          setActiveMode('image')
          setActiveRail('create')
          setPreview({
            id: `local-${Date.now()}`,
            url: dataUrl,
            thumbnailUrl: dataUrl,
            prompt: firstImage.name,
            width: dimensions.width || settings.width,
            height: dimensions.height || settings.height,
            createdAt: new Date().toISOString(),
            mode: 'image',
            modelName: 'Local file',
            source: 'upload',
          })
          setStatusMessage(`Loaded ${firstImage.name} into the image canvas.`)
        }
      } else if (firstVideo) {
        setStatusMessage(`Uploading ${firstVideo.name} to Video Lab...`)
        const probe = await uploadVideoLabFile(firstVideo)
        setActiveRail('tools')
        setActiveMode('video')
        setStatusMessage(
          `Uploaded ${firstVideo.name}: ${probe.width}x${probe.height}, ${probe.frameCount} frames @ ${probe.fps.toFixed(1)} fps.`,
        )
      }
    },
    [activeMode, bootstrap.aspectRatios, bootstrap.models, setPreview, settings.height, settings.width, uploadModelFilesAndRefresh],
  )

  const handleFileDragEnter = useCallback((event: ReactDragEvent<HTMLDivElement>) => {
    if (!event.dataTransfer.types.includes('Files')) {
      return
    }
    event.preventDefault()
    fileDropDepthRef.current += 1
    setFileDropActive(true)
  }, [])

  const handleFileDragOver = useCallback((event: ReactDragEvent<HTMLDivElement>) => {
    if (!event.dataTransfer.types.includes('Files')) {
      return
    }
    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
  }, [])

  const handleFileDragLeave = useCallback((event: ReactDragEvent<HTMLDivElement>) => {
    if (!event.dataTransfer.types.includes('Files')) {
      return
    }
    event.preventDefault()
    fileDropDepthRef.current = Math.max(0, fileDropDepthRef.current - 1)
    if (fileDropDepthRef.current === 0) {
      setFileDropActive(false)
    }
  }, [])

  const handleFileDrop = useCallback(
    (event: ReactDragEvent<HTMLDivElement>) => {
      if (!event.dataTransfer.types.includes('Files')) {
        return
      }
      event.preventDefault()
      fileDropDepthRef.current = 0
      setFileDropActive(false)
      const files = Array.from(event.dataTransfer.files)
      void handleDroppedFiles(files).catch((error: unknown) => {
        const message = formatApiError(error)
        setStatusMessage(message)
        showSupportIssue('File import failed', error, 'file-drop', { fileCount: files.length })
        reportProClientError({
          kind: 'file-drop',
          message,
          source: 'handleFileDrop',
        })
      })
    },
    [handleDroppedFiles, showSupportIssue],
  )

  const handleGenerate = useCallback(async () => {
    if (generationAbortRef.current) {
      if (generationActive) {
        setStatusMessage('Generation is already running.')
        return false
      }
      generationAbortRef.current = null
    }
    if (generationActive) {
      setStatusMessage('Generation is already running in the backend.')
      return false
    }
    if (!settings.prompt.trim()) {
      setStatusMessage('Enter a prompt before generating.')
      return false
    }
    const requestedModel = bootstrap.models.find((model) => model.id === settings.modelId)
    if (requestedModel && !modelFitsCreationMode(requestedModel, settings.mode)) {
      const message =
        settings.mode === 'video'
          ? 'Pick a Wan or Sana Video model before generating video.'
          : 'Video models are only available from the Video tab.'
      setGenerationError(message)
      setStatusMessage(message)
      return false
    }
    if (settings.mode === 'inpaint') {
      if (!settings.initImageDataUrl) {
        setStatusMessage('Load an image into the inpaint canvas first.')
        return false
      }
      if (!settings.maskImageDataUrl) {
        setStatusMessage('Paint a mask over the area you want to regenerate.')
        return false
      }
      if (selectedModel && !['sd15', 'sdxl', 'flux_fill'].includes(selectedModel.engineId ?? 'unknown')) {
        const message = 'Inpainting supports SD 1.5, SDXL, and Flux Fill checkpoints. Pick one of those models.'
        setGenerationError(message)
        setStatusMessage(message)
        return false
      }
    }
    if (settings.controlNetEnabled) {
      const controlNetStatus = getControlNetCompatibility(selectedModel, settings.controlNetModel)
      if (!controlNetStatus.supported) {
        const message = controlNetStatus.message
        setGenerationError(message)
        setStatusMessage(message)
        return false
      }
    }
    if (isModelBlocked(selectedModel)) {
      const message = modelBlockedMessage(selectedModel)
      setGenerationError(message)
      setStatusMessage(message)
      return false
    }
    if (!bootstrap.models.some((model) => model.id === settings.modelId)) {
      const message = 'Selected model is not available in the current Pro model list.'
      setGenerationError(message)
      setStatusMessage(message)
      return false
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
        commitRecentOutputs(stampedOutputs)
      }
      setStatusMessage(result.message || `Generation ${result.status}.`)
      setGenerationError('')
      return true
    } catch (error: unknown) {
      if (isGenerationCancelResult(error)) {
        setGenerationError('')
        setStatusMessage('Generation stop requested.')
        return false
      } else {
        const nextMessage = `Generation failed: ${formatApiError(error)}`
        setGenerationError(nextMessage)
        setStatusMessage(nextMessage)
        showSupportIssue('Generation failed', error, 'generate', {
          mode: settings.mode,
          modelId: settings.modelId,
          route: '/api/pro/generate',
          width: settings.width,
          height: settings.height,
          steps: settings.steps,
        })
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
        return false
      }
    } finally {
      if (generationAbortRef.current === controller) {
        generationAbortRef.current = null
      }
      setIsGenerating(false)
      void fetchProRuntime().then(setRuntime).catch(() => undefined)
      void fetchProLogs().then(setLogStatus).catch(() => undefined)
    }
  }, [bootstrap.models, commitRecentOutputs, generationActive, selectedModel, settings])

  const handleOpenXyPlot = useCallback(() => {
    setXyPlotCells((current) => {
      if (current.length > 0) {
        return normalizeXyPlotCells(current, xyPlotModels, settings)
      }
      return buildDefaultXyPlotCells(settings, xyPlotModels)
    })
    setXyPlotStatus(xyPlotModels.length > 0 ? 'Ready.' : 'No image models are ready for X/Y testing.')
    setActiveModal('xyPlot')
  }, [settings, xyPlotModels])

  const handleResetXyPlot = useCallback(() => {
    setXyPlotCells(buildDefaultXyPlotCells(settings, xyPlotModels))
    setXyPlotStatus('Reset to the current image settings.')
  }, [settings, xyPlotModels])

  const handleXyPlotCellChange = useCallback((id: string, patch: Partial<XyPlotCell>) => {
    setXyPlotCells((current) =>
      current.map((cell) =>
        cell.id === id
          ? {
              ...cell,
              ...patch,
              steps: patch.steps !== undefined ? clamp(Math.round(patch.steps), 1, 150) : cell.steps,
            }
          : cell,
      ),
    )
  }, [])

  const handleAddXyPlotCell = useCallback(() => {
    setXyPlotCells((current) => {
      if (current.length >= XY_PLOT_MAX_CELLS) {
        return current
      }
      const model = xyPlotModels[current.length % Math.max(1, xyPlotModels.length)]
      const previous = current[current.length - 1]
      const nextSteps = clamp((previous?.steps ?? settings.steps) + 5, 1, 150)
      return [
        ...current,
        {
          id: `xy-${Date.now()}-${current.length}`,
          modelId: model?.id ?? settings.modelId,
          steps: nextSteps,
        },
      ]
    })
  }, [settings.modelId, settings.steps, xyPlotModels])

  const handleRemoveXyPlotCell = useCallback((id: string) => {
    setXyPlotCells((current) => (current.length <= 1 ? current : current.filter((cell) => cell.id !== id)))
  }, [])

  const handleRunXyPlot = useCallback(async () => {
    if (generationAbortRef.current) {
      if (generationActive) {
        setXyPlotStatus('Generation is already running.')
        setStatusMessage('Generation is already running.')
        return
      }
      generationAbortRef.current = null
    }
    if (generationActive) {
      setXyPlotStatus('Generation is already running in the backend.')
      setStatusMessage('Generation is already running in the backend.')
      return
    }
    if (!settings.prompt.trim()) {
      setXyPlotStatus('Enter a prompt before running the X/Y plot.')
      setStatusMessage('Enter a prompt before generating.')
      return
    }

    const normalizedCells = normalizeXyPlotCells(xyPlotCells, xyPlotModels, settings)
    const runnableCells = normalizedCells.filter((cell) => xyPlotModels.some((model) => model.id === cell.modelId))
    setXyPlotCells(normalizedCells)
    if (runnableCells.length === 0) {
      setXyPlotStatus('No ready image model is selected.')
      return
    }

    const controller = new AbortController()
    generationAbortRef.current = controller
    setIsGenerating(true)
    setGenerationProgress([])
    setGenerationTimings({})
    setGenerationReceiptPath('')
    setGenerationError('')
    setXyPlotStatus(`Running 1 of ${runnableCells.length}.`)
    setStatusMessage(`Running X/Y plot 1/${runnableCells.length}...`)
    reportProClientEvent({
      action: 'pro-xy-plot-submit',
      detail: `${runnableCells.length} image tests`,
      context: {
        cells: runnableCells.map((cell) => ({ modelId: cell.modelId, steps: cell.steps })),
      },
    })

    const allOutputs: RecentOutput[] = []
    try {
      for (let index = 0; index < runnableCells.length; index += 1) {
        const cell = runnableCells[index]
        const model = xyPlotModels.find((item) => item.id === cell.modelId)
        if (!model) {
          throw new Error(`X/Y cell ${index + 1} has no ready image model selected.`)
        }
        const requestSettings: GenerationSettings = {
          ...settings,
          mode: 'image',
          modelId: model.id,
          steps: clamp(Math.round(cell.steps), 1, 150),
          batchSize: 1,
          batchCount: 1,
          initImageDataUrl: '',
          maskImageDataUrl: '',
        }
        setXyPlotStatus(`Running ${index + 1} of ${runnableCells.length}: ${model.name}, ${requestSettings.steps} steps.`)
        setStatusMessage(`Running X/Y plot ${index + 1}/${runnableCells.length}: ${model.name}.`)
        const result = await generateProOutput(requestSettings, controller.signal)
        setGenerationProgress(result.progress)
        setGenerationTimings(result.timings)
        setGenerationReceiptPath(result.receiptPath ?? '')
        const stampedOutputs = collectGenerateOutputs(result, model.name)
        if (stampedOutputs.length > 0) {
          allOutputs.push(...stampedOutputs)
          setPreview(stampedOutputs[stampedOutputs.length - 1])
          commitRecentOutputs(stampedOutputs)
        }
      }
      setStatusMessage(`X/Y plot complete: ${allOutputs.length} image${allOutputs.length === 1 ? '' : 's'}.`)
      setXyPlotStatus(`Complete: ${allOutputs.length} image${allOutputs.length === 1 ? '' : 's'} generated.`)
      setGenerationError('')
      setActiveModal(null)
    } catch (error: unknown) {
      if (isGenerationCancelResult(error)) {
        setGenerationError('')
        setXyPlotStatus('X/Y plot stopped.')
        setStatusMessage('Generation stop requested.')
      } else {
        const nextMessage = `X/Y plot failed: ${formatApiError(error)}`
        setGenerationError(nextMessage)
        setXyPlotStatus(nextMessage)
        setStatusMessage(nextMessage)
        showSupportIssue('X/Y plot failed', error, 'xy-plot', {
          route: '/api/pro/generate',
          cellCount: runnableCells.length,
          cells: runnableCells.map((cell) => ({ modelId: cell.modelId, steps: cell.steps })),
        })
        reportProClientError({
          kind: 'generation',
          message: nextMessage,
          stack: error instanceof Error ? error.stack : undefined,
          source: 'handleRunXyPlot',
          context: {
            route: '/api/pro/generate',
            cellCount: runnableCells.length,
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
  }, [commitRecentOutputs, generationActive, settings, showSupportIssue, xyPlotCells, xyPlotModels])

  const handleStopGenerate = useCallback(() => {
    continuousGenerateRef.current = false
    setContinuousGenerating(false)
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

  const handleToggleContinuousGenerate = useCallback(() => {
    if (continuousGenerateRef.current) {
      continuousGenerateRef.current = false
      setContinuousGenerating(false)
      handleStopGenerate()
      return
    }
    if (generationActive) {
      setStatusMessage('Wait for the current generation to finish before starting continuous mode.')
      return
    }
    continuousGenerateRef.current = true
    setContinuousGenerating(true)
    setStatusMessage('Continuous generation started.')
    reportProClientEvent({
      action: 'pro-generate-continuous-start',
      detail: `${settings.mode} ${settings.width}x${settings.height}`,
      context: {
        mode: settings.mode,
        modelId: settings.modelId,
        steps: settings.steps,
      },
    })
    void (async () => {
      while (continuousGenerateRef.current) {
        const completed = await handleGenerate()
        if (!completed || !continuousGenerateRef.current) {
          break
        }
      }
      if (continuousGenerateRef.current) {
        continuousGenerateRef.current = false
      }
      setContinuousGenerating(false)
    })()
  }, [generationActive, handleGenerate, handleStopGenerate, settings.height, settings.mode, settings.modelId, settings.steps, settings.width])

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

  const handleApplyOutputSettings = useCallback((output: RecentOutput) => {
    const patch = buildOutputGenerationSettingsPatch(output)
    setSettings((current) => applyGenerationSettingsPatch(current, patch, bootstrap.models, bootstrap.aspectRatios))
    if (patch.mode === 'image' || patch.mode === 'inpaint' || patch.mode === 'video') {
      setActiveMode(patch.mode)
    } else {
      setActiveMode('image')
    }
    setActiveRail('create')
    setStatusMessage('Output settings applied to the current generation controls.')
  }, [bootstrap.aspectRatios, bootstrap.models])

  const readCurrentPreviewDataUrl = useCallback(async () => {
    if (!preview?.url) {
      return ''
    }
    if (preview.url.startsWith('data:')) {
      return preview.url
    }
    const blob = await (await fetch(preview.url)).blob()
    return await new Promise<string>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = () => reject(new Error('Could not read the current preview image.'))
      reader.readAsDataURL(blob)
    })
  }, [preview?.url])

  const handleEnhanceSourceChange = useCallback((event: ReactChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) {
      return
    }
    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        setEnhanceSourceDataUrl(reader.result)
        setEnhanceSourceName(file.name)
        setEnhanceMessage(`Loaded ${file.name}.`)
      }
    }
    reader.readAsDataURL(file)
  }, [])

  const handleUsePreviewForEnhance = useCallback(async () => {
    try {
      const dataUrl = await readCurrentPreviewDataUrl()
      if (!dataUrl) {
        setEnhanceMessage('Generate or select an image first.')
        return
      }
      setEnhanceSourceDataUrl(dataUrl)
      setEnhanceSourceName('Current preview')
      setEnhanceMessage('Current preview loaded for Enhance.')
    } catch (error: unknown) {
      setEnhanceMessage(`Could not load preview: ${formatApiError(error)}`)
    }
  }, [readCurrentPreviewDataUrl])

  const handleRunEnhance = useCallback(async () => {
    let source = enhanceSourceDataUrl
    if (!source) {
      source = await readCurrentPreviewDataUrl()
    }
    if (!source) {
      setEnhanceMessage('Upload an image or use the current preview first.')
      return
    }
    setEnhanceBusy(true)
    setEnhanceMessage(enhanceMode === 'vsr' ? 'Running NVIDIA VSR image upscale...' : 'Running Enhance pipeline...')
    try {
      const result = enhanceMode === 'vsr'
        ? await runVsrImage({
            imageDataUrl: source,
            scale: enhanceVsrScale,
            mode: enhanceVsrMode,
            effect: 'SuperRes',
            strength: enhanceVsrStrength,
          })
        : await runEnhanceImage({
            imageDataUrl: source,
            restoreEnabled: enhanceMode === 'restore' || enhanceMode === 'restore-upscale',
            restoreModel: enhanceRestoreModel,
            restoreVisibility: enhanceRestoreVisibility,
            codeformerWeight: enhanceCodeformerWeight,
            upscaleEnabled: enhanceMode === 'upscale' || enhanceMode === 'restore-upscale',
            upscaleModel: enhanceUpscaleModel,
            upscaleScale: enhanceUpscaleScale,
            tileSize: enhanceTileSize,
            tileOverlap: enhanceTileOverlap,
            restoreFirst: true,
          })
      const outputUrl = result.image || result.url
      if (!outputUrl) {
        setEnhanceMessage(result.message || 'Enhance completed without an image payload.')
        return
      }
      const enhanced: RecentOutput = {
        id: `pro-${enhanceMode}-${Date.now()}`,
        url: outputUrl,
        thumbnailUrl: outputUrl,
        width: result.width || preview?.width || settings.width,
        height: result.height || preview?.height || settings.height,
        prompt: preview?.prompt || 'Pro image post-process',
        negativePrompt: preview?.negativePrompt || '',
        modelName: enhanceMode === 'vsr' ? 'NVIDIA VSR' : 'Pro Enhance',
        mode: 'image',
        seed: preview?.seed,
        steps: preview?.steps,
        cfgScale: preview?.cfgScale,
        sampler: preview?.sampler,
        scheduler: preview?.scheduler,
        infotext: result.infotext || result.message,
        createdAt: new Date().toISOString(),
        status: 'completed',
        path: result.outputPath,
      }
      setPreview(enhanced)
      setBootstrap((current) => ({
        ...current,
        recentOutputs: mergeRecentOutputs([enhanced], current.recentOutputs),
      }))
      setEnhanceMessage(result.message || 'Enhance complete.')
      setStatusMessage(result.message || 'Enhance complete.')
    } catch (error: unknown) {
      const message = `Enhance failed: ${formatApiError(error)}`
      setEnhanceMessage(message)
      showSupportIssue('Enhance failed', error, 'enhance', { route: '/api/pro/enhance/image' })
    } finally {
      setEnhanceBusy(false)
    }
  }, [
    enhanceCodeformerWeight,
    enhanceMode,
    enhanceRestoreModel,
    enhanceRestoreVisibility,
    enhanceSourceDataUrl,
    enhanceTileOverlap,
    enhanceTileSize,
    enhanceUpscaleModel,
    enhanceUpscaleScale,
    enhanceVsrMode,
    enhanceVsrScale,
    enhanceVsrStrength,
    preview,
    readCurrentPreviewDataUrl,
    settings.height,
    settings.width,
    setPreview,
  ])

  const handleLayoutReset = useCallback(() => {
    setLeftPanelWidth(380)
    setRightPanelWidth(320)
    setLeftPanelCollapsed(false)
    setRightPanelCollapsed(false)
    setBottomDockHeight(196)
    setBottomDockVisible(true)
    setOutputPreviewVisible(true)
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

  // Stable callbacks so the memoized shell components (PromptPanel, canvas,
  // dock, menus) skip re-rendering on every runtime stream tick.
  const settingsRef = useRef(settings)
  useEffect(() => {
    settingsRef.current = settings
  }, [settings])
  const openSegmentationModal = useCallback(() => setActiveModal('segmentation'), [])
  const openHiresModal = useCallback(() => setActiveModal('hires'), [])
  const openEnhanceModal = useCallback(() => setActiveModal('enhance'), [])
  const openReactorModal = useCallback(() => setActiveModal('reactor'), [])
  const openControlNetModal = useCallback(() => setActiveModal('controlnet'), [])
  const toggleBottomDock = useCallback(() => setBottomDockVisible((value) => !value), [])
  const toggleOutputPreview = useCallback(() => setOutputPreviewVisible((value) => !value), [])
  const toggleAdvanced = useCallback(() => setShowAdvanced((value) => !value), [])
  const collapseLeftPanel = useCallback(() => setLeftPanelCollapsed(true), [])
  const toggleRightPanel = useCallback(() => setRightPanelCollapsed((value) => !value), [])
  const startLeftDrag = useCallback(() => startHorizontalDrag('left'), [startHorizontalDrag])
  const startRightDrag = useCallback(() => startHorizontalDrag('right'), [startHorizontalDrag])
  const openSettingsRail = useCallback(() => handleRailSelect('settings'), [handleRailSelect])

  const handleMenuAction = useCallback((action: string) => {
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
    } else if (action === 'open-enhance') {
      setActiveModal('enhance')
    } else if (action === 'open-controlnet') {
      setActiveModal('controlnet')
    } else if (action === 'copy-last') {
      void navigator.clipboard?.writeText(settingsRef.current.prompt)
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
  }, [handleLayoutReset, handleRailSelect])

  const workspaceIsFullSurface = FULL_SURFACE_RAILS.has(activeRail)
  const workspaceStyle = activeRail === 'models' || workspaceIsFullSurface
    ? undefined
    : ({
        '--left-panel-width': `${leftPanelWidth}px`,
        '--right-panel-width': `${rightPanelWidth}px`,
      } as CSSProperties)
  const workspaceClassName = [
    'pro-workspace',
    workspaceIsFullSurface ? 'pro-workspace-full-surface' : '',
    activeRail === 'create' && leftPanelCollapsed ? 'pro-workspace-left-collapsed' : '',
    activeRail === 'create' && rightPanelCollapsed ? 'pro-workspace-right-collapsed' : '',
  ].filter(Boolean).join(' ')
  const activeRailItems = railsForMode(activeMode, activeRail)

  return (
    <div
      className={`aiwf-pro-shell theme-preset-1${fileDropActive ? ' is-file-drop-active' : ''}`}
      data-mode={activeMode}
      data-rail={STUDIO_RAIL_ATTRIBUTE[activeRail] ?? activeRail}
      onDragEnter={handleFileDragEnter}
      onDragOver={handleFileDragOver}
      onDragLeave={handleFileDragLeave}
      onDrop={handleFileDrop}
    >
      {startupSplashVisible ? <StartupSplash ready={backendConnected} /> : null}
      {fileDropActive ? (
        <div className="pro-file-drop-overlay" aria-hidden="true">
          <div>
            <HardDrive size={28} aria-hidden="true" />
            <strong>Drop files</strong>
            <span>Images load into the current canvas. Videos go to Video Lab. Models go to the sorter.</span>
          </div>
        </div>
      ) : null}
      {supportIssue ? (
        <SupportIssueDialog
          issue={supportIssue}
          onClose={() => setSupportIssue(null)}
          onCopy={async () => {
            try {
              await copyTextToClipboard(JSON.stringify(supportIssue, null, 2))
              setStatusMessage('Error report copied.')
            } catch {
              setStatusMessage('Could not copy the error report.')
            }
          }}
          onSubmit={() => {
            reportProClientError({
              kind: 'tester-report',
              message: supportIssue.message,
              source: supportIssue.source,
              context: {
                ...supportIssue.context,
                detail: supportIssue.detail,
                createdAt: supportIssue.createdAt,
              },
            })
            setStatusMessage('Local error report saved. Send the copied report if Shawn asks for it.')
          }}
        />
      ) : null}
      <aside className="pro-rail" aria-label="Subnavigation">
        <button
          type="button"
          className="pro-logo-button"
          aria-label="AIWF Studio home"
          onClick={() => handleRailSelect('create')}
        >
          <img className="pro-logo-image" src={PRO_APP_ICON} alt="" />
        </button>
        <nav className="pro-rail-nav">
          {activeRailItems.map((item) => (
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
          onAction={handleMenuAction}
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
          liveStreamConnected={runtimeStreamConnected}
          onRecoverBackend={handleRecoverBackend}
          onOpenSettings={openSettingsRail}
          onCopyGenerationError={handleCopyGenerationError}
        />
        <ModeTabs activeMode={activeMode} onSelect={handleModeSelect} />

        <section
          className={workspaceClassName}
          aria-label="AIWF Pro workspace"
          style={workspaceStyle}
        >
          {activeRail === 'families' ? (
            <ModelFamilyMatrixLayout {...buildLayoutProps()} />
          ) : activeRail === 'foundry' ? (
            <MediaFoundryImageLayout {...buildLayoutProps()} />
          ) : activeRail === 'pipeline' ? (
            <PipelineAtlasLayout {...buildLayoutProps()} />
          ) : activeRail === 'projects' ? (
            <ProjectCenterLayout {...buildLayoutProps()} />
          ) : activeRail === 'assistant' ? (
            <AgenticChatLayout {...buildLayoutProps()} />
          ) : activeRail === 'audiolab' ? (
            <AudioStudioLayout {...buildLayoutProps()} />
          ) : activeRail === 'workflow' ? (
            <section className="pro-workspace-surface" aria-label="Workflow">
              <WorkspaceHeader
                eyebrow="Workflow"
                title="Workflow builder"
                description="Send settings here from any generation tab, then reorder the nodes into the exact pipeline you want."
              />
              <div className="pro-workflow-wrap">
                <WorkflowPanel
                  blocks={workflowBlocks}
                  onChange={setWorkflowBlocks}
                  runStatus={workflowStatus}
                />
              </div>
            </section>
          ) : activeRail === 'models' ? (
            <ModelsWorkspace
              engineFilter={engineFilter}
              engines={bootstrap.engines}
              models={bootstrap.models}
              downloadsStatus={downloadsStatus}
              selectedModelId={settings.modelId}
              onEngineFilterChange={handleEngineFilterChange}
              onModelSelect={handleModelSelect}
              onCatalogDownload={handleCatalogDownload}
              downloadingCatalogKey={downloadingCatalogKey}
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
                onMouseDown={startLeftDrag}
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
                onMouseDown={startRightDrag}
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
                onOpenEnhance={() => setActiveModal('enhance')}
                onOpenReactor={() => setActiveModal('reactor')}
              />
              <ResizeHandle
                axis="vertical"
                label="Resize left panel"
                onMouseDown={startLeftDrag}
              />
              <ToolsWorkspace
                capabilitiesStatus={capabilitiesStatus}
                runtime={runtime}
                wanModels={bootstrap.models.filter((model) => (model.engineId ?? 'unknown') === 'wan')}
                onOpenCreate={() => handleRailSelect('create')}
                onOpenVideo={() => {
                  handleRailSelect('create')
                  handleModeSelect('video')
                }}
                onOpenData={() => handleRailSelect('data')}
                onOpenSegmentation={() => setActiveModal('segmentation')}
                onOpenEnhance={() => setActiveModal('enhance')}
                onOpenReactor={() => setActiveModal('reactor')}
              />
              <ResizeHandle
                axis="vertical"
                label="Resize right panel"
                onMouseDown={startRightDrag}
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
                onMouseDown={startLeftDrag}
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
                onMouseDown={startRightDrag}
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
                onMouseDown={startLeftDrag}
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
                onMouseDown={startRightDrag}
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
                onMouseDown={startLeftDrag}
              />
              <SettingsWorkspace
                bootstrap={bootstrap}
                runtime={runtime}
                settings={settings}
                settingsStatus={settingsStatus}
                recentOutputs={recentOutputs}
                onSettingsChange={setSettings}
                onSettingsStatusChange={setSettingsStatus}
                onSaveSettings={handleSaveProSettings}
                onModelFilesUpload={uploadModelFilesAndRefresh}
                onModelReorganize={reorganizeModelFilesNow}
                onUnloadModel={handleUnloadModel}
                onRestartBackend={handleRecoverBackend}
                onReloadFrontend={handleReloadFrontend}
                settingsSaveStatus={settingsSaveStatus}
                leftPanelWidth={leftPanelWidth}
                rightPanelWidth={rightPanelWidth}
                bottomDockHeight={bottomDockHeight}
                bottomDockVisible={bottomDockVisible}
                showAdvanced={showAdvanced}
              />
              <ResizeHandle
                axis="vertical"
                label="Resize right panel"
                onMouseDown={startRightDrag}
              />
            </>
          ) : (
            <>
              {leftPanelCollapsed ? (
                <CollapsedPanelButton side="left" label="Show prompt column" icon={PanelLeft} onClick={() => setLeftPanelCollapsed(false)} />
              ) : (
                <PromptPanel
                  settings={settings}
                  bootstrap={bootstrap}
                  filteredModels={filteredModels}
                  engineFilter={engineFilter}
                  engines={creationEngines}
                  selectedModelName={selectedModel?.name ?? settings.modelId}
                  activeRatio={activeRatio}
                  showAdvanced={showAdvanced}
                  isGenerating={generationActive}
                  isContinuousGenerating={continuousGenerating}
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
                  onToggleContinuousGenerate={handleToggleContinuousGenerate}
                  onSendToWorkflow={handleSendToWorkflow}
                  onToggleAdvanced={toggleAdvanced}
                  onToggleLeftPanel={collapseLeftPanel}
                  onToggleRightPanel={toggleRightPanel}
                  onOpenXyPlot={handleOpenXyPlot}
                  onPromptAnalyze={handlePromptAnalyze}
                  selectedModelWarning={selectedModelWarning}
                  rightPanelCollapsed={rightPanelCollapsed}
                  dualRuntimeAvailable={dualRuntimeAvailable}
                  sdcppRuntimeAvailable={sdcppRuntimeAvailable}
                />
              )}
              {leftPanelCollapsed ? (
                <div className="pro-panel-gap" aria-hidden="true" />
              ) : (
                <ResizeHandle
                  axis="vertical"
                  label="Resize left panel"
                  onMouseDown={startLeftDrag}
                />
              )}
              <div className="pro-center-column">
                {activeMode === 'inpaint' ? (
                  <InpaintCanvas
                    settings={settings}
                    onSettingsChange={setSettings}
                    statusMessage={statusMessage}
                    preview={preview}
                    onOpenSegmentation={openSegmentationModal}
                    onOpenControlNet={openControlNetModal}
                    controlNetEnabled={settings.controlNetEnabled}
                    controlNetAvailable={controlNetCompatibility.supported}
                    controlNetUnavailableMessage={controlNetCompatibility.message}
                    leftPanelCollapsed={leftPanelCollapsed}
                    isGenerating={generationActive}
                    isContinuousGenerating={continuousGenerating}
                    selectedModelWarning={selectedModelWarning}
                    onGenerate={handleGenerate}
                    onStopGenerate={handleStopGenerate}
                    onToggleContinuousGenerate={handleToggleContinuousGenerate}
                  />
                ) : (
                  <CanvasPreview
                    activeMode={activeMode}
                    preview={preview}
                    statusMessage={statusMessage}
                    width={settings.width}
                    height={settings.height}
                    onOpenSegmentation={openSegmentationModal}
                    onOpenHires={openHiresModal}
                    onOpenEnhance={openEnhanceModal}
                    onOpenReactor={openReactorModal}
                    onOpenControlNet={openControlNetModal}
                    controlNetEnabled={settings.controlNetEnabled}
                    controlNetAvailable={controlNetCompatibility.supported}
                    controlNetUnavailableMessage={controlNetCompatibility.message}
                    bottomDockVisible={bottomDockVisible}
                    outputPreviewVisible={outputPreviewVisible}
                    onToggleBottomDock={toggleBottomDock}
                    onToggleOutputPreview={toggleOutputPreview}
                    leftPanelCollapsed={leftPanelCollapsed}
                    isGenerating={generationActive}
                    isContinuousGenerating={continuousGenerating}
                    selectedModelWarning={selectedModelWarning}
                    onGenerate={handleGenerate}
                    onStopGenerate={handleStopGenerate}
                    onToggleContinuousGenerate={handleToggleContinuousGenerate}
                  />
                )}
                <BottomDock
                  visible={bottomDockVisible}
                  height={bottomDockVisible ? bottomDockHeight : 0}
                  recentOutputs={recentOutputs}
                  selectedOutput={preview}
                  statusMessage={statusMessage}
                  generationError={generationError}
                  selectedModelName={selectedModel?.name ?? settings.modelId}
                  onPreviewSelect={setPreview}
                  onApplyOutputSettings={handleApplyOutputSettings}
                  onResizeStart={startBottomDrag}
                  onToggleVisible={toggleBottomDock}
                />
              </div>
              {rightPanelCollapsed ? (
                <div className="pro-panel-gap" aria-hidden="true" />
              ) : (
                <ResizeHandle
                  axis="vertical"
                  label="Resize right panel"
                  onMouseDown={startRightDrag}
                />
              )}
            </>
          )}
          {workspaceIsFullSurface ? null : activeRail === 'create' && rightPanelCollapsed ? (
            <CollapsedPanelButton side="right" label="Show system column" icon={PanelRight} onClick={() => setRightPanelCollapsed(false)} />
          ) : (
            <RuntimePanel
              runtime={runtime}
              selectedModelName={selectedModel?.name ?? settings.modelId}
              onUnloadModel={handleUnloadModel}
              onToggleRightPanel={() => setRightPanelCollapsed(true)}
            />
          )}
        </section>

      </main>

      <ToolModal open={activeModal === 'xyPlot'} title="X/Y plot" onClose={() => setActiveModal(null)}>
        <XyPlotSetupModal
          cells={xyPlotCells}
          models={xyPlotModels}
          running={generationActive}
          status={xyPlotStatus}
          onCellChange={handleXyPlotCellChange}
          onAddCell={handleAddXyPlotCell}
          onRemoveCell={handleRemoveXyPlotCell}
          onReset={handleResetXyPlot}
          onRun={handleRunXyPlot}
        />
      </ToolModal>

      <ToolModal open={activeModal === 'controlnet'} title="ControlNet" onClose={() => setActiveModal(null)}>
        <ControlNetSettingsModal
          settings={settings}
          onSettingsChange={setSettings}
          compatibility={controlNetCompatibility}
        />
      </ToolModal>

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
          <p className="pro-field-note">
            Quick auto-mask controls are on the Inpaint canvas. Full SAM box, point, and DINO workflows remain in Gradio Lab.
          </p>
        </div>
      </ToolModal>

      <ToolModal open={activeModal === 'hires'} title="High-res fix" onClose={() => setActiveModal(null)}>
        <div className="pro-modal-form">
          <label className="pro-toggle">
            <input
              type="checkbox"
              checked={settings.enableHires}
              onChange={(event) => setSettings((current) => ({ ...current, enableHires: event.target.checked }))}
            />
            <span>Enable high-res pass</span>
          </label>
          <RangeField
            label="Scale"
            min={1}
            max={4}
            step={0.05}
            value={settings.hiresScale}
            onChange={(value) => setSettings((current) => ({ ...current, hiresScale: value }))}
          />
          <RangeField
            label="Denoise"
            min={0}
            max={1}
            step={0.05}
            value={settings.hiresDenoise}
            onChange={(value) => setSettings((current) => ({ ...current, hiresDenoise: value }))}
          />
          <label className="pro-field">
            <FieldLabel
              label="Second-pass steps"
              tooltip="Use fewer high-res steps than the base pass unless the model visibly needs more detail cleanup."
            />
            <input
              type="number"
              min={0}
              max={80}
              value={settings.hiresSteps}
              onChange={(event) => setSettings((current) => ({ ...current, hiresSteps: Number(event.target.value) }))}
            />
          </label>
          <label className="pro-field">
            <FieldLabel
              label="Upscaler"
              tooltip="Diffusers accepts common resize names such as latent, nearest, lanczos, bicubic, or a project upscaler id when supported by the route."
            />
            <input
              value={settings.hiresUpscaler}
              placeholder="latent, lanczos, bicubic, nearest"
              onChange={(event) => setSettings((current) => ({ ...current, hiresUpscaler: event.target.value }))}
            />
          </label>
        </div>
      </ToolModal>

      <ToolModal open={activeModal === 'enhance'} title="Enhance / VSR" onClose={() => setActiveModal(null)}>
        <div className="pro-modal-form">
          <div className="pro-enhance-source-row">
            {enhanceSourceDataUrl ? (
              <img className="pro-enhance-preview" src={enhanceSourceDataUrl} alt="" />
            ) : (
              <div className="pro-enhance-empty">No source image</div>
            )}
            <div className="pro-enhance-source-actions">
              <span>{enhanceSourceName || 'Use the current canvas image or upload a source.'}</span>
              <button type="button" className="pro-secondary-button" onClick={handleUsePreviewForEnhance} disabled={enhanceBusy || !preview?.url}>
                Use current preview
              </button>
              <label className="pro-secondary-button" htmlFor="pro-enhance-source-input">
                <FileImage size={15} aria-hidden="true" />
                <span>Upload image</span>
              </label>
              <input
                id="pro-enhance-source-input"
                className="pro-file-input-hidden"
                type="file"
                accept={IMAGE_FILE_ACCEPT}
                onChange={handleEnhanceSourceChange}
                disabled={enhanceBusy}
              />
            </div>
          </div>
          <label className="pro-field">
            <FieldLabel
              label="Mode"
              tooltip="Face restore uses GFPGAN/CodeFormer-style restorer models. Upscale uses local Enhance upscalers. VSR uses NVIDIA VideoFX when installed."
            />
            <select value={enhanceMode} onChange={(event) => setEnhanceMode(event.target.value as typeof enhanceMode)} disabled={enhanceBusy}>
              <option value="restore">Face restore</option>
              <option value="upscale">Upscale</option>
              <option value="restore-upscale">Face restore + upscale</option>
              <option value="vsr">NVIDIA VSR image upscale</option>
            </select>
          </label>
          {enhanceMode !== 'upscale' && enhanceMode !== 'vsr' ? (
            <>
              <label className="pro-field">
                <FieldLabel label="Face restorer model" />
                <input
                  value={enhanceRestoreModel}
                  onChange={(event) => setEnhanceRestoreModel(event.target.value)}
                  placeholder="gfpgan-v1.4 or codeformer"
                  disabled={enhanceBusy}
                />
              </label>
              <RangeField
                label="Restore strength"
                min={0}
                max={1}
                step={0.05}
                value={enhanceRestoreVisibility}
                onChange={setEnhanceRestoreVisibility}
              />
              <RangeField
                label="CodeFormer weight"
                min={0}
                max={1}
                step={0.05}
                value={enhanceCodeformerWeight}
                onChange={setEnhanceCodeformerWeight}
              />
            </>
          ) : null}
          {enhanceMode === 'upscale' || enhanceMode === 'restore-upscale' ? (
            <>
              <label className="pro-field">
                <FieldLabel label="Upscaler model" />
                <input
                  value={enhanceUpscaleModel}
                  onChange={(event) => setEnhanceUpscaleModel(event.target.value)}
                  placeholder="realesrgan-x4plus"
                  disabled={enhanceBusy}
                />
              </label>
              <RangeField label="Scale" min={1} max={8} step={0.5} value={enhanceUpscaleScale} onChange={setEnhanceUpscaleScale} />
              <div className="pro-control-grid">
                <label className="pro-field">
                  <FieldLabel label="Tile size" />
                  <input
                    type="number"
                    min={0}
                    max={2048}
                    step={64}
                    value={enhanceTileSize}
                    onChange={(event) => setEnhanceTileSize(Number(event.target.value))}
                    disabled={enhanceBusy}
                  />
                </label>
                <label className="pro-field">
                  <FieldLabel label="Tile overlap" />
                  <input
                    type="number"
                    min={0}
                    max={512}
                    step={16}
                    value={enhanceTileOverlap}
                    onChange={(event) => setEnhanceTileOverlap(Number(event.target.value))}
                    disabled={enhanceBusy}
                  />
                </label>
              </div>
            </>
          ) : null}
          {enhanceMode === 'vsr' ? (
            <>
              <RangeField label="VSR scale" min={1} max={4} step={0.5} value={enhanceVsrScale} onChange={setEnhanceVsrScale} />
              <div className="pro-control-grid">
                <label className="pro-field">
                  <FieldLabel label="VSR mode" />
                  <input
                    type="number"
                    min={0}
                    max={19}
                    value={enhanceVsrMode}
                    onChange={(event) => setEnhanceVsrMode(clamp(Number(event.target.value) || 0, 0, 19))}
                    disabled={enhanceBusy}
                  />
                </label>
                <label className="pro-field">
                  <FieldLabel label="Strength" />
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={enhanceVsrStrength}
                    onChange={(event) => setEnhanceVsrStrength(clamp(Number(event.target.value) || 0, 0, 1))}
                    disabled={enhanceBusy}
                  />
                </label>
              </div>
              <p className="pro-field-note">Video VSR is also available from Video Lab after uploading a clip.</p>
            </>
          ) : null}
          <div className="pro-settings-actions">
            <button type="button" className="pro-primary-button" onClick={handleRunEnhance} disabled={enhanceBusy}>
              {enhanceBusy ? 'Working...' : 'Run'}
            </button>
            <span>{enhanceMessage || 'Ready.'}</span>
          </div>
        </div>
      </ToolModal>

      <ToolModal open={activeModal === 'reactor'} title="ReActor" onClose={() => setActiveModal(null)}>
        <div className="pro-modal-form">
          <label className="pro-field">
            <FieldLabel label="Source face" tooltip="Upload a clear photo of the face to transplant onto the current preview image." />
            <input
              type="file"
              accept={IMAGE_FILE_ACCEPT}
              onChange={(event) => {
                const file = event.target.files?.[0]
                event.target.value = ''
                if (!file) {
                  return
                }
                const reader = new FileReader()
                reader.onload = () => {
                  if (typeof reader.result === 'string') {
                    setReactorSourceDataUrl(reader.result)
                    setReactorMessage(`Source face loaded: ${file.name}`)
                  }
                }
                reader.readAsDataURL(file)
              }}
              disabled={reactorBusy}
            />
          </label>
          {reactorSourceDataUrl ? (
            <img className="pro-video-source-preview" src={reactorSourceDataUrl} alt="Source face" />
          ) : null}
          <button
            type="button"
            className="pro-primary-button"
            disabled={reactorBusy || !reactorSourceDataUrl || !preview?.url}
            onClick={async () => {
              if (!preview?.url || !reactorSourceDataUrl) {
                return
              }
              setReactorBusy(true)
              setReactorMessage('Swapping face…')
              try {
                let targetDataUrl = preview.url
                if (!targetDataUrl.startsWith('data:')) {
                  const blob = await (await fetch(targetDataUrl)).blob()
                  targetDataUrl = await new Promise<string>((resolve, reject) => {
                    const reader = new FileReader()
                    reader.onload = () => resolve(reader.result as string)
                    reader.onerror = () => reject(new Error('Could not read the preview image.'))
                    reader.readAsDataURL(blob)
                  })
                }
                const result = await runFaceSwap(targetDataUrl, reactorSourceDataUrl)
                if (result.image) {
                  const swapped: RecentOutput = {
                    ...preview,
                    id: `${preview.id}-reactor-${Date.now()}`,
                    url: result.image,
                    thumbnailUrl: result.image,
                    createdAt: new Date().toISOString(),
                  }
                  setPreview(swapped)
                  setBootstrap((current) => ({
                    ...current,
                    recentOutputs: mergeRecentOutputs([swapped], current.recentOutputs),
                  }))
                }
                setReactorMessage(result.message || 'Face swap complete.')
              } catch (error: unknown) {
                setReactorMessage(`Face swap failed: ${formatApiError(error)}`)
              } finally {
                setReactorBusy(false)
              }
            }}
          >
            {reactorBusy ? 'Swapping…' : 'Swap onto current preview'}
          </button>
          {reactorMessage ? <p className="pro-field-note">{reactorMessage}</p> : null}
          {!preview?.url ? <p className="pro-field-note">Generate or select an image first — the swap targets the current preview.</p> : null}
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

function XyPlotSetupModal({
  cells,
  models,
  running,
  status,
  onCellChange,
  onAddCell,
  onRemoveCell,
  onReset,
  onRun,
}: {
  cells: XyPlotCell[]
  models: ProModelOption[]
  running: boolean
  status: string
  onCellChange: (id: string, patch: Partial<XyPlotCell>) => void
  onAddCell: () => void
  onRemoveCell: (id: string) => void
  onReset: () => void
  onRun: () => void
}) {
  const canRun = !running && cells.length > 0 && models.length > 0
  return (
    <div className="pro-modal-form pro-xy-plot-panel">
      <p className="pro-field-note">
        Run one image per row using the current prompt, resolution, seed, backend, and sampler. Change model and steps for each test.
      </p>
      <div className="pro-xy-plot-grid">
        {cells.map((cell, index) => (
          <section className="pro-xy-plot-cell" key={cell.id}>
            <div className="pro-xy-plot-cell-header">
              <strong>Test {index + 1}</strong>
              <button
                type="button"
                className="pro-secondary-button ghost"
                onClick={() => onRemoveCell(cell.id)}
                disabled={running || cells.length <= 1}
              >
                Remove
              </button>
            </div>
            <label className="pro-field">
              <FieldLabel label="Model" />
              <select
                value={cell.modelId}
                onChange={(event) => onCellChange(cell.id, { modelId: event.target.value })}
                disabled={running || models.length === 0}
              >
                {models.length === 0 ? <option value="">No ready image models</option> : null}
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {formatModelOptionLabel(model)}
                  </option>
                ))}
              </select>
            </label>
            <label className="pro-field">
              <FieldLabel label="Steps" />
              <input
                type="number"
                min={1}
                max={150}
                value={cell.steps}
                onChange={(event) => onCellChange(cell.id, { steps: Number(event.target.value) })}
                disabled={running}
              />
            </label>
          </section>
        ))}
      </div>
      <div className="pro-xy-plot-actions">
        <button type="button" className="pro-secondary-button" onClick={onAddCell} disabled={running || cells.length >= XY_PLOT_MAX_CELLS || models.length === 0}>
          Add test
        </button>
        <button type="button" className="pro-secondary-button" onClick={onReset} disabled={running}>
          Reset
        </button>
        <button type="button" className="pro-primary-button" onClick={onRun} disabled={!canRun}>
          {running ? 'Running...' : 'Run X/Y plot'}
        </button>
      </div>
      <p className="pro-xy-plot-status" role="status">
        {status || 'Ready.'}
      </p>
    </div>
  )
}

const MenuBar = memo(MenuBarImpl)

function MenuBarImpl({
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

function runtimeLightClass(state: string, hasError: boolean, connected = true): string {
  const normalized = state.trim().toLowerCase()
  if (!connected || ['failed', 'error', 'cancelled', 'canceled', 'disconnected', 'offline'].includes(normalized)) {
    return 'is-danger'
  }
  if (hasError) {
    return 'is-warning'
  }
  if (['running', 'connecting', 'loading', 'queued', 'recovering', 'starting'].includes(normalized)) {
    return 'is-processing'
  }
  return 'is-good'
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
  liveStreamConnected,
  onRecoverBackend,
  onOpenSettings,
  onCopyGenerationError,
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
  liveStreamConnected: boolean
  onRecoverBackend: () => void
  onOpenSettings: () => void
  onCopyGenerationError: () => void | Promise<void>
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
          <strong className="pro-generation-status-text">{progressMessage}</strong>
        </div>
        <div className="pro-generation-meter" aria-label={`Generation progress ${progressPercent}%`}>
          <span style={{ width: `${progressPercent}%` }} />
        </div>
        <div className="pro-generation-meta">
          {progressStep ? <span>{progressStep}</span> : null}
          <span>{selectedModelName}</span>
        </div>
        {activeError ? (
          <button
            type="button"
            className="pro-icon-button pro-generation-copy-button"
            aria-label="Copy generation error"
            title="Copy generation error"
            onClick={onCopyGenerationError}
          >
            <Clipboard size={14} aria-hidden="true" />
          </button>
        ) : null}
      </div>
      <div className="pro-topbar-status">
        <div
          className="pro-engine-status"
          data-state={runtime.state.toLowerCase()}
          title={
            !backendConnected
              ? 'Backend unreachable.'
              : liveStreamConnected
                ? 'Live runtime stream connected.'
                : 'Live stream degraded — falling back to polling.'
          }
        >
          <span
            className={`pro-status-dot ${runtimeLightClass(
              backendRecovering ? 'recovering' : runtime.state,
              Boolean(activeError) || (backendConnected && !liveStreamConnected),
              backendConnected,
            )}`}
            aria-hidden="true"
          />
          <span>Local Engine</span>
          <strong>{backendConnected ? runtime.state : 'Disconnected'}</strong>
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

function SupportIssueDialog({
  issue,
  onClose,
  onCopy,
  onSubmit,
}: {
  issue: SupportIssue
  onClose: () => void
  onCopy: () => void | Promise<void>
  onSubmit: () => void
}) {
  const details = JSON.stringify({ detail: issue.detail, context: issue.context }, null, 2)
  return (
    <div className="pro-support-modal-backdrop" role="presentation">
      <section className="pro-support-modal" role="dialog" aria-modal="true" aria-labelledby="support-issue-title">
        <div className="pro-support-modal-header">
          <div>
            <span>Error report</span>
            <strong id="support-issue-title">{issue.title}</strong>
          </div>
          <button type="button" className="pro-icon-button" aria-label="Close error report" onClick={onClose}>
            <X size={18} aria-hidden="true" />
          </button>
        </div>
        <p>{issue.message}</p>
        <dl className="pro-support-summary">
          <div>
            <dt>Source</dt>
            <dd>{issue.source}</dd>
          </div>
          <div>
            <dt>Time</dt>
            <dd>{issue.createdAt}</dd>
          </div>
          <div>
            <dt>Model</dt>
            <dd>{String(issue.context?.selectedModelName ?? issue.context?.selectedModelId ?? 'Unknown')}</dd>
          </div>
        </dl>
        {issue.detail || issue.context ? (
          <details className="pro-support-details">
            <summary>Technical details</summary>
            <pre>{details}</pre>
          </details>
        ) : null}
        <div className="pro-settings-action-row">
          <button type="button" className="pro-primary-button" onClick={onCopy}>
            Copy report
          </button>
          <button type="button" className="pro-secondary-button" onClick={onSubmit}>
            Save local report
          </button>
          <button type="button" className="pro-secondary-button" onClick={onClose}>
            Dismiss
          </button>
        </div>
      </section>
    </div>
  )
}

const ModeTabs = memo(ModeTabsImpl)

function ModeTabsImpl({
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

const PromptPanel = memo(PromptPanelImpl)

function PromptPanelImpl({
  settings,
  bootstrap,
  filteredModels,
  engineFilter,
  engines,
  selectedModelName,
  activeRatio,
  showAdvanced,
  isGenerating,
  isContinuousGenerating,
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
  onToggleContinuousGenerate,
  onSendToWorkflow,
  onToggleAdvanced,
  onToggleLeftPanel,
  onToggleRightPanel,
  onOpenXyPlot,
  onPromptAnalyze,
  selectedModelWarning,
  rightPanelCollapsed,
  dualRuntimeAvailable,
  sdcppRuntimeAvailable,
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
  isContinuousGenerating: boolean
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
  onToggleContinuousGenerate: () => void
  onSendToWorkflow: (source?: string) => void
  onToggleAdvanced: () => void
  onToggleLeftPanel: () => void
  onToggleRightPanel: () => void
  onOpenXyPlot: () => void
  onPromptAnalyze: () => void
  selectedModelWarning: string
  rightPanelCollapsed: boolean
  dualRuntimeAvailable: boolean
  sdcppRuntimeAvailable: boolean
}) {
  const handlePromptKeyDown = (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || !event.shiftKey || event.nativeEvent.isComposing) {
      return
    }
    event.preventDefault()
    onGenerate()
  }
  const selectedModel =
    filteredModels.find((model) => model.id === settings.modelId) ?? filteredModels[0]
  const selectedEngine = selectedModel?.engineId ?? 'unknown'
  // Flow-match DiT families run their own scheduler; the sampler picker has no effect.
  const samplerIgnored = ['flux', 'flux2', 'zimage', 'sd35', 'qwen', 'sana'].includes(selectedEngine)
  // Flux.2 Klein is step-distilled; classifier-free guidance is ignored by the pipeline.
  const cfgIgnored = selectedEngine === 'flux2'
  const modelHelper = useMemo(
    () => buildModelHelper(selectedModel, selectedEngine, samplerIgnored, cfgIgnored),
    [cfgIgnored, samplerIgnored, selectedEngine, selectedModel],
  )
  const resolutionOptions = useMemo(() => {
    const aspect = imageAspect(settings.width, settings.height, activeRatio)
    return RESOLUTION_PRESETS.map((preset) => ({
      ...preset,
      dimensions: dimensionsForShortEdge(aspect, preset.shortEdge),
      active: Math.abs(Math.min(settings.width, settings.height) - preset.shortEdge) <= 8,
    }))
  }, [activeRatio, settings.height, settings.width])

  const handleResolutionSelect = useCallback(
    (shortEdge: number) => {
      onSettingsChange((current) => {
        const dimensions = dimensionsForShortEdge(
          imageAspect(current.width, current.height, activeRatio),
          shortEdge,
        )
        return {
          ...current,
          width: dimensions.width,
          height: dimensions.height,
        }
      })
    },
    [activeRatio, onSettingsChange],
  )

  const handleAspectSwap = useCallback(() => {
    onSettingsChange((current) => {
      const swappedRatio = findMatchingAspectRatio(bootstrap.aspectRatios, current.height, current.width)
      return {
        ...current,
        aspectRatioId: swappedRatio?.id ?? current.aspectRatioId,
        width: current.height,
        height: current.width,
      }
    })
  }, [bootstrap.aspectRatios, onSettingsChange])

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

  const handleAddModelHelperToPrompt = useCallback(() => {
    if (!modelHelper.promptText) {
      return
    }
    onSettingsChange((current) => ({
      ...current,
      prompt: appendPromptText(current.prompt, modelHelper.promptText),
    }))
  }, [modelHelper.promptText, onSettingsChange])

  return (
    <aside className="pro-prompt-panel" aria-label="Prompt and generation settings">
      <PanelHeader title="Prompt" actionLabel="Hide prompt column" icon={PanelLeft} onAction={onToggleLeftPanel} />
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
              accept={IMAGE_FILE_ACCEPT}
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
          className={
            isGenerating
              ? 'pro-generate-button pro-generate-button-stop'
              : selectedModelWarning
                ? 'pro-generate-button pro-generate-button-disabled'
                : 'pro-generate-button'
          }
          disabled={!isGenerating && Boolean(selectedModelWarning)}
          onClick={isGenerating ? onStopGenerate : onGenerate}
          title={!isGenerating && selectedModelWarning ? selectedModelWarning : undefined}
        >
          {isGenerating ? <X size={18} aria-hidden="true" /> : <Sparkles size={18} aria-hidden="true" />}
          <span>{isGenerating ? 'Stop' : settings.mode === 'video' ? 'Generate video' : 'Generate image'}</span>
        </button>
        <button
          type="button"
          className={isContinuousGenerating ? 'pro-secondary-button pro-continuous-generate-button is-active' : 'pro-secondary-button pro-continuous-generate-button'}
          disabled={!isContinuousGenerating && (isGenerating || Boolean(selectedModelWarning))}
          onClick={onToggleContinuousGenerate}
          title={!isContinuousGenerating && selectedModelWarning ? selectedModelWarning : 'Keep generating until stopped.'}
        >
          {isContinuousGenerating ? <X size={16} aria-hidden="true" /> : <RefreshCcw size={16} aria-hidden="true" />}
          <span>{isContinuousGenerating ? 'Stop continuous' : 'Generate continuously'}</span>
        </button>
        <button
          type="button"
          className="pro-secondary-button pro-workflow-send-button"
          onClick={() => onSendToWorkflow('Create panel')}
          title="Capture the current settings as a reorderable workflow node"
        >
          <WorkflowIcon size={16} aria-hidden="true" />
          <span>Send to workflow</span>
        </button>
        <button
          type="button"
          className="pro-secondary-button pro-xy-plot-button"
          disabled={settings.mode !== 'image'}
          onClick={onOpenXyPlot}
          title={settings.mode !== 'image' ? 'X/Y plot is image-only in Pro right now.' : 'Set up a multi-image model and steps test.'}
        >
          <ArrowLeftRight size={16} aria-hidden="true" />
          <span>X/Y plot</span>
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
      <div className="pro-run-options" aria-label="Run options">
        <div className="pro-pipeline-toggle" role="group" aria-label="Generation backend">
          <button
            type="button"
            className={settings.pipelineBackend === 'aiwf' ? 'active' : ''}
            aria-pressed={settings.pipelineBackend === 'aiwf'}
            onClick={() => onSettingsChange((current) => ({ ...current, pipelineBackend: 'aiwf' }))}
          >
            AIWF pipelines
          </button>
          <button
            type="button"
            className={settings.pipelineBackend === 'dual' ? 'active' : ''}
            aria-pressed={settings.pipelineBackend === 'dual'}
            disabled={settings.mode === 'video' || !dualRuntimeAvailable}
            title={
              settings.mode === 'video'
                ? 'Dual mode is image-only in Pro right now.'
                : dualRuntimeAvailable
                  ? 'Use the dual Diffusers plus C++ runtime for this image run.'
                  : 'Restart Pro with the Dual backend before using this option.'
            }
            onClick={() => onSettingsChange((current) => ({ ...current, pipelineBackend: 'dual' }))}
          >
            Dual mode
          </button>
          <button
            type="button"
            className={settings.pipelineBackend === 'sdcpp' ? 'active' : ''}
            aria-pressed={settings.pipelineBackend === 'sdcpp'}
            disabled={settings.mode === 'video' || !sdcppRuntimeAvailable}
            title={
              settings.mode === 'video'
                ? 'C++ backend is image-only in Pro right now.'
                : sdcppRuntimeAvailable
                  ? 'Use stable-diffusion.cpp for this image run.'
                  : 'Restart Pro with the C++ backend before using this option.'
            }
            onClick={() => onSettingsChange((current) => ({ ...current, pipelineBackend: 'sdcpp' }))}
          >
            C++ backend
          </button>
        </div>
        <div className="pro-visibility-actions">
          <button type="button" className="pro-secondary-button" onClick={onToggleLeftPanel}>
            <PanelLeft size={14} aria-hidden="true" />
            <span>Hide left</span>
          </button>
          <button type="button" className="pro-secondary-button" onClick={onToggleRightPanel}>
            <PanelRight size={14} aria-hidden="true" />
            <span>{rightPanelCollapsed ? 'Show right' : 'Hide right'}</span>
          </button>
        </div>
      </div>
      {selectedModelWarning ? (
        <div className="pro-model-readiness-note" role="alert">
          <strong>{isModelBlocked(selectedModel) ? 'Model not ready' : 'Model unavailable'}</strong>
          <span>{selectedModelWarning}</span>
          {selectedModel?.suggestedAction ? <small>{selectedModel.suggestedAction}</small> : null}
        </div>
      ) : null}

      <section className="pro-prompt-insight-card" aria-label="Prompt helper">
        <div className="pro-prompt-insight-header">
          <div>
            <strong>Prompt helper</strong>
            <span>{modelHelper.summary}</span>
          </div>
          <button
            type="button"
            className="pro-secondary-button pro-prompt-helper-add"
            onClick={handleAddModelHelperToPrompt}
            disabled={!modelHelper.promptText}
          >
            Add to prompt
          </button>
        </div>
        <div className="pro-model-helper-grid">
          {modelHelper.lines.map((line) => (
            <span key={line}>{line}</span>
          ))}
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
        <button
          type="button"
          className="pro-icon-button pro-aspect-swap"
          onClick={handleAspectSwap}
          aria-label="Swap aspect ratio"
          title="Swap aspect ratio"
        >
          <ArrowLeftRight size={16} aria-hidden="true" />
        </button>
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

      <fieldset className="pro-aspect-group pro-resolution-group">
        <legend>
          <FieldLabel
            label="Resolution"
            tooltip="Choose the short edge for the active shape. Width and height stay editable below."
          />
        </legend>
        <div className="pro-aspect-chips">
          {resolutionOptions.map((option) => (
            <button
              key={option.id}
              type="button"
              className={option.active ? 'pro-chip pro-chip-active pro-resolution-chip' : 'pro-chip pro-resolution-chip'}
              aria-pressed={option.active}
              onClick={() => handleResolutionSelect(option.shortEdge)}
            >
              <span>{option.label}</span>
              <small>{option.dimensions.width}x{option.dimensions.height}</small>
            </button>
          ))}
        </div>
      </fieldset>

      <div className="pro-settings-block">
        <div className="pro-section-label">
          {settings.mode === 'video'
            ? selectedEngine === 'wan'
              ? 'Wan video settings'
              : 'Sana video settings'
            : 'Image settings'}
        </div>
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
        {cfgIgnored ? (
          <p className="pro-field-note">Flux.2 Klein is distilled - CFG has no effect on this model.</p>
        ) : null}
        {settings.mode === 'video' ? (
          <>
            <RangeField
              label="Frames"
              tooltip="Frame count controls video duration and denoise work. Wan normalizes to 4k+1 frames (e.g. 81). Keep smoke tests short, then increase after timing receipts look sane."
              min={5}
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
            {selectedEngine === 'wan' ? (
              <p className="pro-field-note">
                Wan sampler, flow shift, and offload strategy come from Settings → Video &amp; Performance. The
                source image above is the first frame for image-to-video.
              </p>
            ) : null}
            {selectedEngine !== 'wan' ? (
            <>
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
            ) : null}
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
            {samplerIgnored ? (
              <p className="pro-field-note">This model family uses its own flow-match scheduler - sampler choice is ignored.</p>
            ) : null}
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

function ControlNetSettingsModal({
  settings,
  onSettingsChange,
  compatibility,
}: {
  settings: GenerationSettings
  onSettingsChange: Dispatch<SetStateAction<GenerationSettings>>
  compatibility: ControlNetCompatibility
}) {
  const handleControlNetImageChange = (event: ReactChangeEvent<HTMLInputElement>) => {
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
        controlNetImageDataUrl: value,
        controlNetImageName: file.name,
      }))
    }
    reader.readAsDataURL(file)
  }
  const disabled = !compatibility.supported

  return (
    <div className="pro-modal-form">
      <div className={disabled ? 'pro-controlnet-status is-blocked' : 'pro-controlnet-status'} role={disabled ? 'alert' : undefined}>
        <strong>{disabled ? 'ControlNet unavailable' : 'ControlNet ready'}</strong>
        <span>{compatibility.message}</span>
      </div>
      <section className="pro-controlnet-card" aria-label="ControlNet unit">
        <div className="pro-controlnet-header">
          <label className="pro-toggle">
            <input
              type="checkbox"
              checked={settings.controlNetEnabled && !disabled}
              disabled={disabled}
              onChange={(event) =>
                onSettingsChange((current) => ({ ...current, controlNetEnabled: event.target.checked }))
              }
            />
            <span>ControlNet unit 1</span>
          </label>
          <small>SD 1.5 and SDXL only. Flux, Qwen, Sana, and video routes stay off.</small>
        </div>
        <label className="pro-field pro-compact-field">
          <FieldLabel
            label="ControlNet model"
            tooltip="Use a local ControlNet model id or path that matches the selected SD/SDXL family."
          />
          <input
            value={settings.controlNetModel}
            placeholder="control_v11p_sd15_canny, diffusers folder, or local path"
            onChange={(event) =>
              onSettingsChange((current) => ({ ...current, controlNetModel: event.target.value }))
            }
          />
        </label>
        <label className="pro-field pro-compact-field">
          <FieldLabel
            label="Preprocessor"
            tooltip="Choose none if the image is already prepared. Canny, depth, pose, line, and segmentation preprocessors run before the ControlNet pass when available."
          />
          <select
            value={settings.controlNetModule}
            disabled={disabled}
            onChange={(event) =>
              onSettingsChange((current) => ({ ...current, controlNetModule: event.target.value }))
            }
          >
            {CONTROLNET_MODULE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <div className="pro-controlnet-upload-row">
          {settings.controlNetImageDataUrl ? (
            <img className="pro-controlnet-preview" src={settings.controlNetImageDataUrl} alt="" />
          ) : (
            <div className="pro-controlnet-empty">No control image</div>
          )}
          <div className="pro-controlnet-upload-actions">
            <span>{settings.controlNetImageName || 'Upload an edge, pose, depth, or reference image.'}</span>
            <label className={disabled ? 'pro-secondary-button is-disabled' : 'pro-secondary-button'} htmlFor="pro-controlnet-image-input">
              <FileImage size={15} aria-hidden="true" />
              <span>Upload image</span>
            </label>
            <input
              id="pro-controlnet-image-input"
              className="pro-file-input-hidden"
              type="file"
              accept={IMAGE_FILE_ACCEPT}
              disabled={disabled}
              onChange={handleControlNetImageChange}
            />
            {settings.controlNetImageDataUrl ? (
              <button
                type="button"
                className="pro-secondary-button ghost"
                onClick={() =>
                  onSettingsChange((current) => ({
                    ...current,
                    controlNetImageDataUrl: '',
                    controlNetImageName: '',
                  }))
                }
              >
                Clear
              </button>
            ) : null}
          </div>
        </div>
        <RangeField
          label="Weight"
          tooltip="ControlNet weight decides how strongly the control image steers the generation."
          min={0}
          max={2}
          step={0.05}
          value={settings.controlNetWeight}
          onChange={(value) => onSettingsChange((current) => ({ ...current, controlNetWeight: value }))}
        />
        <div className="pro-controlnet-window">
          <RangeField
            label="Start"
            min={0}
            max={1}
            step={0.01}
            value={settings.controlNetGuidanceStart}
            onChange={(value) => onSettingsChange((current) => ({ ...current, controlNetGuidanceStart: value }))}
          />
          <RangeField
            label="End"
            min={0}
            max={1}
            step={0.01}
            value={settings.controlNetGuidanceEnd}
            onChange={(value) => onSettingsChange((current) => ({ ...current, controlNetGuidanceEnd: value }))}
          />
        </div>
        <label className="pro-field pro-compact-field">
          <FieldLabel label="Processor resolution" compact />
          <input
            type="number"
            min={64}
            max={4096}
            step={64}
            disabled={disabled}
            value={settings.controlNetProcessorRes}
            onChange={(event) =>
              onSettingsChange((current) => ({
                ...current,
                controlNetProcessorRes: Number(event.target.value),
              }))
            }
          />
        </label>
      </section>
    </div>
  )
}

function buildOutputDetailRows(item: RecentOutput): Array<{ label: string; value: string }> {
  return [
    { label: 'Model', value: item.modelName || 'Not stored' },
    { label: 'Size', value: item.width && item.height ? `${item.width}x${item.height}` : 'Not stored' },
    { label: 'Steps', value: formatOutputSettingValue(item.steps) },
    { label: 'Time', value: formatDurationSeconds(item.durationSeconds) },
    { label: 'Speed', value: item.speed || 'Not stored' },
    { label: 'CFG', value: formatOutputSettingValue(item.cfgScale) },
    { label: 'Clip skip', value: formatOutputSettingValue(item.clipSkip) },
    { label: 'Sampler', value: item.sampler || 'Not stored' },
    { label: 'Seed', value: formatOutputSettingValue(item.seed) },
  ]
}

function buildOutputStatusText(item: RecentOutput): string {
  const prompt = item.prompt || item.infotext || 'Local output'
  const details = [
    item.modelName || '',
    item.width && item.height ? `${item.width}x${item.height}` : '',
    typeof item.steps === 'number' ? `${item.steps} steps` : '',
    typeof item.seed === 'number' ? `seed ${item.seed}` : '',
  ].filter((value) => value.length > 0)
  return details.length > 0 ? `${prompt} | ${details.join(' | ')}` : prompt
}

function formatOutputReceiptText(item: RecentOutput): string {
  const duration = formatDurationSeconds(item.durationSeconds)
  const parts = [
    duration !== 'Not stored' ? duration : '',
    item.speed || '',
  ].filter((value) => value.length > 0)
  return parts.join(' | ')
}

function buildOutputButtonLabel(item: RecentOutput): string {
  return `Show output settings for ${buildOutputStatusText(item)}`
}

function buildModelHelper(
  model: ProModelOption | undefined,
  engineId: EngineId,
  samplerIgnored: boolean,
  cfgIgnored: boolean,
): { summary: string; lines: string[]; promptText: string } {
  if (!model) {
    return {
      summary: 'Select a model to see sane settings before analysis.',
      lines: ['No model selected'],
      promptText: '',
    }
  }
  const preset = model.generationPreset ?? {}
  const presetParts = [
    Number.isFinite(preset.width) && Number.isFinite(preset.height) ? `${preset.width}x${preset.height}` : '',
    Number.isFinite(preset.steps) ? `${preset.steps} steps` : '',
    Number.isFinite(preset.cfgScale) ? `CFG ${preset.cfgScale}` : '',
    preset.sampler ? `${preset.sampler}` : '',
  ].filter(Boolean)
  const size = typeof model.sizeBytes === 'number' && model.sizeBytes > 0 ? formatBytes(model.sizeBytes) : ''
  const sourceParts = [
    model.engineLabel || modelEngineFallbackLabel(engineId),
    size,
    model.fileCount && model.fileCount > 1 ? `${model.fileCount} files` : model.assetSummary,
  ].filter(Boolean)
  const lines = [
    sourceParts.join(' / ') || 'Local model',
    presetParts.length > 0 ? `Preset: ${presetParts.join(' / ')}` : 'Preset: use current controls',
  ]
  if (samplerIgnored) {
    lines.push('Sampler is ignored by this family.')
  }
  if (cfgIgnored) {
    lines.push('CFG is ignored by Flux2 Klein.')
  }
  if (model.heavyFor12Gb) {
    lines.push('Heavy for 12 GB VRAM; keep resolution modest.')
  }
  const promptText = modelPromptTextForEngine(engineId)
  return {
    summary: `${model.name} sane-setting hints`,
    lines,
    promptText,
  }
}

function modelPromptTextForEngine(engineId: EngineId): string {
  switch (engineId) {
    case 'sd15':
    case 'sdxl':
      return 'high detail, sharp focus, natural lighting'
    case 'flux':
    case 'flux2':
      return 'natural light, realistic texture, coherent anatomy'
    case 'sana':
      return 'clean composition, crisp subject detail, balanced color'
    case 'wan':
    case 'sana_video':
      return 'smooth motion, stable subject, cinematic framing'
    default:
      return 'clear subject, detailed lighting, clean composition'
  }
}

function appendPromptText(prompt: string, addition: string): string {
  const cleanAddition = addition.trim()
  if (!cleanAddition) {
    return prompt
  }
  const cleanPrompt = prompt.trim()
  if (!cleanPrompt) {
    return cleanAddition
  }
  if (cleanPrompt.toLowerCase().includes(cleanAddition.toLowerCase())) {
    return prompt
  }
  return `${cleanPrompt}, ${cleanAddition}`
}

function formatOutputSettingValue(value: number | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? String(value) : 'Not stored'
}

function formatDurationSeconds(value: number | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    return 'Not stored'
  }
  if (value < 10) {
    return `${value.toFixed(2)}s`
  }
  if (value < 60) {
    return `${value.toFixed(1)}s`
  }
  const minutes = Math.floor(value / 60)
  const seconds = Math.round(value % 60).toString().padStart(2, '0')
  return `${minutes}:${seconds}`
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

const ModelsWorkspace = memo(ModelsWorkspaceImpl)

function ModelsWorkspaceImpl({
  engineFilter,
  engines,
  models,
  downloadsStatus,
  selectedModelId,
  onEngineFilterChange,
  onModelSelect,
  onCatalogDownload,
  downloadingCatalogKey,
}: {
  engineFilter: EngineId
  engines: EngineSummary[]
  models: ProModelOption[]
  downloadsStatus: ProDownloadsStatus | null
  selectedModelId: string
  onEngineFilterChange: (value: EngineId) => void
  onModelSelect: (modelId: string) => void
  onCatalogDownload: (key: string) => void
  downloadingCatalogKey: string
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
          {downloadSummary.items.map((item) => {
            const linkLabel = item.source === 'civitai' ? 'Open CivitAI' : 'Open source'
            const subtitle = item.installed
              ? 'Installed'
              : item.canDownload
                ? `${item.category} | Direct download`
                : item.hfUrl
                  ? `${item.category} | ${linkLabel}`
                  : item.category
            if (item.canDownload && !item.installed) {
              return (
                <button
                  key={item.key}
                  type="button"
                  className="pro-download-chip pro-download-chip-button"
                  title={`${item.destination}${item.notes ? ` - ${item.notes}` : ''}`}
                  onClick={() => onCatalogDownload(item.key)}
                  disabled={downloadingCatalogKey === item.key}
                >
                  <strong>{item.title}</strong>
                  <small>{downloadingCatalogKey === item.key ? 'Downloading...' : subtitle}</small>
                </button>
              )
            }
            return item.hfUrl ? (
              <a
                key={item.key}
                className={item.installed ? 'pro-download-chip pro-download-chip-ready' : 'pro-download-chip pro-download-chip-link'}
                href={item.hfUrl}
                target="_blank"
                rel="noreferrer"
                title={`${item.destination}${item.notes ? ` - ${item.notes}` : ''}`}
              >
                <strong>{item.title}</strong>
                <small>{subtitle}</small>
              </a>
            ) : (
              <span
                key={item.key}
                className={item.installed ? 'pro-download-chip pro-download-chip-ready' : 'pro-download-chip'}
                title={item.destination}
              >
                <strong>{item.title}</strong>
                <small>{subtitle}</small>
              </span>
            )
          })}
        </div>
        <p className="pro-download-note">
          Downloaded-model and installable-model browsers are coming soon.
        </p>
        {false ? (
        <div className="pro-download-chip-row">
          {downloadSummary.items.map((item) =>
            item.hfUrl ? (
              <a
                key={item.key}
                className={item.installed ? 'pro-download-chip pro-download-chip-ready' : 'pro-download-chip pro-download-chip-link'}
                href={item.hfUrl}
                target="_blank"
                rel="noreferrer"
                title={`${item.destination}${item.notes ? ` — ${item.notes}` : ''}`}
              >
                <strong>{item.title}</strong>
                <small>{item.installed ? 'Installed' : `${item.category} · Hugging Face ↗`}</small>
              </a>
            ) : (
              <span
                key={item.key}
                className={item.installed ? 'pro-download-chip pro-download-chip-ready' : 'pro-download-chip'}
                title={item.destination}
              >
                <strong>{item.title}</strong>
                <small>{item.installed ? 'Installed' : item.category}</small>
              </span>
            ),
          )}
        </div>
        ) : null}
      </section>

      {downloadsStatus && downloadsStatus.civitaiLinks.length > 0 ? (
        <section className="pro-download-status-card" aria-label="CivitAI browse links">
          <div>
            <strong>Find more on CivitAI</strong>
            <span>Pre-filtered searches that only show models AIWF's routes can run.</span>
          </div>
          <div className="pro-download-chip-row">
            {downloadsStatus.civitaiLinks
              .filter((link) => engineFilter === 'all' || link.engine === engineFilter)
              .map((link) => (
                <a
                  key={link.url}
                  className="pro-download-chip pro-download-chip-link pro-civitai-link"
                  href={link.url}
                  target="_blank"
                  rel="noreferrer"
                  title={link.note}
                >
                  <strong>{link.label}</strong>
                  <small>Open CivitAI</small>
                  <small>CivitAI ↗</small>
                </a>
              ))}
          </div>
        </section>
      ) : null}

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
                      {model.heavyFor12Gb ? (
                        <div className="pro-model-card-vram-flag" title={`Estimated ~${model.estVramGb ?? '?'} GB of VRAM in use - may exceed 12 GB GPUs`}>
                          High VRAM · ~{model.estVramGb} GB
                        </div>
                      ) : null}
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
        <InfoCard title="Library state" subtitle="The shell now treats data as a workspace with paths, receipts, and inventory.">
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
  onOpenEnhance,
  onOpenReactor,
}: {
  capabilitiesStatus: ProCapabilitiesStatus | null
  runtime: ProRuntimeStatus
  onOpenCreate: () => void
  onOpenVideo: () => void
  onOpenData: () => void
  onOpenSegmentation: () => void
  onOpenEnhance: () => void
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
            <button type="button" className="pro-secondary-button" onClick={onOpenEnhance}>Enhance</button>
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

function ExtensionsCard() {
  const [status, setStatus] = useState<ProExtensionsStatus | null>(null)
  const [message, setMessage] = useState('')
  const [busyId, setBusyId] = useState('')

  const refresh = useCallback(() => {
    fetchProExtensions()
      .then(setStatus)
      .catch((error: unknown) => setMessage(`Could not load extensions: ${formatApiError(error)}`))
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  const handleToggle = async (id: string, enabled: boolean) => {
    setBusyId(id)
    try {
      const result = await toggleProExtension(id, enabled)
      setMessage(result.note || 'Saved.')
      refresh()
    } catch (error: unknown) {
      setMessage(`Toggle failed: ${formatApiError(error)}`)
    } finally {
      setBusyId('')
    }
  }

  return (
    <InfoCard
      title="Extensions"
      subtitle="User extensions loaded from the plugins folder. Add your own by copying plugins/hello-extension — see docs/EXTENSIONS.md."
    >
      <div className="pro-form-stack">
        {status && status.extensions.length === 0 ? (
          <p className="pro-muted">
            No extensions found. Drop a folder with a plugin.py into {status.pluginsDir || 'plugins/'} and restart.
          </p>
        ) : null}
        {(status?.extensions ?? []).map((ext) => (
          <div key={ext.id} className="pro-extension-row">
            <div className="pro-extension-info">
              <strong>
                {ext.name} <small>v{ext.version}</small>
              </strong>
              {ext.description ? <span>{ext.description}</span> : null}
              {ext.error ? <span className="pro-extension-error">Load error: {ext.error}</span> : null}
              {ext.hasApi ? <small className="pro-muted">API: {ext.apiBase}</small> : null}
            </div>
            <button
              type="button"
              className="pro-secondary-button"
              disabled={busyId === ext.id}
              onClick={() => handleToggle(ext.id, !ext.enabled)}
            >
              {ext.enabled ? 'Disable' : 'Enable'}
            </button>
          </div>
        ))}
        {message ? <p className="pro-field-note">{message}</p> : null}
        <p className="pro-field-note">
          Extensions run as Python code inside the app — only install ones you trust. Enable/disable changes apply on
          the next restart.
        </p>
      </div>
    </InfoCard>
  )
}

type VideoLabOp = 'vsr' | 'rife' | 'audio' | 'extend'

function VideoLabCard({ wanModels }: { wanModels: ProModelOption[] }) {
  const [labStatus, setLabStatus] = useState<VideoLabStatus | null>(null)
  const [source, setSource] = useState<VideoLabProbe | null>(null)
  const [op, setOp] = useState<VideoLabOp>('vsr')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [resultUrl, setResultUrl] = useState('')
  const [vsrScale, setVsrScale] = useState(2)
  const [vsrMode, setVsrMode] = useState(0)
  const [rifeMultiplier, setRifeMultiplier] = useState(2)
  const [audioPrompt, setAudioPrompt] = useState('')
  const [extendPrompt, setExtendPrompt] = useState('')
  const [extendFrames, setExtendFrames] = useState(81)
  const [extendModelId, setExtendModelId] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    fetchVideoLabStatus(controller.signal)
      .then(setLabStatus)
      .catch(() => setLabStatus(null))
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!extendModelId && wanModels.length > 0) {
      setExtendModelId(wanModels[0].id)
    }
  }, [extendModelId, wanModels])

  const handleUpload = async (event: ReactChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) {
      return
    }
    setBusy(true)
    setMessage(`Uploading ${file.name}…`)
    setResultUrl('')
    try {
      const probe = await uploadVideoLabFile(file)
      setSource(probe)
      setMessage(
        `Loaded ${file.name}: ${probe.width}x${probe.height}, ${probe.frameCount} frames @ ${probe.fps.toFixed(1)} fps.`,
      )
    } catch (error: unknown) {
      setMessage(`Upload failed: ${formatApiError(error)}`)
    } finally {
      setBusy(false)
    }
  }

  const handleRun = async () => {
    if (!source) {
      setMessage('Upload a video first.')
      return
    }
    setBusy(true)
    setResultUrl('')
    setMessage(
      op === 'extend'
        ? 'Extending video (generates a Wan continuation, then stitches — this takes a while)…'
        : 'Running…',
    )
    try {
      const result = await runVideoLab({
        op,
        videoPath: source.path,
        scale: vsrScale,
        mode: vsrMode,
        multiplier: rifeMultiplier,
        audioPrompt,
        prompt: extendPrompt,
        frames: extendFrames,
        checkpointId: extendModelId,
      })
      setResultUrl(result.url)
      setMessage(result.message || 'Done.')
    } catch (error: unknown) {
      setMessage(`Failed: ${formatApiError(error)}`)
    } finally {
      setBusy(false)
    }
  }

  const opChoices: Array<{ id: VideoLabOp; label: string; enabled: boolean; hint: string }> = [
    {
      id: 'vsr',
      label: 'VSR upscale',
      enabled: Boolean(labStatus?.vsr.available),
      hint: labStatus?.vsr.available
        ? `NVIDIA VideoFX ready (${labStatus.vsr.modelCount} model packs)`
        : 'NVIDIA VideoFX SDK not detected. Install it locally, then run the installer with -WithNvidiaVideoFx.',
    },
    {
      id: 'rife',
      label: 'RIFE interpolation',
      enabled: Boolean(labStatus?.rife.available),
      hint: labStatus?.rife.available
        ? `Checkpoints: ${labStatus.rife.checkpoints.join(', ')}`
        : 'RIFE checkpoints not found under models/rife.',
    },
    {
      id: 'audio',
      label: 'Add audio',
      enabled: true,
      hint:
        (labStatus?.audio.videoAudioModels.length ?? 0) > 0
          ? 'Video-conditioned audio (MMAudio) available.'
          : 'Falls back to text-to-music audio muxed over the clip.',
    },
    {
      id: 'extend',
      label: 'Extend video',
      enabled: wanModels.length > 0,
      hint:
        wanModels.length > 0
          ? labStatus?.extend.note || 'Continues motion from the last frame via Wan 5B.'
          : 'No Wan video models detected.',
    },
  ]
  const activeChoice = opChoices.find((choice) => choice.id === op)

  return (
    <InfoCard
      title="Video Lab"
      subtitle="Upload any video, then upscale it with NVIDIA VSR, smooth it with RIFE, extend it with Wan, or add a soundtrack."
    >
      <div className="pro-form-stack">
        <div className="pro-video-source-actions">
          <label className="pro-secondary-button" htmlFor="pro-video-lab-upload">
            <FileImage size={15} aria-hidden="true" />
            <span>{source ? 'Replace video' : 'Upload video'}</span>
          </label>
          <input
            id="pro-video-lab-upload"
            className="pro-file-input-hidden"
            type="file"
            accept={VIDEO_FILE_ACCEPT}
            onChange={handleUpload}
            disabled={busy}
          />
          {source ? (
            <span className="pro-muted">
              {source.width}x{source.height} · {source.frameCount} frames · {source.fps.toFixed(1)} fps
            </span>
          ) : null}
        </div>
        <label className="pro-field">
          <FieldLabel label="Operation" />
          <select value={op} onChange={(event) => setOp(event.target.value as VideoLabOp)} disabled={busy}>
            {opChoices.map((choice) => (
              <option key={choice.id} value={choice.id} disabled={!choice.enabled}>
                {choice.label}
                {choice.enabled ? '' : ' (unavailable)'}
              </option>
            ))}
          </select>
          {activeChoice ? <p className="pro-field-note">{activeChoice.hint}</p> : null}
        </label>
        {op === 'vsr' ? (
          <div className="pro-control-grid">
            <label className="pro-field">
              <FieldLabel label="Scale" />
              <select value={vsrScale} onChange={(event) => setVsrScale(Number(event.target.value))} disabled={busy}>
                <option value={1.3333333}>1.33x</option>
                <option value={1.5}>1.5x</option>
                <option value={2}>2x</option>
                <option value={3}>3x</option>
                <option value={4}>4x</option>
              </select>
            </label>
            <label className="pro-field">
              <FieldLabel label="Mode" tooltip="0 keeps detail conservative; higher modes trade artifacts for sharpness. Modes 8-15 are same-resolution cleanup." />
              <input
                type="number"
                min={0}
                max={19}
                value={vsrMode}
                onChange={(event) => setVsrMode(clamp(Number(event.target.value) || 0, 0, 19))}
                disabled={busy}
              />
            </label>
          </div>
        ) : null}
        {op === 'rife' ? (
          <label className="pro-field">
            <FieldLabel label="Frame multiplier" />
            <select
              value={rifeMultiplier}
              onChange={(event) => setRifeMultiplier(Number(event.target.value))}
              disabled={busy}
            >
              <option value={2}>2x frames</option>
              <option value={4}>4x frames</option>
              <option value={8}>8x frames</option>
            </select>
          </label>
        ) : null}
        {op === 'audio' ? (
          <label className="pro-field">
            <FieldLabel label="Audio prompt" />
            <input
              value={audioPrompt}
              onChange={(event) => setAudioPrompt(event.target.value)}
              placeholder="e.g. gentle rain with distant thunder"
              disabled={busy}
            />
          </label>
        ) : null}
        {op === 'extend' ? (
          <>
            <label className="pro-field">
              <FieldLabel label="Motion prompt" tooltip="Describe how the scene should continue. The clip's last frame is the starting image." />
              <input
                value={extendPrompt}
                onChange={(event) => setExtendPrompt(event.target.value)}
                placeholder="e.g. the camera keeps panning right across the skyline"
                disabled={busy}
              />
            </label>
            <div className="pro-control-grid">
              <label className="pro-field">
                <FieldLabel label="Extra frames" />
                <input
                  type="number"
                  min={5}
                  max={257}
                  step={4}
                  value={extendFrames}
                  onChange={(event) => setExtendFrames(clamp(Number(event.target.value) || 81, 5, 257))}
                  disabled={busy}
                />
              </label>
              <label className="pro-field">
                <FieldLabel label="Wan model" />
                <select value={extendModelId} onChange={(event) => setExtendModelId(event.target.value)} disabled={busy}>
                  {wanModels.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </>
        ) : null}
        <div className="pro-settings-actions">
          <button
            type="button"
            className="pro-primary-button"
            onClick={handleRun}
            disabled={busy || !source || !(activeChoice?.enabled ?? false)}
          >
            {busy ? 'Working…' : 'Run'}
          </button>
          <span>{message}</span>
        </div>
        {resultUrl ? (
          <video className="pro-video-lab-result" src={resultUrl} controls loop />
        ) : null}
      </div>
    </InfoCard>
  )
}

function ToolsWorkspace({
  capabilitiesStatus,
  runtime,
  wanModels,
  onOpenCreate,
  onOpenVideo,
  onOpenData,
  onOpenSegmentation,
  onOpenEnhance,
  onOpenReactor,
}: {
  capabilitiesStatus: ProCapabilitiesStatus | null
  runtime: ProRuntimeStatus
  wanModels: ProModelOption[]
  onOpenCreate: () => void
  onOpenVideo: () => void
  onOpenData: () => void
  onOpenSegmentation: () => void
  onOpenEnhance: () => void
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
        { label: 'Enhance', onClick: onOpenEnhance },
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
        <VideoLabCard wanModels={wanModels} />
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
        <InfoCard title="Event stream" subtitle="Synthetic rows for now, shaped like the monitor table this shell needs.">
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
  onSettingsChange,
  onSettingsStatusChange,
  onSaveSettings,
  onModelFilesUpload,
  onModelReorganize,
  onUnloadModel,
  onRestartBackend,
  onReloadFrontend,
  settingsSaveStatus,
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
  onSettingsChange: Dispatch<SetStateAction<GenerationSettings>>
  onSettingsStatusChange: Dispatch<SetStateAction<ProSettingsStatus | null>>
  onSaveSettings: () => void
  onModelFilesUpload: (files: File[]) => Promise<string>
  onModelReorganize: () => Promise<string>
  onUnloadModel: () => void
  onRestartBackend: () => void
  onReloadFrontend: () => void
  settingsSaveStatus: string
  leftPanelWidth: number
  rightPanelWidth: number
  bottomDockHeight: number
  bottomDockVisible: boolean
  showAdvanced: boolean
}) {
  const summary = summarizeRecentOutputs(recentOutputs)
  const [activeSection, setActiveSection] = useState<SettingsSectionId>('generation')
  const [settingsQuery, setSettingsQuery] = useState('')
  const [modelSortBusy, setModelSortBusy] = useState(false)
  const [modelSortStatus, setModelSortStatus] = useState('')
  const modelUploadInputRef = useRef<HTMLInputElement>(null)
  const show = useCallback(
    (section: SettingsSectionId, keywords: string) => {
      const query = settingsQuery.trim().toLowerCase()
      if (query) {
        return keywords.toLowerCase().includes(query)
      }
      return activeSection === section
    },
    [activeSection, settingsQuery],
  )
  const uiSettings = settingsStatus?.ui ?? {
    accentPreset: 'mint',
    galleryColumns: 2,
    galleryHeight: 480,
    livePreview: true,
    showProgressEveryNSteps: 5,
    livePreviewDecoder: 'vae',
    livePreviewTitleProgress: true,
    hiddenTabs: [],
  }
  const livePreviewSummary = uiSettings.livePreview
    ? `Preview decode every ${uiSettings.showProgressEveryNSteps} denoise step${uiSettings.showProgressEveryNSteps === 1 ? '' : 's'}. Higher values are lighter on SDXL.`
    : 'Live preview is off. Final images still render normally.'
  const outputSettings = settingsStatus?.output ?? {
    imageFormat: 'png',
    imageQuality: 95,
    embedMetadata: true,
    saveGrid: false,
    saveSidecarTxt: false,
    filenamePattern: '[datetime]',
    saveBeforeHires: false,
    saveInterrupted: false,
    metadataIncludeModelHash: true,
    metadataIncludeVaeHash: true,
    metadataIncludeLoraHashes: true,
    metadataIncludeAppVersion: true,
    metadataIncludeOptimizationProfile: true,
    optimizationProfileId: 'balanced_sdpa_fp16',
  }
  const videoSettings = settingsStatus?.video ?? {
    wanHigh: '',
    wanLow: '',
    wanVae: '',
    wanTextEncoder: '',
    wanOffload: 'balanced',
    wanSampler: 'unipc',
    wanFlowShift: 5,
    wanRuntimeMode: 'fast_5b',
    ltxDtype: 'bf16',
    ltxCpuOffload: 'auto',
    wanGroupOffloadStream: true,
    wanGroupOffloadBlocks: 4,
    ggufCudaKernels: false,
    wanSageAttention: 'auto',
    wanNativeDenoise: true,
    wanManualVaeDecode: false,
    wanVaeChunkFrames: 4,
    wanGroupOffloadRecordStream: true,
    wanGroupOffloadLowCpuMem: true,
    wanResidentMinVramGb: 20,
  }
  const runtimeSettings = settingsStatus?.runtime ?? {
    port: 7860,
    listen: false,
    share: false,
    autolaunch: false,
    api: false,
    gerror: false,
    genlog: false,
    backend: 'diffusers',
    onnxProvider: 'auto',
    attention: 'sage_sdpa',
    xformers: false,
    optSdpAttention: false,
    optSplitAttention: false,
    asyncOffload: true,
    pinnedMemory: true,
    cudaMalloc: false,
    vramProfile: 'normal',
    medvram: false,
    lowvram: false,
    highvram: false,
    noHalf: false,
    fp8: false,
    fluxFp8: false,
    directml: false,
    cpu: false,
    cudaGraphs: false,
    torchao: false,
    fp8Quant: false,
    torchCompile: false,
    channelsLast: false,
    nvenc: false,
    hevc: false,
    blockPrivateDownloadUrls: true,
    apiCorsOrigins: '',
    apiRateLimitPerMinute: 0,
    theme: 'dark',
    modelsDir: '',
    checkpointDir: '',
    outputDir: '',
    extraModelDirs: '',
    extraCheckpointDirs: '',
  }
  const settingsModelOptions = modelsForCreationMode(bootstrap.models, settings.mode)

  const updateGenerationSetting = useCallback(
    (patch: Partial<GenerationSettings>) => {
      onSettingsChange((current) => ({ ...current, ...patch }))
    },
    [onSettingsChange],
  )

  const updateUiSetting = useCallback(
    (patch: Partial<ProSettingsStatus['ui']>) => {
      onSettingsStatusChange((current) =>
        current
          ? {
              ...current,
              ui: {
                ...current.ui,
                ...patch,
              },
            }
          : current,
      )
    },
    [onSettingsStatusChange],
  )

  const updateOutputSetting = useCallback(
    (patch: Partial<ProSettingsStatus['output']>) => {
      onSettingsStatusChange((current) =>
        current
          ? {
              ...current,
              output: {
                ...current.output,
                ...patch,
              },
            }
          : current,
      )
    },
    [onSettingsStatusChange],
  )

  const updateVideoSetting = useCallback(
    (patch: Partial<ProSettingsStatus['video']>) => {
      onSettingsStatusChange((current) =>
        current
          ? {
              ...current,
              video: {
                ...current.video,
                ...patch,
              },
            }
          : current,
      )
    },
    [onSettingsStatusChange],
  )

  const updateRuntimeSetting = useCallback(
    (patch: Partial<ProSettingsStatus['runtime']>) => {
      onSettingsStatusChange((current) =>
        current
          ? {
              ...current,
              runtime: {
                ...current.runtime,
                ...patch,
              },
            }
          : current,
      )
    },
    [onSettingsStatusChange],
  )

  const updateVramProfile = useCallback(
    (profile: string) => {
      updateRuntimeSetting({
        vramProfile: profile,
        cpu: profile === 'cpu',
        lowvram: profile === 'low',
        medvram: profile === 'mid',
        highvram: profile === 'high',
      })
    },
    [updateRuntimeSetting],
  )

  const handleSettingsModelUpload = useCallback(
    (event: ReactChangeEvent<HTMLInputElement>) => {
      const files = Array.from(event.target.files ?? [])
      event.target.value = ''
      if (files.length === 0) {
        return
      }
      setModelSortBusy(true)
      setModelSortStatus(`Sorting ${files.length} file${files.length === 1 ? '' : 's'}...`)
      void onModelFilesUpload(files)
        .then(setModelSortStatus)
        .catch((error: unknown) => setModelSortStatus(formatApiError(error)))
        .finally(() => setModelSortBusy(false))
    },
    [onModelFilesUpload],
  )

  const handleSettingsModelReorganize = useCallback(() => {
    setModelSortBusy(true)
    setModelSortStatus('Re-reading model headers...')
    void onModelReorganize()
      .then(setModelSortStatus)
      .catch((error: unknown) => setModelSortStatus(formatApiError(error)))
      .finally(() => setModelSortBusy(false))
  }, [onModelReorganize])

  const outputToggles: Array<{ key: keyof ProSettingsStatus['output']; label: string }> = [
    { key: 'embedMetadata', label: 'Embed metadata' },
    { key: 'saveSidecarTxt', label: 'Write sidecar txt' },
    { key: 'saveGrid', label: 'Save grids' },
    { key: 'saveBeforeHires', label: 'Save before hi-res' },
    { key: 'saveInterrupted', label: 'Save interrupted images' },
    { key: 'metadataIncludeModelHash', label: 'Include model hash' },
    { key: 'metadataIncludeVaeHash', label: 'Include VAE hash' },
    { key: 'metadataIncludeLoraHashes', label: 'Include LoRA hashes' },
    { key: 'metadataIncludeAppVersion', label: 'Include app version' },
    { key: 'metadataIncludeOptimizationProfile', label: 'Include optimization profile' },
  ]

  const runtimeToggles: Array<{ key: keyof ProSettingsStatus['runtime']; label: string }> = [
    { key: 'listen', label: 'Listen on LAN' },
    { key: 'api', label: 'Enable API' },
    { key: 'gerror', label: 'Funny errors' },
    { key: 'genlog', label: 'Generation log' },
    { key: 'share', label: 'Public share link' },
    { key: 'autolaunch', label: 'Auto launch browser' },
    { key: 'blockPrivateDownloadUrls', label: 'Block private download URLs' },
    { key: 'asyncOffload', label: 'Async offload' },
    { key: 'pinnedMemory', label: 'Pinned memory' },
    { key: 'cudaMalloc', label: 'CUDA malloc tuning' },
    { key: 'noHalf', label: 'Disable half precision' },
    { key: 'fp8', label: 'FP8 mode' },
    { key: 'fluxFp8', label: 'Flux FP8' },
    { key: 'directml', label: 'DirectML' },
    { key: 'xformers', label: 'xFormers flag' },
    { key: 'optSdpAttention', label: 'SDP attention flag' },
    { key: 'optSplitAttention', label: 'Split attention flag' },
    { key: 'cudaGraphs', label: 'CUDA graphs' },
    { key: 'torchao', label: 'TorchAO' },
    { key: 'fp8Quant', label: 'TorchAO FP8 quant' },
    { key: 'torchCompile', label: 'Torch compile' },
    { key: 'channelsLast', label: 'Channels last' },
    { key: 'nvenc', label: 'NVENC' },
    { key: 'hevc', label: 'HEVC' },
  ]

  return (
    <section className="pro-workspace-surface" aria-label="Workspace settings">
      <WorkspaceHeader
        eyebrow="Settings"
        title="Workspace controls"
        description="Saved defaults, display behavior, paths, and runtime policy for the Pro shell."
      />
      <div className="pro-settings-toolbar" role="tablist" aria-label="Settings sections">
        <nav className="pro-settings-nav">
          {SETTINGS_SECTIONS.map((section) => (
            <button
              key={section.id}
              type="button"
              role="tab"
              aria-selected={!settingsQuery && activeSection === section.id}
              className={`pro-settings-nav-item${!settingsQuery && activeSection === section.id ? ' is-active' : ''}`}
              title={section.hint}
              onClick={() => {
                setSettingsQuery('')
                setActiveSection(section.id)
              }}
            >
              {section.label}
            </button>
          ))}
        </nav>
        <input
          type="search"
          className="pro-settings-search"
          placeholder="Search settings…"
          value={settingsQuery}
          onChange={(event) => setSettingsQuery(event.target.value)}
          aria-label="Search settings"
        />
      </div>
      <div className="pro-workspace-grid">
        {show('system', 'support restart backend reload frontend unload model recovery troubleshooting logs') && (
        <InfoCard title="Support and recovery" subtitle="Quick actions for tester machines when something gets stuck.">
          <div className="pro-settings-action-row">
            <button type="button" className="pro-secondary-button" onClick={onUnloadModel}>
              Unload model
            </button>
            <button type="button" className="pro-secondary-button" onClick={onRestartBackend}>
              Restart backend
            </button>
            <button type="button" className="pro-secondary-button" onClick={onReloadFrontend}>
              Reload app window
            </button>
          </div>
          <p className="pro-field-note">
            Recovery stays inside the app. Use Monitor and Logs for diagnostics.
          </p>
        </InfoCard>
        )}
        {show('generation', 'generation defaults model sampler scheduler steps cfg clip skip width height negative prompt save images') && (
        <InfoCard title="Generation defaults" subtitle="Saved through the Pro backend settings file.">
          <div className="pro-form-stack">
            <label className="pro-field">
              <FieldLabel label="Default model" />
              <select
                value={settings.modelId}
                onChange={(event) => updateGenerationSetting({ modelId: event.target.value })}
              >
                {settingsModelOptions.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="pro-field">
              <FieldLabel label="Default sampler" />
              <select
                value={settings.sampler}
                onChange={(event) => updateGenerationSetting({ sampler: event.target.value })}
              >
                {bootstrap.samplers.map((sampler) => (
                  <option key={sampler} value={sampler}>
                    {sampler}
                  </option>
                ))}
              </select>
            </label>
            <label className="pro-field">
              <FieldLabel label="Default scheduler" />
              <select
                value={settings.scheduler}
                onChange={(event) => updateGenerationSetting({ scheduler: event.target.value })}
              >
                {SCHEDULER_OPTIONS.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <div className="pro-control-grid">
              <label className="pro-field">
                <FieldLabel label="Width" />
                <input
                  type="number"
                  min={64}
                  max={2048}
                  step={64}
                  value={settings.width}
                  onChange={(event) => updateGenerationSetting({ width: clamp(Number(event.target.value) || 64, 64, 2048) })}
                />
              </label>
              <label className="pro-field">
                <FieldLabel label="Height" />
                <input
                  type="number"
                  min={64}
                  max={2048}
                  step={64}
                  value={settings.height}
                  onChange={(event) => updateGenerationSetting({ height: clamp(Number(event.target.value) || 64, 64, 2048) })}
                />
              </label>
            </div>
            <RangeField
              label="Steps"
              min={1}
              max={80}
              step={1}
              value={settings.steps}
              onChange={(value) => updateGenerationSetting({ steps: value })}
            />
            <RangeField
              label="CFG scale"
              min={0}
              max={20}
              step={0.5}
              value={settings.cfgScale}
              onChange={(value) => updateGenerationSetting({ cfgScale: value })}
            />
            <RangeField
              label="Clip skip"
              min={1}
              max={12}
              step={1}
              value={settings.clipSkip}
              onChange={(value) => updateGenerationSetting({ clipSkip: value })}
            />
            <label className="pro-field">
              <FieldLabel label="Default negative prompt" />
              <textarea
                value={settings.negativePrompt}
                onChange={(event) => updateGenerationSetting({ negativePrompt: event.target.value })}
              />
            </label>
            <label className="pro-toggle">
              <input
                type="checkbox"
                checked={settings.saveImages}
                onChange={(event) => updateGenerationSetting({ saveImages: event.target.checked })}
              />
              <span>Save generated images to outputs</span>
            </label>
            <div className="pro-settings-actions">
              <button type="button" className="pro-primary-button" onClick={onSaveSettings} disabled={!settingsStatus}>
                Save settings
              </button>
              <span>{settingsSaveStatus || (settingsStatus ? 'Connected' : 'Backend not connected')}</span>
            </div>
          </div>
        </InfoCard>
        )}
        {show('interface', 'ui interface live preview decoder progress steps title slow sdxl') && (
        <InfoCard title="Live preview" subtitle="Control preview decoding while image jobs run.">
          <div className="pro-form-stack">
            <label className="pro-toggle">
              <input
                type="checkbox"
                checked={uiSettings.livePreview}
                onChange={(event) => updateUiSetting({ livePreview: event.target.checked })}
                disabled={!settingsStatus}
              />
              <span>Live preview enabled</span>
            </label>
            <RangeField
              label="Preview interval"
              tooltip="Raise this if SDXL previews make the UI feel slow."
              min={1}
              max={20}
              step={1}
              value={uiSettings.showProgressEveryNSteps}
              onChange={(value) => updateUiSetting({ showProgressEveryNSteps: value })}
            />
            <p className="pro-field-note">{livePreviewSummary}</p>
            <label className="pro-field">
              <FieldLabel label="Preview decoder" />
              <select
                value={uiSettings.livePreviewDecoder}
                onChange={(event) => updateUiSetting({ livePreviewDecoder: event.target.value })}
                disabled={!settingsStatus}
              >
                <option value="vae">VAE</option>
              </select>
              <p className="pro-field-note">Live previews currently decode SD 1.5 and SDXL image routes only.</p>
            </label>
            <label className="pro-toggle">
              <input
                type="checkbox"
                checked={uiSettings.livePreviewTitleProgress}
                onChange={(event) => updateUiSetting({ livePreviewTitleProgress: event.target.checked })}
                disabled={!settingsStatus}
              />
              <span>Show generation progress in the window title</span>
            </label>
            <div className="pro-settings-actions">
              <button type="button" className="pro-primary-button" onClick={onSaveSettings} disabled={!settingsStatus}>
                Save settings
              </button>
              <span>{settingsSaveStatus || (settingsStatus ? 'Connected' : 'Backend not connected')}</span>
            </div>
          </div>
        </InfoCard>
        )}
        {show('interface', 'ui interface gallery columns dock height output dock layout') && (
        <InfoCard title="Output dock" subtitle="Gallery density and dock size.">
          <div className="pro-form-stack">
            <div className="pro-control-grid">
              <RangeField
                label="Gallery columns"
                min={1}
                max={8}
                step={1}
                value={uiSettings.galleryColumns}
                onChange={(value) => updateUiSetting({ galleryColumns: value })}
              />
              <RangeField
                label="Dock height"
                min={160}
                max={1200}
                step={20}
                value={uiSettings.galleryHeight}
                onChange={(value) => updateUiSetting({ galleryHeight: value })}
              />
            </div>
            <div className="pro-settings-actions">
              <button type="button" className="pro-primary-button" onClick={onSaveSettings} disabled={!settingsStatus}>
                Save settings
              </button>
              <span>{settingsSaveStatus || (settingsStatus ? 'Connected' : 'Backend not connected')}</span>
            </div>
          </div>
        </InfoCard>
        )}
        {show('output', 'output metadata image format quality filename pattern sidecar grid hash infotext png jpg webp') && (
        <InfoCard title="Output and metadata" subtitle="Saved image format, filenames, sidecars, and infotext fields.">
          <div className="pro-form-stack">
            <div className="pro-control-grid">
              <label className="pro-field">
                <FieldLabel label="Image format" />
                <select
                  value={outputSettings.imageFormat}
                  onChange={(event) => updateOutputSetting({ imageFormat: event.target.value })}
                  disabled={!settingsStatus}
                >
                  {OUTPUT_FORMAT_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="pro-field">
                <FieldLabel label="Image quality" />
                <input
                  type="number"
                  min={10}
                  max={100}
                  step={1}
                  value={outputSettings.imageQuality}
                  onChange={(event) => updateOutputSetting({ imageQuality: clamp(Number(event.target.value) || 95, 10, 100) })}
                  disabled={!settingsStatus}
                />
              </label>
            </div>
            <label className="pro-field">
              <FieldLabel label="Filename pattern" tooltip="Tokens include [datetime], [date], [time], [seed], [model_name], [width], [height], and [seq]." />
              <input
                value={outputSettings.filenamePattern}
                onChange={(event) => updateOutputSetting({ filenamePattern: event.target.value })}
                disabled={!settingsStatus}
              />
            </label>
            <label className="pro-field">
              <FieldLabel label="Optimization profile id" />
              <input
                value={outputSettings.optimizationProfileId}
                onChange={(event) => updateOutputSetting({ optimizationProfileId: event.target.value })}
                disabled={!settingsStatus}
              />
            </label>
            <div className="pro-settings-columns pro-settings-columns-compact">
              {outputToggles.map((item) => (
                <label className="pro-toggle" key={item.key}>
                  <input
                    type="checkbox"
                    checked={Boolean(outputSettings[item.key])}
                    onChange={(event) =>
                      updateOutputSetting({ [item.key]: event.target.checked } as Partial<ProSettingsStatus['output']>)
                    }
                    disabled={!settingsStatus}
                  />
                  <span>{item.label}</span>
                </label>
              ))}
            </div>
          </div>
        </InfoCard>
        )}
        {show('video', 'video performance ltx precision bfloat16 float16 dtype cpu offload streamed group blocks gguf cuda kernels vram wan speed') && (
        <InfoCard
          title="Video engine performance"
          subtitle="Precision and VRAM strategy for the Wan and LTX pipelines. Changes apply to the next generation — no restart needed."
        >
          <div className="pro-form-stack">
            <div className="pro-control-grid">
              <label className="pro-field">
                <FieldLabel
                  label="LTX precision"
                  tooltip="LTX-Video is calibrated in bfloat16; float16 can overflow and cause artifacts. Use float16 only on GPUs without bf16 support (pre-Ampere)."
                />
                <select
                  value={videoSettings.ltxDtype}
                  onChange={(event) => updateVideoSetting({ ltxDtype: event.target.value })}
                  disabled={!settingsStatus}
                >
                  {LTX_DTYPE_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="pro-field">
                <FieldLabel
                  label="LTX CPU offload"
                  tooltip="Auto keeps small checkpoints fully on the GPU (fastest) and offloads only when the model would not fit."
                />
                <select
                  value={videoSettings.ltxCpuOffload}
                  onChange={(event) => updateVideoSetting({ ltxCpuOffload: event.target.value })}
                  disabled={!settingsStatus}
                >
                  {LTX_OFFLOAD_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <label className="pro-toggle">
              <input
                type="checkbox"
                checked={videoSettings.wanGroupOffloadStream}
                onChange={(event) => updateVideoSetting({ wanGroupOffloadStream: event.target.checked })}
                disabled={!settingsStatus}
              />
              <span>Streamed Wan group offload (overlap block transfers with compute)</span>
            </label>
            <RangeField
              label="Wan offload blocks per group"
              min={1}
              max={40}
              step={1}
              value={videoSettings.wanGroupOffloadBlocks}
              onChange={(value) => updateVideoSetting({ wanGroupOffloadBlocks: value })}
            />
            <label className="pro-toggle">
              <input
                type="checkbox"
                checked={videoSettings.ggufCudaKernels}
                onChange={(event) => updateVideoSetting({ ggufCudaKernels: event.target.checked })}
                disabled={!settingsStatus}
              />
              <span>GGUF optimized CUDA kernels (needs the `kernels` package; Linux only, ignored on Windows)</span>
            </label>
            <p className="pro-field-note">
              Tip for 16 GB cards: in NVIDIA Control Panel set CUDA Sysmem Fallback Policy to "Prefer No Sysmem
              Fallback" so an over-budget run fails fast instead of silently paging at 10x slower speed.
            </p>
          </div>
        </InfoCard>
        )}
        {show('video', 'advanced wan runtime sage attention native denoise vae decode chunk record stream resident vram low cpu memory') && (
        <InfoCard
          title="Advanced Wan runtime"
          subtitle="Exact control over the Wan execution path. Defaults are the shipped behavior — change one knob at a time and compare timings."
        >
          <div className="pro-form-stack">
            <label className="pro-field">
              <FieldLabel
                label="SageAttention"
                tooltip="Auto uses SageAttention when the package is installed (the shipped behavior). Force warns if it is missing; Off falls back to plain torch SDPA — useful when comparing quality or debugging attention artifacts."
              />
              <select
                value={videoSettings.wanSageAttention}
                onChange={(event) => updateVideoSetting({ wanSageAttention: event.target.value })}
                disabled={!settingsStatus}
              >
                <option value="auto">Auto (use when installed)</option>
                <option value="force">Force (warn if missing)</option>
                <option value="off">Off (plain torch SDPA)</option>
              </select>
            </label>
            <label className="pro-toggle">
              <input
                type="checkbox"
                checked={videoSettings.wanNativeDenoise}
                onChange={(event) => updateVideoSetting({ wanNativeDenoise: event.target.checked })}
                disabled={!settingsStatus}
              />
              <span>Native denoise loop (AIWF-owned stepping; off = diffusers pipeline as a black box)</span>
            </label>
            <label className="pro-toggle">
              <input
                type="checkbox"
                checked={videoSettings.wanManualVaeDecode}
                onChange={(event) => updateVideoSetting({ wanManualVaeDecode: event.target.checked })}
                disabled={!settingsStatus}
              />
              <span>Manual chunked VAE decode (lower peak VRAM, slower decode)</span>
            </label>
            {videoSettings.wanManualVaeDecode ? (
              <RangeField
                label="VAE decode chunk frames"
                min={1}
                max={16}
                step={1}
                value={videoSettings.wanVaeChunkFrames}
                onChange={(value) => updateVideoSetting({ wanVaeChunkFrames: value })}
              />
            ) : null}
            <label className="pro-toggle">
              <input
                type="checkbox"
                checked={videoSettings.wanGroupOffloadRecordStream}
                onChange={(event) => updateVideoSetting({ wanGroupOffloadRecordStream: event.target.checked })}
                disabled={!settingsStatus}
              />
              <span>Record CUDA streams during group offload (safer overlap; tiny overhead)</span>
            </label>
            <label className="pro-toggle">
              <input
                type="checkbox"
                checked={videoSettings.wanGroupOffloadLowCpuMem}
                onChange={(event) => updateVideoSetting({ wanGroupOffloadLowCpuMem: event.target.checked })}
                disabled={!settingsStatus}
              />
              <span>Low CPU memory staging for offload (off = faster swaps, more system RAM)</span>
            </label>
            <RangeField
              label="Resident mode minimum VRAM (GB)"
              tooltip="Dual FP8 high/low stages only co-reside on GPUs with at least this much VRAM; below it, Resident falls back to Balanced swapping. Lower at your own risk."
              min={8}
              max={96}
              step={1}
              value={videoSettings.wanResidentMinVramGb}
              onChange={(value) => updateVideoSetting({ wanResidentMinVramGb: value })}
            />
            <p className="pro-field-note">
              Applied to the next generation — no restart needed. If a change makes things slower or unstable, set it
              back: the defaults shown on first load are the tested configuration.
            </p>
          </div>
        </InfoCard>
        )}
        {show('video', 'wan video defaults runtime mode offload sampler flow shift high low model vae text encoder') && (
        <InfoCard title="Wan video defaults" subtitle="Restore the default Wan split, sampler, and offload choices when the video tab opens.">
          <div className="pro-form-stack">
            <div className="pro-control-grid">
              <label className="pro-field">
                <FieldLabel label="Runtime mode" />
                <select
                  value={videoSettings.wanRuntimeMode}
                  onChange={(event) => updateVideoSetting({ wanRuntimeMode: event.target.value })}
                  disabled={!settingsStatus}
                >
                  {WAN_RUNTIME_MODE_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="pro-field">
                <FieldLabel label="Offload" />
                <select
                  value={videoSettings.wanOffload}
                  onChange={(event) => updateVideoSetting({ wanOffload: event.target.value })}
                  disabled={!settingsStatus}
                >
                  {WAN_OFFLOAD_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="pro-control-grid">
              <label className="pro-field">
                <FieldLabel label="Sampler" />
                <select
                  value={videoSettings.wanSampler}
                  onChange={(event) => updateVideoSetting({ wanSampler: event.target.value })}
                  disabled={!settingsStatus}
                >
                  {WAN_SAMPLER_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="pro-field">
                <FieldLabel label="Flow shift" />
                <input
                  type="number"
                  min={0}
                  max={20}
                  step={0.1}
                  value={videoSettings.wanFlowShift}
                  onChange={(event) => updateVideoSetting({ wanFlowShift: clamp(Number(event.target.value) || 0, 0, 20) })}
                  disabled={!settingsStatus}
                />
              </label>
            </div>
            <label className="pro-field">
              <FieldLabel label="High model" />
              <input
                value={videoSettings.wanHigh}
                onChange={(event) => updateVideoSetting({ wanHigh: event.target.value })}
                disabled={!settingsStatus}
              />
            </label>
            <label className="pro-field">
              <FieldLabel label="Low model" />
              <input
                value={videoSettings.wanLow}
                onChange={(event) => updateVideoSetting({ wanLow: event.target.value })}
                disabled={!settingsStatus}
              />
            </label>
            <label className="pro-field">
              <FieldLabel label="VAE" />
              <input
                value={videoSettings.wanVae}
                onChange={(event) => updateVideoSetting({ wanVae: event.target.value })}
                disabled={!settingsStatus}
              />
            </label>
            <label className="pro-field">
              <FieldLabel label="Text encoder" />
              <input
                value={videoSettings.wanTextEncoder}
                onChange={(event) => updateVideoSetting({ wanTextEncoder: event.target.value })}
                disabled={!settingsStatus}
              />
            </label>
          </div>
        </InfoCard>
        )}
        {show('system', 'advanced launch port backend attention onnx theme api cors rate limit vram fp8 directml cpu torch compile nvenc hevc paths directories') && (
        <InfoCard title="Advanced launch controls" subtitle="Saved to launch profile. Bad choices can break startup until changed back.">
          <div className="pro-form-stack">
            <p className="pro-muted">Most changes apply on restart. Loaded models are not rebuilt just because a flag is saved.</p>
            <div className="pro-control-grid">
              <label className="pro-field">
                <FieldLabel label="Port" />
                <input
                  type="number"
                  min={1024}
                  max={65535}
                  step={1}
                  value={runtimeSettings.port}
                  onChange={(event) => updateRuntimeSetting({ port: clamp(Number(event.target.value) || 7860, 1024, 65535) })}
                  disabled={!settingsStatus}
                />
              </label>
              <label className="pro-field">
                <FieldLabel label="Backend" />
                <select
                  value={runtimeSettings.backend}
                  onChange={(event) => updateRuntimeSetting({ backend: event.target.value })}
                  disabled={!settingsStatus}
                >
                  {RUNTIME_BACKEND_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="pro-control-grid">
              <label className="pro-field">
                <FieldLabel label="Attention" />
                <select
                  value={runtimeSettings.attention}
                  onChange={(event) => updateRuntimeSetting({ attention: event.target.value })}
                  disabled={!settingsStatus}
                >
                  {RUNTIME_ATTENTION_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="pro-field">
                <FieldLabel label="ONNX provider" />
                <select
                  value={runtimeSettings.onnxProvider}
                  onChange={(event) => updateRuntimeSetting({ onnxProvider: event.target.value })}
                  disabled={!settingsStatus}
                >
                  {ONNX_PROVIDER_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="pro-control-grid">
              <label className="pro-field">
                <FieldLabel label="Theme" />
                <select
                  value={runtimeSettings.theme}
                  onChange={(event) => updateRuntimeSetting({ theme: event.target.value })}
                  disabled={!settingsStatus}
                >
                  <option value="dark">Dark</option>
                  <option value="light">Light</option>
                </select>
              </label>
              <label className="pro-field">
                <FieldLabel label="API rate limit" />
                <input
                  type="number"
                  min={0}
                  max={6000}
                  step={1}
                  value={runtimeSettings.apiRateLimitPerMinute}
                  onChange={(event) => updateRuntimeSetting({ apiRateLimitPerMinute: clamp(Number(event.target.value) || 0, 0, 6000) })}
                  disabled={!settingsStatus}
                />
              </label>
            </div>
            <label className="pro-field">
              <FieldLabel label="API CORS origins" />
              <input
                value={runtimeSettings.apiCorsOrigins}
                onChange={(event) => updateRuntimeSetting({ apiCorsOrigins: event.target.value })}
                disabled={!settingsStatus}
              />
            </label>
            <label className="pro-field">
              <FieldLabel
                label="VRAM profile"
                tooltip="Low targets 4-8 GB with sequential CPU offload; mid targets 8-16 GB with model CPU offload; high targets 16+ GB resident models. Restart AIWF for this change to take effect."
              />
              <select
                value={runtimeSettings.vramProfile}
                onChange={(event) => updateVramProfile(event.target.value)}
                disabled={!settingsStatus}
              >
                {VRAM_PROFILE_OPTIONS.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <div className="pro-settings-columns pro-settings-columns-compact">
              {runtimeToggles.map((item) => (
                <label className="pro-toggle" key={item.key}>
                  <input
                    type="checkbox"
                    checked={Boolean(runtimeSettings[item.key])}
                    onChange={(event) =>
                      updateRuntimeSetting({ [item.key]: event.target.checked } as Partial<ProSettingsStatus['runtime']>)
                    }
                    disabled={!settingsStatus}
                  />
                  <span>{item.label}</span>
                </label>
              ))}
            </div>
            <label className="pro-field">
              <FieldLabel label="Models directory" />
              <input
                value={runtimeSettings.modelsDir}
                onChange={(event) => updateRuntimeSetting({ modelsDir: event.target.value })}
                disabled={!settingsStatus}
              />
            </label>
            <label className="pro-field">
              <FieldLabel label="Checkpoint directory" />
              <input
                value={runtimeSettings.checkpointDir}
                onChange={(event) => updateRuntimeSetting({ checkpointDir: event.target.value })}
                disabled={!settingsStatus}
              />
            </label>
            <label className="pro-field">
              <FieldLabel label="Output directory" />
              <input
                value={runtimeSettings.outputDir}
                onChange={(event) => updateRuntimeSetting({ outputDir: event.target.value })}
                disabled={!settingsStatus}
              />
            </label>
            <label className="pro-field">
              <FieldLabel label="Extra model directories" />
              <textarea
                value={runtimeSettings.extraModelDirs}
                onChange={(event) => updateRuntimeSetting({ extraModelDirs: event.target.value })}
                disabled={!settingsStatus}
              />
            </label>
            <label className="pro-field">
              <FieldLabel label="Extra checkpoint directories" />
              <textarea
                value={runtimeSettings.extraCheckpointDirs}
                onChange={(event) => updateRuntimeSetting({ extraCheckpointDirs: event.target.value })}
                disabled={!settingsStatus}
              />
            </label>
          </div>
        </InfoCard>
        )}
        {show('generation', 'saved defaults snapshot model sampler resolution version') && (
        <InfoCard title="Saved defaults snapshot" subtitle="Current backend values after the last settings refresh.">
          <div className="pro-stat-grid">
            <StatTile label="Default model" value={settingsStatus?.generationDefaults.modelId ?? settings.modelId} hint="saved default" />
            <StatTile label="Sampler" value={settingsStatus?.generationDefaults.sampler ?? settings.sampler} hint="saved default" />
            <StatTile label="Resolution" value={`${settingsStatus?.generationDefaults.width ?? settings.width}x${settingsStatus?.generationDefaults.height ?? settings.height}`} hint="saved default" />
            <StatTile label="Version" value={bootstrap.version} hint="shell build" />
          </div>
        </InfoCard>
        )}
        {show('system', 'extensions plugins user addons custom api routes enable disable') && (
        <ExtensionsCard />
        )}
        {show('system', 'backend paths config launch profile models checkpoints outputs') && (
        <InfoCard title="Backend paths" subtitle="Real paths reported by the Pro API for tonight's QA pass.">
          <dl className="pro-runtime-list">
            <MetricRow label="Config" value={settingsStatus?.paths.settings || 'Unavailable'} />
            <MetricRow label="Launch profile" value={settingsStatus?.paths.launch || 'Unavailable'} />
            <MetricRow label="Models" value={settingsStatus?.paths.models || 'Unavailable'} />
            <MetricRow label="Checkpoints" value={settingsStatus?.paths.checkpoints || 'Unavailable'} />
            <MetricRow label="Outputs" value={settingsStatus?.paths.outputs || 'Unavailable'} />
          </dl>
        </InfoCard>
        )}
        {show('system', 'models upload drag drop sort reorganize reread headers gguf safetensors checkpoint inventory') && (
        <InfoCard title="Model file sorter" subtitle="Drop or pick model files; AIWF reads headers and moves confident matches.">
          <div className="pro-settings-action-row">
            <button
              type="button"
              className="pro-secondary-button"
              onClick={() => modelUploadInputRef.current?.click()}
              disabled={modelSortBusy || !settingsStatus}
            >
              <HardDrive size={14} aria-hidden="true" />
              Upload model files
            </button>
            <input
              ref={modelUploadInputRef}
              className="pro-file-input-hidden"
              type="file"
              multiple
              accept={MODEL_FILE_ACCEPT}
              onChange={handleSettingsModelUpload}
            />
            <button
              type="button"
              className="pro-secondary-button"
              onClick={handleSettingsModelReorganize}
              disabled={modelSortBusy || !settingsStatus}
            >
              <RefreshCcw size={14} aria-hidden="true" />
              Reorganize models
            </button>
          </div>
          <p className="pro-field-note">
            Reorganize scans the main models directory, reads model headers, and moves confident matches without overwriting files.
          </p>
          {modelSortStatus ? <p className="pro-muted">{modelSortStatus}</p> : null}
        </InfoCard>
        )}
        {show('interface', 'layout memory panel width dock height advanced') && (
        <InfoCard title="Layout memory" subtitle="Local shell layout is already persisted; this page makes those values obvious.">
          <dl className="pro-runtime-list">
            <MetricRow label="Left panel width" value={`${leftPanelWidth}px`} />
            <MetricRow label="Right panel width" value={`${rightPanelWidth}px`} />
            <MetricRow label="Bottom dock height" value={`${bottomDockHeight}px`} />
            <MetricRow label="Bottom dock visible" value={bottomDockVisible ? 'Yes' : 'No'} />
            <MetricRow label="Advanced panel open" value={showAdvanced ? 'Yes' : 'No'} />
          </dl>
        </InfoCard>
        )}
        {show('interface', 'local persistence boundary browser saved runtime session') && (
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
        )}
        {show('system', 'workspace inventory models samplers receipts') && (
        <InfoCard title="Workspace inventory" subtitle="The settings page now doubles as a concise shell inventory.">
          <div className="pro-stat-grid">
            <StatTile label="Models" value={`${bootstrap.models.length}`} hint="available routes" />
            <StatTile label="Samplers" value={`${bootstrap.samplers.length}`} hint="loaded options" />
            <StatTile label="Receipts" value={`${recentOutputs.length}`} hint="local artifacts" />
            <StatTile label="Latest receipt" value={summary.latestCreatedAt} hint="recent output" />
          </div>
        </InfoCard>
        )}
        {show('system', 'runtime policy device precision telemetry') && (
        <InfoCard title="Runtime policy" subtitle="CPU-side surface effects are fine here; GPU cycles stay reserved for model execution.">
          <ul className="pro-bullet-list">
            <li>Use CSS glow and pulse on threshold classes rather than shader effects.</li>
            <li>Keep timers and loading bars driven from sampled runtime state updates.</li>
            <li>Bound graph refresh cadence so telemetry stays light during runs.</li>
            <li>Runtime currently reports {runtime.device} with {runtime.precision} execution.</li>
          </ul>
        </InfoCard>
        )}
        {show('about', 'credits about version build ai embedded systems') && (
        <InfoCard title="Credits" subtitle="Local-first AI tools for consumers.">
          <p className="pro-muted">
            Engineered by <a href="https://www.ai-embedded-systems.com" target="_blank" rel="noreferrer">AI Embedded Systems</a>.
          </p>
        </InfoCard>
        )}
      </div>
    </section>
  )
}

const CanvasPreview = memo(CanvasPreviewImpl)

function CanvasPreviewImpl({
  activeMode,
  preview,
  statusMessage,
  width,
  height,
  onOpenSegmentation,
  onOpenHires,
  onOpenEnhance,
  onOpenReactor,
  onOpenControlNet,
  controlNetEnabled,
  controlNetAvailable,
  controlNetUnavailableMessage,
  bottomDockVisible,
  outputPreviewVisible,
  onToggleBottomDock,
  onToggleOutputPreview,
  leftPanelCollapsed,
  isGenerating,
  isContinuousGenerating,
  selectedModelWarning,
  onGenerate,
  onStopGenerate,
  onToggleContinuousGenerate,
}: {
  activeMode: ProMode
  preview: RecentOutput | null
  statusMessage: string
  width: number
  height: number
  onOpenSegmentation: () => void
  onOpenHires: () => void
  onOpenEnhance: () => void
  onOpenReactor: () => void
  onOpenControlNet: () => void
  controlNetEnabled: boolean
  controlNetAvailable: boolean
  controlNetUnavailableMessage: string
  bottomDockVisible: boolean
  outputPreviewVisible: boolean
  onToggleBottomDock: () => void
  onToggleOutputPreview: () => void
  leftPanelCollapsed: boolean
  isGenerating: boolean
  isContinuousGenerating: boolean
  selectedModelWarning: string
  onGenerate: () => void
  onStopGenerate: () => void
  onToggleContinuousGenerate: () => void
}) {
  const frameWidth = Math.max(1, preview?.width ?? width)
  const frameHeight = Math.max(1, preview?.height ?? height)
  const aspectRatio = `${frameWidth} / ${frameHeight}`
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
      const targetAspect = frameWidth / frameHeight
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
  }, [frameHeight, frameWidth])

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
  const outputReceiptText = preview ? formatOutputReceiptText(preview) : ''
  const previewVisibilityLabel = activeMode === 'video'
    ? outputPreviewVisible ? 'Hide video' : 'Show video'
    : outputPreviewVisible ? 'Hide image' : 'Show image'
  const emptyCanvas = activeMode === 'video'
    ? {
        className: 'pro-empty-preview pro-stage-empty pro-stage-empty-video',
        icon: Video,
        title: 'Video canvas',
        message: 'Generate text-to-video or upload a first frame in the prompt panel. Playback controls appear here.',
      }
    : {
        className: 'pro-empty-preview pro-stage-empty pro-stage-empty-image',
        icon: Image,
        title: 'Image canvas',
        message: 'Generate an image or select one from the output dock. Settings and history stay below the canvas.',
      }
  const EmptyIcon = emptyCanvas.icon
  const showImageTools = activeMode !== 'video'

  return (
    <section className="pro-canvas" aria-label="Canvas and output preview">
      <div className="pro-canvas-header">
        <div className="pro-canvas-title">
          <strong>Canvas</strong>
          <small>{frameWidth}x{frameHeight}</small>
        </div>
        <div className="pro-canvas-tools" aria-label="Canvas tools">
          {leftPanelCollapsed ? (
            <>
              <button
                type="button"
                className={isGenerating ? 'pro-generate-button pro-canvas-generate-button pro-generate-button-stop' : 'pro-generate-button pro-canvas-generate-button'}
                disabled={!isGenerating && Boolean(selectedModelWarning)}
                onClick={isGenerating ? onStopGenerate : onGenerate}
                title={!isGenerating && selectedModelWarning ? selectedModelWarning : undefined}
              >
                {isGenerating ? <X size={16} aria-hidden="true" /> : <Sparkles size={16} aria-hidden="true" />}
                <span>{isGenerating ? 'Stop' : activeMode === 'video' ? 'Generate video' : 'Generate image'}</span>
              </button>
              <button
                type="button"
                className={isContinuousGenerating ? 'pro-tool-chip pro-continuous-generate-button is-active' : 'pro-tool-chip pro-continuous-generate-button'}
                disabled={!isContinuousGenerating && (isGenerating || Boolean(selectedModelWarning))}
                onClick={onToggleContinuousGenerate}
                title={!isContinuousGenerating && selectedModelWarning ? selectedModelWarning : 'Keep generating until stopped.'}
              >
                {isContinuousGenerating ? <X size={14} aria-hidden="true" /> : <RefreshCcw size={14} aria-hidden="true" />}
                <span>{isContinuousGenerating ? 'Stop loop' : 'Continuous'}</span>
              </button>
            </>
          ) : null}
          {showImageTools ? (
            <>
              <button
                type="button"
                className={controlNetEnabled && controlNetAvailable ? 'pro-tool-chip is-active' : 'pro-tool-chip'}
                disabled={!controlNetAvailable}
                title={controlNetAvailable ? 'Configure ControlNet for this SD/SDXL model.' : controlNetUnavailableMessage}
                onClick={onOpenControlNet}
              >
                <SlidersHorizontal size={14} aria-hidden="true" />
                <span>ControlNet</span>
              </button>
              <button type="button" className="pro-tool-chip" onClick={onOpenSegmentation}>
                <ScanSearch size={14} aria-hidden="true" />
                <span>Segment</span>
              </button>
              <button type="button" className="pro-tool-chip" onClick={onOpenHires}>
                <Highlighter size={14} aria-hidden="true" />
                <span>Hi-res</span>
              </button>
              <button type="button" className="pro-tool-chip" onClick={onOpenEnhance}>
                <Sparkles size={14} aria-hidden="true" />
                <span>Enhance</span>
              </button>
              <button type="button" className="pro-tool-chip" onClick={onOpenReactor}>
                <Wand2 size={14} aria-hidden="true" />
                <span>ReActor</span>
              </button>
            </>
          ) : null}
          <button type="button" className="pro-icon-button" aria-label="Pan preview">
            <Hand size={16} aria-hidden="true" />
          </button>
          <button type="button" className="pro-icon-button" aria-label="Fit preview">
            <Maximize2 size={16} aria-hidden="true" />
          </button>
          <button type="button" className="pro-zoom-button">100%</button>
          <button type="button" className="pro-tool-chip" onClick={onToggleOutputPreview}>
            {outputPreviewVisible ? <EyeOff size={14} aria-hidden="true" /> : <Eye size={14} aria-hidden="true" />}
            <span>{previewVisibilityLabel}</span>
          </button>
          <button type="button" className="pro-tool-chip" onClick={onToggleBottomDock}>
            {bottomDockVisible ? <Rows3 size={14} aria-hidden="true" /> : <Layers2 size={14} aria-hidden="true" />}
            <span>{bottomDockVisible ? 'Hide dock' : 'Show dock'}</span>
          </button>
        </div>
      </div>

      <div className="pro-preview-stage" ref={previewStageRef}>
        {outputPreviewVisible ? (
        <div className="pro-output-frame" style={outputFrameStyle}>
          {previewIsVideo ? (
            <video src={preview.url} controls playsInline />
          ) : preview ? (
            <img src={preview.url} alt={preview.prompt} />
          ) : (
            <div className={emptyCanvas.className}>
              <EmptyIcon size={42} aria-hidden="true" />
              <strong>{emptyCanvas.title}</strong>
              <span>{emptyCanvas.message}</span>
            </div>
          )}
        </div>
        ) : (
          <div className="pro-output-hidden">
            {activeMode === 'video' ? <Video size={32} aria-hidden="true" /> : <Image size={32} aria-hidden="true" />}
            <strong>{activeMode === 'video' ? 'Video hidden' : 'Image hidden'}</strong>
            <span>{statusMessage || outputReceiptText}</span>
          </div>
        )}
      </div>

      <div className="pro-canvas-footer">
        <span>{frameWidth}x{frameHeight}</span>
        <span>{preview?.modelName ?? 'Local model'}</span>
        <span>{outputReceiptText || statusMessage}</span>
      </div>
    </section>
  )
}

const InpaintCanvas = memo(InpaintCanvasImpl)

function InpaintCanvasImpl({
  settings,
  onSettingsChange,
  statusMessage,
  preview,
  onOpenSegmentation,
  onOpenControlNet,
  controlNetEnabled,
  controlNetAvailable,
  controlNetUnavailableMessage,
  leftPanelCollapsed,
  isGenerating,
  isContinuousGenerating,
  selectedModelWarning,
  onGenerate,
  onStopGenerate,
  onToggleContinuousGenerate,
}: {
  settings: GenerationSettings
  onSettingsChange: Dispatch<SetStateAction<GenerationSettings>>
  statusMessage: string
  preview: RecentOutput | null
  onOpenSegmentation: () => void
  onOpenControlNet: () => void
  controlNetEnabled: boolean
  controlNetAvailable: boolean
  controlNetUnavailableMessage: string
  leftPanelCollapsed: boolean
  isGenerating: boolean
  isContinuousGenerating: boolean
  selectedModelWarning: string
  onGenerate: () => void
  onStopGenerate: () => void
  onToggleContinuousGenerate: () => void
}) {
  const imageCanvasRef = useRef<HTMLCanvasElement>(null)
  const maskCanvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [brushSize, setBrushSize] = useState(48)
  const [zoom, setZoom] = useState(1)
  const [erasing, setErasing] = useState(false)
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })
  const paintingRef = useRef(false)
  const lastPointRef = useRef<{ x: number; y: number } | null>(null)
  const hasStrokesRef = useRef(false)

  const loadImage = useCallback(
    (dataUrl: string) => {
      const img = new window.Image()
      img.onload = () => {
        setImageSize({ width: img.naturalWidth, height: img.naturalHeight })
        requestAnimationFrame(() => {
          const canvas = imageCanvasRef.current
          const mask = maskCanvasRef.current
          if (!canvas || !mask) {
            return
          }
          canvas.width = img.naturalWidth
          canvas.height = img.naturalHeight
          mask.width = img.naturalWidth
          mask.height = img.naturalHeight
          canvas.getContext('2d')?.drawImage(img, 0, 0)
          mask.getContext('2d')?.clearRect(0, 0, mask.width, mask.height)
          let exported = dataUrl
          if (!dataUrl.startsWith('data:')) {
            try {
              exported = canvas.toDataURL('image/png')
            } catch {
              exported = dataUrl
            }
          }
          onSettingsChange((current) => ({ ...current, initImageDataUrl: exported, maskImageDataUrl: '' }))
        })
        hasStrokesRef.current = false
      }
      img.src = dataUrl
    },
    [onSettingsChange],
  )

  useEffect(() => {
    if (settings.initImageDataUrl) {
      loadImage(settings.initImageDataUrl)
    }
  }, [loadImage, settings.initImageDataUrl])

  const handleFileChange = useCallback(
    (event: ReactChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      if (!file) {
        return
      }
      const reader = new FileReader()
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          loadImage(reader.result)
        }
      }
      reader.readAsDataURL(file)
      event.target.value = ''
    },
    [loadImage],
  )

  const exportMask = useCallback(() => {
    const mask = maskCanvasRef.current
    if (!mask || mask.width === 0) {
      return
    }
    if (!hasStrokesRef.current) {
      onSettingsChange((current) => (current.maskImageDataUrl ? { ...current, maskImageDataUrl: '' } : current))
      return
    }
    const exportCanvas = document.createElement('canvas')
    exportCanvas.width = mask.width
    exportCanvas.height = mask.height
    const ctx = exportCanvas.getContext('2d')
    if (!ctx) {
      return
    }
    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height)
    ctx.drawImage(mask, 0, 0)
    const dataUrl = exportCanvas.toDataURL('image/png')
    onSettingsChange((current) => ({ ...current, maskImageDataUrl: dataUrl }))
  }, [onSettingsChange])

  const [autoMaskBusy, setAutoMaskBusy] = useState(false)
  const [autoMaskStatus, setAutoMaskStatus] = useState('')

  const handleGenerateAutoMask = useCallback(async () => {
    if (!settings.initImageDataUrl) {
      setAutoMaskStatus('Load an image first.')
      return
    }
    const prompt = settings.autoMaskPrompt.trim()
    if (!prompt) {
      setAutoMaskStatus('Enter a SAM + DINO prompt first.')
      return
    }
    setAutoMaskBusy(true)
    setAutoMaskStatus('Segmenting…')
    try {
      const result = await generateAutoMask(settings.initImageDataUrl, prompt, settings.autoMaskBoxThreshold)
      const mask = maskCanvasRef.current
      const ctx = mask?.getContext('2d')
      if (!mask || !ctx || !result.mask) {
        setAutoMaskStatus('Segmentation returned an empty mask.')
        return
      }
      const img = new window.Image()
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve()
        img.onerror = () => reject(new Error('Mask image could not be decoded.'))
        img.src = result.mask
      })
      // Convert the returned luminance mask into the white-with-alpha strokes
      // the paint canvas uses, so brush edits still compose with it.
      const off = document.createElement('canvas')
      off.width = mask.width
      off.height = mask.height
      const offCtx = off.getContext('2d')
      if (!offCtx) {
        setAutoMaskStatus('Canvas is unavailable.')
        return
      }
      offCtx.drawImage(img, 0, 0, mask.width, mask.height)
      const data = offCtx.getImageData(0, 0, mask.width, mask.height)
      const px = data.data
      for (let i = 0; i < px.length; i += 4) {
        const lum = px[i]
        px[i] = 255
        px[i + 1] = 255
        px[i + 2] = 255
        px[i + 3] = lum
      }
      offCtx.putImageData(data, 0, 0)
      ctx.clearRect(0, 0, mask.width, mask.height)
      ctx.globalCompositeOperation = 'source-over'
      ctx.drawImage(off, 0, 0)
      hasStrokesRef.current = true
      exportMask()
      setAutoMaskStatus(result.status || 'Auto mask applied — refine with the brush if needed.')
    } catch (error: unknown) {
      setAutoMaskStatus(formatApiError(error))
    } finally {
      setAutoMaskBusy(false)
    }
  }, [exportMask, settings.autoMaskBoxThreshold, settings.autoMaskPrompt, settings.initImageDataUrl])

  const canvasPoint = useCallback((event: ReactPointerEvent<HTMLCanvasElement>) => {
    const mask = maskCanvasRef.current
    if (!mask) {
      return null
    }
    const rect = mask.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) {
      return null
    }
    return {
      x: ((event.clientX - rect.left) * mask.width) / rect.width,
      y: ((event.clientY - rect.top) * mask.height) / rect.height,
    }
  }, [])

  const paintTo = useCallback(
    (point: { x: number; y: number }) => {
      const mask = maskCanvasRef.current
      const ctx = mask?.getContext('2d')
      if (!mask || !ctx) {
        return
      }
      const scale = mask.getBoundingClientRect().width > 0 ? mask.width / mask.getBoundingClientRect().width : 1
      ctx.globalCompositeOperation = erasing ? 'destination-out' : 'source-over'
      ctx.strokeStyle = '#ffffff'
      ctx.fillStyle = '#ffffff'
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      ctx.lineWidth = brushSize * scale
      const last = lastPointRef.current
      ctx.beginPath()
      if (last) {
        ctx.moveTo(last.x, last.y)
        ctx.lineTo(point.x, point.y)
        ctx.stroke()
      } else {
        ctx.arc(point.x, point.y, (brushSize * scale) / 2, 0, Math.PI * 2)
        ctx.fill()
      }
      lastPointRef.current = point
      if (!erasing) {
        hasStrokesRef.current = true
      }
    },
    [brushSize, erasing],
  )

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLCanvasElement>) => {
      event.currentTarget.setPointerCapture(event.pointerId)
      paintingRef.current = true
      lastPointRef.current = null
      const point = canvasPoint(event)
      if (point) {
        paintTo(point)
      }
    },
    [canvasPoint, paintTo],
  )

  const handlePointerMove = useCallback(
    (event: ReactPointerEvent<HTMLCanvasElement>) => {
      if (!paintingRef.current) {
        return
      }
      const point = canvasPoint(event)
      if (point) {
        paintTo(point)
      }
    },
    [canvasPoint, paintTo],
  )

  const handlePointerUp = useCallback(() => {
    if (!paintingRef.current) {
      return
    }
    paintingRef.current = false
    lastPointRef.current = null
    exportMask()
  }, [exportMask])

  const clearMask = useCallback(() => {
    const mask = maskCanvasRef.current
    const ctx = mask?.getContext('2d')
    if (mask && ctx) {
      ctx.clearRect(0, 0, mask.width, mask.height)
    }
    hasStrokesRef.current = false
    onSettingsChange((current) => ({ ...current, maskImageDataUrl: '' }))
  }, [onSettingsChange])

  const usePreviewImage = useCallback(() => {
    if (preview?.url) {
      loadImage(preview.url)
    }
  }, [loadImage, preview])

  const handleWheel = useCallback((event: React.WheelEvent<HTMLDivElement>) => {
    if (imageSize.width <= 0) {
      return
    }
    event.preventDefault()
    const direction = event.deltaY < 0 ? 0.1 : -0.1
    setZoom((current) => clamp(Number((current + direction).toFixed(2)), 0.25, 4))
  }, [imageSize.width])

  const canvasDisplayStyle = useMemo<CSSProperties>(() => {
    if (imageSize.width <= 0 || imageSize.height <= 0) {
      return {}
    }
    return {
      width: `${imageSize.width * zoom}px`,
      height: `${imageSize.height * zoom}px`,
    }
  }, [imageSize.height, imageSize.width, zoom])

  return (
    <section className="pro-canvas" aria-label="Inpaint canvas">
      <div className="pro-canvas-header">
        <div className="pro-canvas-title">
          <strong>Inpaint</strong>
          <small>{imageSize.width > 0 ? `${imageSize.width}x${imageSize.height}` : 'load an image'}</small>
        </div>
        <div className="pro-canvas-tools" aria-label="Inpaint tools">
          {leftPanelCollapsed ? (
            <>
              <button
                type="button"
                className={isGenerating ? 'pro-generate-button pro-canvas-generate-button pro-generate-button-stop' : 'pro-generate-button pro-canvas-generate-button'}
                disabled={!isGenerating && Boolean(selectedModelWarning)}
                onClick={isGenerating ? onStopGenerate : onGenerate}
                title={!isGenerating && selectedModelWarning ? selectedModelWarning : undefined}
              >
                {isGenerating ? <X size={16} aria-hidden="true" /> : <Sparkles size={16} aria-hidden="true" />}
                <span>{isGenerating ? 'Stop' : 'Generate image'}</span>
              </button>
              <button
                type="button"
                className={isContinuousGenerating ? 'pro-tool-chip pro-continuous-generate-button is-active' : 'pro-tool-chip pro-continuous-generate-button'}
                disabled={!isContinuousGenerating && (isGenerating || Boolean(selectedModelWarning))}
                onClick={onToggleContinuousGenerate}
                title={!isContinuousGenerating && selectedModelWarning ? selectedModelWarning : 'Keep generating until stopped.'}
              >
                {isContinuousGenerating ? <X size={14} aria-hidden="true" /> : <RefreshCcw size={14} aria-hidden="true" />}
                <span>{isContinuousGenerating ? 'Stop loop' : 'Continuous'}</span>
              </button>
            </>
          ) : null}
          <button type="button" className="pro-tool-chip" onClick={() => fileInputRef.current?.click()}>
            <FileImage size={14} aria-hidden="true" />
            <span>Load image</span>
          </button>
          <input ref={fileInputRef} type="file" accept={IMAGE_FILE_ACCEPT} hidden onChange={handleFileChange} />
          {preview ? (
            <button type="button" className="pro-tool-chip" onClick={usePreviewImage}>
              <Image size={14} aria-hidden="true" />
              <span>Use preview</span>
            </button>
          ) : null}
          <button
            type="button"
            className={controlNetEnabled && controlNetAvailable ? 'pro-tool-chip is-active' : 'pro-tool-chip'}
            disabled={!controlNetAvailable}
            title={controlNetAvailable ? 'Configure ControlNet for this SD/SDXL model.' : controlNetUnavailableMessage}
            onClick={onOpenControlNet}
          >
            <SlidersHorizontal size={14} aria-hidden="true" />
            <span>ControlNet</span>
          </button>
          <button type="button" className="pro-tool-chip" onClick={onOpenSegmentation}>
            <ScanSearch size={14} aria-hidden="true" />
            <span>Segment</span>
          </button>
          <button
            type="button"
            className="pro-tool-chip"
            aria-pressed={!erasing}
            onClick={() => setErasing(false)}
          >
            <Brush size={14} aria-hidden="true" />
            <span>Brush</span>
          </button>
          <button
            type="button"
            className="pro-tool-chip"
            aria-pressed={erasing}
            onClick={() => setErasing(true)}
          >
            <X size={14} aria-hidden="true" />
            <span>Eraser</span>
          </button>
          <label className="pro-tool-chip">
            <span>Size {brushSize}px</span>
            <input
              type="range"
              min={4}
              max={160}
              value={brushSize}
              onChange={(event) => setBrushSize(Number(event.target.value))}
            />
          </label>
          <button type="button" className="pro-tool-chip" onClick={clearMask}>
            <RefreshCcw size={14} aria-hidden="true" />
            <span>Clear mask</span>
          </button>
          <button type="button" className="pro-tool-chip" onClick={() => setZoom(1)}>
            <Maximize2 size={14} aria-hidden="true" />
            <span>{Math.round(zoom * 100)}%</span>
          </button>
        </div>
      </div>

      <div className="pro-inpaint-auto-mask">
        <label className="pro-toggle">
          <input
            type="checkbox"
            checked={settings.autoMaskEnabled}
            onChange={(event) => onSettingsChange((current) => ({ ...current, autoMaskEnabled: event.target.checked }))}
          />
          <span>Auto mask</span>
        </label>
        <label className="pro-field">
          <FieldLabel label="SAM + DINO prompt" compact />
          <input
            list="pro-auto-mask-prompts"
            value={settings.autoMaskPrompt}
            placeholder="person, face, shirt, hands..."
            onChange={(event) => onSettingsChange((current) => ({ ...current, autoMaskPrompt: event.target.value }))}
          />
          <datalist id="pro-auto-mask-prompts">
            <option value="person" />
            <option value="face" />
            <option value="hands" />
            <option value="clothing" />
            <option value="background" />
          </datalist>
        </label>
        <label className="pro-field">
          <FieldLabel label="Auto mask route" compact />
          <input
            list="pro-auto-mask-routes"
            value={settings.autoMaskModel}
            onChange={(event) => onSettingsChange((current) => ({ ...current, autoMaskModel: event.target.value }))}
          />
          <datalist id="pro-auto-mask-routes">
            <option value="sam+dino" />
            <option value="sam" />
            <option value="grounding-dino" />
          </datalist>
        </label>
        <button
          type="button"
          className="pro-secondary-button"
          onClick={handleGenerateAutoMask}
          disabled={autoMaskBusy || !settings.initImageDataUrl}
        >
          <ScanSearch size={14} aria-hidden="true" />
          {autoMaskBusy ? 'Segmenting…' : 'Generate mask'}
        </button>
        {autoMaskStatus ? <span className="pro-muted">{autoMaskStatus}</span> : null}
      </div>

      <div className="pro-preview-stage pro-inpaint-stage" onWheel={handleWheel}>
        {imageSize.width > 0 ? (
          <div className="pro-inpaint-canvas-stack" style={canvasDisplayStyle}>
            <canvas
              ref={imageCanvasRef}
              style={{ display: 'block', width: '100%', height: '100%', borderRadius: 8 }}
            />
            <canvas
              ref={maskCanvasRef}
              style={{
                position: 'absolute',
                inset: 0,
                width: '100%',
                height: '100%',
                opacity: settings.inpaintMaskOpacity,
                touchAction: 'none',
                cursor: 'crosshair',
                borderRadius: 8,
              }}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={handlePointerUp}
            />
          </div>
        ) : (
          <div className="pro-empty-preview pro-stage-empty pro-stage-empty-inpaint">
            <Brush size={42} aria-hidden="true" />
            <strong>Load image to inpaint</strong>
            <span>Use Load image or Use preview, then paint the white mask. Scroll the canvas to zoom for edge cleanup.</span>
          </div>
        )}
      </div>

      <div className="pro-canvas-footer">
        <label className="pro-toggle">
          <input
            type="checkbox"
            checked={settings.inpaintOnlyMasked}
            onChange={(event) => onSettingsChange((current) => ({ ...current, inpaintOnlyMasked: event.target.checked }))}
          />
          <span>Only masked</span>
        </label>
        <label className="pro-tool-chip">
          <span>Padding {settings.inpaintMaskedPadding}px</span>
          <input
            type="range"
            min={0}
            max={256}
            value={settings.inpaintMaskedPadding}
            onChange={(event) => onSettingsChange((current) => ({ ...current, inpaintMaskedPadding: Number(event.target.value) }))}
          />
        </label>
        <label className="pro-tool-chip">
          <span>Opacity {Math.round(settings.inpaintMaskOpacity * 100)}%</span>
          <input
            type="range"
            min={0.15}
            max={0.9}
            step={0.01}
            value={settings.inpaintMaskOpacity}
            onChange={(event) => onSettingsChange((current) => ({ ...current, inpaintMaskOpacity: Number(event.target.value) }))}
          />
        </label>
        <label className="pro-field pro-inpaint-content-field">
          <FieldLabel label="Masked content" compact />
          <select
            value={settings.inpaintMaskContent}
            onChange={(event) => onSettingsChange((current) => ({ ...current, inpaintMaskContent: event.target.value }))}
          >
            <option value="original">Original</option>
            <option value="fill">Fill</option>
            <option value="latent noise">Latent noise</option>
            <option value="latent nothing">Latent nothing</option>
          </select>
        </label>
        <label className="pro-tool-chip">
          <span>Denoise {settings.denoisingStrength.toFixed(2)}</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={settings.denoisingStrength}
            onChange={(event) =>
              onSettingsChange((current) => ({ ...current, denoisingStrength: Number(event.target.value) }))
            }
          />
        </label>
        <label className="pro-tool-chip">
          <span>Mask blur {settings.maskBlur}px</span>
          <input
            type="range"
            min={0}
            max={64}
            value={settings.maskBlur}
            onChange={(event) => onSettingsChange((current) => ({ ...current, maskBlur: Number(event.target.value) }))}
          />
        </label>
        <span>{statusMessage}</span>
      </div>
    </section>
  )
}

const BottomDock = memo(BottomDockImpl)

function BottomDockImpl({
  visible,
  height,
  recentOutputs,
  selectedOutput,
  statusMessage,
  generationError,
  selectedModelName,
  onPreviewSelect,
  onApplyOutputSettings,
  onResizeStart,
  onToggleVisible,
}: {
  visible: boolean
  height: number
  recentOutputs: RecentOutput[]
  selectedOutput: RecentOutput | null
  statusMessage: string
  generationError: string
  selectedModelName: string
  onPreviewSelect: (value: RecentOutput) => void
  onApplyOutputSettings: (value: RecentOutput) => void
  onResizeStart: (event: ReactMouseEvent<HTMLButtonElement>) => void
  onToggleVisible: () => void
}) {
  const [copyStatus, setCopyStatus] = useState('')
  const selectedOutputDetails = useMemo(
    () => (selectedOutput ? buildOutputDetailRows(selectedOutput) : []),
    [selectedOutput],
  )
  const outputStatusText = selectedOutput
    ? buildOutputStatusText(selectedOutput)
    : statusMessage

  useEffect(() => {
    setCopyStatus('')
  }, [selectedOutput?.id])

  useEffect(() => {
    if (!copyStatus) {
      return undefined
    }
    const timeoutId = window.setTimeout(() => setCopyStatus(''), 1800)
    return () => window.clearTimeout(timeoutId)
  }, [copyStatus])

  const handleApplySettings = useCallback(() => {
    if (!selectedOutput) {
      return
    }
    onApplyOutputSettings(selectedOutput)
    setCopyStatus('Applied.')
  }, [onApplyOutputSettings, selectedOutput])

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
          <span>{generationError ? 'Generation error' : selectedOutput ? 'Selected output' : 'Status'}</span>
          <strong>{generationError || outputStatusText}</strong>
          {!generationError && selectedOutput ? (
            <>
              <div className="pro-bottom-output-meta">
                {selectedOutputDetails.map((detail) => (
                  <small key={detail.label}>
                    <span>{detail.label}</span>
                    <strong>{detail.value}</strong>
                  </small>
                ))}
              </div>
              <button
                type="button"
                className="pro-secondary-button pro-bottom-copy-button"
                onClick={handleApplySettings}
                disabled={!selectedOutput}
              >
                <SlidersHorizontal size={14} aria-hidden="true" />
                Apply settings
              </button>
              {copyStatus ? <small className="pro-bottom-copy-status">{copyStatus}</small> : null}
            </>
          ) : null}
        </div>
        <div className="pro-bottom-gallery">
          {recentOutputs.map((item) => (
            <button
              key={item.id}
              type="button"
              className={selectedOutput?.id === item.id ? 'pro-bottom-thumb pro-bottom-thumb-active' : 'pro-bottom-thumb'}
              onClick={() => onPreviewSelect(item)}
              aria-pressed={selectedOutput?.id === item.id}
              aria-label={buildOutputButtonLabel(item)}
              title={buildOutputStatusText(item)}
            >
              <OutputMedia item={item} />
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

function CollapsedPanelButton({
  side,
  label,
  icon: Icon,
  onClick,
}: {
  side: 'left' | 'right'
  label: string
  icon: LucideIcon
  onClick: () => void
}) {
  return (
    <aside className={`pro-collapsed-panel pro-collapsed-panel-${side}`} aria-label={label}>
      <button type="button" className="pro-collapsed-panel-button" onClick={onClick}>
        <Icon size={17} aria-hidden="true" />
        <span>{label}</span>
      </button>
    </aside>
  )
}

function RuntimePanel({
  runtime,
  selectedModelName,
  onUnloadModel,
  onToggleRightPanel,
}: {
  runtime: ProRuntimeStatus
  selectedModelName: string
  onUnloadModel: () => void
  onToggleRightPanel: () => void
}) {
  const loadedModelName = runtime.loadedModel.loaded ? runtime.loadedModel.name : 'No model loaded'
  return (
    <aside className="pro-status-panel" aria-label="Runtime status">
      <div className="pro-status-heading">
        <span>System</span>
        <div className="pro-status-heading-actions">
          <strong>
            <span
              className={`pro-status-dot ${runtimeLightClass(runtime.state, Boolean(runtime.job.error))}`}
              aria-hidden="true"
            />
            {runtime.state}
          </strong>
          <button type="button" className="pro-icon-button" aria-label="Hide system column" onClick={onToggleRightPanel}>
            <PanelRight size={15} aria-hidden="true" />
          </button>
        </div>
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
        <button
          type="button"
          className="pro-unload-button"
          onClick={onUnloadModel}
          disabled={!runtime.loadedModel.loaded}
        >
          Unload model
        </button>
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

function getControlNetCompatibility(model: ProModelOption | undefined, controlNetModel: string): ControlNetCompatibility {
  const modelFamily = controlNetFamilyForModel(model)
  const controlNetFamily = controlNetFamilyForName(controlNetModel)
  if (!modelFamily) {
    return {
      supported: false,
      modelFamily: null,
      controlNetFamily,
      message: 'ControlNet is enabled only for SD 1.5 and SDXL routes. Flux, Qwen, Sana, SD3.5, and video routes will not receive SD ControlNet units.',
    }
  }
  if (controlNetFamily && controlNetFamily !== modelFamily) {
    return {
      supported: false,
      modelFamily,
      controlNetFamily,
      message: `The selected model is ${controlNetFamilyLabel(modelFamily)}, but the ControlNet entry looks like ${controlNetFamilyLabel(controlNetFamily)}.`,
    }
  }
  return {
    supported: true,
    modelFamily,
    controlNetFamily,
    message: controlNetFamily
      ? `Ready for ${controlNetFamilyLabel(modelFamily)} ControlNet.`
      : `Ready for ${controlNetFamilyLabel(modelFamily)} ControlNet. Use a matching local model id or path.`,
  }
}

function controlNetFamilyForModel(model: ProModelOption | undefined): 'sd15' | 'sdxl' | null {
  const engineId = model?.engineId ?? 'unknown'
  if (engineId === 'sd15' || engineId === 'sdxl') {
    return engineId
  }
  const text = `${model?.architecture ?? ''} ${model?.name ?? ''} ${model?.id ?? ''}`.toLowerCase()
  if (text.includes('sdxl') || text.includes('stable diffusion xl')) {
    return 'sdxl'
  }
  if (text.includes('sd15') || text.includes('sd1.5') || text.includes('stable diffusion 1.5')) {
    return 'sd15'
  }
  return null
}

function controlNetFamilyForName(value: string): 'sd15' | 'sdxl' | null {
  const text = value.toLowerCase()
  if (!text.trim()) {
    return null
  }
  if (text.includes('sdxl') || text.includes('_xl') || text.includes('-xl') || text.includes('controlnet-xl')) {
    return 'sdxl'
  }
  if (
    text.includes('sd15') ||
    text.includes('sd_15') ||
    text.includes('sd-15') ||
    text.includes('sd1.5') ||
    text.includes('v11') ||
    text.includes('v1-1') ||
    text.includes('control_v')
  ) {
    return 'sd15'
  }
  return null
}

function controlNetFamilyLabel(family: 'sd15' | 'sdxl'): string {
  return family === 'sdxl' ? 'SDXL' : 'SD 1.5'
}

function modelFitsCreationMode(model: ProModelOption, mode: CreationMode): boolean {
  const engineId = model.engineId ?? 'unknown'
  const kind = `${model.kind ?? ''}`.toLowerCase()
  const isVideoModel = kind === 'video' || engineId === 'sana_video' || engineId === 'wan'
  if (mode === 'video') {
    return isVideoModel
  }
  if (engineId === 'unknown') {
    return false
  }
  if (mode === 'inpaint') {
    return !isVideoModel && (engineId === 'sd15' || engineId === 'sdxl' || engineId === 'flux_fill')
  }
  // Flux Fill is an inpaint-only checkpoint; keep it out of plain txt2img.
  return !isVideoModel && engineId !== 'flux_fill'
}

function modelsForCreationMode(models: ProModelOption[], mode: CreationMode): ProModelOption[] {
  return models.filter((model) => modelFitsCreationMode(model, mode))
}

function summarizeEnginesForModels(engines: EngineSummary[], models: ProModelOption[]): EngineSummary[] {
  const labels = new Map<EngineId, string>()
  for (const engine of engines) {
    labels.set(engine.id, engine.label)
  }
  const counts = new Map<EngineId, number>()
  for (const model of models) {
    const id = (model.engineId ?? 'unknown') as EngineId
    counts.set(id, (counts.get(id) ?? 0) + 1)
  }
  return Array.from(counts.entries())
    .map(([id, count]) => ({
      id,
      label: labels.get(id) ?? modelEngineFallbackLabel(id),
      count,
    }))
    .sort((left, right) => left.label.localeCompare(right.label))
}

function modelEngineFallbackLabel(engineId: EngineId): string {
  switch (engineId) {
    case 'flux':
      return 'Flux'
    case 'flux_fill':
      return 'Flux Fill (inpaint)'
    case 'flux2':
      return 'Flux.2 Klein'
    case 'sana_video':
      return 'Sana Video'
    case 'wan':
      return 'Wan Video'
    case 'sd15':
      return 'Stable Diffusion 1.5'
    case 'sdxl':
      return 'Stable Diffusion XL'
    case 'sd35':
      return 'Stable Diffusion 3.5'
    case 'zimage':
      return 'Z-Image'
    case 'qwen':
      return 'Qwen Image'
    case 'sana':
      return 'Sana'
    default:
      return 'Other'
  }
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
    case 'wan':
      return architecture.includes('wan')
    case 'sd15':
      return architecture.includes('sd15') || architecture.includes('sd1.5') || architecture.includes('stable diffusion 1.5')
    case 'sdxl':
      return architecture.includes('sdxl') || architecture.includes('stable diffusion xl')
    case 'sd35':
      return architecture.includes('sd35') || architecture.includes('sd3.5') || architecture.includes('stable diffusion 3.5')
    case 'zimage':
      return architecture.includes('z-image') || architecture.includes('z image') || architecture.includes('zimage')
    case 'qwen':
      return architecture.includes('qwen')
    case 'sana':
      return architecture.includes('sana') && !architecture.includes('video')
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

function isModelBlocked(model: ProModelOption | undefined): boolean {
  if (!model?.status) {
    return false
  }
  return ['blocked-cleanly', 'broken-runtime', 'unsupported-no-route'].includes(model.status)
}

function modelBlockedMessage(model: ProModelOption | undefined): string {
  if (!model) {
    return 'Selected model is not available in the current Pro model list.'
  }
  const reason = model.reason?.trim()
  if (reason) {
    return reason
  }
  return `${model.name} is not ready for Pro generation.`
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
  onAction,
}: {
  title: string
  actionLabel: string
  icon: LucideIcon
  onAction?: () => void
}) {
  return (
    <div className="pro-panel-header">
      <span>{title}</span>
      <button type="button" className="pro-icon-button" aria-label={actionLabel} onClick={onAction}>
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

const RailButton = memo(RailButtonImpl)

function RailButtonImpl({
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
      <div className="pro-modal" role="dialog" aria-modal="true" aria-label={title} onClick={(event) => event.stopPropagation()}>
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

function collectGenerateOutputs(result: ProGenerateResult, modelName: string): RecentOutput[] {
  const sessionOutputs =
    result.recentOutputs.length > 0
      ? result.recentOutputs
      : result.output
        ? [result.output]
        : []
  return sessionOutputs.map((item) => ({
    ...item,
    modelName: item.modelName || modelName,
  }))
}

function buildOutputGenerationSettingsPatch(output: RecentOutput): GenerationSettingsPatch {
  const flatPatch = normalizeGenerationSettingsPatch({
    mode: output.mode,
    prompt: output.prompt,
    negativePrompt: output.negativePrompt,
    width: output.width,
    height: output.height,
    steps: output.steps,
    cfgScale: output.cfgScale,
    clipSkip: output.clipSkip,
    sampler: output.sampler,
    scheduler: output.scheduler,
    seed: output.seed,
    modelName: output.modelName,
  })
  const embeddedPatch = normalizeGenerationSettingsPatch(output.generationSettings)
  return { ...flatPatch, ...embeddedPatch }
}

function normalizeGenerationSettingsPatch(value: unknown): GenerationSettingsPatch {
  const record = looseRecord(value)
  const patch: GenerationSettingsPatch = {}
  const mode = readPatchMode(record, ['mode'])
  if (mode) {
    patch.mode = mode
  }
  assignStringPatch(patch, record, 'prompt', ['prompt'])
  assignStringPatch(patch, record, 'negativePrompt', ['negativePrompt', 'negative_prompt'])
  assignStringPatch(patch, record, 'modelId', ['modelId', 'model_id', 'checkpointId', 'checkpoint_id'])
  assignStringPatch(patch, record, 'modelName', ['modelName', 'model_name', 'model'])
  const pipelineBackend = readPatchPipelineBackend(record, ['pipelineBackend', 'pipeline_backend'])
  if (pipelineBackend) {
    patch.pipelineBackend = pipelineBackend
  }
  assignStringPatch(patch, record, 'sampler', ['sampler'])
  assignStringPatch(patch, record, 'scheduler', ['scheduler'])
  assignStringPatch(patch, record, 'hiresUpscaler', ['hiresUpscaler', 'hires_upscaler', 'hrUpscaler', 'hr_upscaler'])
  assignStringPatch(patch, record, 'sourceImageDataUrl', ['sourceImageDataUrl', 'source_image_data_url'])
  assignStringPatch(patch, record, 'sourceImageName', ['sourceImageName', 'source_image_name'])
  assignStringPatch(patch, record, 'sanaQuantization', ['sanaQuantization', 'sana_quantization'])
  assignStringPatch(patch, record, 'sanaVaeTiling', ['sanaVaeTiling', 'sana_vae_tiling', 'vaeTiling', 'vae_tiling'])
  assignStringPatch(patch, record, 'wanRuntimeMode', ['wanRuntimeMode', 'wan_runtime_mode', 'runtimeMode', 'runtime_mode'])
  assignStringPatch(patch, record, 'highNoiseModelId', ['highNoiseModelId', 'high_noise_model_id'])
  assignStringPatch(patch, record, 'lowNoiseModelId', ['lowNoiseModelId', 'low_noise_model_id'])
  assignStringPatch(patch, record, 'highNoiseLoraId', ['highNoiseLoraId', 'high_noise_lora_id'])
  assignStringPatch(patch, record, 'lowNoiseLoraId', ['lowNoiseLoraId', 'low_noise_lora_id'])
  assignStringPatch(patch, record, 'vaeId', ['vaeId', 'vae_id'])
  assignStringPatch(patch, record, 'textEncoderPath', ['textEncoderPath', 'text_encoder_path'])
  assignStringPatch(patch, record, 'wanOffload', ['wanOffload', 'wan_offload', 'offload'])
  assignStringPatch(patch, record, 'wanSigmaType', ['wanSigmaType', 'wan_sigma_type', 'sigmaType', 'sigma_type'])
  assignStringPatch(patch, record, 'wanSampler', ['wanSampler', 'wan_sampler'])
  assignStringPatch(patch, record, 'initImageDataUrl', ['initImageDataUrl', 'init_image_data_url'])
  assignStringPatch(patch, record, 'maskImageDataUrl', ['maskImageDataUrl', 'mask_image_data_url'])
  assignStringPatch(patch, record, 'inpaintMaskContent', ['inpaintMaskContent', 'inpaint_mask_content'])
  assignStringPatch(patch, record, 'controlNetModel', ['controlNetModel', 'controlnet_model'])
  assignStringPatch(patch, record, 'controlNetModule', ['controlNetModule', 'controlnet_module'])
  assignStringPatch(patch, record, 'controlNetImageDataUrl', ['controlNetImageDataUrl', 'controlnet_image_data_url'])
  assignStringPatch(patch, record, 'controlNetImageName', ['controlNetImageName', 'controlnet_image_name'])

  assignNumberPatch(patch, record, 'width', ['width'])
  assignNumberPatch(patch, record, 'height', ['height'])
  assignNumberPatch(patch, record, 'steps', ['steps'])
  assignNumberPatch(patch, record, 'cfgScale', ['cfgScale', 'cfg_scale'])
  assignNumberPatch(patch, record, 'seed', ['seed'])
  assignNumberPatch(patch, record, 'clipSkip', ['clipSkip', 'clip_skip'])
  assignNumberPatch(patch, record, 'batchSize', ['batchSize', 'batch_size'])
  assignNumberPatch(patch, record, 'batchCount', ['batchCount', 'batch_count'])
  assignNumberPatch(patch, record, 'hiresScale', ['hiresScale', 'hires_scale', 'hrScale', 'hr_scale'])
  assignNumberPatch(patch, record, 'hiresSteps', ['hiresSteps', 'hires_steps', 'hrSteps', 'hr_steps'])
  assignNumberPatch(patch, record, 'hiresDenoise', ['hiresDenoise', 'hires_denoise', 'hrDenoisingStrength', 'hr_denoising_strength'])
  assignNumberPatch(patch, record, 'frames', ['frames'])
  assignNumberPatch(patch, record, 'fps', ['fps'])
  assignNumberPatch(patch, record, 'highNoiseSteps', ['highNoiseSteps', 'high_noise_steps'])
  assignNumberPatch(patch, record, 'lowNoiseSteps', ['lowNoiseSteps', 'low_noise_steps'])
  assignNumberPatch(patch, record, 'boundaryRatio', ['boundaryRatio', 'boundary_ratio'])
  assignNumberPatch(patch, record, 'highNoiseLoraScale', ['highNoiseLoraScale', 'high_noise_lora_scale'])
  assignNumberPatch(patch, record, 'lowNoiseLoraScale', ['lowNoiseLoraScale', 'low_noise_lora_scale'])
  assignNumberPatch(patch, record, 'wanFlowShift', ['wanFlowShift', 'wan_flow_shift', 'flowShift', 'flow_shift'])
  assignNumberPatch(patch, record, 'denoisingStrength', ['denoisingStrength', 'denoising_strength'])
  assignNumberPatch(patch, record, 'maskBlur', ['maskBlur', 'mask_blur'])
  assignNumberPatch(patch, record, 'inpaintMaskedPadding', ['inpaintMaskedPadding', 'inpaint_masked_padding'])
  assignNumberPatch(patch, record, 'inpaintMaskOpacity', ['inpaintMaskOpacity', 'inpaint_mask_opacity'])
  assignNumberPatch(patch, record, 'controlNetWeight', ['controlNetWeight', 'controlnet_weight'])
  assignNumberPatch(patch, record, 'controlNetGuidanceStart', ['controlNetGuidanceStart', 'controlnet_guidance_start'])
  assignNumberPatch(patch, record, 'controlNetGuidanceEnd', ['controlNetGuidanceEnd', 'controlnet_guidance_end'])
  assignNumberPatch(patch, record, 'controlNetProcessorRes', ['controlNetProcessorRes', 'controlnet_processor_res'])

  assignBooleanPatch(patch, record, 'enableHires', ['enableHires', 'enable_hires', 'enableHr', 'enable_hr'])
  assignBooleanPatch(patch, record, 'offloadTextEncoderAfterEncode', ['offloadTextEncoderAfterEncode', 'offload_text_encoder_after_encode'])
  assignBooleanPatch(patch, record, 'useSageAttention', ['useSageAttention', 'use_sage_attention'])
  assignBooleanPatch(patch, record, 'generateAudio', ['generateAudio', 'generate_audio'])
  assignBooleanPatch(patch, record, 'inpaintOnlyMasked', ['inpaintOnlyMasked', 'inpaint_only_masked'])
  assignBooleanPatch(patch, record, 'autoMaskEnabled', ['autoMaskEnabled', 'auto_mask_enabled'])
  assignBooleanPatch(patch, record, 'controlNetEnabled', ['controlNetEnabled', 'controlnet_enabled'])
  assignBooleanPatch(patch, record, 'saveImages', ['saveImages', 'save_images'])
  return patch
}

function applyGenerationSettingsPatch(
  current: GenerationSettings,
  patch: GenerationSettingsPatch,
  models: ProModelOption[],
  aspectRatios: AspectRatioOption[],
): GenerationSettings {
  const model = findImportedModel(models, patch)
  let next: GenerationSettings = model
    ? applyModelPresetSettings({ ...current, modelId: model.id }, model, aspectRatios)
    : { ...current }
  const mode = patch.mode === 'inpaint' ? 'inpaint' : patch.mode === 'video' ? 'video' : patch.mode === 'image' ? 'image' : undefined
  const width = finiteNumber(patch.width)
  const height = finiteNumber(patch.height)
  next = {
    ...next,
    mode: mode ?? next.mode,
    prompt: typeof patch.prompt === 'string' && patch.prompt.length > 0 ? patch.prompt : next.prompt,
    negativePrompt: typeof patch.negativePrompt === 'string' ? patch.negativePrompt : next.negativePrompt,
    width: width ? clamp(Math.round(width), 64, 2048) : next.width,
    height: height ? clamp(Math.round(height), 64, 2048) : next.height,
    steps: patch.steps !== undefined ? clamp(Math.round(finiteNumber(patch.steps) ?? next.steps), 1, 150) : next.steps,
    cfgScale: patch.cfgScale !== undefined ? clamp(finiteNumber(patch.cfgScale) ?? next.cfgScale, 0, 30) : next.cfgScale,
    clipSkip: patch.clipSkip !== undefined ? clamp(Math.round(finiteNumber(patch.clipSkip) ?? next.clipSkip), 1, 12) : next.clipSkip,
    sampler: typeof patch.sampler === 'string' && patch.sampler.length > 0 ? patch.sampler : next.sampler,
    scheduler: typeof patch.scheduler === 'string' && patch.scheduler.length > 0 ? patch.scheduler : next.scheduler,
    pipelineBackend: patch.pipelineBackend === 'dual' || patch.pipelineBackend === 'sdcpp' || patch.pipelineBackend === 'aiwf' ? patch.pipelineBackend : next.pipelineBackend,
    seed: patch.seed !== undefined ? Math.round(finiteNumber(patch.seed) ?? next.seed) : next.seed,
    batchSize: patch.batchSize !== undefined ? clamp(Math.round(finiteNumber(patch.batchSize) ?? next.batchSize), 1, 4) : next.batchSize,
    batchCount: patch.batchCount !== undefined ? clamp(Math.round(finiteNumber(patch.batchCount) ?? next.batchCount), 1, 4) : next.batchCount,
    enableHires: typeof patch.enableHires === 'boolean' ? patch.enableHires : next.enableHires,
    hiresScale: patch.hiresScale !== undefined ? clamp(finiteNumber(patch.hiresScale) ?? next.hiresScale, 1, 4) : next.hiresScale,
    hiresSteps: patch.hiresSteps !== undefined ? clamp(Math.round(finiteNumber(patch.hiresSteps) ?? next.hiresSteps), 1, 150) : next.hiresSteps,
    hiresDenoise: patch.hiresDenoise !== undefined ? clamp(finiteNumber(patch.hiresDenoise) ?? next.hiresDenoise, 0, 1) : next.hiresDenoise,
    hiresUpscaler: typeof patch.hiresUpscaler === 'string' && patch.hiresUpscaler.length > 0 ? patch.hiresUpscaler : next.hiresUpscaler,
    frames: patch.frames !== undefined ? clamp(Math.round(finiteNumber(patch.frames) ?? next.frames), 1, 241) : next.frames,
    fps: patch.fps !== undefined ? clamp(Math.round(finiteNumber(patch.fps) ?? next.fps), 1, 60) : next.fps,
    sourceImageDataUrl: typeof patch.sourceImageDataUrl === 'string' ? patch.sourceImageDataUrl : next.sourceImageDataUrl,
    sourceImageName: typeof patch.sourceImageName === 'string' ? patch.sourceImageName : next.sourceImageName,
    sanaQuantization: typeof patch.sanaQuantization === 'string' && patch.sanaQuantization.length > 0 ? patch.sanaQuantization : next.sanaQuantization,
    sanaVaeTiling: typeof patch.sanaVaeTiling === 'string' && patch.sanaVaeTiling.length > 0 ? patch.sanaVaeTiling : next.sanaVaeTiling,
    offloadTextEncoderAfterEncode: typeof patch.offloadTextEncoderAfterEncode === 'boolean' ? patch.offloadTextEncoderAfterEncode : next.offloadTextEncoderAfterEncode,
    useSageAttention: typeof patch.useSageAttention === 'boolean' ? patch.useSageAttention : next.useSageAttention,
    generateAudio: typeof patch.generateAudio === 'boolean' ? patch.generateAudio : next.generateAudio,
    wanRuntimeMode: typeof patch.wanRuntimeMode === 'string' && patch.wanRuntimeMode.length > 0 ? patch.wanRuntimeMode : next.wanRuntimeMode,
    highNoiseModelId: typeof patch.highNoiseModelId === 'string' ? patch.highNoiseModelId : next.highNoiseModelId,
    lowNoiseModelId: typeof patch.lowNoiseModelId === 'string' ? patch.lowNoiseModelId : next.lowNoiseModelId,
    highNoiseSteps: patch.highNoiseSteps !== undefined ? clamp(Math.round(finiteNumber(patch.highNoiseSteps) ?? next.highNoiseSteps), 1, 150) : next.highNoiseSteps,
    lowNoiseSteps: patch.lowNoiseSteps !== undefined ? clamp(Math.round(finiteNumber(patch.lowNoiseSteps) ?? next.lowNoiseSteps), 1, 150) : next.lowNoiseSteps,
    boundaryRatio: patch.boundaryRatio !== undefined ? clamp(finiteNumber(patch.boundaryRatio) ?? next.boundaryRatio, 0, 1) : next.boundaryRatio,
    highNoiseLoraId: typeof patch.highNoiseLoraId === 'string' ? patch.highNoiseLoraId : next.highNoiseLoraId,
    highNoiseLoraScale: patch.highNoiseLoraScale !== undefined ? clamp(finiteNumber(patch.highNoiseLoraScale) ?? next.highNoiseLoraScale, 0, 3) : next.highNoiseLoraScale,
    lowNoiseLoraId: typeof patch.lowNoiseLoraId === 'string' ? patch.lowNoiseLoraId : next.lowNoiseLoraId,
    lowNoiseLoraScale: patch.lowNoiseLoraScale !== undefined ? clamp(finiteNumber(patch.lowNoiseLoraScale) ?? next.lowNoiseLoraScale, 0, 3) : next.lowNoiseLoraScale,
    vaeId: typeof patch.vaeId === 'string' ? patch.vaeId : next.vaeId,
    textEncoderPath: typeof patch.textEncoderPath === 'string' ? patch.textEncoderPath : next.textEncoderPath,
    wanOffload: typeof patch.wanOffload === 'string' && patch.wanOffload.length > 0 ? patch.wanOffload : next.wanOffload,
    wanSigmaType: typeof patch.wanSigmaType === 'string' && patch.wanSigmaType.length > 0 ? patch.wanSigmaType : next.wanSigmaType,
    wanSampler: typeof patch.wanSampler === 'string' && patch.wanSampler.length > 0 ? patch.wanSampler : next.wanSampler,
    wanFlowShift: patch.wanFlowShift !== undefined ? clamp(finiteNumber(patch.wanFlowShift) ?? next.wanFlowShift, 0, 20) : next.wanFlowShift,
    denoisingStrength: patch.denoisingStrength !== undefined ? clamp(finiteNumber(patch.denoisingStrength) ?? next.denoisingStrength, 0, 1) : next.denoisingStrength,
    maskBlur: patch.maskBlur !== undefined ? clamp(Math.round(finiteNumber(patch.maskBlur) ?? next.maskBlur), 0, 64) : next.maskBlur,
    inpaintOnlyMasked: typeof patch.inpaintOnlyMasked === 'boolean' ? patch.inpaintOnlyMasked : next.inpaintOnlyMasked,
    inpaintMaskedPadding: patch.inpaintMaskedPadding !== undefined ? clamp(Math.round(finiteNumber(patch.inpaintMaskedPadding) ?? next.inpaintMaskedPadding), 0, 256) : next.inpaintMaskedPadding,
    inpaintMaskContent: typeof patch.inpaintMaskContent === 'string' && patch.inpaintMaskContent.length > 0 ? patch.inpaintMaskContent : next.inpaintMaskContent,
    controlNetEnabled: typeof patch.controlNetEnabled === 'boolean' ? patch.controlNetEnabled : next.controlNetEnabled,
    controlNetModel: typeof patch.controlNetModel === 'string' ? patch.controlNetModel : next.controlNetModel,
    controlNetModule: typeof patch.controlNetModule === 'string' && patch.controlNetModule.length > 0 ? patch.controlNetModule : next.controlNetModule,
    controlNetWeight: patch.controlNetWeight !== undefined ? clamp(finiteNumber(patch.controlNetWeight) ?? next.controlNetWeight, 0, 2) : next.controlNetWeight,
    controlNetGuidanceStart: patch.controlNetGuidanceStart !== undefined ? clamp(finiteNumber(patch.controlNetGuidanceStart) ?? next.controlNetGuidanceStart, 0, 1) : next.controlNetGuidanceStart,
    controlNetGuidanceEnd: patch.controlNetGuidanceEnd !== undefined ? clamp(finiteNumber(patch.controlNetGuidanceEnd) ?? next.controlNetGuidanceEnd, 0, 1) : next.controlNetGuidanceEnd,
    controlNetProcessorRes: patch.controlNetProcessorRes !== undefined ? clamp(Math.round(finiteNumber(patch.controlNetProcessorRes) ?? next.controlNetProcessorRes), 64, 4096) : next.controlNetProcessorRes,
    saveImages: typeof patch.saveImages === 'boolean' ? patch.saveImages : next.saveImages,
  }
  const matchingRatio = findMatchingAspectRatio(aspectRatios, next.width, next.height)
  return matchingRatio ? { ...next, aspectRatioId: matchingRatio.id } : next
}

function findImportedModel(models: ProModelOption[], patch: GenerationSettingsPatch): ProModelOption | undefined {
  const modelId = typeof patch.modelId === 'string' ? patch.modelId.trim().toLowerCase() : ''
  const modelName = typeof patch.modelName === 'string' ? patch.modelName.trim().toLowerCase() : ''
  if (!modelId && !modelName) {
    return undefined
  }
  return models.find((model) => {
    const labels = [model.id, model.name, model.assetSummary].map((value) => `${value ?? ''}`.trim().toLowerCase())
    return labels.some((label) => {
      if (!label) {
        return false
      }
      return (modelId && label === modelId) || (modelName && (label === modelName || label.includes(modelName)))
    })
  })
}

function finiteNumber(value: unknown): number | undefined {
  const numberValue = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(numberValue) ? numberValue : undefined
}

function looseRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

function readPatchUnknown(record: Record<string, unknown>, keys: string[]): unknown {
  for (const key of keys) {
    if (Object.prototype.hasOwnProperty.call(record, key)) {
      return record[key]
    }
  }
  return undefined
}

function readPatchString(record: Record<string, unknown>, keys: string[]): string | undefined {
  const value = readPatchUnknown(record, keys)
  if (typeof value === 'string') {
    return value
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  return undefined
}

function readPatchNumber(record: Record<string, unknown>, keys: string[]): number | undefined {
  return finiteNumber(readPatchUnknown(record, keys))
}

function readPatchBoolean(record: Record<string, unknown>, keys: string[]): boolean | undefined {
  const value = readPatchUnknown(record, keys)
  if (typeof value === 'boolean') {
    return value
  }
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase()
    if (['true', '1', 'yes', 'on'].includes(normalized)) {
      return true
    }
    if (['false', '0', 'no', 'off'].includes(normalized)) {
      return false
    }
  }
  if (typeof value === 'number') {
    return value !== 0
  }
  return undefined
}

function readPatchMode(record: Record<string, unknown>, keys: string[]): CreationMode | undefined {
  const value = readPatchString(record, keys)?.toLowerCase()
  if (value === 'image' || value === 'txt2img' || value === 'img2img') {
    return 'image'
  }
  if (value === 'inpaint') {
    return 'inpaint'
  }
  if (value === 'video') {
    return 'video'
  }
  return undefined
}

function readPatchPipelineBackend(record: Record<string, unknown>, keys: string[]): GenerationSettings['pipelineBackend'] | undefined {
  const value = readPatchString(record, keys)?.toLowerCase()
  if (value === 'dual' || value === 'both') {
    return 'dual'
  }
  if (value === 'sdcpp' || value === 'stable-diffusion.cpp' || value === 'stable_diffusion_cpp') {
    return 'sdcpp'
  }
  if (value === 'aiwf' || value === 'diffusers' || value === 'pipeline') {
    return 'aiwf'
  }
  return undefined
}

function assignStringPatch(patch: GenerationSettingsPatch, record: Record<string, unknown>, key: keyof GenerationSettingsPatch, keys: string[]): void {
  const value = readPatchString(record, keys)
  if (value !== undefined) {
    ;(patch as Record<string, unknown>)[key] = value
  }
}

function assignNumberPatch(patch: GenerationSettingsPatch, record: Record<string, unknown>, key: keyof GenerationSettingsPatch, keys: string[]): void {
  const value = readPatchNumber(record, keys)
  if (value !== undefined) {
    ;(patch as Record<string, unknown>)[key] = value
  }
}

function assignBooleanPatch(patch: GenerationSettingsPatch, record: Record<string, unknown>, key: keyof GenerationSettingsPatch, keys: string[]): void {
  const value = readPatchBoolean(record, keys)
  if (value !== undefined) {
    ;(patch as Record<string, unknown>)[key] = value
  }
}

function readReceiptNumber(receipt: Record<string, unknown>, key: string): number | undefined {
  return finiteNumber(receipt[key])
}

function readReceiptSpeed(receipt: Record<string, unknown>): string | undefined {
  const stepsPerSecond = finiteNumber(receipt.steps_per_second)
  return stepsPerSecond !== undefined ? `${stepsPerSecond.toFixed(2)} steps/s` : undefined
}

function readImportedModelName(imported: ImportedGenerationMetadata, settings: GenerationSettingsPatch): string | undefined {
  const model = imported.metadata.model
  if (model && typeof model === 'object' && !Array.isArray(model)) {
    const record = model as Record<string, unknown>
    const label = record.title ?? record.id ?? record.filename
    return typeof label === 'string' && label.length > 0 ? label : undefined
  }
  return settings.modelId || settings.modelName
}

function buildDefaultXyPlotCells(settings: GenerationSettings, models: ProModelOption[]): XyPlotCell[] {
  const fallbackModelId = models.find((model) => model.id === settings.modelId)?.id ?? models[0]?.id ?? settings.modelId
  const baseSteps = clamp(Math.round(settings.steps || 20), 1, 150)
  return Array.from({ length: XY_PLOT_DEFAULT_CELLS }, (_, index) => {
    const model = models[index % Math.max(1, models.length)]
    const stepsOffset = Math.floor(index / Math.max(1, models.length)) * 5
    return {
      id: `xy-${index}`,
      modelId: model?.id ?? fallbackModelId,
      steps: clamp(baseSteps + stepsOffset, 1, 150),
    }
  })
}

function normalizeXyPlotCells(cells: XyPlotCell[], models: ProModelOption[], settings: GenerationSettings): XyPlotCell[] {
  const modelIds = new Set(models.map((model) => model.id))
  const fallbackModelId = models.find((model) => model.id === settings.modelId)?.id ?? models[0]?.id ?? settings.modelId
  const normalized = cells.slice(0, XY_PLOT_MAX_CELLS).map((cell, index) => ({
    id: cell.id || `xy-${index}`,
    modelId: modelIds.has(cell.modelId) ? cell.modelId : fallbackModelId,
    steps: clamp(Math.round(Number.isFinite(cell.steps) ? cell.steps : settings.steps), 1, 150),
  }))
  return normalized.length > 0 ? normalized : buildDefaultXyPlotCells(settings, models)
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

function isSdcppRuntime(runtime: ProRuntimeStatus): boolean {
  const backend = `${runtime.backend} ${runtime.loadedModel.type}`.toLowerCase()
  return backend.includes('dual') || backend.includes('sdcpp') || backend.includes('stable-diffusion.cpp') || backend.includes('stable diffusion.cpp')
}

function isDualRuntime(runtime: ProRuntimeStatus): boolean {
  const backend = `${runtime.backend} ${runtime.loadedModel.type}`.toLowerCase()
  return backend.includes('dual')
}

function mergeBootstrapDefaults(
  current: GenerationSettings,
  nextBootstrap: ProBootstrap,
): GenerationSettings {
  const routeModels = modelsForCreationMode(nextBootstrap.models, current.mode)
  const modelStillExists = routeModels.some((model) => model.id === current.modelId)
  const samplerStillExists = nextBootstrap.samplers.includes(current.sampler)
  const ratioStillExists = nextBootstrap.aspectRatios.some((ratio) => ratio.id === current.aspectRatioId)
  const ratio = ratioStillExists
    ? nextBootstrap.aspectRatios.find((item) => item.id === current.aspectRatioId)
    : nextBootstrap.aspectRatios.find((item) => item.id === nextBootstrap.defaults.aspectRatioId)

  return {
    ...current,
    modelId: modelStillExists ? current.modelId : routeModels[0]?.id ?? nextBootstrap.defaults.modelId,
    sampler: samplerStillExists ? current.sampler : nextBootstrap.defaults.sampler,
    aspectRatioId: ratio?.id ?? nextBootstrap.defaults.aspectRatioId,
    width: ratio?.width ?? current.width,
    height: ratio?.height ?? current.height,
  }
}

function applyModelPresetSettings(
  current: GenerationSettings,
  model: ProModelOption | undefined,
  ratios: AspectRatioOption[],
): GenerationSettings {
  const preset = model?.generationPreset
  if (!preset) {
    return current
  }
  const next: GenerationSettings = { ...current }
  if (Number.isFinite(preset.steps) && Number(preset.steps) > 0) {
    next.steps = Number(preset.steps)
  }
  if (Number.isFinite(preset.cfgScale) && Number(preset.cfgScale) >= 0) {
    next.cfgScale = Number(preset.cfgScale)
  }
  if (preset.sampler) {
    next.sampler = preset.sampler
  }
  if (preset.scheduler) {
    next.scheduler = preset.scheduler
  }
  if (Number.isFinite(preset.clipSkip) && Number(preset.clipSkip) >= 1) {
    next.clipSkip = Number(preset.clipSkip)
  }
  const width = Number(preset.width)
  const height = Number(preset.height)
  if (Number.isFinite(width) && width >= 64 && Number.isFinite(height) && height >= 64) {
    next.width = width
    next.height = height
    next.aspectRatioId = findMatchingAspectRatio(ratios, width, height)?.id ?? current.aspectRatioId
  }
  return next
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
    current.clipSkip === expected.clipSkip &&
    current.batchSize === expected.batchSize &&
    current.batchCount === expected.batchCount &&
    current.sourceImageDataUrl === expected.sourceImageDataUrl &&
    current.sourceImageName === expected.sourceImageName
  )
}

function imageAspect(width: number, height: number, fallback?: AspectRatioOption): number {
  if (width > 0 && height > 0) {
    return width / height
  }
  if (fallback && fallback.width > 0 && fallback.height > 0) {
    return fallback.width / fallback.height
  }
  return 1
}

function dimensionsForShortEdge(aspect: number, shortEdge: number): { width: number; height: number } {
  const safeAspect = Number.isFinite(aspect) && aspect > 0 ? aspect : 1
  let width = safeAspect >= 1 ? roundModelDimension(shortEdge * safeAspect) : roundModelDimension(shortEdge)
  let height = safeAspect >= 1 ? roundModelDimension(shortEdge) : roundModelDimension(shortEdge / safeAspect)
  const maxDimension = Math.max(width, height)
  if (maxDimension > 2048) {
    const scale = 2048 / maxDimension
    width = roundModelDimension(width * scale)
    height = roundModelDimension(height * scale)
  }
  return { width, height }
}

function roundModelDimension(value: number): number {
  return clamp(Math.round(value / 16) * 16, 64, 2048)
}

function findMatchingAspectRatio(
  ratios: AspectRatioOption[],
  width: number,
  height: number,
): AspectRatioOption | undefined {
  const target = imageAspect(width, height)
  let best: { ratio: AspectRatioOption; distance: number } | null = null
  for (const ratio of ratios) {
    const distance = Math.abs(imageAspect(ratio.width, ratio.height) - target)
    if (!best || distance < best.distance) {
      best = { ratio, distance }
    }
  }
  return best && best.distance <= 0.04 ? best.ratio : undefined
}

function isCreationMode(mode: ProMode): mode is CreationMode {
  return mode === 'image' || mode === 'video' || mode === 'inpaint'
}

function railsForMode(mode: ProMode, activeRail?: string): IconItem<string>[] {
  const railIds = RAILS_BY_MODE[mode] ?? RAILS_BY_MODE.image
  const items = railIds
    .map((id) => RAIL_ITEM_BY_ID.get(id))
    .filter((item): item is IconItem<string> => Boolean(item))
  // The active rail must always be visible, even when hash navigation or a
  // menu action lands on a rail outside the current mode's list — otherwise
  // the user is stranded on a surface with no highlighted way back.
  if (activeRail && !railIds.includes(activeRail)) {
    const item = RAIL_ITEM_BY_ID.get(activeRail)
    if (item) {
      items.push(item)
    }
  }
  return items
}

function modeContainingRail(railId: string, currentMode: ProMode): ProMode {
  if ((RAILS_BY_MODE[currentMode] ?? []).includes(railId)) {
    return currentMode
  }
  const fallback = (Object.keys(RAILS_BY_MODE) as ProMode[]).find((mode) =>
    RAILS_BY_MODE[mode].includes(railId),
  )
  return fallback ?? 'image'
}

function fileExtension(file: File): string {
  const name = file.name || ''
  const dot = name.lastIndexOf('.')
  return dot >= 0 ? name.slice(dot).toLowerCase() : ''
}

function isImageFile(file: File): boolean {
  return file.type.startsWith('image/') || IMAGE_FILE_EXTENSIONS.has(fileExtension(file))
}

function isVideoFile(file: File): boolean {
  return file.type.startsWith('video/') || VIDEO_FILE_EXTENSIONS.has(fileExtension(file))
}

function isModelFile(file: File): boolean {
  return MODEL_FILE_EXTENSIONS.has(fileExtension(file))
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : '')
    reader.onerror = () => reject(new Error(`Could not read ${file.name}.`))
    reader.readAsDataURL(file)
  })
}

function imageDimensionsFromDataUrl(dataUrl: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve) => {
    const image = new window.Image()
    image.onload = () => resolve({ width: image.naturalWidth || 0, height: image.naturalHeight || 0 })
    image.onerror = () => resolve({ width: 0, height: 0 })
    image.src = dataUrl
  })
}

function summarizeModelSort(result: ProModelSortResult): string {
  const moved = result.counts.moved
  const left = result.counts.left
  const inventory = result.counts.inventoryCount
  if (moved > 0 && left > 0) {
    return `Moved ${moved} model file${moved === 1 ? '' : 's'}; ${left} need review. Inventory: ${inventory}.`
  }
  if (moved > 0) {
    return `Moved ${moved} model file${moved === 1 ? '' : 's'}. Inventory: ${inventory}.`
  }
  if (left > 0) {
    return `${left} model file${left === 1 ? '' : 's'} need review. Inventory: ${inventory}.`
  }
  return `Model inventory refreshed. Inventory: ${inventory}.`
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

function isGenerationCancelResult(error: unknown): boolean {
  return isAbortError(error) || (error instanceof ProApiError && error.status === 499)
}

function readInitialRail(): string {
  const hash = window.location.hash.replace(/^#/, '').trim()
  if (hash === 'image' || hash === 'inpaint' || hash === 'video') {
    return 'create'
  }
  if (hash === 'audio') {
    return 'audiolab'
  }
  if (hash === 'settings') {
    return 'settings'
  }
  return RAIL_IDS.has(hash) ? hash : 'create'
}

function readInitialMode(): ProMode {
  const hash = window.location.hash.replace(/^#/, '').trim()
  if (hash === 'image' || hash === 'inpaint' || hash === 'video' || hash === 'audio' || hash === 'models' || hash === 'data' || hash === 'settings') {
    return hash
  }
  const rail = readInitialRail()
  if (rail === 'models' || rail === 'data') {
    return rail
  }
  if (rail === 'settings') {
    return 'settings'
  }
  if (rail === 'audiolab') {
    return 'audio'
  }
  return 'image'
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
      outputPreviewVisible:
        typeof parsed.outputPreviewVisible === 'boolean'
          ? parsed.outputPreviewVisible
          : DEFAULT_LAYOUT_PREFERENCES.outputPreviewVisible,
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
      { id: 'open-controlnet', label: 'ControlNet' },
      { id: 'open-enhance', label: 'Enhance / VSR' },
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
