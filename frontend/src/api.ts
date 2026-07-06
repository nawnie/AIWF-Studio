import type {
  AspectRatioOption,
  CreationMode,
  EngineSummary,
  GenerationSettings,
  LoadedModelInfo,
  ProBootstrap,
  ProCapabilitiesStatus,
  ProCapabilityItem,
  ProDataStatus,
  ProDownloadsStatus,
  ProGenerateRequest,
  ProGenerateResult,
  ProLogEvent,
  ProLogFile,
  ProLogStatus,
  ProModelOption,
  ProReadinessCounts,
  ProReadinessFamily,
  ProReadinessItem,
  ProReadinessStatus,
  ProRuntimeStatus,
  ProSettingsStatus,
  ProStopResult,
  RecentOutput,
  ResourceMetric,
  ResourceTone,
} from './types'

type JsonRecord = Record<string, unknown>

const API_BASE = (import.meta.env.VITE_AIWF_API_BASE ?? '').replace(/\/$/, '')
const API_LATENCY_SAMPLE_LIMIT = 40

export interface ProApiLatencySample {
  path: string
  status: number
  clientMs: number
  serverMs: number | null
  createdAt: string
}

export interface ProStartupStatus {
  status: string
  serverReady: boolean
  windowReady: boolean
  startedAt: string
  serverReadyAt: string
  windowReadyAt: string
  minSplashMs: number
  readyHoldMs: number
}

const apiLatencySamples: ProApiLatencySample[] = []

function recordApiLatency(sample: ProApiLatencySample): void {
  apiLatencySamples.push(sample)
  if (apiLatencySamples.length > API_LATENCY_SAMPLE_LIMIT) {
    apiLatencySamples.splice(0, apiLatencySamples.length - API_LATENCY_SAMPLE_LIMIT)
  }
}

export function getProApiLatencySamples(): ProApiLatencySample[] {
  return apiLatencySamples.map((sample) => ({ ...sample }))
}

export interface ProModelSortAction {
  filename: string
  source: string
  family: string
  architecture: string
  destSubdir: string
  status: string
  reason: string
}

export interface ProModelSortResult {
  status: string
  uploadedPath: string
  uploadedBytes: number
  actions: ProModelSortAction[]
  counts: {
    total: number
    moved: number
    left: number
    inventoryCount: number
  }
}

const DEFAULT_ASPECT_RATIOS: AspectRatioOption[] = [
  { id: '1:1', label: '1:1', width: 1024, height: 1024 },
  { id: '3:2', label: '3:2', width: 1536, height: 1024 },
  { id: '16:9', label: '16:9', width: 1344, height: 768 },
  { id: '9:16', label: '9:16', width: 768, height: 1344 },
  { id: '2:3', label: '2:3', width: 1024, height: 1536 },
]

const DEFAULT_MODELS: ProModelOption[] = [
  {
    id: 'sdxl-base-1.0',
    name: 'sdxl-base-1.0 (Diffusers)',
    architecture: 'SDXL 1.0',
    engineId: 'sdxl',
    engineLabel: 'Stable Diffusion XL',
    backend: 'Diffusers',
    status: 'Loaded',
  },
  {
    id: 'models/sana-video/Diffusers/SANA-Video_2B_480p_diffusers',
    name: 'SANA-Video 2B 480p',
    kind: 'video',
    architecture: 'sana_video',
    engineId: 'sana_video',
    engineLabel: 'Sana Video',
    backend: 'Diffusers',
    status: 'Needs snapshot',
  },
]

const DEFAULT_SAMPLERS = ['DPM++ 2M Karras', 'Euler a', 'DPM++ SDE', 'UniPC']

const READINESS_STATUS_KEYS = [
  'working',
  'metadata-only',
  'blocked-cleanly',
  'broken-runtime',
  'unsupported-no-route',
] as const

const EMPTY_READINESS_COUNTS: ProReadinessCounts = {
  working: 0,
  'metadata-only': 0,
  'blocked-cleanly': 0,
  'broken-runtime': 0,
  'unsupported-no-route': 0,
}

const DEFAULT_SETTINGS: GenerationSettings = {
  mode: 'image',
  prompt:
    'A futuristic overgrown city in the rain, neon lights reflecting on wet streets, cinematic, ultra detailed, moody lighting, photorealistic',
  negativePrompt: 'blurry, low quality, distorted, text, watermark, signature',
  modelId: DEFAULT_MODELS[0].id,
  aspectRatioId: '3:2',
  width: 1536,
  height: 1024,
  steps: 30,
  cfgScale: 7,
  sampler: DEFAULT_SAMPLERS[0],
  scheduler: 'automatic',
  seed: -1,
  clipSkip: 1,
  batchSize: 1,
  batchCount: 1,
  enableHires: false,
  hiresScale: 1.75,
  hiresSteps: 20,
  hiresDenoise: 0.3,
  hiresUpscaler: 'lanczos',
  frames: 81,
  fps: 16,
  sourceImageDataUrl: '',
  sourceImageName: '',
  sanaQuantization: 'auto',
  sanaVaeTiling: 'auto',
  offloadTextEncoderAfterEncode: true,
  useSageAttention: true,
  generateAudio: false,
  wanRuntimeMode: 'fast_5b',
  highNoiseModelId: '',
  lowNoiseModelId: '',
  highNoiseSteps: 20,
  lowNoiseSteps: 1,
  boundaryRatio: 0.875,
  highNoiseLoraId: '',
  highNoiseLoraScale: 1.0,
  lowNoiseLoraId: '',
  lowNoiseLoraScale: 1.0,
  vaeId: '',
  textEncoderPath: '',
  wanOffload: 'balanced',
  wanSigmaType: 'simple',
  wanSampler: 'unipc',
  wanFlowShift: 5.0,
  initImageDataUrl: '',
  maskImageDataUrl: '',
  denoisingStrength: 0.75,
  maskBlur: 4,
  inpaintOnlyMasked: false,
  inpaintMaskedPadding: 32,
  inpaintMaskContent: 'original',
  inpaintMaskOpacity: 0.48,
  autoMaskEnabled: false,
  autoMaskPrompt: '',
  autoMaskModel: 'sam+dino',
  autoMaskBoxThreshold: 0.3,
  autoMaskTextThreshold: 0.25,
  controlNetEnabled: false,
  controlNetModel: '',
  controlNetModule: 'canny',
  controlNetImageDataUrl: '',
  controlNetImageName: '',
  controlNetWeight: 1,
  controlNetGuidanceStart: 0,
  controlNetGuidanceEnd: 1,
  controlNetProcessorRes: 512,
  saveImages: true,
}

const FALLBACK_RUNTIME: ProRuntimeStatus = {
  state: 'Connecting',
  job: {
    id: '',
    state: 'idle',
    progress: 0,
    step: 0,
    totalSteps: 0,
    message: '',
    hasResult: false,
    error: '',
    previewUrl: '',
  },
  backend: 'Waiting for API',
  device: 'Local runtime pending',
  precision: 'Unknown',
  attention: 'Unknown',
  maxResolution: 'Unknown',
  queueCount: 0,
  resources: [
    { label: 'VRAM', value: 'Unavailable', percent: 0, tone: 'neutral' },
    { label: 'GPU utilization', value: 'Unavailable', percent: 0, tone: 'neutral' },
    { label: 'RAM', value: 'Unavailable', percent: 0, tone: 'neutral' },
    { label: 'Storage', value: 'Unavailable', percent: 0, tone: 'neutral' },
    { label: 'CPU', value: 'Unavailable', percent: 0, tone: 'neutral' },
  ],
  loadedModel: {
    name: 'No model loaded',
    type: 'Text-to-Image',
    baseModel: 'None',
    sizeOnDisk: 'Unknown',
    precision: 'Unknown',
    vae: '',
    textEncoder: '',
    unet: '',
    loaded: false,
  },
}

const FALLBACK_READINESS: ProReadinessStatus = {
  counts: EMPTY_READINESS_COUNTS,
  families: [],
  working: [],
  needsWork: [],
  metadataOnlyCount: 0,
  total: 0,
  error: '',
}

const FALLBACK_CAPABILITIES: ProCapabilitiesStatus = {
  gradioTabs: [
    { id: 'studio', label: 'Studio', group: 'Create', status: 'ready', count: 0, route: 'create', tab: 'Image', summary: 'Image generation and inpaint.', details: ['Existing image surface is available.'] },
    { id: 'video', label: 'Sana / Wan / LTX Video', group: 'Video', status: 'available', count: 0, route: 'create', tab: 'Video', summary: 'Video tool coverage is tracked.', details: ['React Pro can submit Sana Video.'] },
    { id: 'enhance', label: 'Enhance', group: 'Image', status: 'available', count: 0, route: 'tools', tab: 'Enhance', summary: 'Quick restore, upscale, and VSR image tools.', details: ['Full old-photo and batch workflows remain in Gradio.'] },
    { id: 'segment', label: 'Segment', group: 'Image', status: 'available', count: 0, route: 'modal:segmentation', tab: 'Segment', summary: 'SAM tool coverage is tracked.', details: ['React Pro has a quick popup.'] },
    { id: 'reactor', label: 'ReActor', group: 'Image', status: 'available', count: 0, route: 'modal:reactor', tab: 'ReActor', summary: 'Face swap coverage is tracked.', details: ['React Pro has a quick popup.'] },
  ],
  tools: [
    { id: 'image-generation', label: 'Image generation', group: 'Create', status: 'ready', count: 1, route: 'create', summary: 'React Pro can submit image jobs.', details: ['Fallback mode.'] },
  ],
  counts: {
    gradioTabs: 5,
    reactRails: 7,
    checkpoints: 1,
    blockedCheckpoints: 0,
    loras: 0,
    controlnet: 0,
    sam: 0,
    reactor: 0,
    enhance: 0,
    sanaVideo: 0,
    wan: 0,
  },
  readiness: FALLBACK_READINESS,
  notes: ['Waiting for /api/pro/capabilities.'],
}

export class ProApiError extends Error {
  readonly status: number
  readonly path: string
  readonly detail: unknown
  readonly clientMs?: number
  readonly serverMs?: number | null

  constructor(
    path: string,
    status: number,
    message: string,
    detail?: unknown,
    timing?: { clientMs?: number; serverMs?: number | null },
  ) {
    super(message)
    this.name = 'ProApiError'
    this.path = path
    this.status = status
    this.detail = detail
    this.clientMs = timing?.clientMs
    this.serverMs = timing?.serverMs
  }
}

export function getFallbackBootstrap(): ProBootstrap {
  return {
    workspaceName: 'AIWF Studio',
    subtitle: 'Second GUI',
    version: 'v0.2.0',
    localFirst: true,
    onboardingSeen: false,
    engines: buildEngineSummaries(DEFAULT_MODELS),
    models: DEFAULT_MODELS.map((model) => ({ ...model })),
    blockedModels: [],
    counts: {
      checkpoints: DEFAULT_MODELS.length,
      blockedCheckpoints: 0,
    },
    samplers: [...DEFAULT_SAMPLERS],
    aspectRatios: DEFAULT_ASPECT_RATIOS.map((ratio) => ({ ...ratio })),
    defaults: { ...DEFAULT_SETTINGS },
    recentOutputs: buildFallbackOutputs(),
  }
}

export function getFallbackRuntime(): ProRuntimeStatus {
  return {
    ...FALLBACK_RUNTIME,
    job: { ...FALLBACK_RUNTIME.job },
    resources: FALLBACK_RUNTIME.resources.map((metric) => ({ ...metric })),
    loadedModel: { ...FALLBACK_RUNTIME.loadedModel },
  }
}

export async function fetchProBootstrap(signal?: AbortSignal): Promise<ProBootstrap> {
  const payload = await requestJson('/api/pro/bootstrap', { signal })
  return normalizeBootstrap(payload)
}

export async function fetchProRuntime(signal?: AbortSignal): Promise<ProRuntimeStatus> {
  const payload = await requestJson('/api/pro/runtime', { signal })
  return normalizeRuntime(payload)
}

export function streamProRuntime(
  onRuntime: (runtime: ProRuntimeStatus) => void,
  onConnectionChange?: (connected: boolean) => void,
): () => void {
  if (typeof EventSource === 'undefined') {
    onConnectionChange?.(false)
    return () => undefined
  }

  const source = new EventSource(`${API_BASE}/api/pro/runtime/stream`)
  let lastEventData = ''
  source.addEventListener('runtime', (event) => {
    try {
      // Identical payloads (idle ticks) must not trigger a state update —
      // every setRuntime re-renders the whole shell.
      if (event.data === lastEventData) {
        onConnectionChange?.(true)
        return
      }
      lastEventData = event.data
      onRuntime(normalizeRuntime(JSON.parse(event.data) as unknown))
      onConnectionChange?.(true)
    } catch {
      onConnectionChange?.(false)
    }
  })
  source.addEventListener('error', () => {
    onConnectionChange?.(false)
  })

  return () => {
    source.close()
  }
}

export async function fetchProData(signal?: AbortSignal): Promise<ProDataStatus> {
  const payload = await requestJson('/api/pro/data', { signal })
  return normalizeDataStatus(payload)
}

export async function fetchProDownloads(signal?: AbortSignal): Promise<ProDownloadsStatus> {
  const payload = await requestJson('/api/pro/downloads', { signal })
  return normalizeDownloadsStatus(payload)
}

export async function downloadCatalogModel(key: string): Promise<ProDownloadsStatus> {
  const payload = await requestJson(`/api/pro/downloads/catalog/${encodeURIComponent(key)}`, {
    method: 'POST',
  })
  return normalizeDownloadsStatus(payload)
}

export async function fetchProLogs(signal?: AbortSignal): Promise<ProLogStatus> {
  const payload = await requestJson('/api/pro/logs', { signal })
  return normalizeLogStatus(payload)
}

export async function fetchProSettings(signal?: AbortSignal): Promise<ProSettingsStatus> {
  const payload = await requestJson('/api/pro/settings', { signal })
  return normalizeSettingsStatus(payload)
}

export async function saveProSettings(
  settings: GenerationSettings,
  ui: ProSettingsStatus['ui'] | null | undefined,
  output: ProSettingsStatus['output'] | null | undefined,
  video: ProSettingsStatus['video'] | null | undefined,
  runtime: ProSettingsStatus['runtime'] | null | undefined,
): Promise<ProSettingsStatus> {
  const payload = await requestJson('/api/pro/settings', {
    body: JSON.stringify({
      generationDefaults: {
        modelId: settings.modelId,
        negativePrompt: settings.negativePrompt,
        sampler: settings.sampler,
        scheduler: settings.scheduler,
        steps: settings.steps,
        cfgScale: settings.cfgScale,
        width: settings.width,
        height: settings.height,
        clipSkip: settings.clipSkip,
        saveImages: settings.saveImages,
      },
      ui: ui
        ? {
            galleryColumns: ui.galleryColumns,
            galleryHeight: ui.galleryHeight,
            livePreview: ui.livePreview,
            showProgressEveryNSteps: ui.showProgressEveryNSteps,
            livePreviewDecoder: ui.livePreviewDecoder,
            accentPreset: ui.accentPreset,
            hiddenTabs: ui.hiddenTabs,
          }
        : {},
      output: output
        ? {
            imageFormat: output.imageFormat,
            imageQuality: output.imageQuality,
            embedMetadata: output.embedMetadata,
            saveGrid: output.saveGrid,
            saveSidecarTxt: output.saveSidecarTxt,
            filenamePattern: output.filenamePattern,
            saveBeforeHires: output.saveBeforeHires,
            saveInterrupted: output.saveInterrupted,
            metadataIncludeModelHash: output.metadataIncludeModelHash,
            metadataIncludeVaeHash: output.metadataIncludeVaeHash,
            metadataIncludeLoraHashes: output.metadataIncludeLoraHashes,
            metadataIncludeAppVersion: output.metadataIncludeAppVersion,
            metadataIncludeOptimizationProfile: output.metadataIncludeOptimizationProfile,
            optimizationProfileId: output.optimizationProfileId,
          }
        : {},
      video: video
        ? {
            wanHigh: video.wanHigh,
            wanLow: video.wanLow,
            wanVae: video.wanVae,
            wanTextEncoder: video.wanTextEncoder,
            wanOffload: video.wanOffload,
            wanSampler: video.wanSampler,
            wanFlowShift: video.wanFlowShift,
            wanRuntimeMode: video.wanRuntimeMode,
            ltxDtype: video.ltxDtype,
            ltxCpuOffload: video.ltxCpuOffload,
            wanGroupOffloadStream: video.wanGroupOffloadStream,
            wanGroupOffloadBlocks: video.wanGroupOffloadBlocks,
            ggufCudaKernels: video.ggufCudaKernels,
            wanSageAttention: video.wanSageAttention,
            wanNativeDenoise: video.wanNativeDenoise,
            wanManualVaeDecode: video.wanManualVaeDecode,
            wanVaeChunkFrames: video.wanVaeChunkFrames,
            wanGroupOffloadRecordStream: video.wanGroupOffloadRecordStream,
            wanGroupOffloadLowCpuMem: video.wanGroupOffloadLowCpuMem,
            wanResidentMinVramGb: video.wanResidentMinVramGb,
          }
        : {},
      runtime: runtime
        ? {
            port: runtime.port,
            listen: runtime.listen,
            share: runtime.share,
            autolaunch: runtime.autolaunch,
            api: runtime.api,
            genlog: runtime.genlog,
            backend: runtime.backend,
            onnxProvider: runtime.onnxProvider,
            attention: runtime.attention,
            xformers: runtime.xformers,
            optSdpAttention: runtime.optSdpAttention,
            optSplitAttention: runtime.optSplitAttention,
            asyncOffload: runtime.asyncOffload,
            pinnedMemory: runtime.pinnedMemory,
            cudaMalloc: runtime.cudaMalloc,
            vramProfile: runtime.vramProfile,
            medvram: runtime.medvram,
            lowvram: runtime.lowvram,
            highvram: runtime.highvram,
            noHalf: runtime.noHalf,
            fp8: runtime.fp8,
            fluxFp8: runtime.fluxFp8,
            directml: runtime.directml,
            cpu: runtime.cpu,
            cudaGraphs: runtime.cudaGraphs,
            torchao: runtime.torchao,
            fp8Quant: runtime.fp8Quant,
            torchCompile: runtime.torchCompile,
            channelsLast: runtime.channelsLast,
            nvenc: runtime.nvenc,
            hevc: runtime.hevc,
            blockPrivateDownloadUrls: runtime.blockPrivateDownloadUrls,
            apiCorsOrigins: runtime.apiCorsOrigins,
            apiRateLimitPerMinute: runtime.apiRateLimitPerMinute,
            theme: runtime.theme,
            modelsDir: runtime.modelsDir,
            checkpointDir: runtime.checkpointDir,
            outputDir: runtime.outputDir,
            extraModelDirs: runtime.extraModelDirs,
            extraCheckpointDirs: runtime.extraCheckpointDirs,
          }
        : {},
    }),
    headers: { 'Content-Type': 'application/json' },
    method: 'POST',
  })
  return normalizeSettingsStatus(payload)
}

export async function fetchProCapabilities(signal?: AbortSignal): Promise<ProCapabilitiesStatus> {
  const payload = await requestJson('/api/pro/capabilities', { signal })
  return normalizeCapabilitiesStatus(payload)
}

export async function generateProOutput(
  request: ProGenerateRequest,
  signal?: AbortSignal,
): Promise<ProGenerateResult> {
  const payload = await requestJson('/api/pro/generate', {
    body: JSON.stringify(toGeneratePayload(request)),
    headers: { 'Content-Type': 'application/json' },
    method: 'POST',
    signal,
  })
  return normalizeGenerateResult(payload, request)
}

export interface VideoLabProbe {
  path: string
  url?: string
  width: number
  height: number
  fps: number
  frameCount: number
  durationSeconds: number
}

export interface VideoLabStatus {
  vsr: {
    available: boolean
    upscaleAvailable: boolean
    denoiseAvailable: boolean
    sdkRoot: string
    modelCount: number
    features: string[]
    help: string
  }
  rife: { available: boolean; checkpoints: string[] }
  audio: { videoAudioModels: string[] }
  extend: { available: boolean; note: string }
}

export interface VideoLabResult {
  status: string
  outputPath: string
  url: string
  message: string
  probe: VideoLabProbe
}

export async function fetchVideoLabStatus(signal?: AbortSignal): Promise<VideoLabStatus> {
  const payload = await requestJson('/api/pro/video-lab/status', { signal })
  const record = asRecord(payload)
  const vsr = readRecord(record, ['vsr'])
  const rife = readRecord(record, ['rife'])
  const audio = readRecord(record, ['audio'])
  const extend = readRecord(record, ['extend'])
  return {
    vsr: {
      available: readBoolean(vsr, ['available'], false),
      upscaleAvailable: readBoolean(vsr, ['upscaleAvailable'], false),
      denoiseAvailable: readBoolean(vsr, ['denoiseAvailable'], false),
      sdkRoot: readString(vsr, ['sdkRoot'], ''),
      modelCount: readNumber(vsr, ['modelCount'], 0),
      features: readArray(vsr, ['features']).map((item) => `${item}`),
      help: readString(vsr, ['help'], ''),
    },
    rife: {
      available: readBoolean(rife, ['available'], false),
      checkpoints: readArray(rife, ['checkpoints']).map((item) => `${item}`),
    },
    audio: {
      videoAudioModels: readArray(audio, ['videoAudioModels']).map((item) => `${item}`),
    },
    extend: {
      available: readBoolean(extend, ['available'], false),
      note: readString(extend, ['note'], ''),
    },
  }
}

function normalizeVideoLabProbe(value: unknown): VideoLabProbe {
  const record = asRecord(value)
  return {
    path: readString(record, ['path'], ''),
    url: readString(record, ['url'], ''),
    width: readNumber(record, ['width'], 0),
    height: readNumber(record, ['height'], 0),
    fps: readNumber(record, ['fps'], 0),
    frameCount: readNumber(record, ['frameCount', 'frame_count'], 0),
    durationSeconds: readNumber(record, ['durationSeconds', 'duration_seconds'], 0),
  }
}

function normalizeModelSortAction(value: unknown): ProModelSortAction {
  const record = asRecord(value)
  return {
    filename: readString(record, ['filename'], ''),
    source: readString(record, ['source'], ''),
    family: readString(record, ['family'], ''),
    architecture: readString(record, ['architecture'], ''),
    destSubdir: readString(record, ['destSubdir', 'dest_subdir'], ''),
    status: readString(record, ['status'], ''),
    reason: readString(record, ['reason'], ''),
  }
}

function normalizeModelSortResult(value: unknown): ProModelSortResult {
  const record = asRecord(value)
  const counts = readRecord(record, ['counts'])
  return {
    status: readString(record, ['status'], 'completed'),
    uploadedPath: readString(record, ['uploadedPath', 'uploaded_path'], ''),
    uploadedBytes: readNumber(record, ['uploadedBytes', 'uploaded_bytes'], 0),
    actions: readArray(record, ['actions']).map(normalizeModelSortAction),
    counts: {
      total: readNumber(counts, ['total'], 0),
      moved: readNumber(counts, ['moved'], 0),
      left: readNumber(counts, ['left'], 0),
      inventoryCount: readNumber(counts, ['inventoryCount', 'inventory_count'], 0),
    },
  }
}

export async function uploadVideoLabFile(file: File): Promise<VideoLabProbe> {
  const body = new FormData()
  body.append('file', file)
  const response = await fetch(`${API_BASE}/api/pro/video-lab/upload`, {
    method: 'POST',
    body,
    cache: 'no-store',
    headers: { Accept: 'application/json' },
  })
  const text = await response.text()
  if (!response.ok) {
    throw new ProApiError('/api/pro/video-lab/upload', response.status, formatResponseError(text, response.statusText))
  }
  return normalizeVideoLabProbe(text ? JSON.parse(text) : {})
}

export async function uploadModelFile(file: File): Promise<ProModelSortResult> {
  const body = new FormData()
  body.append('file', file)
  const response = await fetch(`${API_BASE}/api/pro/models/upload`, {
    method: 'POST',
    body,
    cache: 'no-store',
    headers: { Accept: 'application/json' },
  })
  const text = await response.text()
  if (!response.ok) {
    throw new ProApiError('/api/pro/models/upload', response.status, formatResponseError(text, response.statusText))
  }
  return normalizeModelSortResult(text ? JSON.parse(text) : {})
}

export async function reorganizeModels(): Promise<ProModelSortResult> {
  const response = await requestJson('/api/pro/models/reorganize', { method: 'POST' })
  return normalizeModelSortResult(response)
}

export async function runVideoLab(payload: Record<string, unknown>, signal?: AbortSignal): Promise<VideoLabResult> {
  const response = await requestJson('/api/pro/video-lab/run', {
    method: 'POST',
    body: JSON.stringify(payload),
    headers: { 'Content-Type': 'application/json' },
    signal,
  })
  const record = asRecord(response)
  return {
    status: readString(record, ['status'], 'completed'),
    outputPath: readString(record, ['outputPath', 'output_path'], ''),
    url: readString(record, ['url'], ''),
    message: readString(record, ['message'], ''),
    probe: normalizeVideoLabProbe(readUnknown(record, ['probe'])),
  }
}

export async function generateAutoMask(
  imageDataUrl: string,
  prompt: string,
  boxThreshold = 0.3,
): Promise<{ mask: string; preview: string; status: string }> {
  const payload = await requestJson('/api/pro/segment/auto-mask', {
    method: 'POST',
    body: JSON.stringify({ imageDataUrl, prompt, boxThreshold }),
    headers: { 'Content-Type': 'application/json' },
  })
  const record = asRecord(payload)
  return {
    mask: readString(record, ['mask'], ''),
    preview: readString(record, ['preview'], ''),
    status: readString(record, ['status'], ''),
  }
}

export async function runFaceSwap(
  targetImageDataUrl: string,
  sourceImageDataUrl: string,
): Promise<{ image: string; width: number; height: number; message: string }> {
  const payload = await requestJson('/api/pro/faceswap', {
    method: 'POST',
    body: JSON.stringify({ targetImageDataUrl, sourceImageDataUrl }),
    headers: { 'Content-Type': 'application/json' },
  })
  const record = asRecord(payload)
  return {
    image: readString(record, ['image'], ''),
    width: readNumber(record, ['width'], 0),
    height: readNumber(record, ['height'], 0),
    message: readString(record, ['message'], ''),
  }
}

export interface ProEnhanceImageRequest {
  imageDataUrl: string
  restoreEnabled: boolean
  restoreModel: string
  restoreVisibility: number
  codeformerWeight: number
  upscaleEnabled: boolean
  upscaleModel: string
  upscaleScale: number
  tileSize: number
  tileOverlap: number
  restoreFirst: boolean
}

export interface ProImageProcessResult {
  status: string
  image: string
  url: string
  outputPath: string
  width: number
  height: number
  message: string
  infotext: string
}

function normalizeImageProcessResult(payload: unknown): ProImageProcessResult {
  const record = asRecord(payload)
  return {
    status: readString(record, ['status'], 'completed'),
    image: readString(record, ['image'], ''),
    url: readString(record, ['url'], ''),
    outputPath: readString(record, ['outputPath', 'output_path'], ''),
    width: readNumber(record, ['width'], 0),
    height: readNumber(record, ['height'], 0),
    message: readString(record, ['message'], ''),
    infotext: readString(record, ['infotext'], ''),
  }
}

export async function runEnhanceImage(request: ProEnhanceImageRequest): Promise<ProImageProcessResult> {
  const payload = await requestJson('/api/pro/enhance/image', {
    method: 'POST',
    body: JSON.stringify(request),
    headers: { 'Content-Type': 'application/json' },
  })
  return normalizeImageProcessResult(payload)
}

export async function runVsrImage(request: {
  imageDataUrl: string
  scale: number
  mode: number
  effect: string
  strength: number
}): Promise<ProImageProcessResult> {
  const payload = await requestJson('/api/pro/vsr/image', {
    method: 'POST',
    body: JSON.stringify(request),
    headers: { 'Content-Type': 'application/json' },
  })
  return normalizeImageProcessResult(payload)
}

export interface ProExtensionInfo {
  id: string
  name: string
  version: string
  description: string
  path: string
  enabled: boolean
  error: string | null
  hasApi: boolean
  apiBase: string
}

export interface ProExtensionsStatus {
  pluginsDir: string
  disabled: string[]
  extensions: ProExtensionInfo[]
}

export async function fetchProExtensions(signal?: AbortSignal): Promise<ProExtensionsStatus> {
  const payload = await requestJson('/api/pro/extensions', { signal })
  const record = asRecord(payload)
  return {
    pluginsDir: readString(record, ['pluginsDir', 'plugins_dir'], ''),
    disabled: readArray(record, ['disabled']).map((item) => `${item}`),
    extensions: readArray(record, ['extensions']).map((item) => {
      const ext = asRecord(item)
      return {
        id: readString(ext, ['id'], ''),
        name: readString(ext, ['name'], ''),
        version: readString(ext, ['version'], '0.0.0'),
        description: readString(ext, ['description'], ''),
        path: readString(ext, ['path'], ''),
        enabled: readBoolean(ext, ['enabled'], true),
        error: readOptionalString(ext, ['error']) ?? null,
        hasApi: readBoolean(ext, ['hasApi', 'has_api'], false),
        apiBase: readString(ext, ['apiBase', 'api_base'], ''),
      }
    }),
  }
}

export async function toggleProExtension(id: string, enabled: boolean): Promise<{ note: string; disabled: string[] }> {
  const payload = await requestJson('/api/pro/extensions/toggle', {
    method: 'POST',
    body: JSON.stringify({ id, enabled }),
    headers: { 'Content-Type': 'application/json' },
  })
  const record = asRecord(payload)
  return {
    note: readString(record, ['note'], ''),
    disabled: readArray(record, ['disabled']).map((item) => `${item}`),
  }
}

export async function stopProGeneration(): Promise<ProStopResult> {
  const payload = await requestJson('/api/pro/interrupt', {
    method: 'POST',
  })
  const record = asRecord(payload)
  return {
    status: readString(record, ['status'], 'interrupt_requested'),
    videoJobId: readString(record, ['videoJobId', 'video_job_id'], ''),
  }
}

export interface ProClientEventPayload {
  action: string
  detail?: string
  context?: Record<string, unknown>
}

export interface ProClientErrorPayload {
  message: string
  stack?: string
  source?: string
  kind?: string
  context?: Record<string, unknown>
}

export function reportProClientEvent(payload: ProClientEventPayload): void {
  void postClientLog('/api/v1/client-events', payload)
}

export function reportProClientError(payload: ProClientErrorPayload): void {
  void postClientLog('/api/v1/client-errors', {
    ...payload,
    kind: payload.kind ?? 'error',
  })
}

function normalizeStartupStatus(value: unknown): ProStartupStatus {
  const record = asRecord(value)
  return {
    status: readString(record, ['status'], 'server-ready'),
    serverReady: readBoolean(record, ['serverReady', 'server_ready'], true),
    windowReady: readBoolean(record, ['windowReady', 'window_ready'], false),
    startedAt: readString(record, ['startedAt', 'started_at'], ''),
    serverReadyAt: readString(record, ['serverReadyAt', 'server_ready_at'], ''),
    windowReadyAt: readString(record, ['windowReadyAt', 'window_ready_at'], ''),
    minSplashMs: readNumber(record, ['minSplashMs', 'min_splash_ms'], 1800),
    readyHoldMs: readNumber(record, ['readyHoldMs', 'ready_hold_ms'], 1200),
  }
}

export async function fetchProStartup(signal?: AbortSignal): Promise<ProStartupStatus> {
  const payload = await requestJson('/api/pro/startup', { signal })
  return normalizeStartupStatus(payload)
}

export async function notifyProWindowReady(): Promise<ProStartupStatus> {
  const payload = await requestJson('/api/pro/startup/window-ready', {
    method: 'POST',
  })
  return normalizeStartupStatus(payload)
}

export async function requestProRestart(): Promise<{ status: string }> {
  const payload = await requestJson('/api/pro/restart', {
    method: 'POST',
  })
  const record = asRecord(payload)
  return {
    status: readString(record, ['status'], 'restart_requested'),
  }
}

export async function unloadProModel(): Promise<ProRuntimeStatus> {
  const payload = await requestJson('/api/pro/models/unload', {
    method: 'POST',
  })
  const record = asRecord(payload)
  return normalizeRuntime(readUnknown(record, ['runtime']))
}

export async function openSupportTerminal(): Promise<{ status: string; cwd: string; venv: string }> {
  const payload = await requestJson('/api/pro/support/terminal', {
    method: 'POST',
  })
  const record = asRecord(payload)
  return {
    status: readString(record, ['status'], 'opened'),
    cwd: readString(record, ['cwd'], ''),
    venv: readString(record, ['venv'], ''),
  }
}

export function formatApiError(error: unknown): string {
  if (error instanceof DOMException && error.name === 'AbortError') {
    return 'Request was cancelled.'
  }
  if (error instanceof ProApiError) {
    return `${error.path} returned ${error.status}: ${error.message}`
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'Unknown API error.'
}

function formatResponseError(text: string, fallback: string): string {
  return readResponseError(text, fallback).message
}

function readResponseError(text: string, fallback: string): { message: string; detail?: unknown } {
  if (!text.trim()) {
    return { message: fallback }
  }
  try {
    const parsed = JSON.parse(text) as unknown
    const record = asRecord(parsed)
    const detail = readUnknown(record, ['detail', 'message', 'error'])
    if (typeof detail === 'string' && detail.trim()) {
      return { message: detail, detail }
    }
    if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
      const detailRecord = asRecord(detail)
      const message = readString(detailRecord, ['message', 'detail', 'error'], fallback)
      const receipt = readString(detailRecord, ['receiptPath', 'receipt_path', 'failureLogPath', 'failure_log_path'], '')
      return { message: [message, receipt ? `Receipt: ${receipt}` : ''].filter(Boolean).join(' '), detail }
    }
    if (Array.isArray(detail) && detail.length > 0) {
      return {
        message: detail
        .map((item) => {
          const itemRecord = asRecord(item)
          return readString(itemRecord, ['msg', 'message', 'detail'], JSON.stringify(item))
        })
        .filter(Boolean)
        .join('; '),
        detail,
      }
    }
  } catch {
    // Fall through to the raw response body.
  }
  return { message: text }
}

async function requestJson(path: string, init: RequestInit = {}): Promise<unknown> {
  const startedAt = typeof performance !== 'undefined' ? performance.now() : Date.now()
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    cache: init.cache ?? 'no-store',
    headers: {
      Accept: 'application/json',
      ...init.headers,
    },
  })
  const clientMs = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - startedAt
  const serverHeader = response.headers.get('X-AIWF-Elapsed-Ms')
  const parsedServerMs = serverHeader === null ? NaN : Number.parseFloat(serverHeader)
  const serverMs = Number.isFinite(parsedServerMs) ? parsedServerMs : null
  recordApiLatency({
    path,
    status: response.status,
    clientMs,
    serverMs,
    createdAt: new Date().toISOString(),
  })

  const text = await response.text()
  if (!response.ok) {
    const error = readResponseError(text, response.statusText)
    throw new ProApiError(path, response.status, error.message, error.detail, { clientMs, serverMs })
  }
  if (!text) {
    return {}
  }
  try {
    return JSON.parse(text) as unknown
  } catch {
    throw new ProApiError(path, response.status, 'Response was not valid JSON.', undefined, { clientMs, serverMs })
  }
}

async function postClientLog(path: string, payload: object): Promise<void> {
  try {
    await fetch(`${API_BASE}${path}`, {
      body: JSON.stringify({
        ...payload,
        url: typeof window !== 'undefined' ? window.location.href : undefined,
        user_agent: typeof navigator !== 'undefined' ? navigator.userAgent : undefined,
      }),
      cache: 'no-store',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      keepalive: true,
      method: 'POST',
    })
  } catch {
    // Client logging is diagnostic only and must not create a second user-facing failure.
  }
}

function toGeneratePayload(request: ProGenerateRequest): JsonRecord {
  return {
    mode: request.mode,
    prompt: request.prompt,
    negative_prompt: request.negativePrompt,
    checkpoint_id: request.modelId,
    width: request.width,
    height: request.height,
    steps: request.steps,
    cfg_scale: request.cfgScale,
    sampler: request.sampler,
    scheduler: request.scheduler,
    seed: request.seed,
    clip_skip: request.clipSkip,
    batch_size: request.batchSize,
    batch_count: request.batchCount,
    enable_hr: request.enableHires,
    hr_scale: request.hiresScale,
    hr_steps: request.hiresSteps,
    hr_denoising_strength: request.hiresDenoise,
    hr_upscaler: request.hiresUpscaler,
    frames: request.frames,
    fps: request.fps,
    source_image_data_url: request.sourceImageDataUrl,
    source_image_name: request.sourceImageName,
    sana_quantization: request.sanaQuantization,
    sana_vae_tiling: request.sanaVaeTiling,
    offload_text_encoder_after_encode: request.offloadTextEncoderAfterEncode,
    use_sage_attention: request.useSageAttention,
    generate_audio: request.generateAudio,
    wan_runtime_mode: request.wanRuntimeMode,
    high_noise_model_id: request.highNoiseModelId || undefined,
    low_noise_model_id: request.lowNoiseModelId || undefined,
    high_noise_steps: request.highNoiseSteps,
    low_noise_steps: request.lowNoiseSteps,
    boundary_ratio: request.boundaryRatio,
    high_noise_lora_id: request.highNoiseLoraId || undefined,
    high_noise_lora_scale: request.highNoiseLoraScale,
    low_noise_lora_id: request.lowNoiseLoraId || undefined,
    low_noise_lora_scale: request.lowNoiseLoraScale,
    vae_id: request.vaeId || undefined,
    text_encoder_path: request.textEncoderPath || undefined,
    wan_offload: request.wanOffload,
    wan_sigma_type: request.wanSigmaType,
    wan_sampler: request.wanSampler,
    wan_flow_shift: request.wanFlowShift,
    init_image_data_url: request.initImageDataUrl || undefined,
    mask_image_data_url: request.maskImageDataUrl || undefined,
    denoising_strength: request.denoisingStrength,
    mask_blur: request.maskBlur,
    inpaint_only_masked: request.inpaintOnlyMasked,
    inpaint_masked_padding: request.inpaintMaskedPadding,
    inpaint_mask_content: request.inpaintMaskContent,
    controlnet_units: request.controlNetEnabled
      ? [
          {
            enabled: true,
            model: request.controlNetModel,
            module: request.controlNetModule || 'none',
            image: request.controlNetImageDataUrl,
            weight: request.controlNetWeight,
            guidance_start: request.controlNetGuidanceStart,
            guidance_end: request.controlNetGuidanceEnd,
            processor_res: request.controlNetProcessorRes,
            resize_mode: 'resize',
            control_mode: 'balanced',
          },
        ]
      : [],
  }
}

function normalizeBootstrap(value: unknown): ProBootstrap {
  const record = asRecord(value)
  const fallback = getFallbackBootstrap()
  const defaultsRecord = readRecord(record, ['defaults', 'settings'])
  const countsRecord = readRecord(record, ['counts'])
  const aspectRatios = readArray(record, ['aspect_ratios', 'aspectRatios', 'ratios'])
    .map(normalizeAspectRatio)
    .filter(isPresent)
  const models = readArray(record, ['models', 'checkpoints'])
    .map(normalizeModel)
    .filter(isPresent)
  const blockedModels = readArray(record, ['blockedModels', 'blocked_checkpoints', 'blockedCheckpoints'])
    .map(normalizeModel)
    .filter(isPresent)
  const engines = readArray(record, ['engines'])
    .map(normalizeEngineSummary)
    .filter(isPresent)
  const samplers = readArray(record, ['samplers'])
    .map(normalizeSamplerOption)
    .filter((item) => item.length > 0)
  const recentOutputs = readArray(record, ['recent_outputs', 'recentOutputs', 'outputs', 'recentImages'])
    .map((item, index) => normalizeRecentOutput(item, index, fallback.defaults))
    .filter(isPresent)

  const ratios = aspectRatios.length > 0 ? aspectRatios : fallback.aspectRatios
  const modelOptions = models.length > 0 ? models : fallback.models
  const samplerOptions = samplers.length > 0 ? samplers : fallback.samplers
  const defaults = normalizeSettings(defaultsRecord, fallback.defaults, ratios, modelOptions, samplerOptions)

  return {
    workspaceName: readString(record, ['workspace_name', 'workspaceName', 'name'], fallback.workspaceName),
    subtitle: readString(record, ['subtitle', 'edition'], fallback.subtitle),
    version: readString(record, ['version'], fallback.version),
    localFirst: readBoolean(record, ['local_first', 'localFirst'], fallback.localFirst),
    onboardingSeen: readBoolean(record, ['onboarding_seen', 'onboardingSeen'], fallback.onboardingSeen),
    engines: engines.length > 0 ? engines : buildEngineSummaries(modelOptions),
    models: modelOptions,
    blockedModels,
    counts: {
      checkpoints: readNumber(countsRecord, ['checkpoints'], modelOptions.length),
      blockedCheckpoints: readNumber(countsRecord, ['blockedCheckpoints', 'blocked_checkpoints'], blockedModels.length),
    },
    samplers: samplerOptions,
    aspectRatios: ratios,
    defaults,
    recentOutputs: recentOutputs.length > 0 ? recentOutputs : fallback.recentOutputs,
  }
}

function normalizeDataStatus(value: unknown): ProDataStatus {
  const record = asRecord(value)
  const fallback = getFallbackBootstrap()
  const counts = readRecord(record, ['counts'])
  const recentOutputs = readArray(record, ['recent_outputs', 'recentOutputs', 'outputs'])
    .map((item, index) => normalizeRecentOutput(item, index, fallback.defaults))
    .filter(isPresent)
  const engines = readArray(record, ['engines'])
    .map(normalizeEngineSummary)
    .filter(isPresent)
  return {
    outputRoot: readString(record, ['outputRoot', 'output_root'], ''),
    counts: {
      checkpoints: readNumber(counts, ['checkpoints'], fallback.models.length),
      blockedCheckpoints: readNumber(counts, ['blockedCheckpoints', 'blocked_checkpoints'], fallback.blockedModels.length),
      recentOutputs: readNumber(counts, ['recentOutputs', 'recent_outputs'], recentOutputs.length),
      engines: readNumber(counts, ['engines'], engines.length),
    },
    engines,
    recentOutputs,
  }
}

function normalizeDownloadsStatus(value: unknown): ProDownloadsStatus {
  const record = asRecord(value)
  const counts = readRecord(record, ['counts'])
  const bundlesRecord = asRecord(readUnknown(record, ['bundles']))
  const bundles = Object.fromEntries(
    Object.entries(bundlesRecord).map(([key, bundleValue]) => [
      key,
      Array.isArray(bundleValue)
        ? bundleValue.map((item) => readLooseString(item, '')).filter(Boolean)
        : [],
    ]),
  )
  const catalog = readArray(record, ['catalog', 'items'])
    .map(normalizeDownloadCatalogItem)
    .filter(isPresent)
  const categories = readArray(record, ['categories'])
    .map(normalizeDownloadCategory)
    .filter(isPresent)
  const civitaiLinks = readArray(record, ['civitaiLinks', 'civitai_links'])
    .map((item) => {
      const link = asRecord(item)
      const url = readString(link, ['url'], '')
      if (!url) {
        return null
      }
      return {
        label: readString(link, ['label'], url),
        url,
        note: readString(link, ['note'], ''),
        engine: readString(link, ['engine'], 'all'),
      }
    })
    .filter(isPresent)

  return {
    categories,
    bundles,
    catalog,
    civitaiLinks,
    counts: {
      categories: readNumber(counts, ['categories'], categories.length),
      catalog: readNumber(counts, ['catalog', 'items'], catalog.length),
      installed: readNumber(
        counts,
        ['installed'],
        catalog.filter((item) => item.installed).length,
      ),
    },
  }
}

function normalizeLogStatus(value: unknown): ProLogStatus {
  const record = asRecord(value)
  return {
    runtime: normalizeRuntime(readUnknown(record, ['runtime'])),
    files: readArray(record, ['files']).map(normalizeLogFile).filter(isPresent),
    events: readArray(record, ['events']).map(normalizeLogEvent).filter(isPresent),
  }
}

function normalizeSettingsStatus(value: unknown): ProSettingsStatus {
  const record = asRecord(value)
  const fallback = getFallbackBootstrap()
  const paths = readRecord(record, ['paths'])
  const ui = readRecord(record, ['ui'])
  const output = readRecord(record, ['output'])
  const video = readRecord(record, ['video'])
  const runtime = readRecord(record, ['runtime'])
  const legacyProfile = readBoolean(runtime, ['cpu'], false)
    ? 'cpu'
    : readBoolean(runtime, ['lowvram'], false)
      ? 'low'
      : readBoolean(runtime, ['medvram'], false)
        ? 'mid'
        : readBoolean(runtime, ['highvram'], false)
          ? 'high'
          : 'normal'
  const vramProfile = readString(runtime, ['vramProfile', 'vram_profile'], legacyProfile)

  return {
    paths: {
      settings: readString(paths, ['settings'], ''),
      launch: readString(paths, ['launch'], ''),
      models: readString(paths, ['models'], ''),
      checkpoints: readString(paths, ['checkpoints'], ''),
      outputs: readString(paths, ['outputs'], ''),
    },
    generationDefaults: normalizeSettings(readRecord(record, ['generationDefaults', 'generation_defaults']), fallback.defaults, fallback.aspectRatios, fallback.models, fallback.samplers),
    ui: {
      accentPreset: readString(ui, ['accentPreset', 'accent_preset'], 'mint'),
      galleryColumns: readNumber(ui, ['galleryColumns', 'gallery_columns'], 2),
      galleryHeight: readNumber(ui, ['galleryHeight', 'gallery_height'], 480),
      livePreview: readBoolean(ui, ['livePreview', 'live_preview'], true),
      showProgressEveryNSteps: readNumber(ui, ['showProgressEveryNSteps', 'show_progress_every_n_steps'], 5),
      livePreviewDecoder: readString(ui, ['livePreviewDecoder', 'live_preview_decoder'], 'vae'),
      hiddenTabs: readArray(ui, ['hiddenTabs', 'hidden_tabs']).map((item) => readLooseString(item, '')).filter(Boolean),
    },
    output: {
      imageFormat: readString(output, ['imageFormat', 'image_format'], 'png'),
      imageQuality: readNumber(output, ['imageQuality', 'image_quality'], 95),
      embedMetadata: readBoolean(output, ['embedMetadata', 'embed_metadata'], true),
      saveGrid: readBoolean(output, ['saveGrid', 'save_grid'], false),
      saveSidecarTxt: readBoolean(output, ['saveSidecarTxt', 'save_sidecar_txt'], false),
      filenamePattern: readString(output, ['filenamePattern', 'filename_pattern'], '[datetime]'),
      saveBeforeHires: readBoolean(output, ['saveBeforeHires', 'save_before_hires'], false),
      saveInterrupted: readBoolean(output, ['saveInterrupted', 'save_interrupted'], false),
      metadataIncludeModelHash: readBoolean(output, ['metadataIncludeModelHash', 'metadata_include_model_hash'], true),
      metadataIncludeVaeHash: readBoolean(output, ['metadataIncludeVaeHash', 'metadata_include_vae_hash'], true),
      metadataIncludeLoraHashes: readBoolean(output, ['metadataIncludeLoraHashes', 'metadata_include_lora_hashes'], true),
      metadataIncludeAppVersion: readBoolean(output, ['metadataIncludeAppVersion', 'metadata_include_app_version'], true),
      metadataIncludeOptimizationProfile: readBoolean(output, ['metadataIncludeOptimizationProfile', 'metadata_include_optimization_profile'], true),
      optimizationProfileId: readString(output, ['optimizationProfileId', 'optimization_profile_id'], 'balanced_sdpa_fp16'),
    },
    video: {
      wanHigh: readString(video, ['wanHigh', 'wan_high'], ''),
      wanLow: readString(video, ['wanLow', 'wan_low'], ''),
      wanVae: readString(video, ['wanVae', 'wan_vae'], ''),
      wanTextEncoder: readString(video, ['wanTextEncoder', 'wan_text_encoder'], ''),
      wanOffload: readString(video, ['wanOffload', 'wan_offload'], 'balanced'),
      wanSampler: readString(video, ['wanSampler', 'wan_sampler'], 'unipc'),
      wanFlowShift: readNumber(video, ['wanFlowShift', 'wan_flow_shift'], 5),
      wanRuntimeMode: readString(video, ['wanRuntimeMode', 'wan_runtime_mode'], 'fast_5b'),
      ltxDtype: readString(video, ['ltxDtype', 'ltx_dtype'], 'bf16'),
      ltxCpuOffload: readString(video, ['ltxCpuOffload', 'ltx_cpu_offload'], 'auto'),
      wanGroupOffloadStream: readBoolean(video, ['wanGroupOffloadStream', 'wan_group_offload_stream'], true),
      wanGroupOffloadBlocks: readNumber(video, ['wanGroupOffloadBlocks', 'wan_group_offload_blocks'], 4),
      ggufCudaKernels: readBoolean(video, ['ggufCudaKernels', 'gguf_cuda_kernels'], false),
      wanSageAttention: readString(video, ['wanSageAttention', 'wan_sage_attention'], 'auto'),
      wanNativeDenoise: readBoolean(video, ['wanNativeDenoise', 'wan_native_denoise'], true),
      wanManualVaeDecode: readBoolean(video, ['wanManualVaeDecode', 'wan_manual_vae_decode'], false),
      wanVaeChunkFrames: readNumber(video, ['wanVaeChunkFrames', 'wan_vae_chunk_frames'], 4),
      wanGroupOffloadRecordStream: readBoolean(video, ['wanGroupOffloadRecordStream', 'wan_group_offload_record_stream'], true),
      wanGroupOffloadLowCpuMem: readBoolean(video, ['wanGroupOffloadLowCpuMem', 'wan_group_offload_low_cpu_mem'], true),
      wanResidentMinVramGb: readNumber(video, ['wanResidentMinVramGb', 'wan_resident_min_vram_gb'], 20),
    },
    runtime: {
      port: readNumber(runtime, ['port'], 7860),
      listen: readBoolean(runtime, ['listen'], false),
      share: readBoolean(runtime, ['share'], false),
      autolaunch: readBoolean(runtime, ['autolaunch'], false),
      api: readBoolean(runtime, ['api'], false),
      genlog: readBoolean(runtime, ['genlog'], false),
      backend: readString(runtime, ['backend'], 'unknown'),
      onnxProvider: readString(runtime, ['onnxProvider', 'onnx_provider'], 'auto'),
      attention: readString(runtime, ['attention'], 'unknown'),
      xformers: readBoolean(runtime, ['xformers'], false),
      optSdpAttention: readBoolean(runtime, ['optSdpAttention', 'opt_sdp_attention'], false),
      optSplitAttention: readBoolean(runtime, ['optSplitAttention', 'opt_split_attention'], false),
      asyncOffload: readBoolean(runtime, ['asyncOffload', 'async_offload'], true),
      pinnedMemory: readBoolean(runtime, ['pinnedMemory', 'pinned_memory'], true),
      cudaMalloc: readBoolean(runtime, ['cudaMalloc', 'cuda_malloc'], false),
      vramProfile,
      medvram: readBoolean(runtime, ['medvram'], false),
      lowvram: readBoolean(runtime, ['lowvram'], false),
      highvram: readBoolean(runtime, ['highvram'], false),
      noHalf: readBoolean(runtime, ['noHalf', 'no_half'], false),
      fp8: readBoolean(runtime, ['fp8'], false),
      fluxFp8: readBoolean(runtime, ['fluxFp8', 'flux_fp8', 'fluxfp8'], false),
      directml: readBoolean(runtime, ['directml'], false),
      cpu: readBoolean(runtime, ['cpu'], false),
      cudaGraphs: readBoolean(runtime, ['cudaGraphs', 'cuda_graphs'], false),
      torchao: readBoolean(runtime, ['torchao'], false),
      fp8Quant: readBoolean(runtime, ['fp8Quant', 'fp8_quant'], false),
      torchCompile: readBoolean(runtime, ['torchCompile', 'torch_compile'], false),
      channelsLast: readBoolean(runtime, ['channelsLast', 'channels_last'], false),
      nvenc: readBoolean(runtime, ['nvenc'], false),
      hevc: readBoolean(runtime, ['hevc'], false),
      blockPrivateDownloadUrls: readBoolean(runtime, ['blockPrivateDownloadUrls', 'block_private_download_urls'], true),
      apiCorsOrigins: readString(runtime, ['apiCorsOrigins', 'api_cors_origins'], ''),
      apiRateLimitPerMinute: readNumber(runtime, ['apiRateLimitPerMinute', 'api_rate_limit_per_minute'], 0),
      theme: readString(runtime, ['theme'], 'dark'),
      modelsDir: readString(runtime, ['modelsDir', 'models_dir'], ''),
      checkpointDir: readString(runtime, ['checkpointDir', 'checkpoint_dir', 'ckpt_dir'], ''),
      outputDir: readString(runtime, ['outputDir', 'output_dir'], ''),
      extraModelDirs: readString(runtime, ['extraModelDirs', 'extra_model_dirs'], ''),
      extraCheckpointDirs: readString(runtime, ['extraCheckpointDirs', 'extra_checkpoint_dirs', 'extra_ckpt_dirs'], ''),
    },
  }
}

function normalizeCapabilitiesStatus(value: unknown): ProCapabilitiesStatus {
  const record = asRecord(value)
  const fallback = FALLBACK_CAPABILITIES
  const counts = readRecord(record, ['counts'])
  const tools = readArray(record, ['tools']).map(normalizeCapabilityItem).filter(isPresent)
  const gradioTabs = readArray(record, ['gradioTabs', 'gradio_tabs']).map(normalizeCapabilityItem).filter(isPresent)
  return {
    gradioTabs: gradioTabs.length > 0 ? gradioTabs : fallback.gradioTabs,
    tools: tools.length > 0 ? tools : fallback.tools,
    counts: {
      gradioTabs: readNumber(counts, ['gradioTabs', 'gradio_tabs'], fallback.counts.gradioTabs),
      reactRails: readNumber(counts, ['reactRails', 'react_rails'], fallback.counts.reactRails),
      checkpoints: readNumber(counts, ['checkpoints'], fallback.counts.checkpoints),
      blockedCheckpoints: readNumber(counts, ['blockedCheckpoints', 'blocked_checkpoints'], fallback.counts.blockedCheckpoints),
      loras: readNumber(counts, ['loras'], fallback.counts.loras),
      controlnet: readNumber(counts, ['controlnet'], fallback.counts.controlnet),
      sam: readNumber(counts, ['sam'], fallback.counts.sam),
      reactor: readNumber(counts, ['reactor'], fallback.counts.reactor),
      enhance: readNumber(counts, ['enhance'], fallback.counts.enhance),
      sanaVideo: readNumber(counts, ['sanaVideo', 'sana_video'], fallback.counts.sanaVideo),
      wan: readNumber(counts, ['wan'], fallback.counts.wan),
    },
    readiness: normalizeReadinessStatus(readUnknown(record, ['readiness']), fallback.readiness),
    notes: readArray(record, ['notes']).map((item) => readLooseString(item, '')).filter(Boolean),
  }
}

function normalizeReadinessStatus(value: unknown, fallback: ProReadinessStatus): ProReadinessStatus {
  const record = asRecord(value)
  const counts = normalizeReadinessCounts(readRecord(record, ['counts']), fallback.counts)
  const families = readArray(record, ['families']).map(normalizeReadinessFamily).filter(isPresent)
  const working = readArray(record, ['working']).map(normalizeReadinessItem).filter(isPresent)
  const needsWork = readArray(record, ['needsWork', 'needs_work']).map(normalizeReadinessItem).filter(isPresent)
  return {
    counts,
    families,
    working,
    needsWork,
    metadataOnlyCount: readNumber(
      record,
      ['metadataOnlyCount', 'metadata_only_count'],
      counts['metadata-only'] ?? fallback.metadataOnlyCount,
    ),
    total: readNumber(record, ['total'], sumReadinessCounts(counts)),
    error: readString(record, ['error'], fallback.error),
  }
}

function normalizeReadinessCounts(record: JsonRecord, fallback: ProReadinessCounts): ProReadinessCounts {
  const counts: ProReadinessCounts = { ...fallback }
  for (const key of READINESS_STATUS_KEYS) {
    counts[key] = readNumber(record, [key], fallback[key] ?? 0)
  }
  for (const [key, value] of Object.entries(record)) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      counts[key] = value
    } else if (typeof value === 'string') {
      const parsed = Number(value)
      if (Number.isFinite(parsed)) {
        counts[key] = parsed
      }
    }
  }
  return counts
}

function normalizeReadinessFamily(value: unknown): ProReadinessFamily | null {
  const record = asRecord(value)
  const family = readString(record, ['family', 'id', 'label'], '')
  if (!family) {
    return null
  }
  const counts = normalizeReadinessCounts(readRecord(record, ['counts']), EMPTY_READINESS_COUNTS)
  return {
    family,
    counts,
    total: readNumber(record, ['total'], sumReadinessCounts(counts)),
  }
}

function normalizeReadinessItem(value: unknown): ProReadinessItem | null {
  const record = asRecord(value)
  const id = readString(record, ['id'], '')
  const label = readString(record, ['label', 'name'], id)
  if (!id && !label) {
    return null
  }
  return {
    id: id || label,
    family: readString(record, ['family'], 'unknown'),
    assetType: readString(record, ['assetType', 'asset_type'], ''),
    path: readString(record, ['path'], ''),
    label,
    status: readString(record, ['status'], 'metadata-only'),
    route: readString(record, ['route'], ''),
    reason: readString(record, ['reason'], ''),
    storage: readString(record, ['storage'], ''),
    quantization: readString(record, ['quantization'], ''),
    requiredVae: readString(record, ['requiredVae', 'required_vae'], ''),
    requiredTextEncoder: readString(record, ['requiredTextEncoder', 'required_text_encoder'], ''),
    tokenizer: readString(record, ['tokenizer'], ''),
    smokeCommand: readString(record, ['smokeCommand', 'smoke_command'], ''),
    receiptPath: readString(record, ['receiptPath', 'receipt_path'], ''),
    suggestedAction: readString(record, ['suggestedAction', 'suggested_action'], ''),
  }
}

function sumReadinessCounts(counts: ProReadinessCounts): number {
  return READINESS_STATUS_KEYS.reduce((total, key) => total + (counts[key] ?? 0), 0)
}

function normalizeCapabilityItem(value: unknown): ProCapabilityItem | null {
  const record = asRecord(value)
  const id = readString(record, ['id'], '')
  const label = readString(record, ['label', 'name'], id)
  if (!id && !label) {
    return null
  }
  return {
    id: id || label,
    label,
    group: readString(record, ['group'], 'Tools'),
    status: readString(record, ['status'], 'available'),
    count: readNumber(record, ['count'], 0),
    route: readString(record, ['route'], 'tools'),
    tab: readOptionalString(record, ['tab']),
    summary: readString(record, ['summary', 'description'], ''),
    details: readArray(record, ['details']).map((item) => readLooseString(item, '')).filter(Boolean),
  }
}

function normalizeDownloadCategory(value: unknown) {
  const record = asRecord(value)
  const key = readString(record, ['key', 'id', 'value'], '')
  const label = readString(record, ['label', 'name'], key)
  if (!key && !label) {
    return null
  }
  return {
    key: key || label,
    label,
    destination: readString(record, ['destination', 'folder', 'path'], ''),
  }
}

function normalizeDownloadCatalogItem(value: unknown) {
  const record = asRecord(value)
  const key = readString(record, ['key', 'id'], '')
  const title = readString(record, ['title', 'label', 'name'], key)
  if (!key && !title) {
    return null
  }
  return {
    key: key || title,
    title,
    category: readString(record, ['category'], 'other'),
    source: readString(record, ['source'], 'unknown'),
    sizeMb: readOptionalNumber(record, ['sizeMb', 'size_mb']),
    repoId: readOptionalString(record, ['repoId', 'repo_id']),
    filename: readOptionalString(record, ['filename']),
    url: readOptionalString(record, ['url']),
    notes: readOptionalString(record, ['notes']),
    snapshot: readBoolean(record, ['snapshot'], false),
    installed: readBoolean(record, ['installed'], false),
    destination: readString(record, ['destination', 'folder', 'path'], ''),
    engineId: normalizeEngineId(readUnknown(record, ['engineId', 'engine_id', 'engine'])),
    engineLabel: readOptionalString(record, ['engineLabel', 'engine_label']),
    hfUrl: readOptionalString(record, ['hfUrl', 'hf_url']),
    requiresAuth: readBoolean(record, ['requiresAuth', 'requires_auth'], false),
    canDownload: readBoolean(record, ['canDownload', 'can_download'], false),
    comingSoon: readBoolean(record, ['comingSoon', 'coming_soon'], false),
  }
}

function normalizeLogFile(value: unknown): ProLogFile | null {
  const record = asRecord(value)
  const name = readString(record, ['name'], '')
  const path = readString(record, ['path'], '')
  if (!name && !path) {
    return null
  }
  return {
    name: name || path,
    path,
    sizeBytes: readNumber(record, ['sizeBytes', 'size_bytes'], 0),
    modifiedAt: readString(record, ['modifiedAt', 'modified_at'], ''),
  }
}

function normalizeLogEvent(value: unknown): ProLogEvent | null {
  const record = asRecord(value)
  const title = readString(record, ['title'], '')
  const detail = readString(record, ['detail', 'message'], '')
  if (!title && !detail) {
    return null
  }
  return {
    id: readString(record, ['id'], `${title}-${detail}`),
    source: readString(record, ['source'], 'runtime'),
    time: readString(record, ['time', 'createdAt', 'created_at'], ''),
    title: title || 'Event',
    detail,
  }
}

function normalizeRuntime(value: unknown): ProRuntimeStatus {
  const record = asRecord(value)
  const fallback = getFallbackRuntime()
  const resourceValue = readUnknown(record, ['resources', 'usage'])
  const loadedModelRecord = readRecord(record, ['loaded_model', 'loadedModel', 'model'])
  const resources = normalizeResources(resourceValue, fallback.resources)

  return {
    state: readString(record, ['state', 'status'], fallback.state),
    backend: readString(record, ['backend'], fallback.backend),
    device: readString(record, ['device', 'gpu'], fallback.device),
    precision: readString(record, ['precision'], fallback.precision),
    attention: readString(record, ['attention'], fallback.attention),
    maxResolution: readString(
      record,
      ['max_resolution', 'maxResolution'],
      fallback.maxResolution,
    ),
    queueCount: readNumber(record, ['queue_count', 'queueCount', 'queue'], fallback.queueCount),
    resources,
    job: normalizeRuntimeJob(readRecord(record, ['job']), fallback.job),
    loadedModel: normalizeLoadedModel(loadedModelRecord, fallback.loadedModel),
  }
}

function normalizeRuntimeJob(value: Record<string, unknown>, fallback: ProRuntimeStatus['job']): ProRuntimeStatus['job'] {
  return {
    id: readString(value, ['id', 'job_id', 'jobId'], fallback.id),
    state: readString(value, ['state', 'status'], fallback.state),
    progress: clampPercent(readNumber(value, ['progress', 'percent'], fallback.progress)),
    step: readNumber(value, ['step'], fallback.step),
    totalSteps: readNumber(value, ['totalSteps', 'total_steps', 'total'], fallback.totalSteps),
    message: readString(value, ['message', 'detail'], fallback.message),
    hasResult: readBoolean(value, ['hasResult', 'has_result'], fallback.hasResult),
    error: readString(value, ['error'], fallback.error),
    previewUrl: normalizeAssetUrl(readString(value, ['previewUrl', 'preview_url', 'dataUrl', 'data_url'], fallback.previewUrl)),
  }
}

function normalizeGenerateResult(
  value: unknown,
  request: ProGenerateRequest,
): ProGenerateResult {
  const record = asRecord(value)
  const encodedImages = readArray(record, ['images'])
    .map((item, index) => normalizeRecentOutput(item, index, request))
    .filter(isPresent)
  const recent = readArray(record, ['recent_outputs', 'recentOutputs', 'outputs'])
    .map((item, index) => normalizeRecentOutput(item, index, request))
    .filter(isPresent)
  const directOutput =
    recent[0] ??
    normalizeRecentOutput(readUnknown(record, ['output', 'image', 'result']), 0, request) ??
    encodedImages[encodedImages.length - 1] ??
    null
  const sessionOutputs = recent.length > 0 ? recent : encodedImages

  return {
    jobId: readString(record, ['job_id', 'jobId', 'id'], directOutput?.id ?? 'local-job'),
    status: readString(record, ['status', 'state'], directOutput?.status ?? 'completed'),
    message: readString(
      record,
      ['message', 'detail'],
      sessionOutputs.length > 1
        ? `Generated ${sessionOutputs.length} images.`
        : directOutput
          ? 'Generation complete.'
          : 'Generation submitted.',
    ),
    output: directOutput,
    recentOutputs: sessionOutputs,
    progress: readArray(record, ['progress', 'events']).map(normalizeProgressEvent),
    timings: normalizeNumberRecord(readRecord(record, ['timings'])),
    receiptPath: readOptionalString(record, ['receiptPath', 'receipt_path']),
    attentionBackend: readOptionalString(record, ['attentionBackend', 'attention_backend']),
    quantization: readOptionalString(record, ['quantization']),
    vaeTiling: readOptionalString(record, ['vaeTiling', 'vae_tiling']),
  }
}

function normalizeProgressEvent(value: unknown) {
  const record = asRecord(value)
  return {
    stage: readString(record, ['stage'], ''),
    progress: readNumber(record, ['progress'], 0),
    message: readString(record, ['message'], ''),
    step: readNumber(record, ['step'], 0),
    total: readNumber(record, ['total'], 0),
    seconds: readNumber(record, ['seconds'], 0),
  }
}

function normalizeNumberRecord(record: JsonRecord): Record<string, number> {
  return Object.fromEntries(
    Object.entries(record)
      .map(([key, value]) => [key, typeof value === 'number' ? value : Number(value)] as const)
      .filter(([, value]) => Number.isFinite(value)),
  )
}

function normalizeSettings(
  record: JsonRecord,
  fallback: GenerationSettings,
  ratios: AspectRatioOption[],
  models: ProModelOption[],
  samplers: string[],
): GenerationSettings {
  const aspectRatioId = readString(
    record,
    ['aspect_ratio_id', 'aspectRatioId', 'aspect_ratio', 'aspectRatio'],
    fallback.aspectRatioId,
  )
  const matchedRatio = ratios.find((ratio) => ratio.id === aspectRatioId) ?? ratios[0]
  const width = readNumber(record, ['width'], matchedRatio?.width ?? fallback.width)
  const height = readNumber(record, ['height'], matchedRatio?.height ?? fallback.height)

  return {
    mode: normalizeCreationMode(readUnknown(record, ['mode']), fallback.mode),
    prompt: readString(record, ['prompt'], fallback.prompt),
    negativePrompt: readString(
      record,
      ['negative_prompt', 'negativePrompt'],
      fallback.negativePrompt,
    ),
    modelId: readString(
      record,
      ['model_id', 'modelId', 'checkpoint_id', 'checkpointId'],
      models[0]?.id ?? fallback.modelId,
    ),
    aspectRatioId: matchedRatio?.id ?? fallback.aspectRatioId,
    width,
    height,
    steps: readNumber(record, ['steps'], fallback.steps),
    cfgScale: readNumber(record, ['cfg_scale', 'cfgScale'], fallback.cfgScale),
    sampler: readString(record, ['sampler'], samplers[0] ?? fallback.sampler),
    scheduler: readString(record, ['scheduler'], fallback.scheduler),
    seed: readNumber(record, ['seed'], fallback.seed),
    clipSkip: readNumber(record, ['clip_skip', 'clipSkip'], fallback.clipSkip),
    batchSize: readNumber(record, ['batch_size', 'batchSize'], fallback.batchSize),
    batchCount: readNumber(record, ['batch_count', 'batchCount'], fallback.batchCount),
    enableHires: readBoolean(record, ['enable_hr', 'enableHr', 'enableHires', 'enable_hires'], fallback.enableHires),
    hiresScale: readNumber(record, ['hr_scale', 'hrScale', 'hiresScale', 'hires_scale'], fallback.hiresScale),
    hiresSteps: readNumber(record, ['hr_steps', 'hrSteps', 'hiresSteps', 'hires_steps'], fallback.hiresSteps),
    hiresDenoise: readNumber(
      record,
      ['hr_denoising_strength', 'hrDenoisingStrength', 'hiresDenoise', 'hires_denoise'],
      fallback.hiresDenoise,
    ),
    hiresUpscaler: readString(record, ['hr_upscaler', 'hrUpscaler', 'hiresUpscaler', 'hires_upscaler'], fallback.hiresUpscaler),
    frames: readNumber(record, ['frames'], fallback.frames),
    fps: readNumber(record, ['fps'], fallback.fps),
    sourceImageDataUrl: readString(record, ['source_image_data_url', 'sourceImageDataUrl'], fallback.sourceImageDataUrl),
    sourceImageName: readString(record, ['source_image_name', 'sourceImageName'], fallback.sourceImageName),
    sanaQuantization: readString(record, ['sana_quantization', 'sanaQuantization'], fallback.sanaQuantization),
    sanaVaeTiling: readString(record, ['sana_vae_tiling', 'sanaVaeTiling', 'vae_tiling', 'vaeTiling'], fallback.sanaVaeTiling),
    offloadTextEncoderAfterEncode: readBoolean(
      record,
      ['offload_text_encoder_after_encode', 'offloadTextEncoderAfterEncode'],
      fallback.offloadTextEncoderAfterEncode,
    ),
    useSageAttention: readBoolean(record, ['use_sage_attention', 'useSageAttention'], fallback.useSageAttention),
    generateAudio: readBoolean(record, ['generate_audio', 'generateAudio'], fallback.generateAudio),
    wanRuntimeMode: readString(record, ['wan_runtime_mode', 'wanRuntimeMode', 'runtimeMode', 'runtime_mode'], fallback.wanRuntimeMode),
    highNoiseModelId: readString(record, ['high_noise_model_id', 'highNoiseModelId'], fallback.highNoiseModelId),
    lowNoiseModelId: readString(record, ['low_noise_model_id', 'lowNoiseModelId'], fallback.lowNoiseModelId),
    highNoiseSteps: readNumber(record, ['high_noise_steps', 'highNoiseSteps'], fallback.highNoiseSteps),
    lowNoiseSteps: readNumber(record, ['low_noise_steps', 'lowNoiseSteps'], fallback.lowNoiseSteps),
    boundaryRatio: readNumber(record, ['boundary_ratio', 'boundaryRatio'], fallback.boundaryRatio),
    highNoiseLoraId: readString(record, ['high_noise_lora_id', 'highNoiseLoraId'], fallback.highNoiseLoraId),
    highNoiseLoraScale: readNumber(record, ['high_noise_lora_scale', 'highNoiseLoraScale'], fallback.highNoiseLoraScale),
    lowNoiseLoraId: readString(record, ['low_noise_lora_id', 'lowNoiseLoraId'], fallback.lowNoiseLoraId),
    lowNoiseLoraScale: readNumber(record, ['low_noise_lora_scale', 'lowNoiseLoraScale'], fallback.lowNoiseLoraScale),
    vaeId: readString(record, ['vae_id', 'vaeId'], fallback.vaeId),
    textEncoderPath: readString(record, ['text_encoder_path', 'textEncoderPath'], fallback.textEncoderPath),
    wanOffload: readString(record, ['wan_offload', 'wanOffload', 'offload'], fallback.wanOffload),
    wanSigmaType: readString(record, ['wan_sigma_type', 'wanSigmaType', 'sigma_type', 'sigmaType'], fallback.wanSigmaType),
    wanSampler: readString(record, ['wan_sampler', 'wanSampler'], fallback.wanSampler),
    wanFlowShift: readNumber(record, ['wan_flow_shift', 'wanFlowShift', 'flow_shift', 'flowShift'], fallback.wanFlowShift),
    initImageDataUrl: readString(record, ['init_image_data_url', 'initImageDataUrl'], fallback.initImageDataUrl),
    maskImageDataUrl: readString(record, ['mask_image_data_url', 'maskImageDataUrl'], fallback.maskImageDataUrl),
    denoisingStrength: readNumber(record, ['denoising_strength', 'denoisingStrength'], fallback.denoisingStrength),
    maskBlur: readNumber(record, ['mask_blur', 'maskBlur'], fallback.maskBlur),
    inpaintOnlyMasked: readBoolean(record, ['inpaint_only_masked', 'inpaintOnlyMasked'], fallback.inpaintOnlyMasked),
    inpaintMaskedPadding: readNumber(record, ['inpaint_masked_padding', 'inpaintMaskedPadding'], fallback.inpaintMaskedPadding),
    inpaintMaskContent: readString(record, ['inpaint_mask_content', 'inpaintMaskContent'], fallback.inpaintMaskContent),
    inpaintMaskOpacity: readNumber(record, ['inpaint_mask_opacity', 'inpaintMaskOpacity'], fallback.inpaintMaskOpacity),
    autoMaskEnabled: readBoolean(record, ['auto_mask_enabled', 'autoMaskEnabled'], fallback.autoMaskEnabled),
    autoMaskPrompt: readString(record, ['auto_mask_prompt', 'autoMaskPrompt'], fallback.autoMaskPrompt),
    autoMaskModel: readString(record, ['auto_mask_model', 'autoMaskModel'], fallback.autoMaskModel),
    autoMaskBoxThreshold: readNumber(record, ['auto_mask_box_threshold', 'autoMaskBoxThreshold'], fallback.autoMaskBoxThreshold),
    autoMaskTextThreshold: readNumber(record, ['auto_mask_text_threshold', 'autoMaskTextThreshold'], fallback.autoMaskTextThreshold),
    controlNetEnabled: readBoolean(record, ['controlnet_enabled', 'controlNetEnabled'], fallback.controlNetEnabled),
    controlNetModel: readString(record, ['controlnet_model', 'controlNetModel'], fallback.controlNetModel),
    controlNetModule: readString(record, ['controlnet_module', 'controlNetModule'], fallback.controlNetModule),
    controlNetImageDataUrl: readString(record, ['controlnet_image_data_url', 'controlNetImageDataUrl'], fallback.controlNetImageDataUrl),
    controlNetImageName: readString(record, ['controlnet_image_name', 'controlNetImageName'], fallback.controlNetImageName),
    controlNetWeight: readNumber(record, ['controlnet_weight', 'controlNetWeight'], fallback.controlNetWeight),
    controlNetGuidanceStart: readNumber(record, ['controlnet_guidance_start', 'controlNetGuidanceStart'], fallback.controlNetGuidanceStart),
    controlNetGuidanceEnd: readNumber(record, ['controlnet_guidance_end', 'controlNetGuidanceEnd'], fallback.controlNetGuidanceEnd),
    controlNetProcessorRes: readNumber(record, ['controlnet_processor_res', 'controlNetProcessorRes'], fallback.controlNetProcessorRes),
    saveImages: readBoolean(record, ['save_images', 'saveImages'], fallback.saveImages),
  }
}

function normalizeSamplerOption(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }
  const record = asRecord(value)
  return readString(record, ['label', 'name', 'id', 'value'], '')
}

function normalizeAspectRatio(value: unknown): AspectRatioOption | null {
  if (typeof value === 'string') {
    const parsed = parseRatio(value)
    return parsed
      ? { id: value, label: value, width: parsed.width, height: parsed.height }
      : null
  }

  const record = asRecord(value)
  const label = readString(record, ['label', 'name', 'id'], '')
  const parsed = parseRatio(label)
  const width = readNumber(record, ['width', 'w'], parsed?.width ?? 1024)
  const height = readNumber(record, ['height', 'h'], parsed?.height ?? 1024)
  const id = readString(record, ['id', 'value'], label || `${width}:${height}`)

  return label || id ? { id, label: label || id, width, height } : null
}

function normalizeModel(value: unknown): ProModelOption | null {
  if (typeof value === 'string') {
    return { id: value, name: value }
  }

  const record = asRecord(value)
  const id = readString(record, ['id', 'model_id', 'checkpoint_id', 'value'], '')
  const name = readString(record, ['name', 'title', 'label'], id)
  if (!id && !name) {
    return null
  }

  return {
    id: id || name,
    name,
    architecture: readOptionalString(record, ['architecture', 'base_model', 'baseModel']),
    sizeBytes: readNumber(record, ['sizeBytes', 'size_bytes'], 0),
    fileCount: readNumber(record, ['fileCount', 'file_count'], 0),
    assetSummary: readOptionalString(record, ['assetSummary', 'asset_summary']),
    kind: readOptionalString(record, ['kind']),
    engineId: normalizeEngineId(readUnknown(record, ['engineId', 'engine_id', 'engine'])),
    engineLabel: readOptionalString(record, ['engineLabel', 'engine_label']),
    backend: readOptionalString(record, ['backend']),
    status: readOptionalString(record, ['status', 'state']),
    reason: readOptionalString(record, ['reason']),
    suggestedAction: readOptionalString(record, ['suggestedAction', 'suggested_action']),
    estVramGb: readNumber(record, ['estVramGb', 'est_vram_gb'], 0),
    heavyFor12Gb: Boolean(readUnknown(record, ['heavyFor12Gb', 'heavy_for_12gb'])),
    generationPreset: normalizeModelPreset(readRecord(record, ['generationPreset', 'generation_preset'])),
  }
}

function normalizeModelPreset(record: JsonRecord): Partial<GenerationSettings> | undefined {
  const preset: Partial<GenerationSettings> = {}
  if ('steps' in record) {
    preset.steps = readNumber(record, ['steps'], 0)
  }
  if ('cfg_scale' in record || 'cfgScale' in record) {
    preset.cfgScale = readNumber(record, ['cfg_scale', 'cfgScale'], 0)
  }
  if ('sampler' in record) {
    preset.sampler = readString(record, ['sampler'], '')
  }
  if ('scheduler' in record) {
    preset.scheduler = readString(record, ['scheduler'], '')
  }
  if ('clip_skip' in record || 'clipSkip' in record) {
    preset.clipSkip = readNumber(record, ['clip_skip', 'clipSkip'], 1)
  }
  if ('width' in record) {
    preset.width = readNumber(record, ['width'], 0)
  }
  if ('height' in record) {
    preset.height = readNumber(record, ['height'], 0)
  }
  return Object.keys(preset).length > 0 ? preset : undefined
}

function normalizeEngineSummary(value: unknown): EngineSummary | null {
  const record = asRecord(value)
  const id = normalizeEngineId(readUnknown(record, ['id', 'engineId', 'engine_id']))
  const label = readString(record, ['label', 'engineLabel', 'engine_label'], '')
  const count = readNumber(record, ['count'], 0)
  if (!id || !label) {
    return null
  }
  return { id, label, count }
}

function normalizeRecentOutput(
  value: unknown,
  index: number,
  defaults: GenerationSettings,
): RecentOutput | null {
  if (typeof value === 'string') {
    const url = normalizeAssetUrl(value)
    return {
      id: `output-${index}-${url}`,
      url,
      thumbnailUrl: url,
      path: undefined,
      prompt: defaults.prompt,
      width: defaults.width,
      height: defaults.height,
      createdAt: 'now',
      mode: defaults.mode,
      modelName: defaults.modelId,
      status: 'completed',
    }
  }

  const record = asRecord(value)
  const rawUrl = readString(
    record,
    ['url', 'dataUrl', 'data_url', 'video', 'video_url', 'videoUrl', 'image', 'image_url', 'imageUrl', 'path', 'src', 'file'],
    '',
  )
  if (!rawUrl) {
    return null
  }
  const url = normalizeAssetUrl(rawUrl)
  const thumbnailUrl = normalizeAssetUrl(
    readString(record, ['thumbnail_url', 'thumbnailUrl', 'thumbnail'], rawUrl),
  )

  return {
    id: readString(record, ['id', 'job_id', 'jobId'], `output-${index}-${url}`),
    url,
    thumbnailUrl,
    path: readOptionalString(record, ['path', 'file']),
    prompt: readString(record, ['prompt'], defaults.prompt),
    negativePrompt: readOptionalString(record, ['negativePrompt', 'negative_prompt']),
    infotext: readOptionalString(record, ['infotext', 'infoText', 'metadata']),
    width: readNumber(record, ['width'], defaults.width),
    height: readNumber(record, ['height'], defaults.height),
    createdAt: readString(record, ['created_at', 'createdAt', 'time', 'age'], 'now'),
    mode: normalizeCreationMode(readUnknown(record, ['mode']), defaults.mode),
    seed: readOptionalNumber(record, ['seed']),
    steps: readOptionalNumber(record, ['steps']),
    cfgScale: readOptionalNumber(record, ['cfgScale', 'cfg_scale']),
    clipSkip: readOptionalNumber(record, ['clipSkip', 'clip_skip']),
    sampler: readOptionalString(record, ['sampler', 'samplerName', 'sampler_name']),
    scheduler: readOptionalString(record, ['scheduler', 'schedule']),
    modelName: readOptionalString(record, ['model_name', 'modelName', 'model']),
    status: readOptionalString(record, ['status', 'state']),
    source: readOptionalString(record, ['source']),
  }
}

function normalizeResources(
  value: unknown,
  fallback: ResourceMetric[],
): ResourceMetric[] {
  if (Array.isArray(value)) {
    const byLabel = new Map(fallback.map((metric) => [metric.label.toLowerCase(), metric]))
    const parsed = value.map((item, index) => {
      const record = asRecord(item)
      const label = readString(record, ['label', 'name'], fallback[index]?.label ?? `Metric ${index + 1}`)
      return normalizeResourceMetric(record, byLabel.get(label.toLowerCase()) ?? fallback[index] ?? {
        label,
        value: 'Unavailable',
        percent: 0,
        tone: 'neutral',
      })
    })
    return parsed.length > 0 ? parsed : fallback
  }
  const record = asRecord(value)
  const metrics = fallback.map((metric) => {
    const key = metric.label.toLowerCase()
    const metricRecord = readRecord(record, [key])
    return normalizeResourceMetric(metricRecord, metric)
  })
  return metrics.length > 0 ? metrics : fallback
}

function normalizeResourceMetric(record: JsonRecord, fallback: ResourceMetric): ResourceMetric {
  const percent = clampPercent(readNumber(record, ['percent', 'usage', 'value_percent'], fallback.percent))
  const value = readString(record, ['value', 'label', 'text'], fallback.value)
  const tone = normalizeTone(readUnknown(record, ['tone']), fallback.tone)
  return {
    label: readString(record, ['label', 'name'], fallback.label),
    value,
    percent,
    tone,
  }
}

function normalizeLoadedModel(record: JsonRecord, fallback: LoadedModelInfo): LoadedModelInfo {
  return {
    name: readString(record, ['name', 'title'], fallback.name),
    type: readString(record, ['type'], fallback.type),
    baseModel: readString(record, ['base_model', 'baseModel'], fallback.baseModel),
    sizeOnDisk: readString(record, ['size_on_disk', 'sizeOnDisk', 'size'], fallback.sizeOnDisk),
    precision: readString(record, ['precision'], fallback.precision),
    vae: readString(record, ['vae'], fallback.vae),
    textEncoder: readString(record, ['text_encoder', 'textEncoder'], fallback.textEncoder),
    unet: readString(record, ['unet'], fallback.unet),
    loaded: readBoolean(record, ['loaded'], fallback.loaded),
  }
}

function buildFallbackOutputs(): RecentOutput[] {
  const specs = [
    ['city', 'Overgrown city', '#14201b', '#67d6ba', '#1c5a73', 1536, 1024],
    ['alpine', 'Alpine lake', '#1b2637', '#91cdf7', '#c8d7df', 1536, 1024],
    ['forest', 'Rain forest path', '#0f1a15', '#578f67', '#d7bb83', 1536, 1024],
    ['orbit', 'Planet horizon', '#090d1b', '#8ca7ff', '#f2f4ff', 1024, 1024],
    ['space', 'Astronaut dock', '#0d1118', '#8a9fb7', '#f2efe6', 1536, 1024],
    ['room', 'Warm studio room', '#281a13', '#f0b875', '#d7dbd5', 1024, 1365],
    ['desert', 'Desert ridge', '#211810', '#d7aa68', '#8eb6ce', 1536, 1024],
    ['neon', 'Neon alley', '#140d18', '#f05b5b', '#5ee4d0', 1024, 1365],
  ] as const

  return specs.map(([id, title, bg, primary, secondary, width, height], index) => {
    const url = buildPreviewDataUri(title, bg, primary, secondary, width, height)
    return {
      id,
      url,
      thumbnailUrl: url,
      prompt: title,
      width,
      height,
      createdAt: `${index * 7 + 2}m ago`,
      mode: 'image',
      modelName: DEFAULT_MODELS[0].name,
      status: 'completed',
    }
  })
}

function buildPreviewDataUri(
  title: string,
  bg: string,
  primary: string,
  secondary: string,
  width: number,
  height: number,
): string {
  const horizon = Math.round(height * 0.62)
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop stop-color="${bg}"/><stop offset=".58" stop-color="${primary}"/><stop offset="1" stop-color="${secondary}"/></linearGradient><filter id="blur"><feGaussianBlur stdDeviation="18"/></filter></defs><rect width="${width}" height="${height}" fill="${bg}"/><rect width="${width}" height="${height}" fill="url(#g)" opacity=".56"/><circle cx="${Math.round(width * 0.72)}" cy="${Math.round(height * 0.22)}" r="${Math.round(width * 0.16)}" fill="${secondary}" opacity=".24" filter="url(#blur)"/><path d="M0 ${horizon} C ${Math.round(width * 0.2)} ${Math.round(height * 0.5)} ${Math.round(width * 0.35)} ${Math.round(height * 0.76)} ${Math.round(width * 0.55)} ${horizon} S ${Math.round(width * 0.82)} ${Math.round(height * 0.44)} ${width} ${Math.round(height * 0.68)} V ${height} H 0 Z" fill="#05070a" opacity=".46"/><g opacity=".26" stroke="#fff" stroke-width="3">${Array.from({ length: 9 }, (_, i) => {
    const x = Math.round((width / 10) * (i + 1))
    return `<path d="M${x} ${Math.round(height * 0.22)} V${Math.round(height * 0.86)}"/>`
  }).join('')}</g><text x="${Math.round(width * 0.07)}" y="${Math.round(height * 0.87)}" fill="#f8fffb" font-family="system-ui, Segoe UI, sans-serif" font-size="${Math.round(width * 0.045)}" font-weight="650">${escapeSvgText(title)}</text></svg>`
  return `data:image/svg+xml,${encodeURIComponent(svg)}`
}

function escapeSvgText(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function parseRatio(value: string): { width: number; height: number } | null {
  const [left, right] = value.split(':').map((part) => Number(part.trim()))
  if (!Number.isFinite(left) || !Number.isFinite(right) || left <= 0 || right <= 0) {
    return null
  }
  const longEdge = 1536
  if (left >= right) {
    return {
      width: longEdge,
      height: Math.round((longEdge * right) / left / 8) * 8,
    }
  }
  return {
    width: Math.round((longEdge * left) / right / 8) * 8,
    height: longEdge,
  }
}

function normalizeAssetUrl(value: string): string {
  const trimmed = value.trim()
  if (!trimmed) {
    return ''
  }
  if (trimmed.startsWith('data:') || trimmed.startsWith('http://') || trimmed.startsWith('https://')) {
    return trimmed
  }
  const normalized = trimmed.replace(/\\/g, '/')
  const apiUrl = (path: string) => `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`
  const outputsIndex = normalized.toLowerCase().lastIndexOf('/outputs/')
  if (outputsIndex >= 0) {
    return apiUrl(`/api/pro/outputs/${normalized.slice(outputsIndex + '/outputs/'.length)}`)
  }
  if (normalized.startsWith('outputs/')) {
    return apiUrl(`/api/pro/outputs/${normalized.slice('outputs/'.length)}`)
  }
  return normalized.startsWith('/') ? apiUrl(normalized) : apiUrl(normalized)
}

function normalizeCreationMode(value: unknown, fallback: CreationMode): CreationMode {
  if (value === 'image' || value === 'video' || value === 'inpaint') {
    return value
  }
  if (value === 'txt2img') {
    return 'image'
  }
  if (value === 'img2img') {
    return 'image'
  }
  return fallback
}

function normalizeTone(value: unknown, fallback: ResourceTone): ResourceTone {
  if (value === 'mint' || value === 'blue' || value === 'amber' || value === 'red' || value === 'neutral') {
    return value
  }
  return fallback
}

function normalizeEngineId(value: unknown): ProModelOption['engineId'] {
  if (
    value === 'all' ||
    value === 'flux' ||
    value === 'flux_fill' ||
    value === 'flux2' ||
    value === 'sana_video' ||
    value === 'wan' ||
    value === 'sd15' ||
    value === 'sdxl' ||
    value === 'sd35' ||
    value === 'zimage' ||
    value === 'qwen' ||
    value === 'sana' ||
    value === 'unknown'
  ) {
    return value
  }
  return undefined
}

function buildEngineSummaries(models: ProModelOption[]): EngineSummary[] {
  const groups = new Map<string, EngineSummary>()
  for (const model of models) {
    const id = model.engineId ?? 'unknown'
    const label = model.engineLabel ?? model.engineId ?? 'Other'
    const existing = groups.get(id)
    if (existing) {
      existing.count += 1
    } else {
      groups.set(id, { id, label, count: 1 })
    }
  }
  return Array.from(groups.values())
}

function asRecord(value: unknown): JsonRecord {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as JsonRecord : {}
}

function readRecord(record: JsonRecord, keys: string[]): JsonRecord {
  const value = readUnknown(record, keys)
  return asRecord(value)
}

function readArray(record: JsonRecord, keys: string[]): unknown[] {
  const value = readUnknown(record, keys)
  return Array.isArray(value) ? value : []
}

function readUnknown(record: JsonRecord, keys: string[]): unknown {
  for (const key of keys) {
    if (Object.prototype.hasOwnProperty.call(record, key)) {
      return record[key]
    }
  }
  return undefined
}

function readString(record: JsonRecord, keys: string[], fallback: string): string {
  return readLooseString(readUnknown(record, keys), fallback)
}

function readOptionalString(record: JsonRecord, keys: string[]): string | undefined {
  const value = readLooseString(readUnknown(record, keys), '')
  return value || undefined
}

function readLooseString(value: unknown, fallback: string): string {
  if (typeof value === 'string') {
    return value
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  return fallback
}

function readNumber(record: JsonRecord, keys: string[], fallback: number): number {
  const value = readUnknown(record, keys)
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof value === 'string') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : fallback
  }
  return fallback
}

function readOptionalNumber(record: JsonRecord, keys: string[]): number | undefined {
  const value = readUnknown(record, keys)
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof value === 'string') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : undefined
  }
  return undefined
}

function readBoolean(record: JsonRecord, keys: string[], fallback: boolean): boolean {
  const value = readUnknown(record, keys)
  if (typeof value === 'boolean') {
    return value
  }
  if (typeof value === 'string') {
    const lowered = value.toLowerCase()
    if (lowered === 'true' || lowered === '1' || lowered === 'yes') {
      return true
    }
    if (lowered === 'false' || lowered === '0' || lowered === 'no') {
      return false
    }
  }
  return fallback
}

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, Math.round(value)))
}

function isPresent<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined
}
