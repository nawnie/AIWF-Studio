export type CreationMode = 'image' | 'video' | 'inpaint'

export type ProMode = CreationMode | 'models' | 'data'

export type EngineId = 'all' | 'flux' | 'flux2' | 'sana_video' | 'sd15' | 'sdxl' | 'sd35' | 'zimage' | 'unknown'

export type ResourceTone = 'mint' | 'blue' | 'amber' | 'red' | 'neutral'

export interface AspectRatioOption {
  id: string
  label: string
  width: number
  height: number
}

export interface ProModelOption {
  id: string
  name: string
  architecture?: string
  sizeBytes?: number
  fileCount?: number
  assetSummary?: string
  engineId?: EngineId
  engineLabel?: string
  backend?: string
  status?: string
  reason?: string
  suggestedAction?: string
}

export interface EngineSummary {
  id: EngineId
  label: string
  count: number
}

export interface GenerationSettings {
  mode: CreationMode
  prompt: string
  negativePrompt: string
  modelId: string
  aspectRatioId: string
  width: number
  height: number
  steps: number
  cfgScale: number
  sampler: string
  scheduler: string
  seed: number
  batchSize: number
  batchCount: number
  frames: number
  fps: number
  sourceImageDataUrl: string
  sourceImageName: string
  sanaQuantization: string
  sanaVaeTiling: string
  offloadTextEncoderAfterEncode: boolean
  useSageAttention: boolean
  generateAudio: boolean
}

export interface RecentOutput {
  id: string
  url: string
  thumbnailUrl: string
  path?: string
  prompt: string
  width: number
  height: number
  createdAt: string
  mode: CreationMode
  seed?: number
  modelName?: string
  status?: string
  source?: string
}

export interface ResourceMetric {
  label: string
  value: string
  percent: number
  tone: ResourceTone
}

export interface LoadedModelInfo {
  name: string
  type: string
  baseModel: string
  sizeOnDisk: string
  precision: string
  vae: string
  textEncoder: string
  unet: string
  loaded: boolean
}

export interface RuntimeJobStatus {
  id: string
  state: string
  progress: number
  step: number
  totalSteps: number
  message: string
  hasResult: boolean
  error: string
}

export interface ProRuntimeStatus {
  state: string
  job: RuntimeJobStatus
  backend: string
  device: string
  precision: string
  attention: string
  maxResolution: string
  queueCount: number
  resources: ResourceMetric[]
  loadedModel: LoadedModelInfo
}

export interface ProBootstrap {
  workspaceName: string
  subtitle: string
  version: string
  localFirst: boolean
  onboardingSeen: boolean
  engines: EngineSummary[]
  models: ProModelOption[]
  blockedModels: ProModelOption[]
  counts: {
    checkpoints: number
    blockedCheckpoints: number
  }
  samplers: string[]
  aspectRatios: AspectRatioOption[]
  defaults: GenerationSettings
  recentOutputs: RecentOutput[]
}

export type ProGenerateRequest = GenerationSettings

export interface ProGenerateResult {
  jobId: string
  status: string
  message: string
  output: RecentOutput | null
  recentOutputs: RecentOutput[]
  progress: GenerationProgressEvent[]
  timings: Record<string, number>
  receiptPath?: string
  attentionBackend?: string
  quantization?: string
  vaeTiling?: string
}

export interface ProStopResult {
  status: string
  videoJobId: string
}

export interface GenerationProgressEvent {
  stage: string
  progress: number
  message: string
  step: number
  total: number
  seconds: number
}

export interface ProDataStatus {
  outputRoot: string
  counts: {
    checkpoints: number
    blockedCheckpoints: number
    recentOutputs: number
    engines: number
  }
  engines: EngineSummary[]
  recentOutputs: RecentOutput[]
}

export interface ProDownloadCategory {
  key: string
  label: string
  destination: string
}

export interface ProDownloadCatalogItem {
  key: string
  title: string
  category: string
  source: string
  sizeMb?: number
  repoId?: string
  filename?: string
  url?: string
  notes?: string
  snapshot: boolean
  installed: boolean
  destination: string
  engineId?: EngineId
  engineLabel?: string
}

export interface ProDownloadsStatus {
  categories: ProDownloadCategory[]
  bundles: Record<string, string[]>
  catalog: ProDownloadCatalogItem[]
  counts: {
    categories: number
    catalog: number
    installed: number
  }
}

export interface ProLogFile {
  name: string
  path: string
  sizeBytes: number
  modifiedAt: string
}

export interface ProLogEvent {
  id: string
  source: string
  time: string
  title: string
  detail: string
}

export interface ProLogStatus {
  runtime: ProRuntimeStatus
  files: ProLogFile[]
  events: ProLogEvent[]
}

export interface ProSettingsStatus {
  paths: {
    settings: string
    launch: string
    models: string
    checkpoints: string
    outputs: string
  }
  generationDefaults: GenerationSettings
  ui: {
    accentPreset: string
    galleryColumns: number
    galleryHeight: number
    livePreview: boolean
    hiddenTabs: string[]
  }
  runtime: {
    listen: boolean
    api: boolean
    genlog: boolean
    backend: string
    attention: string
  }
}

export interface ProCapabilityItem {
  id: string
  label: string
  group: string
  status: string
  count: number
  route: string
  summary: string
  details: string[]
  tab?: string
}

export type ProReadinessStatusName =
  | 'working'
  | 'metadata-only'
  | 'blocked-cleanly'
  | 'broken-runtime'
  | 'unsupported-no-route'

export type ProReadinessCounts = Record<ProReadinessStatusName, number> & Record<string, number>

export interface ProReadinessFamily {
  family: string
  counts: ProReadinessCounts
  total: number
}

export interface ProReadinessItem {
  id: string
  family: string
  assetType: string
  path: string
  label: string
  status: string
  route: string
  reason: string
  storage: string
  quantization: string
  requiredVae: string
  requiredTextEncoder: string
  tokenizer: string
  smokeCommand: string
  receiptPath: string
  suggestedAction: string
}

export interface ProReadinessStatus {
  counts: ProReadinessCounts
  families: ProReadinessFamily[]
  working: ProReadinessItem[]
  needsWork: ProReadinessItem[]
  metadataOnlyCount: number
  total: number
  error: string
}

export interface ProCapabilitiesStatus {
  gradioTabs: ProCapabilityItem[]
  tools: ProCapabilityItem[]
  counts: {
    gradioTabs: number
    reactRails: number
    checkpoints: number
    blockedCheckpoints: number
    loras: number
    controlnet: number
    sam: number
    reactor: number
    enhance: number
    sanaVideo: number
    wan: number
  }
  readiness: ProReadinessStatus
  notes: string[]
}

export interface PromptInsight {
  status: 'idle' | 'loading' | 'ready' | 'error'
  summary: string
  modelLabel: string
  modelScore: number
  modelId: string
  progress: number
  signals: Array<{
    label: string
    value: string
    tone: ResourceTone
  }>
  suggestions: string[]
}
