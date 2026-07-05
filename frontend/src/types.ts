export type CreationMode = 'image' | 'video' | 'inpaint'

export type ProMode = CreationMode | 'models' | 'data'

export type EngineId = 'all' | 'flux' | 'flux_fill' | 'flux2' | 'sana_video' | 'wan' | 'sd15' | 'sdxl' | 'sd35' | 'zimage' | 'qwen' | 'sana' | 'unknown'

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
  kind?: string
  engineId?: EngineId
  engineLabel?: string
  backend?: string
  status?: string
  reason?: string
  suggestedAction?: string
  estVramGb?: number
  heavyFor12Gb?: boolean
  generationPreset?: Partial<GenerationSettings>
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
  clipSkip: number
  batchSize: number
  batchCount: number
  enableHires: boolean
  hiresScale: number
  hiresSteps: number
  hiresDenoise: number
  hiresUpscaler: string
  frames: number
  fps: number
  sourceImageDataUrl: string
  sourceImageName: string
  sanaQuantization: string
  sanaVaeTiling: string
  offloadTextEncoderAfterEncode: boolean
  useSageAttention: boolean
  generateAudio: boolean
  wanRuntimeMode: string
  highNoiseModelId: string
  lowNoiseModelId: string
  highNoiseSteps: number
  lowNoiseSteps: number
  boundaryRatio: number
  highNoiseLoraId: string
  highNoiseLoraScale: number
  lowNoiseLoraId: string
  lowNoiseLoraScale: number
  vaeId: string
  textEncoderPath: string
  wanOffload: string
  wanSigmaType: string
  wanSampler: string
  wanFlowShift: number
  initImageDataUrl: string
  maskImageDataUrl: string
  denoisingStrength: number
  maskBlur: number
  inpaintOnlyMasked: boolean
  inpaintMaskedPadding: number
  inpaintMaskContent: string
  inpaintMaskOpacity: number
  autoMaskEnabled: boolean
  autoMaskPrompt: string
  autoMaskModel: string
  autoMaskBoxThreshold: number
  autoMaskTextThreshold: number
  controlNetEnabled: boolean
  controlNetModel: string
  controlNetModule: string
  controlNetImageDataUrl: string
  controlNetImageName: string
  controlNetWeight: number
  controlNetGuidanceStart: number
  controlNetGuidanceEnd: number
  controlNetProcessorRes: number
  saveImages: boolean
}

export interface RecentOutput {
  id: string
  url: string
  thumbnailUrl: string
  path?: string
  prompt: string
  negativePrompt?: string
  infotext?: string
  width: number
  height: number
  createdAt: string
  mode: CreationMode
  seed?: number
  steps?: number
  cfgScale?: number
  clipSkip?: number
  sampler?: string
  scheduler?: string
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
  previewUrl: string
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
  hfUrl?: string
}

export interface CivitaiBrowseLink {
  label: string
  url: string
  note: string
  engine: string
}

export interface ProDownloadsStatus {
  categories: ProDownloadCategory[]
  bundles: Record<string, string[]>
  catalog: ProDownloadCatalogItem[]
  civitaiLinks: CivitaiBrowseLink[]
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
    showProgressEveryNSteps: number
    livePreviewDecoder: string
    hiddenTabs: string[]
  }
  output: {
    imageFormat: string
    imageQuality: number
    embedMetadata: boolean
    saveGrid: boolean
    saveSidecarTxt: boolean
    filenamePattern: string
    saveBeforeHires: boolean
    saveInterrupted: boolean
    metadataIncludeModelHash: boolean
    metadataIncludeVaeHash: boolean
    metadataIncludeLoraHashes: boolean
    metadataIncludeAppVersion: boolean
    metadataIncludeOptimizationProfile: boolean
    optimizationProfileId: string
  }
  video: {
    wanHigh: string
    wanLow: string
    wanVae: string
    wanTextEncoder: string
    wanOffload: string
    wanSampler: string
    wanFlowShift: number
    wanRuntimeMode: string
    ltxDtype: string
    ltxCpuOffload: string
    wanGroupOffloadStream: boolean
    wanGroupOffloadBlocks: number
    ggufCudaKernels: boolean
    wanSageAttention: string
    wanNativeDenoise: boolean
    wanManualVaeDecode: boolean
    wanVaeChunkFrames: number
    wanGroupOffloadRecordStream: boolean
    wanGroupOffloadLowCpuMem: boolean
    wanResidentMinVramGb: number
  }
  runtime: {
    port: number
    listen: boolean
    share: boolean
    autolaunch: boolean
    api: boolean
    genlog: boolean
    backend: string
    onnxProvider: string
    attention: string
    xformers: boolean
    optSdpAttention: boolean
    optSplitAttention: boolean
    asyncOffload: boolean
    pinnedMemory: boolean
    cudaMalloc: boolean
    medvram: boolean
    lowvram: boolean
    noHalf: boolean
    fp8: boolean
    fluxFp8: boolean
    directml: boolean
    cpu: boolean
    cudaGraphs: boolean
    torchao: boolean
    fp8Quant: boolean
    torchCompile: boolean
    channelsLast: boolean
    nvenc: boolean
    hevc: boolean
    blockPrivateDownloadUrls: boolean
    apiCorsOrigins: string
    apiRateLimitPerMinute: number
    theme: string
    modelsDir: string
    checkpointDir: string
    outputDir: string
    extraModelDirs: string
    extraCheckpointDirs: string
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
