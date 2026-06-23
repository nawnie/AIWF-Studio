import type {
  AspectRatioOption,
  CreationMode,
  GenerationSettings,
  LoadedModelInfo,
  ProBootstrap,
  ProGenerateRequest,
  ProGenerateResult,
  ProModelOption,
  ProRuntimeStatus,
  RecentOutput,
  ResourceMetric,
  ResourceTone,
} from './types'

type JsonRecord = Record<string, unknown>

const API_BASE = (import.meta.env.VITE_AIWF_API_BASE ?? '').replace(/\/$/, '')

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
    backend: 'Diffusers',
    status: 'Loaded',
  },
]

const DEFAULT_SAMPLERS = ['DPM++ 2M Karras', 'Euler a', 'DPM++ SDE', 'UniPC']

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
  seed: -1,
  batchSize: 1,
}

const FALLBACK_RUNTIME: ProRuntimeStatus = {
  state: 'Idle',
  backend: 'PyTorch 2.2.2+cu121',
  device: 'NVIDIA RTX 3090',
  precision: 'FP16',
  attention: 'xFormers',
  maxResolution: '1024 x 1024',
  queueCount: 0,
  resources: [
    { label: 'VRAM', value: '11.2 / 23.9 GB', percent: 46, tone: 'mint' },
    { label: 'RAM', value: '23.6 / 63.9 GB', percent: 37, tone: 'blue' },
    { label: 'Storage', value: '512 / 2048 GB', percent: 25, tone: 'amber' },
    { label: 'CPU', value: '18%', percent: 18, tone: 'blue' },
  ],
  loadedModel: {
    name: 'sdxl-base-1.0 (Diffusers)',
    type: 'Text-to-Image',
    baseModel: 'SDXL 1.0',
    sizeOnDisk: '6.94 GB',
    precision: 'FP16',
    vae: 'sdxl_vae.safetensors',
    textEncoder: 'CLIP ViT-large',
    unet: 'sdxl_unet.safetensors',
    loaded: true,
  },
}

export class ProApiError extends Error {
  readonly status: number
  readonly path: string

  constructor(path: string, status: number, message: string) {
    super(message)
    this.name = 'ProApiError'
    this.path = path
    this.status = status
  }
}

export function getFallbackBootstrap(): ProBootstrap {
  return {
    workspaceName: 'AIWF Studio',
    subtitle: 'Second GUI',
    version: 'v0.2.0',
    localFirst: true,
    onboardingSeen: false,
    models: DEFAULT_MODELS.map((model) => ({ ...model })),
    samplers: [...DEFAULT_SAMPLERS],
    aspectRatios: DEFAULT_ASPECT_RATIOS.map((ratio) => ({ ...ratio })),
    defaults: { ...DEFAULT_SETTINGS },
    recentOutputs: buildFallbackOutputs(),
  }
}

export function getFallbackRuntime(): ProRuntimeStatus {
  return {
    ...FALLBACK_RUNTIME,
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

async function requestJson(path: string, init: RequestInit = {}): Promise<unknown> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...init.headers,
    },
  })

  const text = await response.text()
  if (!response.ok) {
    throw new ProApiError(path, response.status, text || response.statusText)
  }
  if (!text) {
    return {}
  }
  try {
    return JSON.parse(text) as unknown
  } catch {
    throw new ProApiError(path, response.status, 'Response was not valid JSON.')
  }
}

function toGeneratePayload(request: ProGenerateRequest): JsonRecord {
  return {
    mode: request.mode,
    prompt: request.prompt,
    negative_prompt: request.negativePrompt,
    checkpoint_id: request.modelId,
    model_id: request.modelId,
    width: request.width,
    height: request.height,
    steps: request.steps,
    cfg_scale: request.cfgScale,
    sampler: request.sampler,
    seed: request.seed,
    batch_size: request.batchSize,
  }
}

function normalizeBootstrap(value: unknown): ProBootstrap {
  const record = asRecord(value)
  const fallback = getFallbackBootstrap()
  const defaultsRecord = readRecord(record, ['defaults', 'settings'])
  const aspectRatios = readArray(record, ['aspect_ratios', 'aspectRatios', 'ratios'])
    .map(normalizeAspectRatio)
    .filter(isPresent)
  const models = readArray(record, ['models', 'checkpoints'])
    .map(normalizeModel)
    .filter(isPresent)
  const samplers = readArray(record, ['samplers'])
    .map((item) => readLooseString(item, ''))
    .filter((item) => item.length > 0)
  const recentOutputs = readArray(record, ['recent_outputs', 'recentOutputs', 'outputs'])
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
    models: modelOptions,
    samplers: samplerOptions,
    aspectRatios: ratios,
    defaults,
    recentOutputs: recentOutputs.length > 0 ? recentOutputs : fallback.recentOutputs,
  }
}

function normalizeRuntime(value: unknown): ProRuntimeStatus {
  const record = asRecord(value)
  const fallback = getFallbackRuntime()
  const resourceRecord = readRecord(record, ['resources', 'usage'])
  const loadedModelRecord = readRecord(record, ['loaded_model', 'loadedModel', 'model'])
  const resources = normalizeResources(resourceRecord, fallback.resources)

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
    loadedModel: normalizeLoadedModel(loadedModelRecord, fallback.loadedModel),
  }
}

function normalizeGenerateResult(
  value: unknown,
  request: ProGenerateRequest,
): ProGenerateResult {
  const record = asRecord(value)
  const recent = readArray(record, ['recent_outputs', 'recentOutputs', 'outputs', 'images'])
    .map((item, index) => normalizeRecentOutput(item, index, request))
    .filter(isPresent)
  const directOutput =
    normalizeRecentOutput(readUnknown(record, ['output', 'image', 'result']), 0, request) ??
    recent[0] ??
    null

  return {
    jobId: readString(record, ['job_id', 'jobId', 'id'], directOutput?.id ?? 'local-job'),
    status: readString(record, ['status', 'state'], directOutput?.status ?? 'completed'),
    message: readString(record, ['message', 'detail'], directOutput ? 'Generation complete.' : 'Generation submitted.'),
    output: directOutput,
    recentOutputs: recent,
  }
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
    seed: readNumber(record, ['seed'], fallback.seed),
    batchSize: readNumber(record, ['batch_size', 'batchSize'], fallback.batchSize),
  }
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
    backend: readOptionalString(record, ['backend']),
    status: readOptionalString(record, ['status', 'state']),
  }
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
    ['url', 'image_url', 'imageUrl', 'path', 'src', 'file'],
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
    prompt: readString(record, ['prompt'], defaults.prompt),
    width: readNumber(record, ['width'], defaults.width),
    height: readNumber(record, ['height'], defaults.height),
    createdAt: readString(record, ['created_at', 'createdAt', 'time', 'age'], 'now'),
    mode: normalizeCreationMode(readUnknown(record, ['mode']), defaults.mode),
    seed: readOptionalNumber(record, ['seed']),
    modelName: readOptionalString(record, ['model_name', 'modelName', 'model']),
    status: readOptionalString(record, ['status', 'state']),
  }
}

function normalizeResources(
  record: JsonRecord,
  fallback: ResourceMetric[],
): ResourceMetric[] {
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
  const outputsIndex = normalized.toLowerCase().lastIndexOf('/outputs/')
  if (outputsIndex >= 0) {
    return normalized.slice(outputsIndex)
  }
  if (normalized.startsWith('outputs/')) {
    return `/${normalized}`
  }
  return normalized.startsWith('/') ? normalized : `/${normalized}`
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
