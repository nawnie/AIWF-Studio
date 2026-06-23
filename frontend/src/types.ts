export type CreationMode = 'image' | 'video' | 'inpaint'

export type ProMode = CreationMode | 'models' | 'data'

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
  backend?: string
  status?: string
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
  seed: number
  batchSize: number
}

export interface RecentOutput {
  id: string
  url: string
  thumbnailUrl: string
  prompt: string
  width: number
  height: number
  createdAt: string
  mode: CreationMode
  seed?: number
  modelName?: string
  status?: string
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

export interface ProRuntimeStatus {
  state: string
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
  models: ProModelOption[]
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
}
