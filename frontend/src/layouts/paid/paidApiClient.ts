import type { GenerationSettings, RecentOutput } from '../../types'
import type { PaidUserTab } from './PaidLayoutTypes'

export interface PaidAgentModel { id: string; name: string; size?: string; modifiedAt?: string }
export interface PaidAgentTool { id: string; label: string; group: string; status: string; description: string }
export interface PaidAgentMessage { role: 'system' | 'user' | 'assistant' | 'tool'; content: string }
export interface PaidProjectSummary { id: string; name: string; updatedAt?: string; sceneCount?: number }
export interface PaidProject { id: string; name: string; updatedAt?: string; scenes?: PaidScene[]; tracks?: PaidTrack[]; versions?: PaidVersion[]; promptStudio?: PaidPromptStudio; characters?: PaidCharacter[]; ui?: Record<string, unknown> }
export interface PaidScene { id: string; title: string; status?: string; assetIds?: string[]; notes?: string }
export interface PaidTrack { id: string; label: string; type: string; items?: unknown[] }
export interface PaidVersion { id: string; label?: string; createdAt?: string; parentId?: string; assetPath?: string; summary?: string }
export interface PaidPromptStudio { subject?: string; style?: string; camera?: string; lighting?: string; world?: string; negative?: string; assembledPrompt?: string }
export interface PaidCharacter { id: string; name: string; description?: string; reference?: string; notes?: string }
export interface PaidNodeDefinition { id: string; label: string; group: string; requires: string[]; produces: string[]; compatibleEngines?: string[] }
export interface PaidWorkflowTemplate { id: string; label: string; summary: string; stages: string[] }
export interface PaidWorkflowValidation { valid: boolean; errors: string[]; availableClasses?: string[]; stages?: Array<Record<string, unknown>> }
export interface PaidQueueJob { id: string; kind: string; label: string; status: string; progress?: number; createdAt?: string; updatedAt?: string; payload?: Record<string, unknown> }
export interface PaidAsset { id: string; kind: string; name: string; path: string; url?: string; sizeBytes?: number; modifiedAt?: string; tags?: string[]; projectId?: string }
export interface PaidReceipt { id: string; name: string; path: string; kind: string; sizeBytes?: number; modifiedAt?: string }
export interface PaidExportPreset { id: string; label: string; outputs: string[]; summary: string }
export interface PaidPlugin { id: string; name: string; version: string; enabled?: boolean; permissions?: string[]; installedAt?: string; manifest?: Record<string, unknown> }
export interface PaidAgentPermissions { observe: boolean; suggest: boolean; draft: boolean; executeWithApproval: boolean; trustedLocal: boolean; allowedTools: string[] }
export interface PaidQaCheck { id: string; status: string; label: string; suggestion: string }

export interface PaidModelFamilyPrecision { name: string; status: string; loader: string; notes?: string }
export interface PaidModelFamilyRoute { id: string; status: string; kind: string; entrypoint: string; notes?: string }
export interface PaidModelFamily {
  id: string
  label: string
  category: string
  status: string
  summary: string
  storage: string[]
  precisions: PaidModelFamilyPrecision[]
  routes: PaidModelFamilyRoute[]
  sidecars: string[]
  lora: string
  blockers: string[]
  modules: string[]
  localReadiness?: Record<string, number>
  localDetectedPrecisions?: Record<string, number>
}
export interface PaidModelFamilyMatrix {
  schema: string
  generatedAt: string
  source: string
  precisionVocabulary: string[]
  readiness?: { recordCount?: number; countsByFamily?: Record<string, Record<string, number>>; precisionByFamily?: Record<string, Record<string, number>>; error?: string }
  blockedExamples?: Array<{ family: string; status: string; path: string; route: string; reason: string; suggestedAction: string }>
  families: PaidModelFamily[]
}

const USER_TAB_STORAGE_KEY = 'aiwf.paid.userTabs.v4'
const PROJECT_AUTOSAVE_STORAGE_KEY = 'aiwf.paid.projectAutosave.v4'
const API_BASE = (import.meta.env.VITE_AIWF_API_BASE ?? '').replace(/\/$/, '')

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'content-type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const payload = await response.json()
      detail = typeof payload?.detail === 'string' ? payload.detail : JSON.stringify(payload)
    } catch {
      // Keep the HTTP status text.
    }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

function normalizeTabs(value: unknown): PaidUserTab[] {
  const rows = Array.isArray(value) ? value : Array.isArray((value as { tabs?: unknown })?.tabs) ? (value as { tabs: unknown[] }).tabs : []
  return rows
    .filter((item): item is Record<string, unknown> => item !== null && typeof item === 'object')
    .map((item) => ({
      id: String(item.id || item.label || `workspace-${Date.now()}`),
      label: String(item.label || item.id || 'Custom Workspace'),
      icon: String(item.icon || 'plugin'),
      color: String(item.color || '#8b5cf6'),
      hidden: Boolean(item.hidden),
      workspaceType: String(item.workspaceType || item.workspace_type || 'empty') as PaidUserTab['workspaceType'],
      description: String(item.description || ''),
    }))
}

function fallbackProject(): PaidProject {
  return {
    id: 'default',
    name: 'AIWF Media Project',
    updatedAt: new Date().toISOString(),
    scenes: [
      { id: 'scene-01', title: 'Opening image', status: 'draft', notes: 'Start with a hero still.' },
      { id: 'scene-02', title: 'Motion pass', status: 'planned', notes: 'Send selected image to video.' },
    ],
    tracks: [
      { id: 'video', label: 'Video', type: 'video', items: [] },
      { id: 'image', label: 'Image', type: 'image', items: [] },
      { id: 'audio', label: 'Audio', type: 'audio', items: [] },
      { id: 'metadata', label: 'Receipts', type: 'metadata', items: [] },
    ],
    versions: [],
    promptStudio: {},
    characters: [],
    ui: {},
  }
}

export function loadPaidUserTabsFromStorage(): PaidUserTab[] {
  try {
    const raw = window.localStorage.getItem(USER_TAB_STORAGE_KEY)
    return raw ? normalizeTabs(JSON.parse(raw)) : []
  } catch {
    return []
  }
}

export function savePaidUserTabsToStorage(tabs: PaidUserTab[]): void {
  try { window.localStorage.setItem(USER_TAB_STORAGE_KEY, JSON.stringify(tabs)) } catch { /* local convenience only */ }
}

export async function fetchPaidUserTabs(): Promise<PaidUserTab[]> {
  try { return normalizeTabs(await requestJson('/api/pro/extensions/tabs')) } catch { return loadPaidUserTabsFromStorage() }
}

export async function savePaidUserTabs(tabs: PaidUserTab[]): Promise<PaidUserTab[]> {
  savePaidUserTabsToStorage(tabs)
  try {
    const payload = await requestJson<{ tabs?: PaidUserTab[] }>('/api/pro/extensions/tabs', { method: 'POST', body: JSON.stringify({ tabs }) })
    return normalizeTabs(payload)
  } catch { return tabs }
}

export async function fetchPaidAgentModels(): Promise<PaidAgentModel[]> {
  const payload = await requestJson<{ models?: unknown[] }>('/api/pro/agent/ollama/models')
  return (Array.isArray(payload.models) ? payload.models : []).map((item) => {
    const row = (item ?? {}) as Record<string, unknown>
    return { id: String(row.id || row.name || row.model || 'local-model'), name: String(row.name || row.model || row.id || 'Local Ollama model'), size: row.size ? String(row.size) : '', modifiedAt: row.modifiedAt ? String(row.modifiedAt) : '' }
  })
}

export async function fetchPaidAgentTools(): Promise<PaidAgentTool[]> {
  const payload = await requestJson<{ tools?: unknown[] }>('/api/pro/agent/tools')
  return (Array.isArray(payload.tools) ? payload.tools : []).map((item) => {
    const row = (item ?? {}) as Record<string, unknown>
    return { id: String(row.id || row.label || 'tool'), label: String(row.label || row.id || 'Tool'), group: String(row.group || 'Tools'), status: String(row.status || 'available'), description: String(row.description || '') }
  })
}

export async function sendPaidAgentChat(model: string, messages: PaidAgentMessage[], enabledTools: string[]): Promise<string> {
  const payload = await requestJson<{ message?: { content?: string }; response?: string; error?: string }>('/api/pro/agent/chat', { method: 'POST', body: JSON.stringify({ model, messages, enabledTools }) })
  return payload.message?.content || payload.response || payload.error || 'No response returned.'
}

export async function streamPaidAgentChat(
  model: string,
  messages: PaidAgentMessage[],
  enabledTools: string[],
  onDelta: (text: string) => void,
): Promise<string> {
  const response = await fetch('/api/pro/agent/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, messages, enabledTools }),
  })
  if (!response.ok || !response.body) {
    // Fall back to the non-streaming endpoint on failure
    return sendPaidAgentChat(model, messages, enabledTools)
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let full = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed) continue
      try {
        const chunk = JSON.parse(trimmed) as { message?: { content?: string }; error?: string; done?: boolean }
        if (chunk.error) { full += `\n[${chunk.error}]`; onDelta(full); continue }
        const delta = chunk.message?.content ?? ''
        if (delta) { full += delta; onDelta(full) }
      } catch { /* ignore malformed chunk */ }
    }
  }
  return full || 'No response returned.'
}

export interface PaidWorkerStatus { running: boolean; currentJobId?: string | null; completed: number; failed: number; registeredKinds: string[] }

export async function fetchPaidWorkerStatus(): Promise<PaidWorkerStatus | null> {
  try {
    const payload = await requestJson<{ worker?: PaidWorkerStatus }>('/api/pro/queue/worker')
    return payload.worker ?? null
  } catch { return null }
}

export async function controlPaidWorker(action: 'start' | 'stop' | 'run-next'): Promise<PaidWorkerStatus | null> {
  try {
    const payload = await requestJson<{ worker?: PaidWorkerStatus }>(`/api/pro/queue/worker/${action}`, { method: 'POST', body: JSON.stringify({}) })
    return payload.worker ?? null
  } catch { return null }
}

export async function fetchPaidJobLog(jobId: string): Promise<string> {
  try {
    const payload = await requestJson<{ log?: string }>(`/api/pro/queue/jobs/${encodeURIComponent(jobId)}/log`)
    return payload.log ?? ''
  } catch { return '' }
}

export interface PaidContextEntry { name: string; path: string; type: 'dir' | 'file'; sizeBytes: number; readable: boolean }

export async function fetchPaidContextTree(root = 'data', path = ''): Promise<PaidContextEntry[]> {
  try {
    const payload = await requestJson<{ entries?: PaidContextEntry[] }>(`/api/pro/agent/context/tree?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`)
    return payload.entries ?? []
  } catch { return [] }
}

export async function fetchPaidContextFile(root: string, path: string): Promise<string> {
  const payload = await requestJson<{ content?: string }>(`/api/pro/agent/context/file?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`)
  return payload.content ?? ''
}

export async function fetchPaidAgentPermissions(): Promise<PaidAgentPermissions> {
  try {
    return await requestJson<PaidAgentPermissions>('/api/pro/agent/permissions')
  } catch {
    return { observe: true, suggest: true, draft: true, executeWithApproval: false, trustedLocal: false, allowedTools: ['project-reader', 'workflow-json', 'prompt-refiner'] }
  }
}

export async function savePaidAgentPermissions(value: PaidAgentPermissions): Promise<PaidAgentPermissions> {
  return requestJson<PaidAgentPermissions>('/api/pro/agent/permissions', { method: 'POST', body: JSON.stringify(value) })
}

export async function fetchPaidProjects(): Promise<PaidProjectSummary[]> {
  try {
    const payload = await requestJson<{ projects?: PaidProjectSummary[] }>('/api/pro/projects')
    return Array.isArray(payload.projects) ? payload.projects : []
  } catch { return [{ id: 'default', name: 'AIWF Media Project', updatedAt: new Date().toISOString(), sceneCount: 2 }] }
}

export async function fetchPaidProject(projectId = 'default'): Promise<PaidProject> {
  try { return await requestJson<PaidProject>(`/api/pro/projects/${encodeURIComponent(projectId)}`) } catch { return fallbackProject() }
}

export async function savePaidProject(project: PaidProject): Promise<PaidProject> {
  try {
    return await requestJson<PaidProject>(`/api/pro/projects/${encodeURIComponent(project.id || 'default')}`, { method: 'POST', body: JSON.stringify(project) })
  } catch {
    return project
  }
}

export async function autosavePaidProject(payload: Record<string, unknown>): Promise<void> {
  try { window.localStorage.setItem(PROJECT_AUTOSAVE_STORAGE_KEY, JSON.stringify({ savedAt: new Date().toISOString(), payload })) } catch { /* best effort */ }
  try { await requestJson('/api/pro/projects/default/autosave', { method: 'POST', body: JSON.stringify(payload) }) } catch { /* backend may still be booting */ }
}

export async function fetchPaidNodeRegistry(): Promise<{ classes: string[]; nodes: PaidNodeDefinition[] }> {
  try { return await requestJson('/api/pro/workflows/node-registry') } catch { return { classes: ['prompt', 'image', 'mask', 'video', 'audio', 'metadata', 'artifact'], nodes: [] } }
}

export async function fetchPaidWorkflowTemplates(): Promise<PaidWorkflowTemplate[]> {
  try {
    const payload = await requestJson<{ templates?: PaidWorkflowTemplate[] }>('/api/pro/workflows/templates')
    return Array.isArray(payload.templates) ? payload.templates : []
  } catch { return [] }
}

export async function validatePaidWorkflow(workflow: Record<string, unknown>): Promise<PaidWorkflowValidation> {
  try { return await requestJson<PaidWorkflowValidation>('/api/pro/workflows/validate', { method: 'POST', body: JSON.stringify({ workflow }) }) } catch { return { valid: true, errors: [] } }
}

export async function savePaidWorkflow(workflowId: string, workflow: Record<string, unknown>): Promise<Record<string, unknown>> {
  return requestJson(`/api/pro/workflows/${encodeURIComponent(workflowId)}`, { method: 'POST', body: JSON.stringify(workflow) })
}

export async function fetchPaidWorkflow(workflowId = 'main'): Promise<Record<string, unknown>> {
  return requestJson(`/api/pro/workflows/${encodeURIComponent(workflowId)}`)
}

export async function fetchPaidQueue(): Promise<PaidQueueJob[]> {
  try {
    const payload = await requestJson<{ jobs?: PaidQueueJob[] }>('/api/pro/queue')
    return Array.isArray(payload.jobs) ? payload.jobs : []
  } catch { return [] }
}

export async function addPaidQueueJob(label: string, payload: Record<string, unknown>, kind = 'workflow'): Promise<PaidQueueJob | null> {
  try {
    const result = await requestJson<{ job?: PaidQueueJob }>('/api/pro/queue/jobs', { method: 'POST', body: JSON.stringify({ kind, label, payload }) })
    return result.job ?? null
  } catch { return null }
}

export async function updatePaidQueueJob(jobId: string, action: string): Promise<PaidQueueJob[]> {
  try {
    const result = await requestJson<{ queue?: { jobs?: PaidQueueJob[] } }>(`/api/pro/queue/jobs/${encodeURIComponent(jobId)}/${encodeURIComponent(action)}`, { method: 'POST', body: JSON.stringify({}) })
    return result.queue?.jobs ?? []
  } catch { return [] }
}

export async function fetchPaidAssets(): Promise<PaidAsset[]> {
  try {
    const payload = await requestJson<{ assets?: PaidAsset[] }>('/api/pro/assets/library')
    return Array.isArray(payload.assets) ? payload.assets : []
  } catch { return [] }
}

export async function fetchPaidReceipts(): Promise<PaidReceipt[]> {
  try {
    const payload = await requestJson<{ receipts?: PaidReceipt[] }>('/api/pro/receipts')
    return Array.isArray(payload.receipts) ? payload.receipts : []
  } catch { return [] }
}

export async function addPaidVersion(projectId: string, version: Partial<PaidVersion>): Promise<PaidVersion[]> {
  try {
    const payload = await requestJson<{ versions?: PaidVersion[] }>(`/api/pro/projects/${encodeURIComponent(projectId)}/versions`, { method: 'POST', body: JSON.stringify(version) })
    return payload.versions ?? []
  } catch { return [] }
}

export async function analyzePaidQa(prompt: string, workflow: Record<string, unknown>, assetPath = ''): Promise<{ score: number; checks: PaidQaCheck[] }> {
  try { return await requestJson('/api/pro/qa/analyze', { method: 'POST', body: JSON.stringify({ projectId: 'default', prompt, workflow, assetPath }) }) } catch { return { score: 72, checks: [{ id: 'offline', status: 'info', label: 'QA backend unavailable', suggestion: 'Run again after backend finishes starting.' }] } }
}

export async function fetchPaidExportPresets(): Promise<PaidExportPreset[]> {
  try {
    const payload = await requestJson<{ presets?: PaidExportPreset[] }>('/api/pro/export/presets')
    return Array.isArray(payload.presets) ? payload.presets : []
  } catch { return [] }
}

export async function createPaidExportPlan(preset: string, assetIds: string[], settings: Record<string, unknown>): Promise<Record<string, unknown>> {
  return requestJson('/api/pro/export/plan', { method: 'POST', body: JSON.stringify({ preset, projectId: 'default', assetIds, settings }) })
}

export async function fetchPaidPlugins(): Promise<PaidPlugin[]> {
  try {
    const payload = await requestJson<{ plugins?: PaidPlugin[] }>('/api/pro/plugins/registry')
    return Array.isArray(payload.plugins) ? payload.plugins : []
  } catch { return [] }
}

export async function installPaidPluginManifest(manifest: Record<string, unknown>): Promise<PaidPlugin | null> {
  try {
    const payload = await requestJson<{ plugin?: PaidPlugin }>('/api/pro/plugins/install-manifest', { method: 'POST', body: JSON.stringify({ manifest }) })
    return payload.plugin ?? null
  } catch { return null }
}

export function buildAutosavePayload(settings: GenerationSettings, activeRail: string, recentOutputs: RecentOutput[]): Record<string, unknown> {
  return {
    activeRail,
    settings: {
      mode: settings.mode,
      prompt: settings.prompt,
      negativePrompt: settings.negativePrompt,
      modelId: settings.modelId,
      width: settings.width,
      height: settings.height,
      steps: settings.steps,
      cfgScale: settings.cfgScale,
      sampler: settings.sampler,
      scheduler: settings.scheduler,
      seed: settings.seed,
    },
    recentOutputIds: recentOutputs.slice(0, 12).map((item) => item.id),
  }
}

export function fallbackPaidModelFamilyMatrix(): PaidModelFamilyMatrix {
  return {
    schema: 'aiwf.model-family-support.v1',
    generatedAt: new Date().toISOString(),
    source: 'offline fallback; backend /api/pro/model-families unavailable',
    precisionVocabulary: ['FP32', 'FP16', 'BF16', 'FP8', 'INT8', 'NF4', 'FP4', 'NVFP4', 'INT4', 'Q4_K_M', 'Q5_K_M', 'Q8_0'],
    families: [
      { id: 'flux', label: 'Flux / Flux Fill', category: 'image', status: 'supported-experimental-quants', summary: 'Flux single-transformer route with shared text encoders, VAE, GGUF and 4-bit safetensors loaders.', storage: ['.safetensors', '.gguf'], precisions: [{ name: 'BF16', status: 'preferred', loader: 'FluxTransformer2DModel' }, { name: 'FP8', status: 'partial', loader: '--fluxfp8 / patched converter' }, { name: 'NF4/FP4', status: 'supported', loader: 'bitsandbytes 4-bit loader' }], routes: [{ id: 'flux', status: 'supported', kind: 'txt2img', entrypoint: 'DiffusersBackend._load_flux_checkpoint' }], sidecars: ['CLIP-L', 'T5-XXL', 'ae.safetensors'], lora: 'Base Flux runtime LoRA supported.', blockers: ['Known bad FP8/GGUF-NF4 assets stay blocked.'], modules: ['aiwf.infrastructure.diffusers.backend'] },
      { id: 'wan', label: 'Wan Video', category: 'video', status: 'supported-plus-sidecars', summary: 'Wan fast 5B and high/low sidecar-aware routes.', storage: ['Diffusers folder', '.safetensors', '.gguf'], precisions: [{ name: 'BF16/FP16', status: 'supported', loader: 'WanI2VBackend' }, { name: 'FP8', status: 'experimental', loader: 'Comfy FP8/native path' }, { name: 'GGUF Q4/Q5', status: 'experimental', loader: 'Wan GGUF runtime' }], routes: [{ id: 'wan-fast-5b', status: 'supported', kind: 'i2v', entrypoint: 'WanService.generate' }], sidecars: ['high model', 'low model', 'VAE', 'UMT5', 'LoRA', 'offload plan'], lora: 'Single high/low LoRA fields exist; stack support is next.', blockers: ['T2V, Animate, Fun-Control need dedicated routes.'], modules: ['aiwf.services.wan'] },
    ],
  }
}

export async function fetchPaidModelFamilies(): Promise<PaidModelFamilyMatrix> {
  try {
    return await requestJson<PaidModelFamilyMatrix>('/api/pro/model-families')
  } catch {
    return fallbackPaidModelFamilyMatrix()
  }
}
