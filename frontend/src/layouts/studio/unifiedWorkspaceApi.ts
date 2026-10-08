// Typed client for the same-origin unified workspace routes (/api/pro/unified/*).
// The browser never calls Dataset Studio, ReTrain or Qwen Chat directly; AIWF Pro
// makes those loopback calls server-side (aiwf/web/unified_api.py).
import { API_BASE } from '../../apiBase'

// --- response types (schema_version "1") -------------------------------------------
export type CapabilityState = 'ready' | 'partial' | 'not_running' | 'not_configured' | 'auth_rejected' | 'route_missing' | 'unexpected_response'

export type Capability = {
  available: boolean
  state: CapabilityState | string
  reason: string
  package_listing?: boolean
  catalog_studio_outputs?: boolean
  training_start_available?: boolean
  models?: Array<{ model_id: string; label?: string; family?: string; text_capable?: boolean; local_weights_present?: boolean; loaded?: boolean }>
  workflow_prerequisites_ready?: boolean
  missing_nodes?: string[]
  missing_models?: string[]
}

export type UnifiedStatus = {
  schema_version: '1'
  checked_at: string
  capabilities: { dataset_studio: Capability; retrain: Capability; qwen_chat: Capability; comfyui?: Capability }
}

export type ProjectSummary = { project_id: string; name: string; created_at: string; event_counts: Record<string, number> }
export type ProjectEvent = { event_id: string; at: string; kind: string; [key: string]: unknown }
export type ProjectDetail = ProjectSummary & { events: ProjectEvent[] }

export type DatasetPackage = {
  package_name: string
  status: 'ready' | 'invalid'
  reason?: string
  manifest_sha256?: string
  revision_id?: string
  created_at?: string
  recipe_id?: string
  modality?: string
  counts?: { assets?: number; rows?: number; train?: number; validation?: number }
}

export type CatalogResult = {
  status: 'cataloged'
  collection: { id: number; name: string }
  project_tag: string
  assets: Array<{ asset_id: number; relative_path: string; sha256: string; caption_written: boolean }>
  note: string
}

export type ImportedDataset = {
  dataset_id: string
  package_name: string
  manifest_sha256: string
  revision_id?: string
  counts: { assets?: number; rows?: number; train?: number; validation?: number }
  modality?: string
  reused: boolean
}

export type PreflightResult = {
  status: 'preflight'
  start_enabled: false
  execution_requested: false
  message: string
  result: {
    dataset_id: string
    manifest_sha256: string
    model_id: string
    plan_status: string | null
    // gate details arrive with absolute server paths already replaced by "<server path>"
    gates: Array<{ gate: string; state: string; detail: string }>
    estimate: { fit_state?: string; estimated_gb?: number; limit_gb?: number; headroom_gb?: number; percent?: number; warnings?: string[] }
    dependencies: Array<{ label: string; available: boolean }>
    notes: string[]
    summary: Record<string, unknown>
    training_args: Record<string, unknown>
  }
}

export type QwenAnswer = { model_id: string; answer: string; context_sent: string }
export type ModelFamily = { artifact: string; family: string; produced_by: string; usable_in: string[]; not_usable_in: string[] }

// --- error shape: the backend returns detail {code, message} ---------------------------
export class UnifiedApiError extends Error {
  readonly status: number
  readonly code: string
  constructor(status: number, code: string, message: string) {
    super(message)
    this.status = status
    this.code = code
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}/api/pro/unified${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...(init?.body ? { 'Content-Type': 'application/json' } : {}) },
    })
  } catch {
    throw new UnifiedApiError(0, 'network', 'AIWF Pro did not respond. Is the backend running?')
  }
  let body: unknown
  try {
    body = await response.json()
  } catch {
    body = null
  }
  if (!response.ok) {
    // this block turns FastAPI's {detail} into a readable message, whatever its shape
    const detail = (body as { detail?: unknown } | null)?.detail
    if (detail && typeof detail === 'object' && 'message' in detail) {
      const typed = detail as { code?: string; message?: string }
      throw new UnifiedApiError(response.status, typed.code ?? 'error', typed.message ?? `HTTP ${response.status}`)
    }
    if (response.status === 404 && path === '/status') {
      throw new UnifiedApiError(404, 'bridge_missing', 'This AIWF Pro backend has no unified workspace bridge (HTTP 404).')
    }
    const text = typeof detail === 'string' ? detail : Array.isArray(detail) ? 'The request was rejected by validation.' : `HTTP ${response.status}`
    throw new UnifiedApiError(response.status, 'error', text)
  }
  if (!body || typeof body !== 'object' || (body as { schema_version?: unknown }).schema_version !== '1') {
    throw new UnifiedApiError(response.status, 'schema', 'AIWF Pro returned an unknown unified-workspace schema.')
  }
  return body as T
}

function post<T>(path: string, payload: unknown): Promise<T> {
  return request<T>(path, { method: 'POST', body: JSON.stringify(payload) })
}

// --- route wrappers ---------------------------------------------------------------------
export function fetchUnifiedStatus(): Promise<UnifiedStatus> {
  return request<UnifiedStatus>('/status').then((status) => {
    const caps = status.capabilities
    for (const key of ['dataset_studio', 'retrain', 'qwen_chat'] as const) {
      if (!caps?.[key] || typeof caps[key].available !== 'boolean') {
        throw new UnifiedApiError(200, 'schema', 'AIWF Pro returned an incomplete capability status.')
      }
    }
    return status
  })
}

export const fetchProjects = () => request<{ projects: ProjectSummary[] }>('/projects').then((body) => body.projects)
export const createProject = (name: string) => post<{ project: ProjectDetail }>('/projects', { name }).then((body) => body.project)
export const fetchProject = (projectId: string) => request<{ project: ProjectDetail }>(`/projects/${encodeURIComponent(projectId)}`).then((body) => body.project)
export const fetchPackages = () => request<{ packages: DatasetPackage[] }>('/datasets/packages').then((body) => body.packages)
export const fetchModelFamilies = () => request<{ families: ModelFamily[] }>('/model-families').then((body) => body.families)
export const fetchQwenContext = (projectId: string) => request<{ context: string }>(`/projects/${encodeURIComponent(projectId)}/qwen-context`).then((body) => body.context)

export const catalogOutputs = (projectId: string, outputPaths: string[]) =>
  post<CatalogResult>(`/projects/${encodeURIComponent(projectId)}/catalog-outputs`, { output_paths: outputPaths })

export const importPackage = (projectId: string, pkg: { package_name: string; manifest_sha256: string }) =>
  post<{ dataset: ImportedDataset }>('/retrain/import', { project_id: projectId, package_name: pkg.package_name, manifest_sha256: pkg.manifest_sha256 }).then((body) => body.dataset)

export const runPreflight = (projectId: string, dataset: ImportedDataset, modelId: string, method: string) =>
  post<PreflightResult>('/retrain/preflight', {
    project_id: projectId,
    dataset_id: dataset.dataset_id,
    manifest_sha256: dataset.manifest_sha256,
    model_id: modelId,
    settings: { method },
  })

export const askQwen = (projectId: string, modelId: string, question: string) =>
  post<QwenAnswer>(`/projects/${encodeURIComponent(projectId)}/qwen-ask`, { model_id: modelId, question })

// --- model weights: catalog, Hugging Face key, downloads --------------------------------------
// ReTrain keeps the model folders and runs the downloads; Studio proxies it same-origin.
export type ModelAccessState = 'open' | 'gated_ok' | 'gated_no_key' | 'gated_no_access' | 'unavailable' | 'unknown'

export type ModelEntry = {
  model_id: string
  label: string
  family: string
  size_b: number | null
  hf_repo: string | null
  present: boolean
  status: 'present' | 'missing'
  text_capable: boolean
  download: { job_id: string; status: string } | null
  access?: { state: ModelAccessState; gated: boolean | null; has_key: boolean; download_bytes: number | null; file_count?: number; message?: string }
}

export type ModelDownload = {
  job_id: string
  model_id: string
  repo: string
  status: 'starting' | 'running' | 'completed' | 'failed' | 'cancelled'
  total_bytes: number
  done_bytes: number
  files_total: number
  files_done: number
  message: string
  destination_name: string
}

export const fetchModels = (refresh = false) =>
  request<{ has_key: boolean; models: ModelEntry[] }>(`/models${refresh ? '?refresh=true' : ''}`)
export const startModelDownload = (modelId: string) => post<{ download: ModelDownload }>('/models/download', { model_id: modelId }).then((body) => body.download)
export const fetchModelDownload = (jobId: string) => request<{ download: ModelDownload }>(`/models/downloads/${encodeURIComponent(jobId)}`).then((body) => body.download)
export const cancelModelDownload = (jobId: string) => post<{ download: ModelDownload }>(`/models/downloads/${encodeURIComponent(jobId)}/cancel`, {}).then((body) => body.download)
// the key is sent once, to the local server, and is never kept in the page's state after saving
export const saveHfToken = (token: string) => post<{ ok: boolean; user: string }>('/models/hf-token', { token })
export const clearHfToken = () => post<{ ok: boolean }>('/models/hf-token/clear', {})
