export type WorkflowStepType =
  | 'txt2img'
  | 'img2img'
  | 'inpaint'
  | 'segment'
  | 'upscale'
  | 'restore'
  | 'enhance'
  | 'photo_restore'

export type WorkflowRunServerStatus =
  | 'queued'
  | 'running'
  | 'cancelling'
  | 'finalizing'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type WorkflowRunStepStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'skipped'

export interface WorkflowDefinitionStep {
  id: string
  type: WorkflowStepType
  label?: string
  params: Record<string, unknown>
}

export interface WorkflowDefinitionRequest {
  name: string
  description?: string
  version?: number
  save_intermediate?: boolean
  steps: WorkflowDefinitionStep[]
}

export interface WorkflowRunRequest {
  workflow: WorkflowDefinitionRequest
  sourceImageDataUrl?: string | null
  idempotencyKey?: string
}

export interface WorkflowRunStepReceipt {
  message?: string
  seed?: number | null
  image_url?: string | null
  infotext_sha256?: string
  error?: string
}

export interface WorkflowRunStepSnapshot {
  stepId: string
  type: WorkflowStepType
  label: string
  status: WorkflowRunStepStatus
  paramsSha256?: string | null
  startedAt: string | null
  completedAt: string | null
  receiptId: string | null
  receipt: WorkflowRunStepReceipt | null
}

export interface WorkflowRunSnapshot {
  runId: string
  workflowName: string
  status: WorkflowRunServerStatus
  createdAt: string
  updatedAt: string
  startedAt: string | null
  completedAt: string | null
  currentStep: { step_id: string; index: number; total: number } | null
  steps: WorkflowRunStepSnapshot[]
  output: { url: string; width?: number; height?: number } | null
  summary: string | null
  error: { code: string; message: string } | null
  recovery: string | null
}

export interface WorkflowRunRecoveryRef {
  schema: 'aiwf.workflow-run-recovery.v1'
  idempotencyKey: string
  workflowName: string
  runId?: string
  pendingRequest?: WorkflowRunRequest
}

export function isWorkflowRunServerStatus(value: unknown): value is WorkflowRunServerStatus {
  return typeof value === 'string' &&
    ['queued', 'running', 'cancelling', 'finalizing', 'completed', 'failed', 'cancelled'].includes(value)
}

export function isWorkflowRunStepType(value: unknown): value is WorkflowStepType {
  return typeof value === 'string' &&
    ['txt2img', 'img2img', 'inpaint', 'segment', 'upscale', 'restore', 'enhance', 'photo_restore'].includes(value)
}

function isWorkflowRunStepStatus(value: unknown): value is WorkflowRunStepStatus {
  return typeof value === 'string' && ['queued', 'running', 'completed', 'failed', 'cancelled', 'skipped'].includes(value)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === 'string'
}

function parseStepReceipt(value: unknown): WorkflowRunStepReceipt | null {
  if (value === null) return null
  if (!isRecord(value)) return null
  if (value.message !== undefined && typeof value.message !== 'string') return null
  if (value.seed !== undefined && value.seed !== null && typeof value.seed !== 'number') return null
  if (value.image_url !== undefined && !isNullableString(value.image_url)) return null
  if (value.infotext_sha256 !== undefined && typeof value.infotext_sha256 !== 'string') return null
  if (value.error !== undefined && typeof value.error !== 'string') return null
  return {
    ...(typeof value.message === 'string' ? { message: value.message } : {}),
    ...(typeof value.seed === 'number' || value.seed === null ? { seed: value.seed } : {}),
    ...(typeof value.image_url === 'string' || value.image_url === null ? { image_url: value.image_url } : {}),
    ...(typeof value.infotext_sha256 === 'string' ? { infotext_sha256: value.infotext_sha256 } : {}),
    ...(typeof value.error === 'string' ? { error: value.error } : {}),
  }
}

function parseStep(value: unknown): WorkflowRunStepSnapshot | null {
  if (!isRecord(value) || typeof value.stepId !== 'string' || !isWorkflowRunStepType(value.type)) return null
  if (typeof value.label !== 'string' || !isWorkflowRunStepStatus(value.status)) return null
  if (!isNullableString(value.startedAt) || !isNullableString(value.completedAt)) return null
  if (value.paramsSha256 !== undefined && !isNullableString(value.paramsSha256)) return null
  if (value.receiptId !== undefined && !isNullableString(value.receiptId)) return null
  const receipt = value.receipt === undefined ? null : parseStepReceipt(value.receipt)
  if (value.receipt !== undefined && value.receipt !== null && receipt === null) return null
  return {
    stepId: value.stepId,
    type: value.type,
    label: value.label,
    status: value.status,
    startedAt: value.startedAt,
    completedAt: value.completedAt,
    receiptId: typeof value.receiptId === 'string' ? value.receiptId : null,
    receipt,
    ...(typeof value.paramsSha256 === 'string' || value.paramsSha256 === null ? { paramsSha256: value.paramsSha256 } : {}),
  }
}

export function parseWorkflowRunSnapshot(value: unknown): WorkflowRunSnapshot | null {
  if (!isRecord(value)) return null
  if (typeof value.runId !== 'string' || !/^[0-9a-f]{32}$/i.test(value.runId)) return null
  if (typeof value.workflowName !== 'string' || !isWorkflowRunServerStatus(value.status)) return null
  if (typeof value.createdAt !== 'string' || typeof value.updatedAt !== 'string') return null
  if (!isNullableString(value.startedAt) || !isNullableString(value.completedAt)) return null
  if (!Array.isArray(value.steps)) return null
  const steps = value.steps.map(parseStep)
  if (steps.some((step) => step === null)) return null
  let currentStep: WorkflowRunSnapshot['currentStep'] = null
  if (value.currentStep !== null) {
    if (!isRecord(value.currentStep) || typeof value.currentStep.step_id !== 'string') return null
    if (typeof value.currentStep.index !== 'number' || !Number.isInteger(value.currentStep.index) || typeof value.currentStep.total !== 'number' || !Number.isInteger(value.currentStep.total)) return null
    currentStep = {
      step_id: value.currentStep.step_id,
      index: value.currentStep.index,
      total: value.currentStep.total,
    }
  }
  let output: WorkflowRunSnapshot['output'] = null
  if (value.output !== null) {
    if (!isRecord(value.output) || typeof value.output.url !== 'string') return null
    if (value.output.width !== undefined && typeof value.output.width !== 'number') return null
    if (value.output.height !== undefined && typeof value.output.height !== 'number') return null
    output = {
      url: value.output.url,
      ...(typeof value.output.width === 'number' ? { width: value.output.width } : {}),
      ...(typeof value.output.height === 'number' ? { height: value.output.height } : {}),
    }
  }
  let error: WorkflowRunSnapshot['error'] = null
  if (value.error !== null) {
    if (!isRecord(value.error) || typeof value.error.code !== 'string' || typeof value.error.message !== 'string') return null
    error = { code: value.error.code, message: value.error.message }
  }
  if (!isNullableString(value.summary) || !isNullableString(value.recovery)) return null
  return {
    runId: value.runId,
    workflowName: value.workflowName,
    status: value.status,
    createdAt: value.createdAt,
    updatedAt: value.updatedAt,
    startedAt: value.startedAt,
    completedAt: value.completedAt,
    currentStep,
    steps: steps.filter((step): step is WorkflowRunStepSnapshot => step !== null),
    output,
    summary: value.summary,
    error,
    recovery: value.recovery,
  }
}

export function parseWorkflowRunRequest(value: unknown): WorkflowRunRequest | null {
  if (!isRecord(value) || !isRecord(value.workflow)) return null
  const workflow = value.workflow
  if (typeof workflow.name !== 'string' || !Array.isArray(workflow.steps) || workflow.steps.length < 1 || workflow.steps.length > 16) return null
  const ids = new Set<string>()
  const steps: WorkflowDefinitionStep[] = []
  for (const stepValue of workflow.steps) {
    if (!isRecord(stepValue) || typeof stepValue.id !== 'string' || !isWorkflowRunStepType(stepValue.type)) return null
    if (ids.has(stepValue.id) || !isRecord(stepValue.params)) return null
    ids.add(stepValue.id)
    if (stepValue.label !== undefined && typeof stepValue.label !== 'string') return null
    steps.push({
      id: stepValue.id,
      type: stepValue.type,
      params: stepValue.params,
      ...(typeof stepValue.label === 'string' ? { label: stepValue.label } : {}),
    })
  }
  if (workflow.description !== undefined && typeof workflow.description !== 'string') return null
  if (workflow.version !== undefined && typeof workflow.version !== 'number') return null
  if (workflow.save_intermediate !== undefined && typeof workflow.save_intermediate !== 'boolean') return null
  if (value.sourceImageDataUrl !== undefined && !isNullableString(value.sourceImageDataUrl)) return null
  if (value.idempotencyKey !== undefined && (typeof value.idempotencyKey !== 'string' || value.idempotencyKey.length > 128)) return null
  return {
    workflow: {
      name: workflow.name,
      steps,
      ...(typeof workflow.description === 'string' ? { description: workflow.description } : {}),
      ...(typeof workflow.version === 'number' ? { version: workflow.version } : {}),
      ...(typeof workflow.save_intermediate === 'boolean' ? { save_intermediate: workflow.save_intermediate } : {}),
    },
    ...(typeof value.sourceImageDataUrl === 'string' || value.sourceImageDataUrl === null ? { sourceImageDataUrl: value.sourceImageDataUrl } : {}),
    ...(typeof value.idempotencyKey === 'string' ? { idempotencyKey: value.idempotencyKey } : {}),
  }
}

export function parseWorkflowRunRecoveryRef(value: unknown): WorkflowRunRecoveryRef | null {
  if (!isRecord(value) || value.schema !== 'aiwf.workflow-run-recovery.v1') return null
  if (typeof value.idempotencyKey !== 'string' || value.idempotencyKey.length > 128 || typeof value.workflowName !== 'string') return null
  if (value.runId !== undefined && (typeof value.runId !== 'string' || !/^[0-9a-f]{32}$/i.test(value.runId))) return null
  let pendingRequest: WorkflowRunRequest | undefined
  if (value.pendingRequest !== undefined) {
    pendingRequest = parseWorkflowRunRequest(value.pendingRequest) ?? undefined
    if (!pendingRequest || pendingRequest.idempotencyKey !== value.idempotencyKey || value.runId !== undefined) return null
  }
  if (value.runId === undefined && !pendingRequest) return null
  return {
    schema: 'aiwf.workflow-run-recovery.v1',
    idempotencyKey: value.idempotencyKey,
    workflowName: value.workflowName,
    ...(typeof value.runId === 'string' ? { runId: value.runId } : {}),
    ...(pendingRequest ? { pendingRequest } : {}),
  }
}
