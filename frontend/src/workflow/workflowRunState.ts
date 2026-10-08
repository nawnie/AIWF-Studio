import type { WorkflowRunSnapshot } from './workflowRunContract.ts'

export interface WorkflowRunState {
  snapshot: WorkflowRunSnapshot | null
  requestKey: string | null
  submitting: boolean
  localError: string | null
}

export const EMPTY_WORKFLOW_RUN_STATE: WorkflowRunState = {
  snapshot: null,
  requestKey: null,
  submitting: false,
  localError: null,
}

export type WorkflowRunAction =
  | { type: 'submit'; requestKey: string }
  | { type: 'snapshot'; requestKey: string; snapshot: WorkflowRunSnapshot }
  | { type: 'localError'; requestKey: string; message: string }
  | { type: 'restore'; snapshot: WorkflowRunSnapshot | null }
  | { type: 'clear' }

export function isWorkflowRunActive(snapshot: WorkflowRunSnapshot | null): boolean {
  return Boolean(snapshot && ['queued', 'running', 'cancelling', 'finalizing'].includes(snapshot.status))
}

function isTerminal(status: WorkflowRunSnapshot['status']): boolean {
  return ['completed', 'failed', 'cancelled'].includes(status)
}

export function compareWorkflowTimestamps(left: string, right: string): number {
  const parts = (value: string) => {
    const match = value.match(/^(.+?)(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})$/)
    if (!match) return { base: Date.parse(value), fraction: '' }
    return { base: Date.parse(`${match[1]}${match[3]}`), fraction: (match[2] ?? '').padEnd(9, '0').slice(0, 9) }
  }
  const a = parts(left)
  const b = parts(right)
  if (a.base !== b.base) return a.base < b.base ? -1 : 1
  return a.fraction === b.fraction ? 0 : a.fraction < b.fraction ? -1 : 1
}

function isNotNewer(current: string, incoming: string): boolean {
  return compareWorkflowTimestamps(current, incoming) >= 0
}

export function reduceWorkflowRunState(state: WorkflowRunState, action: WorkflowRunAction): WorkflowRunState {
  if (action.type === 'clear') return EMPTY_WORKFLOW_RUN_STATE
  if (action.type === 'restore') return { ...EMPTY_WORKFLOW_RUN_STATE, snapshot: action.snapshot }
  if (action.type === 'submit') {
    if (state.submitting || isWorkflowRunActive(state.snapshot)) return state
    return { ...state, requestKey: action.requestKey, submitting: true, localError: null }
  }
  if (state.requestKey !== action.requestKey) return state
  if (action.type === 'localError') return { ...state, submitting: false, localError: action.message }
  if (action.type === 'snapshot') {
    const current = state.snapshot
    // A delayed poll must never move a run backwards after a newer snapshot arrived.
    if (current?.runId === action.snapshot.runId && (isTerminal(current.status) || isNotNewer(current.updatedAt, action.snapshot.updatedAt))) return state
    return { snapshot: action.snapshot, requestKey: action.requestKey, submitting: false, localError: null }
  }
  return state
}

export const WORKFLOW_RUN_RECOVERY_KEY = 'aiwf.workflowRunRecovery.v1'

export function saveRunId(runId: string): void {
  try { window.localStorage.setItem(WORKFLOW_RUN_RECOVERY_KEY, JSON.stringify({ runId })) } catch { /* storage is optional */ }
}

export function readRunId(): string | null {
  try {
    const value: unknown = JSON.parse(window.localStorage.getItem(WORKFLOW_RUN_RECOVERY_KEY) ?? 'null')
    if (value && typeof value === 'object' && 'runId' in value && typeof value.runId === 'string' && /^[0-9a-f]{32}$/i.test(value.runId)) return value.runId
  } catch { /* malformed or unavailable storage */ }
  return null
}

export function clearRunId(): void {
  try { window.localStorage.removeItem(WORKFLOW_RUN_RECOVERY_KEY) } catch { /* storage is optional */ }
}
