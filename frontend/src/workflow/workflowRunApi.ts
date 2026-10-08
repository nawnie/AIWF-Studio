import { parseWorkflowRunSnapshot, type WorkflowRunRequest, type WorkflowRunSnapshot } from './workflowRunContract.ts'

export type WorkflowFetch = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

export class WorkflowRunHttpError extends Error {
  readonly status: number
  constructor(status: number, message: string) { super(message); this.status = status; this.name = 'WorkflowRunHttpError' }
}

async function snapshotResponse(response: Response): Promise<WorkflowRunSnapshot> {
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = body && typeof body === 'object' && 'detail' in body && typeof body.detail === 'string' ? body.detail : `Workflow request failed (${response.status}).`
    throw new WorkflowRunHttpError(response.status, detail)
  }
  const snapshot = parseWorkflowRunSnapshot(body)
  if (!snapshot) throw new Error('The workflow service returned an invalid run snapshot.')
  return snapshot
}

export function createWorkflowRunApi(fetcher: WorkflowFetch = fetch, apiBase = '') {
  const url = (path: string) => `${apiBase}${path}`
  return {
    async submit(request: WorkflowRunRequest, idempotencyKey: string): Promise<WorkflowRunSnapshot> {
      return snapshotResponse(await fetcher(url('/api/pro/workflows/runs'), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...request, idempotencyKey }),
      }))
    },
    async get(runId: string): Promise<WorkflowRunSnapshot> {
      if (!/^[0-9a-f]{32}$/i.test(runId)) return Promise.reject(new Error('Invalid workflow run ID.'))
      return snapshotResponse(await fetcher(url(`/api/pro/workflows/runs/${runId}`), { method: 'GET' }))
    },
    async cancel(runId: string): Promise<WorkflowRunSnapshot> {
      if (!/^[0-9a-f]{32}$/i.test(runId)) return Promise.reject(new Error('Invalid workflow run ID.'))
      return snapshotResponse(await fetcher(url(`/api/pro/workflows/runs/${runId}/cancel`), { method: 'POST' }))
    },
  }
}
