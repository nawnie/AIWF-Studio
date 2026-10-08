import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import type { WorkflowCodeBlock } from '../types'
import { safeWorkflowHref } from './workflowRunLinks'
import { createWorkflowRunApi, WorkflowRunHttpError } from './workflowRunApi'
import { parseWorkflowRunRecoveryRef, type WorkflowRunRecoveryRef, type WorkflowRunSnapshot } from './workflowRunContract'
import { prepareWorkflowRun } from './workflowRunAdapter'
import { clearRunId, compareWorkflowTimestamps, EMPTY_WORKFLOW_RUN_STATE, isWorkflowRunActive, readRunId, reduceWorkflowRunState, saveRunId, WORKFLOW_RUN_RECOVERY_KEY } from './workflowRunState'
import { API_BASE } from '../apiBase'

function newKey(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`
}

function readRecovery(): WorkflowRunRecoveryRef | null {
  try { return parseWorkflowRunRecoveryRef(JSON.parse(window.localStorage.getItem(WORKFLOW_RUN_RECOVERY_KEY) ?? 'null')) } catch { return null }
}

function storeRecovery(ref: WorkflowRunRecoveryRef): void {
  try { window.localStorage.setItem(WORKFLOW_RUN_RECOVERY_KEY, JSON.stringify(ref)) } catch { /* storage is optional */ }
}

export function WorkflowRunPanel({ blocks }: { blocks: WorkflowCodeBlock[] }) {
  const [state, dispatch] = useReducer(reduceWorkflowRunState, EMPTY_WORKFLOW_RUN_STATE)
  const api = useRef(createWorkflowRunApi(fetch, API_BASE))
  const requestKeyRef = useRef<string | null>(null)
  const mountRequestKeyRef = useRef<string | null>(null)
  if (mountRequestKeyRef.current === null) mountRequestKeyRef.current = newKey()
  const acceptedSnapshotRef = useRef<WorkflowRunSnapshot | null>(null)
  const [pendingCancelRunIds, setPendingCancelRunIds] = useState<Set<string>>(() => new Set())
  const mountedRef = useRef(false)
  const prepared = prepareWorkflowRun(blocks)
  const active = isWorkflowRunActive(state.snapshot)
  const retryRef = readRecovery()
  const recoverableRequest = retryRef?.pendingRequest
  const runBlocked = (!prepared.request && !recoverableRequest) || state.submitting || active

  const accept = useCallback((requestKey: string, snapshot: WorkflowRunSnapshot) => {
    if (!mountedRef.current || requestKeyRef.current !== requestKey) return
    const prior = acceptedSnapshotRef.current
    if (prior?.runId === snapshot.runId) {
      if (['completed', 'failed', 'cancelled'].includes(prior.status) || compareWorkflowTimestamps(prior.updatedAt, snapshot.updatedAt) >= 0) return
    }
    acceptedSnapshotRef.current = snapshot
    saveRunId(snapshot.runId)
    const existing = readRecovery()
    storeRecovery({ schema: 'aiwf.workflow-run-recovery.v1', idempotencyKey: existing?.idempotencyKey ?? requestKey, workflowName: snapshot.workflowName, runId: snapshot.runId })
    dispatch({ type: 'snapshot', requestKey, snapshot })
    if (!isWorkflowRunActive(snapshot)) clearRunId()
  }, [])

  useEffect(() => {
    mountedRef.current = true
    const requestKey = mountRequestKeyRef.current!
    requestKeyRef.current = requestKey
    const recovery = readRecovery()
    const knownRunId = recovery?.runId ?? readRunId()
    if (!recovery && !knownRunId) return () => { mountedRef.current = false }
    dispatch({ type: 'submit', requestKey })
    const reconcile = async () => {
      try {
        if (knownRunId) {
          accept(requestKey, await api.current.get(knownRunId))
          return
        }
        if (recovery?.pendingRequest) {
          const accepted = await api.current.submit(recovery.pendingRequest, recovery.idempotencyKey)
          accept(requestKey, accepted)
        }
      } catch (error) {
        if (mountedRef.current && requestKeyRef.current === requestKey) {
          if (!knownRunId && error instanceof WorkflowRunHttpError && error.status >= 400 && error.status < 500) {
            try { window.localStorage.removeItem(WORKFLOW_RUN_RECOVERY_KEY) } catch { /* storage is optional */ }
          }
          dispatch({ type: 'localError', requestKey, message: error instanceof Error ? error.message : 'Could not recover workflow run.' })
        }
      }
    }
    void reconcile()
    return () => { mountedRef.current = false }
  }, [accept])

  useEffect(() => {
    const snapshot = state.snapshot
    if (!snapshot || !isWorkflowRunActive(snapshot)) return
    const requestKey = state.requestKey
    if (!requestKey) return
    const timer = window.setInterval(() => {
      void api.current.get(snapshot.runId).then((fresh) => accept(requestKey, fresh)).catch((error: unknown) => {
        const latest = acceptedSnapshotRef.current
        // A delayed failed poll must not add an error after a newer response
        // already established that this run is terminal.
        if (mountedRef.current && requestKeyRef.current === requestKey && latest?.runId === snapshot.runId && !['completed', 'failed', 'cancelled'].includes(latest.status)) {
          dispatch({ type: 'localError', requestKey, message: error instanceof Error ? error.message : 'Could not refresh workflow status.' })
        }
      })
    }, 1200)
    return () => window.clearInterval(timer)
  }, [state.requestKey, state.snapshot, accept])

  const run = async () => {
    if ((!prepared.request && !recoverableRequest) || runBlocked) return
    const pending = readRecovery()
    const existingRequest = pending?.pendingRequest
    const idempotencyKey = existingRequest?.idempotencyKey ?? newKey()
    const requestKey = idempotencyKey
    requestKeyRef.current = requestKey
    dispatch({ type: 'submit', requestKey })
    const request = { ...(existingRequest ?? prepared.request!), idempotencyKey }
    storeRecovery({ schema: 'aiwf.workflow-run-recovery.v1', idempotencyKey, workflowName: request.workflow.name, pendingRequest: request })
    try { accept(requestKey, await api.current.submit(request, idempotencyKey)) }
    catch (error) {
      if (mountedRef.current && requestKeyRef.current === requestKey) {
        if (error instanceof WorkflowRunHttpError && error.status >= 400 && error.status < 500) {
          try { window.localStorage.removeItem(WORKFLOW_RUN_RECOVERY_KEY) } catch { /* storage is optional */ }
        }
        dispatch({ type: 'localError', requestKey, message: error instanceof Error ? error.message : 'Could not submit workflow.' })
      }
    }
  }

  const cancel = async () => {
    const snapshot = state.snapshot
    const requestKey = state.requestKey
    if (!snapshot || pendingCancelRunIds.has(snapshot.runId) || !['queued', 'running'].includes(snapshot.status) || !requestKey) return
    setPendingCancelRunIds((current) => new Set(current).add(snapshot.runId))
    try { accept(requestKey, await api.current.cancel(snapshot.runId)) }
    catch (error) {
      const latest = acceptedSnapshotRef.current
      const terminal = latest?.runId === snapshot.runId && ['completed', 'failed', 'cancelled'].includes(latest.status)
      if (mountedRef.current && requestKeyRef.current === requestKey && !terminal) {
        dispatch({ type: 'localError', requestKey, message: error instanceof Error ? error.message : 'Could not cancel workflow.' })
      }
    }
    finally {
      setPendingCancelRunIds((current) => {
        const next = new Set(current)
        next.delete(snapshot.runId)
        return next
      })
    }
  }

  const snapshot = state.snapshot
  const countDone = snapshot?.steps.filter((step) => ['completed', 'failed', 'cancelled', 'skipped'].includes(step.status)).length ?? 0
  const current = snapshot?.currentStep
  const outputHref = snapshot?.output?.url ? safeWorkflowHref(snapshot.output.url, API_BASE) : null
  const canCancel = snapshot && ['queued', 'running'].includes(snapshot.status)
  const cancelPending = Boolean(snapshot && pendingCancelRunIds.has(snapshot.runId))
  const priorCancelPending = Array.from(pendingCancelRunIds).some((runId) => runId !== snapshot?.runId)
  return (
    <section className="pro-workflow-run-panel" aria-label="Workflow run" aria-busy={cancelPending}>
      <div className="pro-workflow-panel-actions">
        <button type="button" className="pro-primary-button" onClick={() => void run()} disabled={runBlocked}>{recoverableRequest ? 'Retry pending submission' : 'Run supported workflow'}</button>
        {canCancel ? <button type="button" className="pro-secondary-button" onClick={() => void cancel()} disabled={cancelPending}>{cancelPending ? 'Cancelling…' : 'Cancel run'}</button> : null}
      </div>
      {prepared.blockedNodes.length ? <ul className="pro-workflow-validation" aria-label="Blocked workflow nodes">{prepared.blockedNodes.map((node, index) => <li key={`${node.blockId}:${node.reason}:${index}`}><strong>{node.label}:</strong> {node.reason}</li>)}</ul> : null}
      {!prepared.request && !prepared.blockedNodes.length ? <p className="pro-field-note">Add at least one supported generation node to run.</p> : null}
      {state.localError ? <p className="pro-workflow-run-error" role="alert">{state.localError}</p> : null}
      {state.submitting ? <p role="status">Submitting workflow…</p> : null}
      {cancelPending && !canCancel ? <p role="status">Cancel request still pending.</p> : null}
      {priorCancelPending ? <p role="status">A previous workflow cancellation request is still pending.</p> : null}
      {snapshot ? <div className="pro-workflow-run-status" aria-live="polite" aria-atomic="true">
        <p><strong>Status:</strong> <span data-testid="workflow-run-status">{snapshot.status}</span></p>
        <p><strong>Run ID:</strong> <code>{snapshot.runId}</code></p>
        {current ? <div className="pro-workflow-run-progress"><p>Step {current.index} of {current.total}: {snapshot.steps.find((step) => step.stepId === current.step_id)?.label ?? current.step_id}</p><progress aria-label="Workflow progress" max={snapshot.steps.length || 1} value={countDone} /></div> : null}
        {snapshot.status === 'finalizing' ? <p>All steps finished. Saving the final output; cancellation is no longer available.</p> : null}
        {snapshot.recovery === 'interrupted_without_replay' ? <p role="alert">This run was interrupted by a service restart. It was not replayed.</p> : null}
        {snapshot.summary ? <p>{snapshot.summary}</p> : null}
        {snapshot.error ? <p className="pro-workflow-run-error" role="alert">{snapshot.error.code}: {snapshot.error.message}</p> : null}
        <ol aria-label="Workflow step receipts">{snapshot.steps.map((step) => {
          const receiptImage = step.receipt?.image_url ? safeWorkflowHref(step.receipt.image_url, API_BASE) : null
          return <li key={step.stepId}><strong>{step.label}</strong>: {step.status}{step.receiptId ? ` · receipt ${step.receiptId}` : ''}{step.receipt?.message ? ` — ${step.receipt.message}` : ''}{step.receipt?.seed !== undefined && step.receipt.seed !== null ? ` (seed ${step.receipt.seed})` : ''}{receiptImage ? <> · <a href={receiptImage}>Step output</a></> : null}</li>
        })}</ol>
        {outputHref ? <p><a href={outputHref}>Open workflow output</a>{snapshot.output?.width && snapshot.output.height ? ` (${snapshot.output.width} × ${snapshot.output.height})` : ''}</p> : null}
      </div> : <p className="pro-field-note">No workflow run has been started.</p>}
    </section>
  )
}
