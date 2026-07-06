import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { DragEvent as ReactDragEvent, ChangeEvent as ReactChangeEvent } from 'react'
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  Boxes,
  CheckCircle2,
  Clipboard,
  Copy,
  Download,
  FileJson,
  GitBranch,
  GripVertical,
  ListChecks,
  Plus,
  Save,
  Settings2,
  Trash2,
  Upload,
} from 'lucide-react'
import type { LayoutProps, WorkflowCodeBlock } from './LayoutTypes'
import { displayDate } from './LayoutTypes'
import { addQueueJob, fetchWorkflowTemplates, saveWorkflow, validateWorkflow } from './studioApiClient'
import type { WorkflowTemplate } from './studioApiClient'
import {
  createWorkflowBlocksFromSettings,
  duplicateWorkflowBlock,
  normalizeWorkflowBlocks,
  renumberWorkflowBlocks,
  validateWorkflowBlocks,
  workflowPayloadFromBlocks,
} from './workflowBlocks'
import './studioLayouts.css'

function routeFromBlock(block: WorkflowCodeBlock): string {
  const route = block.payload.route
  return typeof route === 'string' && route.trim() ? route : block.nodeId
}

function familyFromBlock(block: WorkflowCodeBlock): string {
  const family = block.payload.family
  if (typeof family === 'string' && family.trim()) {
    return family
  }
  const model = block.payload.model
  if (model && typeof model === 'object') {
    const row = model as Record<string, unknown>
    return String(row.engineLabel || row.engineId || row.architecture || row.name || 'Model family')
  }
  return 'Model family'
}

function precisionFromBlock(block: WorkflowCodeBlock): string {
  const precision = block.payload.precision
  return typeof precision === 'string' && precision.trim() ? precision : 'auto'
}

function blockCodeLineCount(block: WorkflowCodeBlock): number {
  return Math.max(1, block.code.split('\n').length)
}

function makeTemplateBlock(template: WorkflowTemplate, order: number): WorkflowCodeBlock {
  const payload = {
    schema: 'aiwf.workflow-template-reference.v1',
    templateId: template.id,
    label: template.label,
    summary: template.summary,
    stages: template.stages,
    capturedAt: new Date().toISOString(),
    note: 'Template reference block. Expand into concrete generation blocks when settings are chosen.',
  }
  return {
    id: `template-block-${template.id}-${Date.now()}`,
    label: template.label,
    kind: 'workflow',
    nodeId: 'template-reference',
    source: 'Template Library',
    createdAt: new Date().toISOString(),
    summary: template.summary,
    order,
    classes: { requires: [], produces: ['artifact'] },
    payload,
    code: JSON.stringify(payload, null, 2),
  }
}

function fallbackImportBlock(payload: Record<string, unknown>, order: number): WorkflowCodeBlock {
  const code = JSON.stringify(payload, null, 2)
  return {
    id: `imported-block-${Date.now()}`,
    label: String(payload.label || payload.id || 'Imported workflow JSON'),
    kind: 'workflow',
    nodeId: 'imported-json',
    source: 'Imported JSON',
    createdAt: new Date().toISOString(),
    summary: 'Raw workflow JSON preserved as a movable code block.',
    order,
    classes: { requires: [], produces: ['artifact'] },
    payload,
    code,
  }
}

export function PipelineAtlasLayout({
  settings,
  bootstrap,
  runtime,
  selectedModel,
  selectedModelName,
  statusMessage,
  isGenerating,
  workflowBlocks,
  onWorkflowBlocksChange,
  onOpenModels,
  onOpenSettings,
}: LayoutProps) {
  const [internalBlocks, setInternalBlocks] = useState<WorkflowCodeBlock[]>([])
  const [selectedBlockId, setSelectedBlockId] = useState('')
  const [draggingBlockId, setDraggingBlockId] = useState<string | null>(null)
  const [localMessage, setLocalMessage] = useState('Linear workflow mode: capture settings as self-contained code blocks, then drag to reorder the queue.')
  const [workflowTemplates, setWorkflowTemplates] = useState<WorkflowTemplate[]>([])
  const [backendErrors, setBackendErrors] = useState<string[]>([])
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const blocks = workflowBlocks ?? internalBlocks
  const setBlocks = onWorkflowBlocksChange ?? setInternalBlocks

  const commitBlocks = useCallback((updater: WorkflowCodeBlock[] | ((current: WorkflowCodeBlock[]) => WorkflowCodeBlock[])) => {
    setBlocks((current) => {
      const next = typeof updater === 'function' ? updater(current) : updater
      return renumberWorkflowBlocks(next)
    })
  }, [setBlocks])

  useEffect(() => {
    fetchWorkflowTemplates().then(setWorkflowTemplates).catch(() => undefined)
  }, [])

  const selectedBlock = useMemo(
    () => blocks.find((block) => block.id === selectedBlockId) ?? blocks[0],
    [blocks, selectedBlockId],
  )

  useEffect(() => {
    if (!selectedBlock && blocks[0]) {
      setSelectedBlockId(blocks[0].id)
    }
  }, [blocks, selectedBlock])

  const localValidation = useMemo(() => validateWorkflowBlocks(blocks), [blocks])
  const totalLines = useMemo(() => blocks.reduce((total, block) => total + blockCodeLineCount(block), 0), [blocks])
  const routeSummary = useMemo(() => {
    const routes = Array.from(new Set(blocks.map(routeFromBlock)))
    return routes.length ? routes.slice(0, 4).join(' · ') : 'No captured routes yet'
  }, [blocks])

  const addCurrentSettingsBlock = () => {
    const newBlocks = createWorkflowBlocksFromSettings({
      settings,
      bootstrap,
      runtime,
      selectedModel,
      selectedModelName,
      source: 'Pipeline Atlas',
    }, blocks.length)
    commitBlocks((current) => [...current, ...newBlocks])
    setSelectedBlockId(newBlocks[0]?.id ?? '')
    setLocalMessage('Captured the current generate settings as a workflow code block. No render was started.')
  }

  const addTemplateBlock = (template: WorkflowTemplate) => {
    const block = makeTemplateBlock(template, blocks.length + 1)
    commitBlocks((current) => [...current, block])
    setSelectedBlockId(block.id)
    setLocalMessage(`${template.label} added as a template reference block.`)
  }

  const removeBlock = (blockId: string) => {
    commitBlocks((current) => current.filter((block) => block.id !== blockId))
    setSelectedBlockId((current) => (current === blockId ? '' : current))
    setLocalMessage('Removed workflow block and renumbered the queue.')
  }

  const duplicateBlockById = (blockId: string) => {
    const source = blocks.find((block) => block.id === blockId)
    if (!source) return
    const copy = duplicateWorkflowBlock(source, source.order + 1)
    commitBlocks((current) => {
      const index = current.findIndex((block) => block.id === blockId)
      const insertion = index >= 0 ? index + 1 : current.length
      return [...current.slice(0, insertion), copy, ...current.slice(insertion)]
    })
    setSelectedBlockId(copy.id)
    setLocalMessage(`${source.label} duplicated for adapter or precision testing.`)
  }

  const moveBlock = (blockId: string, direction: -1 | 1) => {
    commitBlocks((current) => {
      const index = current.findIndex((block) => block.id === blockId)
      const target = index + direction
      if (index < 0 || target < 0 || target >= current.length) return current
      const next = [...current]
      const [item] = next.splice(index, 1)
      next.splice(target, 0, item)
      return next
    })
    setLocalMessage('Workflow queue order updated. Blocks remain self-contained.')
  }

  const dropBlockOn = (targetId: string) => {
    if (!draggingBlockId || draggingBlockId === targetId) {
      setDraggingBlockId(null)
      return
    }
    commitBlocks((current) => {
      const sourceIndex = current.findIndex((block) => block.id === draggingBlockId)
      const targetIndex = current.findIndex((block) => block.id === targetId)
      if (sourceIndex < 0 || targetIndex < 0) return current
      const next = [...current]
      const [item] = next.splice(sourceIndex, 1)
      next.splice(targetIndex, 0, item)
      return next
    })
    setDraggingBlockId(null)
    setLocalMessage('Drag/drop reorder complete. The JSON payloads were not mutated.')
  }

  const updateBlockCode = (blockId: string, code: string) => {
    commitBlocks((current) => current.map((block) => {
      if (block.id !== blockId) return block
      try {
        const payload = JSON.parse(code) as Record<string, unknown>
        return { ...block, code, payload, summary: String(payload.route || block.summary) }
      } catch {
        return { ...block, code }
      }
    }))
  }

  const copyBlockCode = async (block: WorkflowCodeBlock) => {
    try {
      await navigator.clipboard.writeText(block.code)
      setLocalMessage(`${block.label} JSON copied to clipboard.`)
    } catch {
      setLocalMessage('Clipboard unavailable. The JSON is still visible in the code block.')
    }
  }

  const workflowPayload = () => workflowPayloadFromBlocks(blocks)

  const saveWorkflowJson = async () => {
    const validation = validateWorkflowBlocks(blocks)
    if (!validation.valid) {
      setBackendErrors(validation.errors)
      setLocalMessage('Local workflow QA found invalid code blocks before save.')
      return
    }
    const payload = workflowPayload()
    const backendValidation = await validateWorkflow(payload)
    setBackendErrors(backendValidation.errors || [])
    if (!backendValidation.valid) {
      setLocalMessage(`Backend workflow QA found issues: ${(backendValidation.errors || []).join('; ')}`)
      return
    }
    await saveWorkflow('main', payload).catch(() => undefined)
    setLocalMessage('Workflow code-block queue saved. No pipeline execution was started.')
  }

  const queueWorkflowPlan = async () => {
    const validation = validateWorkflowBlocks(blocks)
    if (!blocks.length) {
      setLocalMessage('Capture at least one code block before queueing a workflow plan.')
      return
    }
    if (!validation.valid) {
      setBackendErrors(validation.errors)
      setLocalMessage('Queue blocked by local JSON QA.')
      return
    }
    const payload = workflowPayload()
    const backendValidation = await validateWorkflow(payload)
    setBackendErrors(backendValidation.errors || [])
    if (!backendValidation.valid) {
      setLocalMessage(`Queue blocked by backend QA: ${(backendValidation.errors || []).join('; ')}`)
      return
    }
    await saveWorkflow('main', payload).catch(() => undefined)
    const job = await addQueueJob('Workflow code block plan', payload, 'workflow-plan')
    setLocalMessage(job ? `Queued ${job.label} as validation-only plan.` : 'Saved locally. Queue API unavailable, so nothing was run.')
  }

  const exportWorkflowJson = () => {
    const payload = workflowPayload()
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `aiwf-workflow-code-blocks-${Date.now()}.json`
    anchor.click()
    URL.revokeObjectURL(url)
    setLocalMessage('Workflow JSON exported as linear code blocks.')
  }

  const loadWorkflowJson = (file: File | undefined) => {
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      try {
        const payload = JSON.parse(String(reader.result || '{}')) as Record<string, unknown>
        const loaded = normalizeWorkflowBlocks(payload)
        const nextBlocks = loaded.length ? loaded : [fallbackImportBlock(payload, blocks.length + 1)]
        commitBlocks(nextBlocks)
        setSelectedBlockId(nextBlocks[0]?.id ?? '')
        setLocalMessage(`Loaded ${nextBlocks.length} workflow code block${nextBlocks.length === 1 ? '' : 's'}.`)
      } catch (error) {
        setLocalMessage(error instanceof Error ? error.message : 'Could not load workflow JSON.')
      }
    }
    reader.readAsText(file)
  }

  const clearBlocks = () => {
    commitBlocks([])
    setSelectedBlockId('')
    setBackendErrors([])
    setLocalMessage('Workflow queue cleared. Capture settings again when ready.')
  }

  const status = localValidation.valid && !backendErrors.length ? 'Ready' : 'Needs QA'

  return (
    <div className="studio-atlas studio-full-surface studio-atlas-linear" aria-label="Pipeline Atlas linear workflow code-block queue">
      <aside className="studio-atlas-sidebar studio-linear-sidebar">
        <div className="studio-product-lockup">
          <span className="studio-logo-orb">A</span>
          <div>
            <strong>AIWF Studio</strong>
            <small>Pipeline Atlas · Linear Queue</small>
          </div>
        </div>
        <button className="studio-wide-button" type="button" onClick={addCurrentSettingsBlock}>
          <Plus size={15} /> Capture Current Settings
        </button>
        <div className="studio-system-card">
          <span>Queue Status</span>
          <strong>{status}</strong>
          <small>{blocks.length} blocks · {totalLines} JSON lines</small>
        </div>
        <div className="studio-template-card">
          <span>Template References</span>
          {workflowTemplates.slice(0, 6).map((template) => (
            <button type="button" key={template.id} onClick={() => addTemplateBlock(template)}>
              <strong>{template.label}</strong>
              <small>{template.summary}</small>
            </button>
          ))}
        </div>
        <div className="studio-system-card muted">
          <span>Runtime</span>
          <strong>{runtime.state || 'Idle'}</strong>
          <small>{runtime.device}</small>
          <small>{runtime.backend} · {runtime.precision}</small>
        </div>
      </aside>

      <main className="studio-atlas-main studio-linear-main">
        <header className="studio-atlas-toolbar studio-linear-toolbar">
          <div>
            <span className="studio-eyebrow">WORKFLOW CODE BLOCKS</span>
            <strong>{routeSummary}</strong>
            <small>{localMessage} · {statusMessage}</small>
          </div>
          <div className="studio-toolbar-actions">
            <button type="button" onClick={saveWorkflowJson}><Save size={14} /> Save</button>
            <button type="button" onClick={queueWorkflowPlan}><ListChecks size={14} /> Queue Plan</button>
            <button type="button" onClick={exportWorkflowJson}><Download size={14} /> Export JSON</button>
            <button type="button" onClick={() => fileInputRef.current?.click()}><Upload size={14} /> Load JSON</button>
            <input
              ref={fileInputRef}
              type="file"
              accept="application/json,.json"
              className="studio-hidden-input"
              onChange={(event: ReactChangeEvent<HTMLInputElement>) => {
                loadWorkflowJson(event.target.files?.[0])
                event.target.value = ''
              }}
            />
            <button type="button" onClick={onOpenModels}>Models</button>
            <button type="button" onClick={onOpenSettings} aria-label="Settings"><Settings2 size={16} /></button>
          </div>
        </header>

        <section className="studio-linear-queue-shell">
          <div className="studio-linear-queue-header">
            <div>
              <GitBranch size={18} />
              <strong>Drag/drop queue</strong>
              <span>Order changes queue priority only. Each block keeps its captured settings, models, route, sidecars, and adapters.</span>
            </div>
            <button type="button" onClick={clearBlocks} disabled={!blocks.length}><Trash2 size={14} /> Clear</button>
          </div>

          {blocks.length ? (
            <div className="studio-linear-block-list">
              {blocks.map((block) => {
                const selected = selectedBlock?.id === block.id
                const blockValidation = validateWorkflowBlocks([block])
                return (
                  <article
                    key={block.id}
                    className={`studio-linear-block ${selected ? 'selected' : ''} ${blockValidation.valid ? '' : 'blocked'}`}
                    draggable
                    onDragStart={() => setDraggingBlockId(block.id)}
                    onDragOver={(event: ReactDragEvent<HTMLElement>) => event.preventDefault()}
                    onDrop={() => dropBlockOn(block.id)}
                    onClick={() => setSelectedBlockId(block.id)}
                  >
                    <div className="studio-linear-grip" aria-hidden="true"><GripVertical size={18} /></div>
                    <div className="studio-linear-block-index"><span>{block.order}</span></div>
                    <div className="studio-linear-block-body">
                      <header>
                        <div>
                          <strong>{block.label}</strong>
                          <small>{block.summary}</small>
                        </div>
                        <em>{block.kind}</em>
                      </header>
                      <div className="studio-linear-block-tags">
                        <span>{routeFromBlock(block)}</span>
                        <span>{familyFromBlock(block)}</span>
                        <span>{precisionFromBlock(block)}</span>
                        <span>{blockCodeLineCount(block)} lines</span>
                      </div>
                      <pre>{block.code.split('\n').slice(0, 10).join('\n')}{blockCodeLineCount(block) > 10 ? '\n  …' : ''}</pre>
                    </div>
                    <div className="studio-linear-block-actions">
                      <button type="button" title="Move up" onClick={(event) => { event.stopPropagation(); moveBlock(block.id, -1) }}><ArrowUp size={14} /></button>
                      <button type="button" title="Move down" onClick={(event) => { event.stopPropagation(); moveBlock(block.id, 1) }}><ArrowDown size={14} /></button>
                      <button type="button" title="Duplicate" onClick={(event) => { event.stopPropagation(); duplicateBlockById(block.id) }}><Copy size={14} /></button>
                      <button type="button" title="Copy JSON" onClick={(event) => { event.stopPropagation(); void copyBlockCode(block) }}><Clipboard size={14} /></button>
                      <button type="button" title="Remove" onClick={(event) => { event.stopPropagation(); removeBlock(block.id) }}><Trash2 size={14} /></button>
                    </div>
                  </article>
                )
              })}
            </div>
          ) : (
            <div className="studio-linear-empty">
              <Boxes size={44} />
              <strong>No workflow blocks yet</strong>
              <p>Use <b>Send to workflow</b> beside Generate, or capture the current settings here. The block stores the model, precision guess, prompt, dimensions, video settings, and Wan sidecar hooks as JSON.</p>
              <button type="button" className="studio-wide-button" onClick={addCurrentSettingsBlock}><Plus size={15} /> Capture Current Settings</button>
            </div>
          )}
        </section>
      </main>

      <aside className="studio-atlas-inspector studio-linear-inspector">
        <header>
          <div>
            <FileJson size={18} />
            <strong>{selectedBlock?.label ?? 'Workflow QA'}</strong>
          </div>
          {selectedBlock ? <button type="button" onClick={() => removeBlock(selectedBlock.id)}>Remove</button> : null}
        </header>

        {selectedBlock ? (
          <>
            <section>
              <span className="studio-eyebrow">Captured Route</span>
              <div className="studio-summary-list">
                <span>Order <strong>{selectedBlock.order}</strong></span>
                <span>Route <strong>{routeFromBlock(selectedBlock)}</strong></span>
                <span>Family <strong>{familyFromBlock(selectedBlock)}</strong></span>
                <span>Precision <strong>{precisionFromBlock(selectedBlock)}</strong></span>
                <span>Source <strong>{selectedBlock.source}</strong></span>
                <span>Captured <strong>{displayDate(selectedBlock.createdAt)}</strong></span>
              </div>
            </section>
            <section>
              <span className="studio-eyebrow">Editable Code Block</span>
              <textarea
                className="studio-code-textarea"
                spellCheck={false}
                value={selectedBlock.code}
                onChange={(event) => updateBlockCode(selectedBlock.id, event.target.value)}
              />
            </section>
            <section>
              <span className="studio-eyebrow">QA Result</span>
              {validateWorkflowBlocks([selectedBlock]).valid ? (
                <p className="studio-ok-note"><CheckCircle2 size={15} /> JSON parses and the block has the required code payload.</p>
              ) : (
                <p className="studio-error-note"><AlertTriangle size={15} /> {validateWorkflowBlocks([selectedBlock]).errors.join('; ')}</p>
              )}
              {backendErrors.length ? <p className="studio-error-note"><AlertTriangle size={15} /> {backendErrors.join('; ')}</p> : null}
              <button type="button" className="studio-wide-button" onClick={() => void copyBlockCode(selectedBlock)}><Clipboard size={14} /> Copy Block JSON</button>
            </section>
          </>
        ) : (
          <section>
            <span className="studio-eyebrow">QA Notes</span>
            <p>Nothing selected. Capture a block to inspect the exact model and settings payload.</p>
          </section>
        )}

        <section>
          <span className="studio-eyebrow">Current Selection</span>
          <div className="studio-summary-list">
            <span>Model <strong>{selectedModelName}</strong></span>
            <span>Mode <strong>{settings.mode}</strong></span>
            <span>Size <strong>{settings.width}×{settings.height}</strong></span>
            <span>Steps <strong>{settings.steps}</strong></span>
            <span>Generator <strong>{isGenerating ? 'busy' : 'idle'}</strong></span>
          </div>
        </section>
      </aside>
    </div>
  )
}
