import { useCallback, useRef, useState } from 'react'
import { GripVertical, ArrowUp, ArrowDown, X, Copy, Play, Trash2 } from 'lucide-react'
import type { WorkflowCodeBlock } from '../types'
import { duplicateWorkflowBlock, renumberWorkflowBlocks, validateWorkflowBlocks } from './workflowBlocks'

interface WorkflowPanelProps {
  blocks: WorkflowCodeBlock[]
  onChange: (next: WorkflowCodeBlock[]) => void
  onRun?: (blocks: WorkflowCodeBlock[]) => void
  runLabel?: string
  runStatus?: string
}

const KIND_LABEL: Record<WorkflowCodeBlock['kind'], string> = {
  generation: 'Generate',
  workflow: 'Workflow',
  qa: 'QA',
  export: 'Export',
}

// A manually reorderable list of workflow nodes. Reorder by dragging a node,
// or with the up/down arrows (keyboard-friendly fallback). Order is 1-based and
// re-numbered after every change so the sequence is always contiguous.
export function WorkflowPanel({ blocks, onChange, onRun, runLabel = 'Run workflow', runStatus = '' }: WorkflowPanelProps) {
  const [dragId, setDragId] = useState<string>('')
  const [overId, setOverId] = useState<string>('')
  const dragIndexRef = useRef<number>(-1)

  const move = useCallback(
    (from: number, to: number) => {
      if (to < 0 || to >= blocks.length || from === to) {
        return
      }
      const next = blocks.slice()
      const [item] = next.splice(from, 1)
      next.splice(to, 0, item)
      onChange(renumberWorkflowBlocks(next))
    },
    [blocks, onChange],
  )

  const remove = useCallback(
    (id: string) => onChange(renumberWorkflowBlocks(blocks.filter((b) => b.id !== id))),
    [blocks, onChange],
  )

  const duplicate = useCallback(
    (block: WorkflowCodeBlock, index: number) => {
      const next = blocks.slice()
      next.splice(index + 1, 0, duplicateWorkflowBlock(block, block.order + 1))
      onChange(renumberWorkflowBlocks(next))
    },
    [blocks, onChange],
  )

  const validation = validateWorkflowBlocks(blocks)

  return (
    <section className="pro-workflow-panel" aria-label="Workflow nodes">
      <header className="pro-workflow-panel-head">
        <div>
          <strong>Workflow</strong>
          <span>{blocks.length === 0 ? 'Empty — send settings here from any generation tab.' : `${blocks.length} node${blocks.length === 1 ? '' : 's'}, top to bottom.`}</span>
        </div>
        <div className="pro-workflow-panel-actions">
          {onRun ? (
            <button
              type="button"
              className="pro-primary-button"
              onClick={() => onRun(blocks)}
              disabled={blocks.length === 0 || !validation.valid}
            >
              <Play size={15} aria-hidden="true" />
              <span>{runLabel}</span>
            </button>
          ) : null}
          <button
            type="button"
            className="pro-secondary-button ghost"
            onClick={() => onChange([])}
            disabled={blocks.length === 0}
            title="Remove all nodes"
          >
            <Trash2 size={14} aria-hidden="true" />
            <span>Clear</span>
          </button>
        </div>
      </header>

      {!validation.valid && validation.errors.length > 0 ? (
        <div className="pro-workflow-validation">{validation.errors.join(' · ')}</div>
      ) : null}

      <ol className="pro-workflow-node-list">
        {blocks.map((block, index) => (
          <li
            key={block.id}
            className={`pro-workflow-node${dragId === block.id ? ' is-dragging' : ''}${overId === block.id ? ' is-over' : ''}`}
            draggable
            onDragStart={() => {
              setDragId(block.id)
              dragIndexRef.current = index
            }}
            onDragOver={(event) => {
              event.preventDefault()
              if (overId !== block.id) {
                setOverId(block.id)
              }
            }}
            onDrop={(event) => {
              event.preventDefault()
              if (dragIndexRef.current >= 0) {
                move(dragIndexRef.current, index)
              }
              setDragId('')
              setOverId('')
              dragIndexRef.current = -1
            }}
            onDragEnd={() => {
              setDragId('')
              setOverId('')
              dragIndexRef.current = -1
            }}
          >
            <span className="pro-workflow-node-grip" title="Drag to reorder" aria-hidden="true">
              <GripVertical size={16} />
            </span>
            <span className="pro-workflow-node-order">{block.order}</span>
            <div className="pro-workflow-node-body">
              <div className="pro-workflow-node-title">
                <strong>{block.label || KIND_LABEL[block.kind]}</strong>
                <small>{KIND_LABEL[block.kind]} · {block.source}</small>
              </div>
              {block.summary ? <p className="pro-workflow-node-summary">{block.summary}</p> : null}
            </div>
            <div className="pro-workflow-node-controls">
              <button type="button" title="Move up" aria-label="Move up" disabled={index === 0} onClick={() => move(index, index - 1)}>
                <ArrowUp size={14} aria-hidden="true" />
              </button>
              <button type="button" title="Move down" aria-label="Move down" disabled={index === blocks.length - 1} onClick={() => move(index, index + 1)}>
                <ArrowDown size={14} aria-hidden="true" />
              </button>
              <button type="button" title="Duplicate" aria-label="Duplicate" onClick={() => duplicate(block, index)}>
                <Copy size={14} aria-hidden="true" />
              </button>
              <button type="button" title="Remove" aria-label="Remove" onClick={() => remove(block.id)}>
                <X size={14} aria-hidden="true" />
              </button>
            </div>
          </li>
        ))}
      </ol>

      {runStatus ? <p className="pro-field-note">{runStatus}</p> : null}
    </section>
  )
}
