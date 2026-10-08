import type { WorkflowCodeBlock } from '../types'
import type { WorkflowRunRequest, WorkflowStepType } from './workflowRunContract.ts'

export interface BlockedWorkflowNode { blockId: string; label: string; reason: string }
export interface PreparedWorkflowRun { request: WorkflowRunRequest | null; blockedNodes: BlockedWorkflowNode[] }

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
}

function packetOf(block: WorkflowCodeBlock): Record<string, unknown> | null {
  return record(record(block.payload)?.packet)
}

function blockToTxt2Img(block: WorkflowCodeBlock): { step: { id: string; type: WorkflowStepType; label: string; params: Record<string, unknown> }; reason?: never } | { step?: never; reason: string } {
  if (block.kind !== 'generation') return { reason: `The ${block.kind} node has no workflow executor mapping yet.` }
  const packet = packetOf(block)
  if (!packet || packet.schema !== 'aiwf.studio-generation-packet.v1') return { reason: 'This saved block does not contain a recognized generation packet.' }
  if (packet.mode !== 'image' || packet.route !== 'image-generate') return { reason: `Route “${String(packet.route ?? 'unknown')}” is not supported by the first workflow adapter.` }
  const imageTools = record(packet.imageTools)
  if (imageTools?.hasInitImage === true) return { reason: 'Image-conditioned generation is not supported by this workflow adapter.' }
  const model = record(packet.model)
  const gate = record(packet.selectionGate)
  const family = packet.family
  if (!model || typeof model.id !== 'string' || !model.id.trim()) return { reason: 'Choose a model with a saved model ID before running this node.' }
  if (gate?.normalSelectable !== true) return { reason: 'The saved model is not marked as selectable for normal generation.' }
  if (!['sd15', 'sdxl', 'sd35'].includes(String(family))) return { reason: `Model family “${String(family ?? 'unknown')}” is not supported by the current generation route.` }
  const prompt = record(packet.prompt)
  const generation = record(packet.generation)
  if (!prompt || !generation || typeof prompt.positive !== 'string' || !prompt.positive.trim()) return { reason: 'The saved generation prompt or settings are incomplete.' }
  if (generation.batchSize !== 1 || generation.batchCount !== 1) return { reason: 'Workflow generation currently supports one image per node (batch size and count must both be 1).' }
  if (record(packet.imageTools)?.enableHires === true) return { reason: 'High-resolution second-pass generation is not supported by this workflow adapter.' }
  const params: Record<string, unknown> = {
    checkpoint_id: model.id,
    prompt: prompt.positive,
    negative_prompt: typeof prompt.negative === 'string' ? prompt.negative : '',
  }
  const mapping: Array<[string, string, 'string' | 'number']> = [
    ['width', 'width', 'number'], ['height', 'height', 'number'], ['steps', 'steps', 'number'],
    ['cfgScale', 'cfg_scale', 'number'], ['sampler', 'sampler', 'string'], ['scheduler', 'scheduler', 'string'],
    ['clipSkip', 'clip_skip', 'number'], ['batchSize', 'batch_size', 'number'], ['batchCount', 'batch_count', 'number'],
  ]
  for (const [source, target, kind] of mapping) {
    const value = generation[source]
    if (kind === 'number' && typeof value === 'number' && Number.isFinite(value)) params[target] = value
    if (kind === 'string' && typeof value === 'string') params[target] = value
  }
  if (typeof prompt.seed === 'number' && Number.isInteger(prompt.seed)) params.seed = prompt.seed
  return { step: { id: block.id, type: 'txt2img', label: block.label, params } }
}

export function prepareWorkflowRun(blocks: WorkflowCodeBlock[], name = 'Saved workflow'): PreparedWorkflowRun {
  const steps: Array<{ id: string; type: WorkflowStepType; label: string; params: Record<string, unknown> }> = []
  const blockedNodes: BlockedWorkflowNode[] = []
  for (const block of blocks) {
    const result = blockToTxt2Img(block)
    if (result.step) steps.push(result.step)
    else blockedNodes.push({ blockId: block.id, label: block.label, reason: result.reason })
  }
  if (steps.length > 1) blockedNodes.push({ blockId: 'workflow-packet-limit', label: 'Workflow', reason: 'This workflow adapter accepts one saved generation packet per run. Run a single generation block at a time.' })
  if (blocks.length > 16) blockedNodes.push({ blockId: 'workflow-limit', label: 'Workflow', reason: 'The workflow service accepts at most 16 nodes per run.' })
  const ids = new Set<string>()
  for (const step of steps) {
    if (ids.has(step.id)) blockedNodes.push({ blockId: step.id, label: step.label, reason: 'Workflow node IDs must be unique.' })
    ids.add(step.id)
  }
  if (steps.length === 0 || blockedNodes.length > 0) return { request: null, blockedNodes }
  return { request: { workflow: { name, steps }, sourceImageDataUrl: null }, blockedNodes }
}
