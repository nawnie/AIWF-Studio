import type { GenerationSettings, ProBootstrap, ProModelOption, ProRuntimeStatus } from '../../types'
import type { WorkflowCodeBlock } from './LayoutTypes'

export const WORKFLOW_BLOCK_STORAGE_KEY = 'aiwf.workflowCodeBlocks.v12'

interface WorkflowBlockSource {
  settings: GenerationSettings
  bootstrap: ProBootstrap
  runtime: ProRuntimeStatus
  selectedModel: ProModelOption | undefined
  selectedModelName: string
  source: string
}

interface WorkflowBlockValidation {
  valid: boolean
  errors: string[]
}

function nowIso(): string {
  return new Date().toISOString()
}

function nextId(prefix: string): string {
  const suffix = typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID().slice(0, 8)
    : `${Date.now().toString(36)}-${Math.round(Math.random() * 10000).toString(36)}`
  return `${prefix}-${suffix}`
}

function engineLabel(model: ProModelOption | undefined): string {
  return model?.engineLabel || model?.engineId || model?.architecture || model?.backend || 'Unknown family'
}

function inferFamily(model: ProModelOption | undefined, settings: GenerationSettings, selectedModelName: string): string {
  const text = `${settings.modelId} ${selectedModelName} ${model?.engineId ?? ''} ${model?.architecture ?? ''} ${model?.backend ?? ''}`.toLowerCase()
  if (text.includes('wan')) return 'wan'
  if (text.includes('sana_video') || text.includes('sana-video')) return 'sana_video'
  if (text.includes('ltx')) return 'ltx'
  if (text.includes('flux2') || text.includes('flux.2') || text.includes('klein')) return 'flux2_klein'
  if (text.includes('flux')) return 'flux'
  if (text.includes('qwen') || text.includes('nunchaku')) return 'qwen_image'
  if (text.includes('z-image') || text.includes('zimage')) return 'z_image'
  if (text.includes('sana')) return 'sana'
  if (text.includes('sdxl')) return 'sdxl'
  if (text.includes('sd3')) return 'sd35'
  if (text.includes('sd15') || text.includes('sd1.5') || text.includes('stable diffusion 1.5')) return 'sd15'
  return 'unknown'
}

function selectionGate(model: ProModelOption | undefined): Record<string, unknown> {
  const status = String(model?.status ?? 'metadata-only').toLowerCase()
  const reason = model?.reason ?? ''
  const suggestedAction = model?.suggestedAction ?? ''
  if (['broken-runtime', 'blocked-cleanly', 'unsupported-no-route'].includes(status)) {
    return { normalSelectable: false, level: 'block', status, reason: reason || 'Blocked from normal generation selection.', suggestedAction }
  }
  if (['metadata-only', 'needs-smoke', 'experimental', 'candidate'].includes(status)) {
    return { normalSelectable: true, requiresWarning: true, level: 'warn', status, reason: reason || 'Discovered but not smoke-certified.', suggestedAction }
  }
  return { normalSelectable: true, requiresWarning: false, level: 'pass', status, reason, suggestedAction }
}

function detectPrecision(value: string): string {
  const lower = value.toLowerCase()
  if (/q8[_-]?0/.test(lower)) return 'Q8_0'
  if (/q6[_-]?k/.test(lower)) return 'Q6_K'
  if (/q5[_-]?k[_-]?m/.test(lower)) return 'Q5_K_M'
  if (/q5[_-]?k[_-]?s/.test(lower)) return 'Q5_K_S'
  if (/q4[_-]?k[_-]?m/.test(lower)) return 'Q4_K_M'
  if (/q4[_-]?k[_-]?s/.test(lower)) return 'Q4_K_S'
  if (/q3[_-]?k[_-]?m/.test(lower)) return 'Q3_K_M'
  if (/q3[_-]?k[_-]?s/.test(lower)) return 'Q3_K_S'
  if (/q2[_-]?k/.test(lower)) return 'Q2_K'
  if (lower.includes('nvfp4')) return 'NVFP4'
  if (lower.includes('nf4')) return 'NF4'
  if (lower.includes('fp4')) return 'FP4'
  if (lower.includes('int8') || lower.includes('i8')) return 'INT8'
  if (lower.includes('fp8')) return 'FP8'
  if (lower.includes('bf16')) return 'BF16'
  if (lower.includes('fp16') || lower.includes('float16')) return 'FP16'
  if (lower.includes('fp32') || lower.includes('float32')) return 'FP32'
  return 'auto'
}

function routeFor(settings: GenerationSettings, family: string): string {
  if (settings.mode === 'inpaint') {
    if (family === 'flux') return 'flux-fill-inpaint'
    return 'inpaint'
  }
  if (settings.mode === 'video') {
    if (family === 'wan') return 'wan-video'
    if (family === 'sana_video') return 'sana-video'
    if (family === 'ltx') return 'ltx-video'
    return 'image-to-video'
  }
  if (family === 'flux2_klein') return 'flux2-klein-image'
  if (family === 'qwen_image') return 'qwen-image'
  if (family === 'z_image') return 'z-image'
  if (family === 'sana') return 'sana-image'
  if (family === 'flux') return 'flux-image'
  return 'image-generate'
}

function payloadFor({ settings, bootstrap, runtime, selectedModel, selectedModelName, source }: WorkflowBlockSource): Record<string, unknown> {
  const family = inferFamily(selectedModel, settings, selectedModelName)
  const route = routeFor(settings, family)
  const selectedModelRecord = (selectedModel ?? {}) as Record<string, unknown>
  const familyLabel = engineLabel(selectedModel)
  const precision = detectPrecision(`${selectedModelName} ${selectedModel?.architecture ?? ''} ${selectedModel?.assetSummary ?? ''} ${selectedModel?.backend ?? ''}`)
  const basePayload: Record<string, unknown> = {
    schema: 'aiwf.studio-generation-packet.v1',
    capturedAt: nowIso(),
    source,
    route,
    family,
    familyLabel,
    precision,
    mode: settings.mode,
    prompt: {
      positive: settings.prompt,
      negative: settings.negativePrompt,
      seed: settings.seed,
    },
    model: {
      id: settings.modelId,
      name: selectedModelName,
      engineId: selectedModel?.engineId ?? '',
      engineLabel: selectedModel?.engineLabel ?? '',
      architecture: selectedModel?.architecture ?? '',
      backend: selectedModel?.backend ?? '',
      status: selectedModel?.status ?? '',
      reason: selectedModel?.reason ?? '',
      suggestedAction: selectedModel?.suggestedAction ?? '',
      assetSummary: selectedModel?.assetSummary ?? '',
      estVramGb: selectedModel?.estVramGb ?? null,
      heavyFor12Gb: selectedModel?.heavyFor12Gb ?? false,
    },
    selectionGate: selectionGate(selectedModel),
    generation: {
      width: settings.width,
      height: settings.height,
      aspectRatioId: settings.aspectRatioId,
      steps: settings.steps,
      cfgScale: settings.cfgScale,
      sampler: settings.sampler,
      scheduler: settings.scheduler,
      clipSkip: settings.clipSkip,
      batchSize: settings.batchSize,
      batchCount: settings.batchCount,
      saveImages: settings.saveImages,
    },
    imageTools: {
      enableHires: settings.enableHires,
      hiresScale: settings.hiresScale,
      hiresSteps: settings.hiresSteps,
      hiresDenoise: settings.hiresDenoise,
      hiresUpscaler: settings.hiresUpscaler,
      initImageName: settings.sourceImageName,
      hasInitImage: Boolean(settings.sourceImageDataUrl || settings.initImageDataUrl),
    },
    inpaint: {
      hasMask: Boolean(settings.maskImageDataUrl),
      denoisingStrength: settings.denoisingStrength,
      maskBlur: settings.maskBlur,
      inpaintOnlyMasked: settings.inpaintOnlyMasked,
      inpaintMaskedPadding: settings.inpaintMaskedPadding,
      inpaintMaskContent: settings.inpaintMaskContent,
      inpaintMaskOpacity: settings.inpaintMaskOpacity,
      autoMaskEnabled: settings.autoMaskEnabled,
      autoMaskPrompt: settings.autoMaskPrompt,
      autoMaskModel: settings.autoMaskModel,
      autoMaskBoxThreshold: settings.autoMaskBoxThreshold,
      autoMaskTextThreshold: settings.autoMaskTextThreshold,
    },
    video: {
      frames: settings.frames,
      fps: settings.fps,
      generateAudio: settings.generateAudio,
      sanaQuantization: settings.sanaQuantization,
      sanaVaeTiling: settings.sanaVaeTiling,
      offloadTextEncoderAfterEncode: settings.offloadTextEncoderAfterEncode,
      useSageAttention: settings.useSageAttention,
      wan: {
        runtimeMode: settings.wanRuntimeMode,
        highNoiseModelId: settings.highNoiseModelId,
        lowNoiseModelId: settings.lowNoiseModelId,
        highNoiseSteps: settings.highNoiseSteps,
        lowNoiseSteps: settings.lowNoiseSteps,
        boundaryRatio: settings.boundaryRatio,
        highNoiseLoraId: settings.highNoiseLoraId,
        highNoiseLoraScale: settings.highNoiseLoraScale,
        lowNoiseLoraId: settings.lowNoiseLoraId,
        lowNoiseLoraScale: settings.lowNoiseLoraScale,
        vaeId: settings.vaeId,
        textEncoderPath: settings.textEncoderPath,
        offload: settings.wanOffload,
        sigmaType: settings.wanSigmaType,
        sampler: settings.wanSampler,
        flowShift: settings.wanFlowShift,
      },
    },
    runtime: {
      state: runtime.state,
      backend: runtime.backend,
      device: runtime.device,
      precision: runtime.precision,
      attention: runtime.attention,
      queueCount: runtime.queueCount,
      resources: runtime.resources.map((metric) => ({ label: metric.label, value: metric.value, percent: metric.percent, tone: metric.tone })),
    },
    bootstrap: {
      workspaceName: bootstrap.workspaceName,
      version: bootstrap.version,
    },
  }

  if (route === 'wan-video') {
    basePayload.sidecars = {
      wanModelPack: {
        runtimeMode: settings.wanRuntimeMode,
        highNoiseModelId: settings.highNoiseModelId || selectedModelRecord.highNoiseModelId || selectedModelRecord.high_noise_model_id || null,
        lowNoiseModelId: settings.lowNoiseModelId || selectedModelRecord.lowNoiseModelId || selectedModelRecord.low_noise_model_id || null,
        highNoiseSteps: settings.highNoiseSteps,
        lowNoiseSteps: settings.lowNoiseSteps,
        boundaryRatio: settings.boundaryRatio,
        vaeId: settings.vaeId || selectedModelRecord.vaeId || selectedModelRecord.vae_id || null,
        textEncoderPath: settings.textEncoderPath || selectedModelRecord.textEncoderPath || selectedModelRecord.text_encoder_path || null,
        note: 'High/low, VAE, text encoder, and compatible precision are captured from the same settings packet Pro sends to /api/pro/generate.',
      },
      loraStack: {
        entries: [
          { target: 'high', id: settings.highNoiseLoraId, scale: settings.highNoiseLoraScale },
          { target: 'low', id: settings.lowNoiseLoraId, scale: settings.lowNoiseLoraScale },
        ].filter((entry) => entry.id),
        note: 'Adapter testing remains explicit; this block preserves target stage and weight for later smoke receipts.',
      },
      offloadPlan: {
        mode: settings.wanOffload,
        offloadTextEncoderAfterEncode: settings.offloadTextEncoderAfterEncode,
        useSageAttention: settings.useSageAttention,
        vaeTiling: settings.sanaVaeTiling,
      },
    }
  }

  return basePayload
}

export function createWorkflowBlocksFromSettings(sourceData: WorkflowBlockSource, existingCount: number): WorkflowCodeBlock[] {
  const payload = payloadFor(sourceData)
  const route = String(payload.route || 'generation-request')
  const family = String(payload.family || 'Model family')
  const precision = String(payload.precision || 'auto')
  const mode = sourceData.settings.mode
  const label = `${mode === 'video' ? 'Video' : mode === 'inpaint' ? 'Inpaint' : 'Image'} · ${family}`
  const summary = `${route} · ${sourceData.settings.width}×${sourceData.settings.height} · ${sourceData.settings.steps} steps · ${precision}`
  const block: WorkflowCodeBlock = {
    id: nextId('workflow-block'),
    label,
    kind: 'generation',
    nodeId: 'generation-request',
    source: sourceData.source,
    createdAt: nowIso(),
    summary,
    order: existingCount + 1,
    classes: { requires: [], produces: ['artifact'] },
    payload: { schema: 'aiwf.workflow-code-block.payload.v1', packet: payload },
    code: JSON.stringify({ schema: 'aiwf.workflow-code-block.payload.v1', packet: payload }, null, 2),
  }
  return [block]
}

export function normalizeWorkflowBlocks(value: unknown): WorkflowCodeBlock[] {
  const rows = Array.isArray(value)
    ? value
    : Array.isArray((value as { blocks?: unknown })?.blocks)
      ? (value as { blocks: unknown[] }).blocks
      : []
  return rows
    .filter((item): item is Record<string, unknown> => item !== null && typeof item === 'object')
    .map((item, index) => {
      const payload = item.payload && typeof item.payload === 'object' ? item.payload as Record<string, unknown> : {}
      const code = typeof item.code === 'string' && item.code.trim() ? item.code : JSON.stringify(payload, null, 2)
      return {
        id: String(item.id || nextId('workflow-block')),
        label: String(item.label || item.title || `Workflow block ${index + 1}`),
        kind: ['generation', 'workflow', 'qa', 'export'].includes(String(item.kind)) ? String(item.kind) as WorkflowCodeBlock['kind'] : 'generation',
        nodeId: String(item.nodeId || item.node_id || 'generation-request'),
        source: String(item.source || 'Imported JSON'),
        createdAt: String(item.createdAt || item.created_at || nowIso()),
        summary: String(item.summary || 'Imported workflow code block'),
        order: Number(item.order || index + 1),
        classes: {
          requires: Array.isArray((item.classes as { requires?: unknown[] } | undefined)?.requires) ? (item.classes as { requires: unknown[] }).requires.map(String) : [],
          produces: Array.isArray((item.classes as { produces?: unknown[] } | undefined)?.produces) ? (item.classes as { produces: unknown[] }).produces.map(String) : ['artifact'],
        },
        payload,
        code,
      }
    })
    .sort((a, b) => a.order - b.order)
    .map((block, index) => ({ ...block, order: index + 1 }))
}

export function loadWorkflowBlocksFromStorage(): WorkflowCodeBlock[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(WORKFLOW_BLOCK_STORAGE_KEY)
    return raw ? normalizeWorkflowBlocks(JSON.parse(raw)) : []
  } catch {
    return []
  }
}

export function saveWorkflowBlocksToStorage(blocks: WorkflowCodeBlock[]): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(WORKFLOW_BLOCK_STORAGE_KEY, JSON.stringify({ schema: 'aiwf.workflow-code-blocks.local.v1', blocks }, null, 2))
  } catch {
    // Browser storage can be full or disabled; workflow still lives in React state.
  }
}

export function renumberWorkflowBlocks(blocks: WorkflowCodeBlock[]): WorkflowCodeBlock[] {
  return blocks.map((block, index) => ({ ...block, order: index + 1 }))
}

export function duplicateWorkflowBlock(block: WorkflowCodeBlock, insertOrder: number): WorkflowCodeBlock {
  const payload = { ...block.payload } as Record<string, unknown>
  const packet = payload.packet && typeof payload.packet === 'object' ? { ...(payload.packet as Record<string, unknown>) } : undefined
  if (packet) {
    packet.capturedAt = nowIso()
    packet.duplicatedFrom = block.id
    payload.packet = packet
  } else {
    payload.capturedAt = nowIso()
    payload.duplicatedFrom = block.id
  }
  return {
    ...block,
    id: nextId('workflow-block'),
    label: `${block.label} copy`,
    createdAt: nowIso(),
    order: insertOrder,
    payload,
    code: JSON.stringify(payload, null, 2),
  }
}

export function workflowPayloadFromBlocks(blocks: WorkflowCodeBlock[]): Record<string, unknown> {
  const ordered = renumberWorkflowBlocks(blocks)
  return {
    schema: 'aiwf.workflow-code-blocks.v1',
    savedAt: nowIso(),
    id: 'main',
    label: 'Workflow code block queue',
    blocks: ordered,
    stages: ordered.map((block) => ({
      uid: block.id,
      nodeId: block.nodeId,
      templateId: block.nodeId,
      template: block.label,
      order: block.order,
      classes: block.classes,
    })),
    routing: {
      mode: 'linear-code-blocks',
      note: 'Blocks are self-contained snapshots. Drag/drop changes queue order; no canvas wires are required.',
    },
  }
}

export function validateWorkflowBlocks(blocks: WorkflowCodeBlock[]): WorkflowBlockValidation {
  const errors: string[] = []
  const seen = new Set<string>()
  blocks.forEach((block, index) => {
    if (!block.id) errors.push(`Block ${index + 1} is missing an id.`)
    if (seen.has(block.id)) errors.push(`Block ${index + 1} duplicates id ${block.id}.`)
    seen.add(block.id)
    if (!block.label.trim()) errors.push(`Block ${index + 1} is missing a label.`)
    if (!block.nodeId.trim()) errors.push(`${block.label || `Block ${index + 1}`} is missing nodeId.`)
    if (!block.code.trim()) errors.push(`${block.label || `Block ${index + 1}`} has an empty code block.`)
    try {
      const parsed = JSON.parse(block.code)
      const packet = parsed && typeof parsed === 'object' && 'packet' in parsed ? (parsed as { packet?: unknown }).packet : undefined
      const gate = packet && typeof packet === 'object' ? (packet as { selectionGate?: { normalSelectable?: unknown, reason?: unknown, status?: unknown } }).selectionGate : undefined
      if (gate?.normalSelectable === false) {
        errors.push(`${block.label || `Block ${index + 1}`} uses a blocked model: ${String(gate.reason || gate.status || 'blocked')}.`)
      }
    } catch {
      errors.push(`${block.label || `Block ${index + 1}`} code is not valid JSON.`)
    }
  })
  return { valid: errors.length === 0, errors }
}
