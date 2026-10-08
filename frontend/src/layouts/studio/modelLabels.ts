import type { ProModelOption } from '../../types'

/** A user-facing model label that keeps the family visible across Studio pickers. */
export function formatStudioModelLabel(model: ProModelOption, models: ProModelOption[] = []): string {
  const family = formatStudioModelFamily(model)
  const name = modelNameWithoutStaleGeneratedFamily(model)
  const details = [model.assetSummary, family]
    .map((value) => value?.trim())
    .filter((value): value is string =>
      typeof value === 'string' && value.length > 0 && !name.toLowerCase().includes(value.toLowerCase()),
    )
  const base = details.length > 0 ? `${name} (${details.join(' · ')})` : name
  const hasDuplicate = models.some((candidate) =>
    candidate.id !== model.id && candidate.name.trim().toLowerCase() === model.name.trim().toLowerCase(),
  )
  return hasDuplicate ? `${base} — ${model.id.slice(0, 8)}` : base
}

function modelNameWithoutStaleGeneratedFamily(model: ProModelOption): string {
  const name = model.name.trim()
  const architectureKey = (model.architecture ?? '').trim().toLowerCase().replace(/[.\s-]+/g, '_')
  if (architectureKey !== 'flux_kontext') return name

  // Checkpoint titles are generated as "<id> [<detected family>] [<asset summary>]".
  // Repair only that exact generated prefix, leaving user-authored names alone.
  const stalePrefix = `${model.id} [Flux]`
  if (
    name.toLowerCase().startsWith(stalePrefix.toLowerCase()) &&
    (name.length === stalePrefix.length || /\s/.test(name[stalePrefix.length] ?? ''))
  ) {
    return `${model.id}${name.slice(stalePrefix.length)}`.trim()
  }
  return name
}

export function formatStudioModelFamily(model: ProModelOption): string {
  const architectureKey = (model.architecture ?? '').trim().toLowerCase().replace(/[.\s-]+/g, '_')
  if (architectureKey === 'flux_kontext') return 'Flux Kontext'
  const architecture = model.architecture?.trim()
  if (architecture && architecture.toLowerCase() !== 'other') return humanizeFamily(architecture)
  // Engine/backend identify the loader or route group, not necessarily the
  // model architecture. Showing "Diffusers" or "Other" here mislabels models.
  return 'Model family not reported'
}

/** Labels an engine bucket without assuming a specific model variant. */
export function formatStudioEngineLabel(engineId: string, reportedLabel?: string): string {
  const reported = reportedLabel?.trim()
  if (reported && reported.toLowerCase() !== 'other') return reported
  const labels: Record<string, string> = {
    flux: 'Flux',
    flux_fill: 'Flux Fill',
    // Pro keeps generic Flux.2 in its own `flux2_generic` bucket.
    flux2: 'Flux.2 Klein',
    flux2_generic: 'Flux.2 (generic)',
    krea2: 'Krea 2',
    sana_video: 'Sana Video',
    wan: 'Wan Video',
    ltx: 'LTX Video',
    sd15: 'Stable Diffusion 1.5',
    sdxl: 'Stable Diffusion XL',
    sd35: 'Stable Diffusion 3.5',
    zimage: 'Z-Image',
    qwen: 'Qwen Image',
    sana: 'Sana',
  }
  return labels[engineId] ?? 'Other'
}

/** Label model inventory rows without calling other modality weights image checkpoints. */
export function formatInventoryAssetLabel(familyValue: string, architectureValue: string): string {
  const family = familyValue.trim().toLowerCase().replace(/[\s-]+/g, '_')
  const architecture = architectureValue.trim().toLowerCase().replace(/[\s-]+/g, '_')
  if (family === 'audio') return 'Audio model'
  if (family === 'llm') return 'Chat model'
  if (family === 'preprocessor') return 'ControlNet preprocessor'
  if (family === 'runtime_asset' && architecture === 'rife') return 'Frame interpolation model'
  if (family === 'runtime_asset' && architecture === 'lama') return 'LaMa inpainting model'
  const labels: Record<string, string> = {
    checkpoint: 'checkpoint',
    lora: 'LoRA',
    vae: 'VAE',
    text_encoder: 'text encoder',
    controlnet: 'ControlNet',
    runtime_asset: 'runtime asset',
    upscaler: 'upscaler',
    embedding: 'embedding',
    face_embedding: 'face embedding',
    ip_adapter: 'IP-Adapter',
    hypernetwork: 'hypernetwork',
    wan: 'Wan model',
    ltx: 'LTX model',
    invalid_asset: 'invalid model file',
  }
  const category = labels[family] ?? family.replace(/_/g, ' ')
  const modelFamily = architecture && architecture !== 'unknown' && architecture !== family
    ? humanizeFamily(architecture)
    : ''
  return modelFamily ? `${modelFamily} ${category}` : category
}

export function formatStudioModelStatus(status?: string): string {
  const key = (status ?? '').trim().toLowerCase().replace(/[\s_]+/g, '-')
  const labels: Record<string, string> = {
    // Inventory/preflight readiness is not proof that a generation completed.
    ready: 'Preflight ready · generation unverified',
    'missing-assets': 'Needs supporting files',
    'blocked-runtime': 'Runtime unavailable',
    'broken-runtime': 'Runtime unavailable',
    'blocked-cleanly': 'Unavailable',
    'unsupported-no-route': 'No generation route',
    'metadata-only': 'Metadata only',
    'needs-smoke': 'Needs runtime check',
    experimental: 'Experimental',
    candidate: 'Candidate',
  }
  return labels[key] ?? (status?.trim() || '')
}

export function formatRouteLifecycleStatus(status?: string, resident: boolean | null = null): string {
  const key = (status ?? '').trim().toLowerCase().replace(/[\s_]+/g, '-')
  const labels: Record<string, string> = {
    'needs-setup': 'Needs model setup',
    'setup-ready': resident === false
      ? 'Route setup ready · model not loaded'
      : 'Route setup ready · residency unconfirmed',
    preparing: 'Preparing route',
    running: 'Generation running',
    completed: 'Generation completed',
    loaded: 'Model loaded',
    failed: 'Route failed',
    cancelled: 'Generation cancelled',
    prepared: resident === true
      ? 'Route prepared · model loaded'
      : resident === false
        ? 'Route prepared · model not loaded'
        : 'Route prepared · residency unconfirmed',
  }
  return labels[key] ?? (status?.trim() || 'Route status unavailable')
}

export function formatStudioModelAvailability(model: ProModelOption): string {
  if (model.checkpointPathStatus === 'missing') return 'Model files missing'
  // Mode-specific preflight details do not override an aggregate blocked route:
  // model selection and generation are rejected until the route itself is ready.
  if (model.routeStatus === 'blocked') return 'Route blocked'
  if (model.generationModes) {
    const { textToVideo = false, imageToVideo = false } = model.generationModes
    if (textToVideo && imageToVideo) return 'Text/image-to-video preflight ready · generation unverified'
    if (textToVideo) return 'Text-to-video preflight ready · image-to-video unavailable'
    if (imageToVideo) return 'Image-to-video preflight ready · text-to-video unavailable'
  }
  if (model.routeStatus === 'request-eligible') {
    return model.checkpointPathStatus === 'present'
      ? 'Selectable · generation unverified'
      : 'Route available · generation unverified'
  }
  const status = formatStudioModelStatus(model.status)
  if (model.routeStatus === 'unknown' && status.toLowerCase() === 'installed') {
    return 'Installed · route not checked'
  }
  if (status) return status
  if (model.checkpointPathStatus === 'present') return 'Files present · readiness not checked'
  return 'Readiness not checked'
}

function humanizeFamily(value: string): string {
  const key = value.trim().toLowerCase().replace(/[.\s-]+/g, '_')
  const labels: Record<string, string> = {
    sd15: 'SD 1.5',
    sd_1_5: 'SD 1.5',
    inpaint: 'SD 1.5 Inpaint',
    sdxl: 'SDXL',
    sdxl_inpaint: 'SDXL Inpaint',
    sdxl_refiner: 'SDXL Refiner',
    sd_3_5: 'SD 3.5',
    sd35: 'SD 3.5',
    flux: 'Flux.1',
    flux_fill: 'Flux Fill',
    flux2: 'Flux.2',
    flux_2: 'Flux.2',
    flux2_klein: 'Flux.2 Klein',
    flux_2_klein: 'Flux.2 Klein',
    flux_kontext: 'Flux Kontext',
    krea2: 'Krea 2',
    qwen: 'Qwen Image',
    qwen_image: 'Qwen Image',
    qwen_image_edit: 'Qwen Image Edit (unsupported)',
    qwen_image_edit_plus: 'Qwen Image Edit Plus (unsupported)',
    qwen_image_nunchaku: 'Qwen Image Nunchaku',
    zimage: 'Z-Image',
    z_image: 'Z-Image',
    sana: 'Sana',
    sana_video: 'Sana Video',
    wan: 'Wan',
    ltx: 'LTX',
  }
  if (labels[key]) return labels[key]
  return value.replace(/[_-]+/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}
