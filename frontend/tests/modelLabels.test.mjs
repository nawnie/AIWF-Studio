import assert from 'node:assert/strict'
import test from 'node:test'

import { formatInventoryAssetLabel, formatRouteLifecycleStatus, formatStudioEngineLabel, formatStudioModelAvailability, formatStudioModelFamily, formatStudioModelLabel, formatStudioModelStatus } from '../src/layouts/studio/modelLabels.ts'

test('engine bucket labels match the Pro route taxonomy for Flux.2 variants', () => {
  assert.equal(formatStudioEngineLabel('flux2'), 'Flux.2 Klein')
  assert.equal(formatStudioEngineLabel('flux2_generic'), 'Flux.2 (generic)')
  assert.equal(formatStudioEngineLabel('sdxl', 'Other'), 'Stable Diffusion XL')
  assert.equal(formatStudioModelFamily({ id: 'klein', name: 'Klein', architecture: 'flux2_klein' }), 'Flux.2 Klein')
})

test('model inventory labels preserve audio and chat modality instead of calling weights checkpoints', () => {
  assert.equal(formatInventoryAssetLabel('audio', 'audio'), 'Audio model')
  assert.equal(formatInventoryAssetLabel('llm', 'llm'), 'Chat model')
  assert.equal(formatInventoryAssetLabel('checkpoint', 'sdxl'), 'SDXL checkpoint')
  assert.equal(formatInventoryAssetLabel('vae', 'flux'), 'Flux.1 VAE')
  assert.equal(formatInventoryAssetLabel('preprocessor', 'unknown'), 'ControlNet preprocessor')
  assert.equal(formatInventoryAssetLabel('runtime_asset', 'rife'), 'Frame interpolation model')
  assert.equal(formatInventoryAssetLabel('runtime_asset', 'lama'), 'LaMa inpainting model')
})

test('Flux Kontext keeps its specific architecture label while remaining Flux engine-grouped', () => {
  const model = {
    id: 'flux-kontext-example',
    name: 'Flux Kontext dev',
    architecture: 'flux_kontext',
    engineId: 'flux',
    engineLabel: 'Flux.1',
  }

  assert.equal(model.engineId, 'flux')
  assert.equal(formatStudioModelFamily(model), 'Flux Kontext')
  assert.equal(formatStudioModelLabel(model), 'Flux Kontext dev')
})

test('Flux Kontext removes only the generated stale Flux family suffix from inventory titles', () => {
  const model = {
    id: 'flux1-kontext-dev-Q5_K_M',
    name: 'flux1-kontext-dev-Q5_K_M [Flux] [GGUF Q5_K_M]',
    architecture: 'flux_kontext',
    assetSummary: 'GGUF Q5_K_M',
  }
  assert.equal(formatStudioModelLabel(model), 'flux1-kontext-dev-Q5_K_M [GGUF Q5_K_M] (Flux Kontext)')
})

test('Flux Kontext leaves custom model names intact', () => {
  const model = {
    id: 'flux1-kontext-dev-Q5_K_M',
    name: 'My personal Kontext checkpoint [Flux]',
    architecture: 'flux_kontext',
  }
  assert.equal(formatStudioModelLabel(model), 'My personal Kontext checkpoint [Flux] (Flux Kontext)')
})

test('architecture wins over a stale engine label in shared model labels', () => {
  const model = {
    id: 'qwen-image-model',
    name: 'Qwen Image',
    architecture: 'qwen_image',
    engineId: 'qwen',
    engineLabel: 'Other',
  }

  assert.equal(formatStudioModelFamily(model), 'Qwen Image')
})

test('unsupported Qwen edit variants are labeled separately from generic Qwen Image', () => {
  assert.equal(formatStudioModelFamily({ id: 'edit', name: 'Qwen Edit', architecture: 'qwen_image_edit' }), 'Qwen Image Edit (unsupported)')
  assert.equal(formatStudioModelFamily({ id: 'edit-plus', name: 'Qwen Edit Plus', architecture: 'qwen_image_edit_plus' }), 'Qwen Image Edit Plus (unsupported)')
})

test('generic engine and backend names are not presented as model families', () => {
  assert.equal(formatStudioModelFamily({ id: 'unknown', name: 'Unknown', engineId: 'diffusers', backend: 'diffusers' }), 'Model family not reported')
  assert.equal(formatStudioModelFamily({ id: 'unknown-flux-route', name: 'Unknown', engineId: 'flux', engineLabel: 'Flux' }), 'Model family not reported')
  assert.equal(formatStudioModelFamily({ id: 'other', name: 'Other', architecture: 'other', engineLabel: 'Other', engineId: 'other' }), 'Model family not reported')
})

test('specific architecture wins over a conflicting engine family label', () => {
  const model = {
    id: 'flux-fill-model',
    name: 'Flux Fill',
    architecture: 'flux_fill',
    engineId: 'flux',
    engineLabel: 'Flux.1',
  }

  assert.equal(formatStudioModelFamily(model), 'Flux Fill')
})

test('generic Flux.2 labels stay distinct from explicit Klein labels', () => {
  assert.equal(formatStudioModelFamily({ id: 'flux2-generic', name: 'Flux.2', architecture: 'flux2' }), 'Flux.2')
  assert.equal(formatStudioModelFamily({ id: 'flux2-klein', name: 'Flux.2 Klein', architecture: 'flux2_klein' }), 'Flux.2 Klein')
})

test('same-name model selections retain family and ID across the app shell label', () => {
  const flux = { id: 'flux-12345678-model', name: 'Community Checkpoint', architecture: 'flux', assetSummary: 'safetensors' }
  const sdxl = { id: 'sdxl-abcdef12-model', name: 'Community Checkpoint', architecture: 'sdxl', assetSummary: 'safetensors' }

  assert.match(formatStudioModelLabel(flux, [flux, sdxl]), /Flux\.1/)
  assert.match(formatStudioModelLabel(flux, [flux, sdxl]), /flux-123/)
  assert.match(formatStudioModelLabel(sdxl, [flux, sdxl]), /SDXL/)
  assert.match(formatStudioModelLabel(sdxl, [flux, sdxl]), /sdxl-ab/)
})

test('inpaint and refiner architecture labels keep the SDXL branding', () => {
  assert.equal(formatStudioModelFamily({ id: 'sd15-inpaint', name: 'SD 1.5 Inpaint', architecture: 'inpaint' }), 'SD 1.5 Inpaint')
  assert.equal(formatStudioModelFamily({ id: 'sdxl-inpaint', name: 'SDXL Inpaint', architecture: 'sdxl_inpaint' }), 'SDXL Inpaint')
  assert.equal(formatStudioModelFamily({ id: 'sdxl-refiner', name: 'SDXL Refiner', architecture: 'sdxl_refiner' }), 'SDXL Refiner')
})

test('Sana image and Sana Video labels remain distinct', () => {
  assert.equal(formatStudioModelFamily({ id: 'sana-image', name: 'Sana', architecture: 'sana' }), 'Sana')
  assert.equal(formatStudioModelFamily({ id: 'sana-video', name: 'Sana Video', architecture: 'sana_video' }), 'Sana Video')
})

test('model status uses clear user-facing labels', () => {
  assert.equal(formatStudioModelStatus('missing-assets'), 'Needs supporting files')
  assert.equal(formatStudioModelStatus('blocked-runtime'), 'Runtime unavailable')
  assert.equal(formatStudioModelStatus('ready'), 'Preflight ready · generation unverified')
})

test('model availability distinguishes route eligibility from generation proof', () => {
  assert.equal(formatStudioModelAvailability({
    id: 'local-checkpoint',
    name: 'Local checkpoint',
    checkpointPathStatus: 'present',
    routeStatus: 'request-eligible',
  }), 'Selectable · generation unverified')
  assert.equal(formatStudioModelAvailability({
    id: 'wan-installed-unchecked',
    name: 'Wan',
    status: 'Installed',
    checkpointPathStatus: 'present',
    routeStatus: 'unknown',
  }), 'Installed · route not checked')
  assert.equal(formatStudioModelAvailability({
    id: 'missing-checkpoint',
    name: 'Missing checkpoint',
    checkpointPathStatus: 'missing',
    routeStatus: 'blocked',
  }), 'Model files missing')
  assert.equal(formatStudioModelAvailability({
    id: 'wan-ready',
    name: 'Wan',
    status: 'Ready',
    checkpointPathStatus: 'present',
    routeStatus: 'request-eligible',
  }), 'Selectable · generation unverified')
})

test('Sana Video availability names the exact generation modes confirmed by runtime checks', () => {
  assert.equal(formatStudioModelAvailability({
    id: 'sana-video-480p',
    name: 'Sana Video 480p',
    architecture: 'sana_video',
    checkpointPathStatus: 'present',
    routeStatus: 'request-eligible',
    generationModes: { textToVideo: true, imageToVideo: false },
}), 'Text-to-video preflight ready · image-to-video unavailable')
  assert.equal(formatStudioModelAvailability({
    id: 'sana-video-i2v-only-blocked',
    name: 'Sana Video I2V only',
    architecture: 'sana_video',
    checkpointPathStatus: 'present',
    routeStatus: 'blocked',
    generationModes: { textToVideo: false, imageToVideo: true },
  }), 'Route blocked')
})

test('explicit missing or blocked route evidence outranks a contradictory ready status', () => {
  assert.equal(formatStudioModelAvailability({
    id: 'missing-ready',
    name: 'Missing model',
    status: 'Ready',
    checkpointPathStatus: 'missing',
    routeStatus: 'request-eligible',
  }), 'Model files missing')
  assert.equal(formatStudioModelAvailability({
    id: 'blocked-ready',
    name: 'Blocked model',
    status: 'Ready',
    checkpointPathStatus: 'present',
    routeStatus: 'blocked',
  }), 'Route blocked')
})

test('route lifecycle labels distinguish setup readiness, completion, and residency', () => {
  assert.equal(formatRouteLifecycleStatus('needs-setup'), 'Needs model setup')
  assert.equal(formatRouteLifecycleStatus('setup-ready', false), 'Route setup ready · model not loaded')
  assert.equal(formatRouteLifecycleStatus('setup-ready'), 'Route setup ready · residency unconfirmed')
  assert.equal(formatRouteLifecycleStatus('completed', false), 'Generation completed')
  assert.equal(formatRouteLifecycleStatus('loaded', true), 'Model loaded')
})

test('prepared route status follows setup completion and backend-confirmed residency', () => {
  assert.equal(formatRouteLifecycleStatus('prepared', true), 'Route prepared · model loaded')
  assert.equal(formatRouteLifecycleStatus('prepared', false), 'Route prepared · model not loaded')
  assert.equal(formatRouteLifecycleStatus('prepared'), 'Route prepared · residency unconfirmed')
})

test('unknown route statuses remain visible without implying residency', () => {
  assert.equal(formatRouteLifecycleStatus('future-state', false), 'future-state')
  assert.equal(formatRouteLifecycleStatus(undefined, null), 'Route status unavailable')
})
