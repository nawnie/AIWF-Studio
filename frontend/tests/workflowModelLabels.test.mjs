import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { createWorkflowBlocksFromSettings } from '../src/workflow/workflowBlocks.ts'
import { createWorkflowBlocksFromSettings as createLegacyWorkflowBlocks } from '../src/layouts/studio/workflowBlocks.ts'

const sourcePath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../src/workflow/workflowBlocks.ts')
const source = await readFile(sourcePath, 'utf8')
const workflowBuilders = [
  ['current', createWorkflowBlocksFromSettings],
  ['legacy', createLegacyWorkflowBlocks],
]

function makePacket(build, model, mode = 'image') {
  const [block] = build({
    settings: {
      mode,
      modelId: model.id,
      width: 768,
      height: 768,
      steps: 20,
      prompt: 'test prompt',
      negativePrompt: '',
      seed: 1,
      batchSize: 1,
      batchCount: 1,
      sampler: 'euler',
      scheduler: 'automatic',
      aspectRatioId: 'square',
      saveImages: true,
    },
    bootstrap: { workspaceName: 'test', version: 'test' },
    runtime: {
      state: 'idle', backend: 'test', device: 'cpu', precision: 'fp16', attention: 'native',
      queueCount: 0, resources: [],
    },
    selectedModel: model,
    selectedModelName: model.name,
    source: 'test',
  }, 0)
  return JSON.parse(block.code).packet
}

test('workflow model details use the same canonical label as the packet family', () => {
  assert.match(source, /const familyLabel = packetFamilyLabel\(selectedModel, family\)/)
  assert.match(source, /engineLabel: familyLabel/)
  assert.doesNotMatch(source, /engineLabel: selectedModel\?\.engineLabel/)
})

test('workflow selection gate blocks models whose route assets are missing', () => {
  const [block] = createWorkflowBlocksFromSettings({
    settings: { mode: 'image', modelId: 'sdxl-test', width: 768, height: 768, steps: 20, prompt: 'test', negativePrompt: '', seed: 1 },
    bootstrap: { workspaceName: 'test', version: 'test' },
    runtime: { state: 'idle', backend: 'test', device: 'cpu', precision: 'fp16', attention: 'native', queueCount: 0, resources: [] },
    selectedModel: { id: 'sdxl-test', name: 'Test SDXL', architecture: 'sdxl', status: 'missing-assets', routeStatus: 'blocked' },
    selectedModelName: 'Test SDXL',
    source: 'test',
  }, 0)
  const packet = JSON.parse(block.code).packet

  assert.equal(packet.selectionGate.normalSelectable, false)
  assert.equal(packet.selectionGate.level, 'block')
  assert.equal(packet.selectionGate.status, 'missing-assets')
})

test('legacy workflow selection gate does not call unknown or missing-assets readiness a pass', () => {
  const [block] = createLegacyWorkflowBlocks({
    settings: { mode: 'image', modelId: 'sdxl-test', width: 768, height: 768, steps: 20, prompt: 'test', negativePrompt: '', seed: 1 },
    bootstrap: { workspaceName: 'test', version: 'test' },
    runtime: { state: 'idle', backend: 'test', device: 'cpu', precision: 'fp16', attention: 'native', queueCount: 0, resources: [] },
    selectedModel: { id: 'sdxl-test', name: 'Test SDXL', architecture: 'sdxl', status: 'missing_assets', routeStatus: 'blocked' },
    selectedModelName: 'Test SDXL',
    source: 'test',
  }, 0)
  const packet = JSON.parse(block.code).packet

  assert.equal(packet.selectionGate.normalSelectable, false)
  assert.equal(packet.selectionGate.level, 'block')
  assert.equal(packet.selectionGate.status, 'missing-assets')
})

test('workflow packets keep generic Flux.2 distinct from Flux.2 Klein', () => {
  for (const [builderName, build] of workflowBuilders) {
    const generic = makePacket(build, {
      id: 'flux2-generic', name: 'Flux.2 base Q4', architecture: 'flux2', engineId: 'flux2', status: 'ready',
    })
    assert.equal(generic.family, 'flux2', `${builderName} family`)
    assert.equal(generic.familyLabel, 'Flux.2', `${builderName} family label`)
    assert.equal(generic.route, 'unsupported-generic-flux2', `${builderName} route`)
    assert.equal(generic.selectionGate.normalSelectable, false, `${builderName} generic Flux.2 gate`)
    assert.equal(generic.selectionGate.level, 'block', `${builderName} generic Flux.2 gate level`)

    const ambiguousEngine = makePacket(build, {
      id: 'local-community-model', name: 'Community checkpoint', engineId: 'flux2', status: 'ready',
    })
    assert.equal(ambiguousEngine.family, 'flux2', `${builderName} engine-only family`)
    assert.equal(ambiguousEngine.route, 'unsupported-generic-flux2', `${builderName} engine-only route`)
    assert.equal(ambiguousEngine.selectionGate.level, 'block', `${builderName} engine-only gate`)

    const klein = makePacket(build, {
      id: 'flux2-klein-4b', name: 'Flux.2 Klein 4B', architecture: 'flux2_klein', engineId: 'flux2',
    })
    assert.equal(klein.family, 'flux2_klein', `${builderName} Klein family`)
    assert.equal(klein.route, 'flux2-klein-image', `${builderName} Klein route`)

    const identityKlein = makePacket(build, {
      id: 'f2k-model', name: 'Klein checkpoint', architecture: 'flux2', engineId: 'flux2',
    })
    assert.equal(identityKlein.family, 'flux2_klein', `${builderName} identity-specific Klein family`)
    assert.equal(identityKlein.route, 'flux2-klein-image', `${builderName} identity-specific Klein route`)
  }
})

test('workflow packets preserve Flux Kontext and never route it as generic Flux.1', () => {
  for (const [builderName, build] of workflowBuilders) {
    const kontext = makePacket(build, {
      id: 'flux1-kontext-dev-Q5_K_M',
      name: 'flux1-kontext-dev-Q5_K_M [Flux]',
      architecture: 'flux_kontext',
      engineId: 'flux',
      status: 'blocked-cleanly',
      routeStatus: 'blocked',
      reason: 'Flux Kontext requires a complete Diffusers folder.',
    })
    assert.equal(kontext.family, 'flux_kontext', `${builderName} family`)
    assert.equal(kontext.familyLabel, 'Flux Kontext', `${builderName} family label`)
    assert.equal(kontext.route, 'pro.image.flux-kontext', `${builderName} image route`)
    assert.equal(kontext.selectionGate.level, 'block', `${builderName} blocked state`)

    const inpaint = makePacket(build, {
      id: 'flux1-kontext-folder', name: 'Flux Kontext', architecture: 'flux_kontext', engineId: 'flux',
    }, 'inpaint')
    assert.equal(inpaint.family, 'flux_kontext', `${builderName} inpaint family`)
    assert.equal(inpaint.route, 'unsupported-flux-kontext-inpaint', `${builderName} inpaint route`)

    const staleBroadFlux = makePacket(build, {
      id: 'flux1-kontext-dev-Q5_K_M',
      name: 'flux1-kontext-dev-Q5_K_M [Flux]',
      architecture: 'flux',
      engineId: 'flux',
      engineLabel: 'Flux',
    })
    assert.equal(staleBroadFlux.family, 'flux_kontext', `${builderName} stale architecture family`)
    assert.equal(staleBroadFlux.familyLabel, 'Flux Kontext', `${builderName} stale architecture label`)
    assert.equal(staleBroadFlux.route, 'pro.image.flux-kontext', `${builderName} stale architecture route`)
  }
})

test('Qwen model option reports only the backend loaded state', async () => {
  const uiPath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../src/layouts/studio/UnifiedWorkspaceLayout.tsx')
  const ui = await readFile(uiPath, 'utf8')

  assert.match(ui, /item\.loaded \? ' \(loaded\)' : ' \(not reported loaded\)'/)
  assert.doesNotMatch(ui, /loads on send/)
})
