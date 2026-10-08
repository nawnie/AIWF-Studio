import assert from 'node:assert/strict'
import { mkdir, mkdtemp, rm } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import test, { after, before } from 'node:test'
import { chromium } from 'playwright'
import { createServer } from 'vite'

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const chromePath = process.env.AIWF_TEST_CHROME_PATH ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe'
const fallbackPng = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII='
let browser
let vite
let baseUrl
let artifactRoot

function model(id, name, changes = {}) {
  return {
    id,
    title: name,
    filename: `${id}.safetensors`,
    architecture: 'sdxl',
    engineId: 'sdxl',
    engineLabel: 'Stable Diffusion XL',
    checkpointPathStatus: 'present',
    routeStatus: 'request-eligible',
    generationPreset: { width: 512, height: 512, steps: 4, cfgScale: 5, sampler: 'euler_a' },
    ...changes,
  }
}

function makeBootstrap(state) {
  return {
    workspaceName: 'AIWF Studio',
    subtitle: 'Pro',
    version: 'playwright-test',
    localFirst: true,
    onboardingSeen: true,
    engines: state.bootstrap.models.length > 0 ? [{ id: 'sdxl', label: 'Stable Diffusion XL', count: state.bootstrap.models.length }] : [],
    models: state.bootstrap.models,
    blockedModels: state.bootstrap.blockedModels ?? [],
    counts: { checkpoints: state.bootstrap.models.length, blockedCheckpoints: state.bootstrap.blockedModels?.length ?? 0 },
    samplers: [{ id: 'euler_a', label: 'Euler a' }],
    aspectRatios: [{ id: 'square', label: '1:1', width: 512, height: 512 }],
    defaults: {
      mode: state.bootstrap.mode ?? 'image',
      prompt: 'A small red apple on a wooden table',
      negativePrompt: '',
      modelId: state.bootstrapRequestCount > 1 && state.bootstrapRefreshDefaultModelIds
        ? state.bootstrapRefreshDefaultModelIds[Math.min(state.bootstrapRequestCount - 2, state.bootstrapRefreshDefaultModelIds.length - 1)]
        : state.bootstrapRequestCount > 1 && state.bootstrapRefreshDefaultModelId
          ? state.bootstrapRefreshDefaultModelId
        : state.bootstrap.defaultModelId,
      width: 512,
      height: 512,
      steps: 4,
      cfgScale: 5,
      sampler: 'euler_a',
      scheduler: 'automatic',
      seed: 420026,
      batchSize: 1,
      batchCount: 1,
      aspectRatioId: 'square',
      saveImages: true,
      ...(state.bootstrap.defaultSettings ?? {}),
    },
    recentOutputs: state.outputs,
  }
}

function runtime() {
  return {
    state: 'idle',
    backend: 'fake-diffusers',
    device: 'CPU fake backend',
    precision: 'fp32',
    attention: 'none',
    maxResolution: '512x512',
    queueCount: 0,
    resources: [],
    job: { id: '', state: 'idle', progress: 0, step: 0, totalSteps: 0, message: '', hasResult: false, error: '', previewUrl: '' },
    loadedModel: { name: '', type: '', baseModel: '', sizeOnDisk: '', precision: '', vae: '', textEncoder: '', unet: '', loaded: false },
    modelLoad: { status: 'not-started', modelId: '', detail: '' },
    routeLifecycle: [],
  }
}

function output(prompt = 'A small red apple on a wooden table') {
  return {
    id: 'playwright-output-1',
    url: fallbackPng,
    thumbnailUrl: fallbackPng,
    path: 'task-temp/fake-output.png',
    prompt,
    width: 512,
    height: 512,
    createdAt: new Date().toISOString(),
    mode: 'image',
    seed: 420026,
    steps: 4,
    cfgScale: 5,
    sampler: 'euler_a',
    modelName: 'Test SDXL',
    receiptPath: 'task-temp/fake-output.json',
    status: 'completed',
    source: 'generation',
  }
}

async function openStudio(state, { waitForPrompt = true } = {}) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 })
  const page = await context.newPage()
  const requests = []
  const consoleErrors = []
  page.on('pageerror', (error) => consoleErrors.push(error.message))
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text())
  })

  await page.route('**/api/**', async (route) => {
    const request = route.request()
    const { pathname } = new URL(request.url())
    requests.push({ pathname, url: request.url(), method: request.method(), body: request.postData() })

    if (pathname === '/api/v1/client-events' || pathname === '/api/v1/client-errors') {
      return route.fulfill({ status: 204, body: '' })
    }
    if (pathname === '/api/pro/runtime/stream') {
      return route.fulfill({ status: 200, contentType: 'text/event-stream', body: ': connected\n\n' })
    }
    if (pathname === '/api/pro/engines/ltx/install' && request.method() === 'POST') {
      state.ltxInstallRequests += 1
      return route.fulfill({ json: { status: 'started', pid: 1234, logPath: 'outputs/engine-installs/ltx.log', message: 'LTX engine setup started.' } })
    }
    if (pathname === '/api/pro/engines/qwen_nunchaku/install' && request.method() === 'POST') {
      state.qwenNunchakuInstallRequests += 1
      return route.fulfill({ json: { status: 'started', pid: 2345, logPath: 'outputs/engine-installs/qwen-nunchaku.log', message: 'Qwen Nunchaku runtime setup started.' } })
    }
    if (pathname === '/api/pro/engines/qwen_nunchaku/install-status') {
      if (state.qwenNunchakuInstallRequests > 0) {
        return route.fulfill({ json: { status: 'finished', running: false, exitCode: 0, pid: 2345, logPath: 'outputs/engine-installs/qwen-nunchaku.log', runtimeReady: true, runtimeMessages: [] } })
      }
      return route.fulfill({ json: { status: 'idle', running: false, logPath: '', runtimeReady: false, runtimeMessages: ['Runtime is not installed.'] } })
    }
    if (pathname === '/api/pro/engines/ltx/install-status') {
      if (state.ltxInstallRequests > 0) {
        state.bootstrap.models = [state.ltxReadyModel]
        state.bootstrap.blockedModels = []
        return route.fulfill({ json: { status: 'finished', running: false, exitCode: 0, pid: 1234, logPath: 'outputs/engine-installs/ltx.log' } })
      }
      return route.fulfill({ json: { status: 'idle', running: false, logPath: '' } })
    }
    if (pathname === '/api/pro/bootstrap') {
      state.bootstrapRequestCount += 1
      return route.fulfill({ json: makeBootstrap(state) })
    }
    if (pathname === '/api/pro/startup' || pathname === '/api/pro/startup/window-ready') {
      return route.fulfill({ json: { status: 'ready', serverReady: true, windowReady: true, minSplashMs: 0, readyHoldMs: 0 } })
    }
    if (pathname === '/api/pro/runtime') return route.fulfill({ json: state.runtime ?? runtime() })
    if (pathname === '/api/pro/models/load' && request.method() === 'POST') {
      const body = request.postDataJSON()
      state.modelLoadRequests.push(body)
      if (state.modelLoadResponseGate && body.modelId === state.modelLoadResponseGateId) {
        await state.modelLoadResponseGate
      }
      if (state.modelLoadTerminalFailures > 0) {
        state.modelLoadTerminalFailures -= 1
        return route.fulfill({ status: 500, json: { detail: 'Injected terminal model load failure.' } })
      }
      if (state.modelLoadFailures > 0) {
        state.modelLoadFailures -= 1
        if (state.modelLoadFailureGate) await state.modelLoadFailureGate
        return route.fulfill({ status: 409, json: { detail: 'Another model operation is in progress.' } })
      }
      return route.fulfill({ json: { runtime: { ...runtime(), loadedModel: { ...runtime().loadedModel, name: body.modelId, loaded: true }, modelLoad: { status: 'loaded', modelId: body.modelId, detail: 'Fake model loaded.' } } } })
    }
    if (pathname === '/api/pro/models/scan' && request.method() === 'POST') {
      const url = new URL(request.url())
      const assets = state.modelScanAssets ?? []
      const limit = Number(url.searchParams.get('limit') ?? 50)
      return route.fulfill({ json: {
        scanId: 'fixture-model-scan',
        inventoryCount: assets.length,
        roots: [{ label: 'Models directory', path: 'F:/models', status: 'scanned', assetCount: assets.length, familyCounts: { checkpoint: assets.length }, errorCount: 0, errors: [] }],
        assets: assets.slice(0, limit),
        offset: 0,
        limit,
        matchedCount: assets.length,
        assetsTruncated: Math.max(0, assets.length - limit),
        hasMore: assets.length > limit,
        nextOffset: assets.length > limit ? limit : null,
        query: '',
      } })
    }
    if (pathname === '/api/pro/models/scan/fixture-model-scan' && request.method() === 'GET') {
      const url = new URL(request.url())
      const assets = state.modelScanAssets ?? []
      const query = (url.searchParams.get('query') ?? '').trim().toLowerCase()
      const matches = query
        ? assets.filter((asset) => Object.values(asset).some((value) => String(value).toLowerCase().includes(query)))
        : assets
      const offset = Number(url.searchParams.get('offset') ?? 0)
      const limit = Number(url.searchParams.get('limit') ?? 50)
      const page = matches.slice(offset, offset + limit)
      const nextOffset = offset + page.length
      return route.fulfill({ json: {
        scanId: 'fixture-model-scan',
        inventoryCount: assets.length,
        roots: [{ label: 'Models directory', path: 'F:/models', status: 'scanned', assetCount: assets.length, familyCounts: { checkpoint: assets.length }, errorCount: 0, errors: [] }],
        assets: page,
        offset,
        limit,
        matchedCount: matches.length,
        assetsTruncated: Math.max(0, matches.length - nextOffset),
        hasMore: nextOffset < matches.length,
        nextOffset: nextOffset < matches.length ? nextOffset : null,
        query,
      } })
    }
    if (pathname.endsWith('/placements/preview') && request.method() === 'POST') {
      const body = request.postDataJSON()
      state.placementPreviewRequests.push(body)
      if ((state.placementPreviewFailures ?? []).includes(body.path)) {
        return route.fulfill({ status: 409, json: { detail: 'Source changed after scan; review a fresh scan.' } })
      }
      return route.fulfill({ json: {
        planId: `placement-plan-${state.placementPreviewRequests.length}`,
        scanId: body.scanId,
        source: body.path,
        destination: 'F:/AIWF_Studio/models/flux/GGUF/shared-model.gguf',
        sizeBytes: 1024,
        requiredFreeBytes: 2048,
        availableFreeBytes: 1024 * 1024,
        collision: false,
        canApply: true,
        status: 'ready',
        expiresInSeconds: 300,
      } })
    }
    if (pathname === '/api/pro/models/scan/placements/apply' && request.method() === 'POST') {
      const body = request.postDataJSON()
      state.placementApplyRequests.push(body)
      return route.fulfill({ json: {
        status: 'copied_from_shared_root',
        source: 'F:/shared-models/shared-model.gguf',
        destination: 'F:/AIWF_Studio/models/flux/GGUF/shared-model.gguf',
        sourcePreserved: true,
        inventoryRefresh: 'complete',
      } })
    }
    if (pathname === '/api/pro/models/prepare' && request.method() === 'POST') {
      const body = request.postDataJSON()
      state.routePrepareRequests.push(body)
      if (state.routePrepareFailures > 0) {
        state.routePrepareFailures -= 1
        return route.fulfill({ status: 409, json: { detail: 'Another model load or route preparation is already in progress.' } })
      }
      const ready = state.routePrepareReadiness.shift() ?? true
      const selectedModel = state.bootstrap.models.find((model) => model.id === body.checkpoint_id)
      return route.fulfill({ json: {
        ready,
        loaded: false,
        routeLifecycle: {
          route: selectedModel?.architecture === 'wan'
            ? `video.wan.${body.wan_runtime_mode ?? 'fast_5b'}`
            : 'video.test',
          modelId: body.checkpoint_id,
          supportRevision: 'fake-support-revision',
          status: ready ? 'setup-ready' : 'needs-setup',
          operationId: '',
          resident: null,
          detail: ready ? 'Fake route readiness passed.' : 'Selected video support assets are missing. Install the suggested setup bundle and retry the route check.',
        },
      } })
    }
    if (pathname === '/api/pro/downloads') {
      return route.fulfill({ json: state.downloads ?? { catalog: [], bundles: {}, counts: { catalog: 0, installed: 0 } } })
    }
    const catalogDownloadMatch = pathname.match(/^\/api\/pro\/downloads\/catalog\/([^/]+)$/)
    if (catalogDownloadMatch && request.method() === 'POST') {
      const key = decodeURIComponent(catalogDownloadMatch[1])
      state.catalogDownloads.push(key)
      const installedModelByKey = {
        'cn15-canny': { id: 'control_v11p_sd15_canny_fp16', title: 'SD 1.5 Canny', path: 'F:/models/ControlNet/control_v11p_sd15_canny_fp16.safetensors' },
        'cnxl-canny': { id: 'controlnet-canny-sdxl-1.0', title: 'SDXL Canny', path: 'F:/models/ControlNet/controlnet-canny-sdxl-1.0' },
      }
      const installedModel = installedModelByKey[key]
      if (installedModel && !state.controlNetModels.some((item) => item.id === installedModel.id)) {
        state.controlNetModels.push(installedModel)
      }
      return route.fulfill({ json: {
        bundles: state.downloads?.bundles ?? {},
        catalog: [],
        counts: { catalog: 0, installed: state.controlNetModels.length },
        catalogAction: { key, status: 'already_installed' },
      } })
    }
    const sharedSnapshotImportMatch = pathname.match(/^\/api\/pro\/downloads\/catalog\/([^/]+)\/import-shared$/)
    if (sharedSnapshotImportMatch && request.method() === 'POST') {
      state.sharedSnapshotImports.push({
        key: decodeURIComponent(sharedSnapshotImportMatch[1]),
        query: new URL(request.url()).searchParams.toString(),
      })
      return route.fulfill({ json: {
        bundles: state.downloads?.bundles ?? {},
        catalog: [],
        counts: { catalog: 0, installed: 0 },
        catalogAction: { key: decodeURIComponent(sharedSnapshotImportMatch[1]), status: 'copied_snapshot_from_shared_root' },
      } })
    }
    if (pathname === '/api/pro/enhance/models' && request.method() === 'GET') {
      return route.fulfill({ json: { models: state.enhanceModels } })
    }
    if (pathname === '/api/pro/video-lab/status') {
      return route.fulfill({ json: state.videoLabStatus ?? { vsr: { available: false, upscaleAvailable: false } } })
    }
    if (pathname === '/api/pro/controlnet/models') {
      return route.fulfill({ json: {
        models: state.controlNetModels ?? [],
        setupOptions: [
          { key: 'cn15-canny', family: 'sd15', label: 'ControlNet v1.1 Canny (SD1.5)', modelId: 'control_v11p_sd15_canny_fp16', sizeMb: 689 },
          { key: 'cn15-depth', family: 'sd15', label: 'ControlNet v1.1 Depth (SD1.5)', modelId: 'control_v11f1p_sd15_depth_fp16', sizeMb: 689 },
          { key: 'cnxl-canny', family: 'sdxl', label: 'ControlNet Canny SDXL (xinsir, diffusers folder)', modelId: 'controlnet-canny-sdxl-1.0', sizeMb: 2494 },
          { key: 'cnxl-depth', family: 'sdxl', label: 'ControlNet Depth SDXL (diffusers folder)', modelId: 'controlnet-depth-sdxl-1.0', sizeMb: 2494 },
        ],
      } })
    }
    const enhanceInstallMatch = pathname.match(/^\/api\/pro\/enhance\/models\/([^/]+)\/install$/)
    if (enhanceInstallMatch && request.method() === 'POST') {
      const modelId = decodeURIComponent(enhanceInstallMatch[1])
      state.enhanceInstallRequests.push(modelId)
      const model = state.enhanceModels.find((entry) => entry.id === modelId)
      if (!model) return route.fulfill({ status: 404, json: { detail: 'Unknown Enhance model.' } })
      model.installed = true
      return route.fulfill({ json: { model, path: `F:/AIWF_Studio/models/${model.filename}`, message: `${model.title} installed.` } })
    }
    if (pathname === '/api/pro/enhance/image' && request.method() === 'POST') {
      state.enhanceRuns.push(request.postDataJSON())
      return route.fulfill({ json: {
        status: 'completed',
        image: fallbackPng,
        url: fallbackPng,
        outputPath: 'task-temp/enhanced.png',
        width: 1,
        height: 1,
        message: 'GFPGAN v1.4 restored the image.',
        infotext: 'Restore: GFPGAN v1.4',
      } })
    }
    if (pathname.startsWith('/api/pro/downloads/bundles/') && request.method() === 'POST') {
      const bundleKey = pathname.split('/').at(-1)
      state.bundleInstalls.push(bundleKey)
      if (bundleKey === state.bundleReadyBundleKey && state.bundleReadyModel) {
        state.bootstrap.models = [state.bundleReadyModel]
        state.bootstrap.blockedModels = []
      }
      const fixtureBundleResponse = state.bundleInstallResponses?.[bundleKey]
      if (fixtureBundleResponse) return route.fulfill({ json: fixtureBundleResponse })
      return route.fulfill({
        json: {
          bundles: state.downloads?.bundles ?? {},
          catalog: [],
          counts: { catalog: 0, installed: 0 },
          bundleInstall: {
          items: ['ltx23', 'zimage', 'qwen-nunchaku'].includes(bundleKey) ? [
            { key: `${bundleKey}-model`, status: 'already_installed' },
          ] : bundleKey === 'wan-ti2v-support' ? [
            { key: 'wan-vae-22', status: 'already_installed' },
            { key: 'wan-ti2v-components', status: 'already_installed' },
          ] : [
              { key: 'flux-clip-l', status: 'already_installed' },
              { key: 'flux-ae-vae', status: 'copied_from_shared_root', source: 'F:/shared/ae.safetensors', path: 'F:/AIWF_Studio/models/flux/VAE/ae.safetensors' },
              { key: 'flux-t5-fp8', status: 'unavailable' },
              { key: 'flux-t5-tokenizer', status: 'deferred-active-operation', error: 'Another model operation is in progress.' },
            ],
          },
        },
      })
    }
    if (pathname === '/api/pro/settings') return route.fulfill({ json: state.settingsStatus ?? {} })
    if (pathname === '/api/pro/capabilities') return route.fulfill({ json: state.capabilities ?? {} })
    if (pathname === '/api/pro/data') {
      return route.fulfill({ json: { outputRoot: 'task-temp', counts: { recentOutputs: state.outputs.length }, recentOutputs: state.outputs } })
    }
    if (pathname === '/api/pro/outputs-index') return route.fulfill({ json: { outputs: state.outputs } })
    if (pathname === '/api/pro/generate') {
      const body = request.postDataJSON()
      state.generateRequests.push(body)
      if (state.generationMode === 'error') {
        return route.fulfill({ status: 500, json: { detail: 'Fake backend generation failure.' } })
      }
      if (state.generationMode === 'cancel') {
        state.generateStartedResolve?.()
        await new Promise((resolve) => setTimeout(resolve, 4000))
        try { return await route.fulfill({ status: 200, json: {} }) } catch { return }
      }
      const item = output(body.prompt)
      state.outputs = [item, ...state.outputs]
      return route.fulfill({
        json: {
          status: 'completed',
          jobId: 'playwright-job-1',
          message: 'Generated 1 image.',
          output: item,
          recentOutputs: [item],
          images: [fallbackPng],
          verificationStatus: 'verified',
          progress: [{ stage: 'complete', progress: 1, message: 'Complete', step: 4, total: 4, seconds: 0.2 }],
          timings: { elapsedSeconds: 0.2, stepsPerSecond: 20 },
          receiptPath: item.receiptPath,
        },
      })
    }
    // unified workspace routes go to a per-test fake bridge; without one they are absent (404)
    if (pathname.startsWith('/api/pro/unified/')) {
      if (state.unified) return state.unified(route, request, pathname)
      return route.fulfill({ status: 404, json: { detail: 'Not Found' } })
    }
    if (pathname === '/api/pro/interrupt') {
      state.interruptCalls += 1
      return route.fulfill({ json: { status: 'interrupt_requested', videoJobId: '' } })
    }
    return route.fulfill({ json: {} })
  })

  await page.goto(`${baseUrl}${state.initialHash ?? ''}`, { waitUntil: 'domcontentloaded' })
  if (waitForPrompt) await page.locator('textarea[aria-label="Prompt"]').waitFor({ state: 'visible' })
  await page.getByText('Connected to /api/pro/bootstrap.').first().waitFor({ state: 'visible' })
  return { context, page, requests, consoleErrors }
}

function createState(models, defaultModelId, changes = {}) {
  return {
    bootstrap: { models, defaultModelId, blockedModels: [], ...changes.bootstrap },
    bootstrapRequestCount: 0,
    bootstrapRefreshDefaultModelId: changes.bootstrapRefreshDefaultModelId,
    bootstrapRefreshDefaultModelIds: changes.bootstrapRefreshDefaultModelIds,
    runtime: changes.runtime,
    generationMode: changes.generationMode ?? 'success',
    generateRequests: [],
    modelLoadRequests: [],
    modelLoadFailures: 0,
    placementPreviewRequests: [],
    placementApplyRequests: [],
    modelLoadFailureGate: null,
    routePrepareRequests: [],
    routePrepareFailures: 0,
    routePrepareReadiness: [],
    bundleInstalls: [],
    catalogDownloads: [],
    sharedSnapshotImports: [],
    bundleInstallResponses: changes.bundleInstallResponses,
    ltxInstallRequests: 0,
    qwenNunchakuInstallRequests: 0,
    ltxReadyModel: null,
    bundleReadyModel: null,
    bundleReadyBundleKey: changes.bundleReadyBundleKey ?? 'zimage',
    enhanceModels: changes.enhanceModels ?? [],
    videoLabStatus: changes.videoLabStatus,
    enhanceInstallRequests: [],
    enhanceRuns: [],
    controlNetModels: changes.controlNetModels ?? [],
    downloads: changes.downloads,
    settingsStatus: changes.settingsStatus,
    capabilities: changes.capabilities,
    interruptCalls: 0,
    outputs: [],
    generateStartedResolve: null,
  }
}

before(async () => {
  artifactRoot = process.env.AIWF_E2E_ARTIFACT_DIR ?? await mkdtemp(path.join(tmpdir(), 'aiwf-studio-e2e-'))
  await mkdir(path.join(artifactRoot, 'vite-cache'), { recursive: true })
  process.env.AIWF_PRO_API_TARGET = 'http://127.0.0.1:9'
  vite = await createServer({
    root: frontendRoot,
    configFile: path.join(frontendRoot, 'vite.config.ts'),
    configLoader: 'runner',
    cacheDir: path.join(artifactRoot, 'vite-cache'),
    logLevel: 'error',
    server: { host: '127.0.0.1', port: 0, strictPort: false },
  })
  await vite.listen()
  const address = vite.httpServer.address()
  assert.ok(address && typeof address !== 'string')
  baseUrl = `http://127.0.0.1:${address.port}/`
  browser = await chromium.launch({
    headless: true,
    ...(existsSync(chromePath) ? { executablePath: chromePath } : {}),
    args: ['--disable-gpu'],
  })
})

after(async () => {
  await browser?.close()
  await vite?.close()
  if (artifactRoot && !process.env.AIWF_E2E_ARTIFACT_DIR) await rm(artifactRoot, { recursive: true, force: true })
})

test('Enhance blocks Run until a missing built-in model is explicitly installed', async () => {
  const image = model('image-base', 'Image Base')
  const state = createState([image], image.id, {
    enhanceModels: [
      { id: 'gfpgan-v1.4', title: 'GFPGAN v1.4', filename: 'GFPGANv1.4.pth', kind: 'restorer', architecture: 'GFPGAN', scale: 1, installed: false, installAvailable: true },
      { id: 'realesrgan-x4plus', title: 'RealESRGAN 4x+', filename: 'RealESRGAN_x4plus.pth', kind: 'upscaler', architecture: 'ESRGAN', scale: 4, installed: false, installAvailable: true },
    ],
  })
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'Enhance', exact: true }).first().click()
    const runButton = page.getByRole('button', { name: 'Run', exact: true })
    await page.getByRole('button', { name: 'Install GFPGAN v1.4', exact: true }).waitFor({ state: 'visible' })
    assert.equal(await runButton.isDisabled(), true)

    await page.getByRole('button', { name: 'Install GFPGAN v1.4', exact: true }).click()
    await runButton.waitFor({ state: 'visible' })
    const deadline = Date.now() + 5000
    while (await runButton.isDisabled() && Date.now() < deadline) await page.waitForTimeout(50)
    assert.equal(state.enhanceInstallRequests.length, 1)
    assert.equal(state.enhanceModels[0].installed, true)
    assert.equal(await runButton.isDisabled(), false)

    await runButton.click()
    await page.getByRole('dialog', { name: 'Enhance / VSR' }).getByRole('status')
      .getByText('GFPGAN v1.4 restored the image.', { exact: true }).waitFor({ state: 'visible' })
    assert.equal(state.enhanceRuns.length, 1)
    assert.equal(state.enhanceRuns[0].restoreModel, 'gfpgan-v1.4')
  } finally {
    await context.close()
  }
})

test('Enhance VSR stays blocked when VideoFX upscale is unavailable', async () => {
  const image = model('image-base', 'Image Base')
  const state = createState([image], image.id, {
    videoLabStatus: { vsr: { available: true, upscaleAvailable: false } },
  })
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'Enhance', exact: true }).first().click()
    const dialog = page.getByRole('dialog', { name: 'Enhance / VSR' })
    await dialog.locator('select').first().selectOption('vsr')
    await dialog.getByText('NVIDIA VideoFX is unavailable. Install or configure the VideoFX runtime before using VSR.', { exact: true }).waitFor({ state: 'visible' })
    assert.equal(await dialog.getByRole('button', { name: 'Run', exact: true }).isDisabled(), true)
    assert.equal(state.enhanceRuns.length, 0)
  } finally {
    await context.close()
  }
})

test('Enhance VSR becomes runnable only after VideoFX upscale readiness is confirmed', async () => {
  const image = model('image-base', 'Image Base')
  const state = createState([image], image.id, {
    videoLabStatus: { vsr: { available: true, upscaleAvailable: true } },
  })
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'Enhance', exact: true }).first().click()
    const dialog = page.getByRole('dialog', { name: 'Enhance / VSR' })
    await dialog.locator('select').first().selectOption('vsr')
    const runButton = dialog.getByRole('button', { name: 'Run', exact: true })
    await runButton.waitFor({ state: 'visible' })
    const deadline = Date.now() + 5000
    while (await runButton.isDisabled() && Date.now() < deadline) await page.waitForTimeout(50)
    assert.equal(await runButton.isDisabled(), false)
    assert.equal(state.enhanceRuns.length, 0)
  } finally {
    await context.close()
  }
})

test('a stale selected ID after refresh stays blocked until the user chooses a recovery model', async () => {
  const healthy = model('healthy-sdxl', 'Healthy SDXL')
  const state = createState([healthy], 'removed-sdxl')
  const { context, page, requests } = await openStudio(state)
  try {
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    assert.equal(await modelSelect.inputValue(), 'removed-sdxl')
    assert.match(await modelSelect.locator('option:checked').innerText(), /Unavailable model: removed-sdxl/)
    await page.getByRole('alert').filter({ hasText: 'Selected model is not available' }).waitFor({ state: 'visible' })
    const generate = page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button')
    assert.equal(await generate.isDisabled(), true)
    await generate.click({ force: true })
    assert.equal(state.generateRequests.length, 0)
    await page.screenshot({ path: path.join(artifactRoot, 'stale-model-blocked.png'), fullPage: true })

    await page.getByRole('button', { name: 'Use Healthy SDXL' }).click()
    assert.equal(await modelSelect.inputValue(), 'healthy-sdxl')
    assert.equal(await page.getByRole('alert').count(), 0)
    await generate.click()
    await page.getByRole('button', { name: /Show output settings for A small red apple/ }).waitFor({ state: 'visible' })
    assert.equal(state.generateRequests[0]?.checkpoint_id, 'healthy-sdxl')
    await page.screenshot({ path: path.join(artifactRoot, 'stale-model-recovery.png'), fullPage: true })
    assert.deepEqual(requests.filter((item) => item.pathname === '/api/pro/generate').map((item) => item.method), ['POST'])
  } finally {
    await context.close()
  }
})

test('Settings model inventory proposals can page past 250 and search later assets', async () => {
  const assets = Array.from({ length: 300 }, (_, index) => ({
    path: `F:/models/asset-${String(index).padStart(3, '0')}.safetensors`,
    filename: `asset-${String(index).padStart(3, '0')}.safetensors`,
    family: 'checkpoint',
    architecture: 'sdxl',
    currentSubdir: '',
    recommendedSubdir: 'Stable-diffusion',
    placement: 'candidate',
    signals: {},
  }))
  const state = createState([], '')
  state.modelScanAssets = assets
  const { context, page, requests } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'Open settings' }).click()
    await page.getByRole('tab', { name: 'System' }).click()
    await page.getByRole('button', { name: 'Scan model roots' }).click()
    await page.getByText('Scanned 300 model assets across configured roots.', { exact: false }).waitFor({ state: 'visible' })
    await page.getByText(/Review model asset placements/).click()
    const proposals = page.getByRole('list', { name: 'Discovered model asset placement proposals' })
    await proposals.getByText('asset-000.safetensors', { exact: false }).first().waitFor({ state: 'visible' })
    assert.equal(await proposals.getByText('asset-250.safetensors', { exact: false }).count(), 0)
    await page.getByText(/250 more available on later pages/).waitFor({ state: 'visible' })

    await page.getByRole('button', { name: 'Next' }).click()
    await proposals.getByText('asset-050.safetensors', { exact: false }).first().waitFor({ state: 'visible' })
    assert.equal(await proposals.getByText('asset-000.safetensors', { exact: false }).count(), 0)
    await page.getByRole('button', { name: 'Next' }).click()
    await proposals.getByText('asset-100.safetensors', { exact: false }).first().waitFor({ state: 'visible' })

    const search = page.getByRole('searchbox', { name: 'Search discovered model assets' })
    await search.fill('asset-275.safetensors')
    await page.getByRole('button', { name: 'Search proposals' }).click()
    await proposals.getByText('asset-275.safetensors', { exact: false }).first().waitFor({ state: 'visible' })
    assert.match(await page.locator('.pro-model-scan-assets').innerText(), /Showing 1–1 of 1 matching proposals/)
    assert.ok(requests.some((request) => request.pathname === '/api/pro/models/scan/fixture-model-scan' && request.url.includes('asset-275.safetensors')))
  } finally {
    await context.close()
  }
})

test('a model removed from the refreshed inventory is not replaced by a different ID', async () => {
  const selected = model('selected-sdxl', 'Selected SDXL')
  const recovery = model('recovery-sdxl', 'Recovery SDXL')
  const state = createState([selected, recovery], selected.id)
  const { context, page, requests } = await openStudio(state)
  try {
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    assert.equal(await modelSelect.inputValue(), selected.id)
    state.bootstrap.models = [recovery]
    await page.reload({ waitUntil: 'domcontentloaded' })
    await page.locator('textarea[aria-label="Prompt"]').waitFor({ state: 'visible' })
    await page.getByRole('alert').filter({ hasText: 'Selected model is not available' }).waitFor({ state: 'visible' })
    assert.equal(await modelSelect.inputValue(), 'selected-sdxl')
    assert.match(await modelSelect.locator('option:checked').innerText(), /Unavailable model: selected-sdxl/)
    const generate = page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button')
    assert.equal(await generate.isDisabled(), true)
    await page.getByRole('button', { name: 'Use Recovery SDXL' }).click()
    assert.equal(await modelSelect.inputValue(), 'recovery-sdxl')
    assert.equal(await page.getByRole('alert').count(), 0)
  } finally {
    await context.close()
  }
})

test('returning to Create preserves an unavailable model until explicit recovery', async () => {
  const healthy = model('healthy-sdxl', 'Healthy SDXL')
  const state = createState([healthy], 'removed-sdxl')
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('complementary', { name: 'Subnavigation' }).getByRole('button', { name: 'Data' }).click()
    await page.getByRole('complementary', { name: 'Subnavigation' }).getByRole('button', { name: 'Create' }).click()
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    await page.getByRole('alert').filter({ hasText: 'Selected model is not available' }).waitFor({ state: 'visible' })
    assert.equal(await modelSelect.inputValue(), 'removed-sdxl')
    assert.equal(await page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').isDisabled(), true)
    await page.getByRole('button', { name: 'Use Healthy SDXL' }).click()
    assert.equal(await modelSelect.inputValue(), 'healthy-sdxl')
  } finally {
    await context.close()
  }
})

test('an explicitly empty inventory does not invent fallback models', async () => {
  const state = createState([], 'phantom-model')
  const { context, page } = await openStudio(state)
  try {
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    assert.equal(await modelSelect.inputValue(), 'phantom-model')
    assert.equal(await modelSelect.locator('option').count(), 1)
    assert.match(await modelSelect.locator('option:checked').innerText(), /Unavailable model: phantom-model/)
    await page.getByRole('alert').filter({ hasText: 'Selected model is not available' }).waitFor({ state: 'visible' })
    assert.equal(await page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').isDisabled(), true)
    assert.equal(await page.getByRole('button', { name: 'Open model inventory' }).count(), 1)
  } finally {
    await context.close()
  }
})

test('switching from image to video selects the first ready video model', async () => {
  const image = model('image-base', 'Image Base')
  const wan = model('wan-video', 'Wan Video', {
    architecture: 'wan',
    engineId: 'wan',
    engineLabel: 'Wan',
    kind: 'video',
    mode: 'video',
    status: 'Ready',
    checkpointPathStatus: undefined,
    routeStatus: undefined,
  })
  const state = createState([image, wan], image.id, { bootstrap: { mode: 'image' } })
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'Video', exact: true }).click()
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    assert.equal(await modelSelect.inputValue(), 'wan-video')
    assert.equal(await page.getByRole('alert').count(), 0)
    assert.equal(await page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').isDisabled(), false)
    const runtimePanel = page.getByRole('complementary', { name: 'Runtime status' })
    await runtimePanel.getByText('Image model cache (video route selected)', { exact: true }).waitFor({ state: 'visible' })
    await runtimePanel.getByText('Route setup ready · residency unconfirmed').waitFor({ state: 'visible' })
    await runtimePanel.getByText('Fake route readiness passed. Residency: not reported by this backend.').waitFor({ state: 'visible' })
    assert.equal(state.routePrepareRequests.length, 1)
    assert.equal(state.routePrepareRequests[0].checkpoint_id, 'wan-video')
    await runtimePanel.getByText('Different model is loaded').waitFor({ state: 'hidden' })
  } finally {
    await context.close()
  }
})

test('startup restores Video mode when the saved default model is a video route', async () => {
  const image = model('image-base', 'Image Base')
  const wan = model('wan-video', 'Wan Video', {
    architecture: 'wan',
    engineId: 'wan',
    engineLabel: 'Wan',
    kind: 'video',
    mode: 'video',
    status: 'Ready',
    checkpointPathStatus: undefined,
    routeStatus: undefined,
  })
  const state = createState([image, wan], wan.id, { bootstrap: { mode: 'image' } })
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('complementary', { name: 'Runtime status' })
      .getByText('Route setup ready · residency unconfirmed').waitFor({ state: 'visible' })
    assert.equal(state.routePrepareRequests.length, 1)
    assert.equal(state.routePrepareRequests[0].checkpoint_id, 'wan-video')
    assert.equal(state.routePrepareRequests[0].mode, 'video')
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    assert.equal(await modelSelect.inputValue(), 'wan-video')
    assert.equal(await page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').isDisabled(), false)
  } finally {
    await context.close()
  }
})

test('startup selects a confirmed image fallback when the saved choice is blocked', async () => {
  const fallback = model('startup-fallback-sdxl', 'Startup fallback SDXL')
  const blocked = model('saved-blocked-sdxl', 'Saved blocked SDXL', {
    status: 'Blocked',
    routeStatus: 'blocked',
    reason: 'Required support assets are missing.',
  })
  const runtimeSnapshot = runtime()
  runtimeSnapshot.loadedModel = { ...runtimeSnapshot.loadedModel, id: fallback.id, name: fallback.title, loaded: true }
  runtimeSnapshot.modelLoad = { status: 'loaded', modelId: fallback.id, detail: 'Startup fallback loaded.' }
  const state = createState([fallback], blocked.id, {
    bootstrap: { blockedModels: [blocked] },
    bootstrapRefreshDefaultModelId: fallback.id,
    runtime: runtimeSnapshot,
  })
  const { context, page } = await openStudio(state)
  try {
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    await page.waitForFunction((modelId) => {
      const select = document.querySelectorAll('aside[aria-label="Prompt and generation settings"] select')[1]
      return select instanceof HTMLSelectElement && select.value === modelId
    }, fallback.id)
    assert.equal(await modelSelect.inputValue(), fallback.id)
    assert.equal(await page.getByRole('alert').count(), 0)
    await page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').click()
    await page.getByText('Generated 1 image.').waitFor({ state: 'visible' })
    assert.equal(state.generateRequests.length, 1)
    assert.equal(state.generateRequests[0].checkpoint_id, fallback.id)
  } finally {
    await context.close()
  }
})

test('startup retries fallback reconciliation when the first bootstrap read is stale', async () => {
  const fallback = model('startup-retry-fallback-sdxl', 'Startup retry fallback SDXL')
  const blocked = model('startup-retry-blocked-sdxl', 'Startup retry blocked SDXL', {
    status: 'Blocked',
    routeStatus: 'blocked',
    reason: 'Required support assets are missing.',
  })
  const runtimeSnapshot = runtime()
  runtimeSnapshot.loadedModel = { ...runtimeSnapshot.loadedModel, id: fallback.id, name: fallback.title, loaded: true }
  runtimeSnapshot.modelLoad = { status: 'loaded', modelId: fallback.id, detail: 'Startup fallback loaded.' }
  const state = createState([fallback], blocked.id, {
    bootstrap: { blockedModels: [blocked] },
    bootstrapRefreshDefaultModelIds: [blocked.id, fallback.id],
    runtime: runtimeSnapshot,
  })
  const { context, page } = await openStudio(state)
  try {
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    await page.waitForFunction((modelId) => {
      const select = document.querySelectorAll('aside[aria-label="Prompt and generation settings"] select')[1]
      return select instanceof HTMLSelectElement && select.value === modelId
    }, fallback.id)
    assert.ok(state.bootstrapRequestCount >= 3, 'initial bootstrap plus stale and current reconciliation reads')
    assert.equal(await modelSelect.inputValue(), fallback.id)
  } finally {
    await context.close()
  }
})

test('Wan runtime status does not reuse a resident receipt for a different variant', async () => {
  const wan = model('wan-video', 'Wan Video', {
    architecture: 'wan',
    engineId: 'wan',
    engineLabel: 'Wan',
    kind: 'video',
    mode: 'video',
    status: 'Ready',
    checkpointPathStatus: undefined,
    routeStatus: undefined,
  })
  const runtimeSnapshot = runtime()
  runtimeSnapshot.routeLifecycle = [{
    route: 'video.wan.high_low',
    modelId: wan.id,
    status: 'ready',
    resident: true,
    detail: 'Previously prepared high-low variant.',
  }]
  const state = createState([wan], wan.id, {
    bootstrap: { mode: 'video' },
    runtime: runtimeSnapshot,
  })
  const { context, page } = await openStudio(state)
  try {
    const runtimePanel = page.getByRole('complementary', { name: 'Runtime status' })
    await runtimePanel.getByText('Route readiness is checked before a job starts', { exact: true }).waitFor({ state: 'visible' })
    await runtimePanel.getByText('Previously prepared high-low variant.', { exact: true }).waitFor({ state: 'hidden' })
  } finally {
    await context.close()
  }
})

test('explicit video startup hash still restores the saved video model selection', async () => {
  const wan = model('wan-video', 'Wan Video', {
    architecture: 'wan',
    engineId: 'wan',
    engineLabel: 'Wan',
    kind: 'video',
    mode: 'video',
    status: 'Ready',
    checkpointPathStatus: undefined,
    routeStatus: undefined,
  })
  const state = createState([wan], wan.id, { bootstrap: { mode: 'image' }, initialHash: '#video' })
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('complementary', { name: 'Runtime status' })
      .getByText('Route setup ready · residency unconfirmed').waitFor({ state: 'visible' })
    assert.equal(state.routePrepareRequests.length, 1)
    assert.equal(state.routePrepareRequests[0].checkpoint_id, wan.id)
    assert.equal(await page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1).inputValue(), wan.id)
  } finally {
    await context.close()
  }
})

test('an unrecognized startup hash does not suppress saved video model restoration', async () => {
  const wan = model('wan-video', 'Wan Video', {
    architecture: 'wan',
    engineId: 'wan',
    engineLabel: 'Wan',
    kind: 'video',
    mode: 'video',
    status: 'Ready',
    checkpointPathStatus: undefined,
    routeStatus: undefined,
  })
  const state = createState([wan], wan.id, { bootstrap: { mode: 'image' }, initialHash: '#unknown-route' })
  const { context, page } = await openStudio(state)
  try {
    const deadline = Date.now() + 5000
    while (state.routePrepareRequests.length === 0 && Date.now() < deadline) {
      await page.waitForTimeout(50)
    }
    assert.equal(state.routePrepareRequests.length, 1)
    assert.equal(state.routePrepareRequests[0].mode, 'video')
  } finally {
    await context.close()
  }
})

test('video route preparation retries a transient model-operation conflict on selection', async () => {
  const image = model('image-base', 'Image Base')
  const wan = model('wan-video', 'Wan Video', {
    architecture: 'wan',
    engineId: 'wan',
    engineLabel: 'Wan',
    kind: 'video',
    mode: 'video',
    status: 'Ready',
    checkpointPathStatus: undefined,
    routeStatus: undefined,
  })
  const state = createState([image, wan], image.id, { bootstrap: { mode: 'image' } })
  state.settingsStatus = { video: { wanHigh: 'wan-high-q4.gguf', wanLow: 'wan-low-q4.gguf', wanVae: 'wan-video-vae.safetensors', wanTextEncoder: 'umt5-xxl-encoder.safetensors' } }
  state.routePrepareFailures = 1
  const { context, page } = await openStudio(state)
  try {
    await page.waitForTimeout(250)
    await page.getByRole('button', { name: 'Video', exact: true }).click()
    const runtimePanel = page.getByRole('complementary', { name: 'Runtime status' })
    const routeDeadline = Date.now() + 5000
    while (state.routePrepareRequests.length < 2 && Date.now() < routeDeadline) await page.waitForTimeout(50)
    assert.equal(state.routePrepareRequests.length, 2, `selection should retry the transient 409 once; remainingFailures=${state.routePrepareFailures}`)
    await runtimePanel.getByText('Route setup ready · residency unconfirmed').waitFor({ state: 'visible' })
    const generateButton = page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button')
    assert.equal(await generateButton.isDisabled(), false)
    assert.equal(state.routePrepareRequests[1].checkpoint_id, 'wan-video')
    for (const request of [state.routePrepareRequests[0], state.routePrepareRequests[1]]) {
      assert.equal(request.high_noise_model_id, 'wan-high-q4.gguf')
      assert.equal(request.low_noise_model_id, 'wan-low-q4.gguf')
      assert.equal(request.vae_id, 'wan-video-vae.safetensors')
      assert.equal(request.text_encoder_path, 'umt5-xxl-encoder.safetensors')
    }
    await generateButton.click()
    await page.getByText('Generated 1 image.').first().waitFor({ state: 'visible' })
    assert.equal(state.generateRequests.length, 1)
    assert.equal(state.generateRequests[0].checkpoint_id, 'wan-video')
    assert.equal(state.generateRequests[0].high_noise_model_id, 'wan-high-q4.gguf')
    assert.equal(state.generateRequests[0].low_noise_model_id, 'wan-low-q4.gguf')
    assert.equal(state.generateRequests[0].vae_id, 'wan-video-vae.safetensors')
    assert.equal(state.generateRequests[0].text_encoder_path, 'umt5-xxl-encoder.safetensors')
  } finally {
    await context.close()
  }
})

test('video route with incomplete setup stays blocked until an explicit readiness retry succeeds', async () => {
  const image = model('image-base', 'Image Base')
  const wan = model('wan-video', 'Wan Video', {
    architecture: 'wan',
    engineId: 'wan',
    engineLabel: 'Wan',
    kind: 'video',
    mode: 'video',
    status: 'Ready',
    checkpointPathStatus: undefined,
    routeStatus: undefined,
  })
  const state = createState([image, wan], image.id, { bootstrap: { mode: 'image' } })
  state.routePrepareReadiness = [false, true]
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'Video', exact: true }).click()
    const alert = page.getByRole('alert')
    await alert.getByText('Video route setup incomplete', { exact: true }).waitFor({ state: 'visible' })
    await alert.getByText(/Selected video support assets are missing/).waitFor({ state: 'visible' })
    const generateButton = page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button')
    assert.equal(await generateButton.isDisabled(), true)
    assert.equal(state.routePrepareRequests.length, 1)

    await page.keyboard.press('Control+k')
    const commandPalette = page.getByRole('dialog', { name: 'Command palette' })
    await commandPalette.getByRole('button', { name: /Run current generation/ }).click()
    assert.equal(state.generateRequests.length, 0, 'command palette must respect the failed video readiness guard')

    await page.getByRole('button', { name: 'Retry video route check' }).click()
    await page.getByRole('complementary', { name: 'Runtime status' })
      .getByText('Route setup ready · residency unconfirmed').waitFor({ state: 'visible' })
    assert.equal(state.routePrepareRequests.length, 2)
    assert.equal(await generateButton.isDisabled(), false)
  } finally {
    await context.close()
  }
})

test('LTX route setup installs model assets and worker before refreshing readiness', async () => {
  const ltx = model('ltx:distilled', 'LTX 2.3 Distilled', {
    architecture: 'ltx',
    engineId: 'ltx',
    engineLabel: 'LTX Video',
    kind: 'video',
    mode: 'video',
    status: 'blocked-runtime',
    routeStatus: 'blocked',
    setupBundleKey: 'ltx23',
    setupRoute: {
      routeKey: 'pro.video.ltx.distilled',
      modality: 'video',
      supportState: 'supported',
      preflightKey: 'preflight_ltx_pipeline',
      setupBundleKey: 'ltx23',
      setupAction: 'POST /api/pro/engines/ltx/install',
    },
    reason: 'LTX engine worker is unavailable.',
    suggestedAction: 'Install or enable the LTX 2.3 engine worker from Gradio Settings.',
  })
  const ready = { ...ltx, status: 'Ready', routeStatus: 'request-eligible', reason: '', suggestedAction: '' }
  const state = createState([], ltx.id, { bootstrap: { mode: 'video', blockedModels: [ltx] }, initialHash: '#video' })
  state.ltxReadyModel = ready
  const { context, page, requests } = await openStudio(state)
  try {
    const generateButton = page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button')
    assert.equal(await generateButton.isDisabled(), true)
    await page.getByRole('button', { name: 'Set up LTX model and worker' }).click()
    await page.getByText(/LTX 2\.3 Distilled route is ready on demand/).first().waitFor({ state: 'visible' })
    await page.getByRole('complementary', { name: 'Runtime status' })
      .getByText('Route setup ready · residency unconfirmed').waitFor({ state: 'visible' })
    assert.equal(await generateButton.isDisabled(), false)
    assert.deepEqual(state.bundleInstalls, ['ltx23'])
    assert.equal(state.ltxInstallRequests, 1)
    assert.equal(requests.filter((item) => item.pathname === '/api/pro/downloads/bundles/ltx23').length, 1)
    assert.ok(requests.filter((item) => item.pathname === '/api/pro/bootstrap').length >= 2)
    assert.equal(state.routePrepareRequests.length, 2)
    assert.equal(state.generateRequests.length, 0, 'readiness verification must not run generation')
  } finally {
    await context.close()
  }
})

test('Flux universal conditioning explains manual prerequisites when no generic installer exists', async () => {
  const base = model('base-sdxl', 'Base SDXL')
  const universal = model('flux-universal', 'Flux Universal Conditioning', {
    engineId: 'flux',
    engineLabel: 'Flux',
    status: 'missing-assets',
    routeStatus: 'missing-assets',
    setupRoute: {
      routeKey: 'pro.image.flux-universal-conditioning',
      modality: 'image',
      supportState: 'runtime-dependent',
      preflightKey: 'flux_conditioning',
      limitation: 'Manual setup required: AIWF does not currently provide a verified installation source for the ute.integrations runtime package. Do not install it by package name alone.',
    },
    reason: 'Flux universal conditioning assets are missing.',
  })
  const state = createState([base], base.id, { bootstrap: { mode: 'image', blockedModels: [universal] } })
  const { context, page } = await openStudio(state)
  try {
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    const optionValues = await modelSelect.locator('option').evaluateAll((options) => options.map((option) => option.value))
    assert.ok(optionValues.includes(universal.id), 'manual-setup route should remain discoverable in the model picker')
    await modelSelect.selectOption(universal.id)
    await page.getByText(/AIWF does not currently provide a verified installation source/).waitFor({ state: 'visible' })
    assert.equal(await page.getByRole('button', { name: /Set up Flux Universal Conditioning/ }).count(), 0)
    assert.equal(await page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').isDisabled(), true)
    assert.equal(state.bundleInstalls.length, 0)
  } finally {
    await context.close()
  }
})

test('Qwen Nunchaku setup installs assets and runtime while keeping generation blocked pending a real smoke test', async () => {
  const qwen = model('qwen-image-nunchaku-int4', 'Qwen Image Nunchaku INT4', {
    architecture: 'qwen_image_nunchaku',
    engineId: 'qwen_nunchaku',
    engineLabel: 'Qwen Image Nunchaku',
    kind: 'image',
    mode: 'image',
    status: 'blocked-runtime',
    routeStatus: 'blocked',
    setupBundleKey: 'qwen-nunchaku',
    setupRoute: {
      routeKey: 'pro.image.qwen-nunchaku',
      modality: 'image',
      supportState: 'setup-available',
      preflightKey: 'preflight_qwen_nunchaku_pipeline',
      setupBundleKey: 'qwen-nunchaku',
      setupAction: 'POST /api/pro/engines/qwen_nunchaku/install',
    },
    reason: 'Runtime setup or a real generation smoke test is pending.',
    suggestedAction: 'Set up model assets and the isolated Qwen Nunchaku runtime.',
  })
  const state = createState([], qwen.id, { bootstrap: { mode: 'image', blockedModels: [qwen] } })
  const { context, page, requests } = await openStudio(state)
  try {
    const generateButton = page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button')
    assert.equal(await generateButton.isDisabled(), true)
    assert.match(await page.locator('body').innerText(), /Qwen Image Nunchaku/, 'setup candidate should be visible in the model picker')
    const buttonLabels = (await page.locator('button').allTextContents()).join(' | ')
    const selectedId = await page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1).inputValue()
    assert.equal(selectedId, qwen.id, `unexpected selected model; available buttons: ${buttonLabels}`)
    assert.ok(buttonLabels.includes('Set up Qwen Nunchaku model and runtime'), `setup action not rendered; available buttons: ${buttonLabels}`)
    await page.getByRole('button', { name: 'Set up Qwen Nunchaku model and runtime' }).click()
    await page.getByText(/Generation remains blocked pending a real route smoke test/).waitFor({ state: 'visible' })
    assert.equal(state.qwenNunchakuInstallRequests, 1)
    assert.equal(await generateButton.isDisabled(), true)
    assert.deepEqual(state.bundleInstalls, ['qwen-nunchaku'])
    assert.equal(state.qwenNunchakuInstallRequests, 1)
    assert.equal(requests.filter((item) => item.pathname === '/api/pro/downloads/bundles/qwen-nunchaku').length, 1)
    assert.equal(requests.filter((item) => item.pathname === '/api/pro/engines/qwen_nunchaku/install-status').length >= 1, true)
    assert.equal(state.generateRequests.length, 0)
  } finally {
    await context.close()
  }
})

test('tools readiness card explains when its counts come from a cached snapshot', async () => {
  const image = model('image-base', 'Image Base')
  const state = createState([image], image.id, {
    capabilities: {
      readiness: {
        counts: { working: 1, 'metadata-only': 0, 'blocked-cleanly': 0, 'broken-runtime': 0, 'unsupported-no-route': 0 },
        families: [{ family: 'Flux', total: 1, counts: { working: 1 } }],
        working: [],
        needsWork: [],
        total: 1,
        error: '',
        sourceMessage: 'Using cached readiness ledger from snapshot.json (2026-06-28); live refresh runs in the background.',
      },
    },
  })
  state.initialHash = '#tools'
  const { context, page } = await openStudio(state, { waitForPrompt: false })
  try {
    assert.equal(await page.locator('.aiwf-pro-shell').getAttribute('data-rail'), 'tools')
    assert.match(await page.locator('body').innerText(), /Using cached readiness ledger from snapshot\.json/)
  } finally {
    await context.close()
  }
})

test('switching to Wan TI2V 5B does not restore the saved Wan 2.1 VAE', async () => {
  const wan = model('wan-ti2v-5b', 'Wan TI2V 5B', {
    architecture: 'wan',
    engineId: 'wan',
    engineLabel: 'Wan',
    kind: 'video',
    mode: 'video',
    status: 'Ready',
    checkpointPathStatus: undefined,
    routeStatus: undefined,
  })
  const state = createState([wan], wan.id, { bootstrap: { mode: 'video' } })
  state.settingsStatus = { video: {
    wanHigh: 'wan-high-noise-14b.safetensors',
    wanLow: 'wan-low-noise-14b.safetensors',
    wanVae: 'wan2.1_vae.safetensors',
    wanTextEncoder: 'umt5-xxl-encoder.safetensors',
    wanRuntimeMode: 'fast_5b',
  } }
  const { context, page } = await openStudio(state)
  try {
    const deadline = Date.now() + 5000
    while (state.routePrepareRequests.length === 0 && Date.now() < deadline) await page.waitForTimeout(50)
    await page.waitForTimeout(250)
    assert.ok(state.routePrepareRequests.length > 0)
    for (const request of state.routePrepareRequests) {
      assert.equal(request.wan_runtime_mode, 'fast_5b')
      assert.notEqual(request.vae_id, 'wan2.1_vae.safetensors')
    }
  } finally {
    await context.close()
  }
})

test('video mode selects a ready Wan model when the saved Sana model is unavailable', async () => {
  const sana = model('sana-video', 'Sana Video', {
    architecture: 'sana_video',
    engineId: 'sana_video',
    engineLabel: 'Sana Video',
    kind: 'video',
    status: 'Needs snapshot',
  })
  const wan = model('wan-video', 'Wan Video', {
    architecture: 'wan',
    engineId: 'wan',
    engineLabel: 'Wan',
    kind: 'video',
    mode: 'video',
    status: 'Ready',
    checkpointPathStatus: undefined,
    routeStatus: undefined,
  })
  const state = createState([sana, wan], sana.id, { bootstrap: { mode: 'video' } })
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'Video', exact: true }).click()
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    assert.equal(await modelSelect.inputValue(), 'wan-video')
    assert.equal(await page.getByRole('alert').count(), 0)
    assert.equal(await page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').isDisabled(), false)
  } finally {
    await context.close()
  }
})

test('changing Sana audio requirements reruns route preparation', async () => {
  const sana = model('sana-video', 'Sana Video', {
    architecture: 'sana_video',
    engineId: 'sana_video',
    engineLabel: 'Sana Video',
    kind: 'video',
    status: 'Ready',
  })
  const state = createState([sana], sana.id, {
    bootstrap: { mode: 'video', defaultSettings: { generateAudio: false } },
    initialHash: '#video',
  })
  const { context, page } = await openStudio(state)
  try {
    const runtimePanel = page.getByRole('complementary', { name: 'Runtime status' })
    await runtimePanel.getByText('Route setup ready · residency unconfirmed').waitFor({ state: 'visible' })
    assert.equal(state.routePrepareRequests.length, 1)
    assert.equal(state.routePrepareRequests[0].generate_audio, false)
    const addAudio = page.locator('label.pro-toggle').filter({ hasText: 'Add audio' }).locator('input[type="checkbox"]')
    const nextPrepare = page.waitForResponse((response) => new URL(response.url()).pathname === '/api/pro/models/prepare')
    await addAudio.check()
    await nextPrepare
    assert.equal(state.routePrepareRequests.length, 2)
    assert.equal(state.routePrepareRequests[1].generate_audio, true)
  } finally {
    await context.close()
  }
})

test('LTX selection prefers an eligible route, submits its family payload, and constrains the 2B path', async () => {
  const image = model('image-base', 'Image Base')
  const sana = model('sana-video-ready', 'Sana Video', {
    architecture: 'sana_video', engineId: 'sana_video', engineLabel: 'Sana Video', kind: 'video', status: 'Ready',
  })
  const ltx23 = model('ltx:distilled', 'LTX 2.3 Distilled', {
    architecture: 'ltx', engineId: 'ltx', engineLabel: 'LTX Video', kind: 'video', status: 'Ready',
    generationPreset: { width: 768, height: 512, steps: 20, cfgScale: 1 },
  })
  const state = createState([image, sana, ltx23], image.id, {
    bootstrap: { mode: 'image', defaultSettings: { frames: 80 } },
  })
  const first = await openStudio(state)
  try {
    await first.page.getByRole('button', { name: 'Video', exact: true }).click()
    const modelSelect = first.page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    assert.equal(await modelSelect.inputValue(), 'ltx:distilled')
    const framesRange = first.page.locator('.pro-range-field').filter({ hasText: 'Frames' }).locator('input[type="range"]')
    assert.equal(await framesRange.getAttribute('min'), '9')
    assert.equal(await framesRange.getAttribute('step'), '8')
    assert.equal(await framesRange.inputValue(), '81')
    await first.page.locator('#pro-video-source-input').setInputFiles({
      name: 'first-frame.png',
      mimeType: 'image/png',
      buffer: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=', 'base64'),
    })
    await first.page.locator('.pro-video-source-preview').waitFor({ state: 'visible' })
    await first.page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').click()
    await first.page.getByText('Generated 1 image.', { exact: true }).first().waitFor({ state: 'visible' })
    const request = state.generateRequests[0]
    assert.equal(request.checkpoint_id, 'ltx:distilled')
    assert.equal(request.ltx_pipeline, 'distilled')
    assert.equal(request.ltx_image_strength, 0.8)
    assert.equal(request.ltx_offload, 'disk')
    assert.equal(request.ltx_quantization, 'fp8-cast')
    assert.equal(request.ltx_enhance_prompt, false)
    assert.equal(request.frames, 81)
    assert.match(request.source_image_data_url, /^data:image\/png;base64,/)
    assert.equal(request.source_image_name, 'first-frame.png')
    assert.equal(Object.hasOwn(request, 'generate_audio'), false)
    assert.equal(state.modelLoadRequests.length, 0)
  } finally {
    await first.context.close()
  }

  const ltx2b = model('ltx:diffusers_2b', 'LTX Video 0.9.5 2B', {
    architecture: 'ltx', engineId: 'ltx', engineLabel: 'LTX Video', kind: 'video', status: 'Ready',
    generationPreset: { width: 768, height: 512, steps: 1, cfgScale: 3 },
  })
  const diffusersState = createState([image, ltx2b], image.id, {
    bootstrap: { mode: 'image', defaultSettings: { frames: 80 } },
  })
  const second = await openStudio(diffusersState)
  try {
    await second.page.getByRole('button', { name: 'Video', exact: true }).click()
    assert.equal(await second.page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1).inputValue(), 'ltx:diffusers_2b')
    assert.equal(await second.page.locator('#pro-video-source-input').count(), 0)
    assert.equal(await second.page.locator('.pro-range-field').filter({ hasText: 'Steps' }).locator('output').innerText(), '1')
    const framesRange = second.page.locator('.pro-range-field').filter({ hasText: 'Frames' }).locator('input[type="range"]')
    assert.equal(await framesRange.inputValue(), '81')
    await second.page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').click()
    await second.page.getByText('Generated 1 image.', { exact: true }).first().waitFor({ state: 'visible' })
    const request = diffusersState.generateRequests[0]
    assert.equal(request.ltx_pipeline, 'diffusers_2b')
    assert.equal(request.steps, 1)
    assert.equal(request.frames, 81)
    assert.equal(Object.hasOwn(request, 'source_image_data_url'), false)
    assert.equal(Object.hasOwn(request, 'ltx_image_strength'), false)
    assert.equal(Object.hasOwn(request, 'ltx_offload'), false)
    assert.equal(Object.hasOwn(request, 'ltx_quantization'), false)
    assert.equal(Object.hasOwn(request, 'ltx_enhance_prompt'), false)
    assert.equal(Object.hasOwn(request, 'generate_audio'), false)
  } finally {
    await second.context.close()
  }
})

test('Sana Video blocks source images when image-to-video is unavailable', async () => {
  const image = model('image-base', 'Image Base')
  const sana = model('sana-video-t2v-only', 'Sana Video', {
    architecture: 'sana_video', engineId: 'sana_video', engineLabel: 'Sana Video', kind: 'video', status: 'Ready',
    generationModes: { textToVideo: true, imageToVideo: false },
  })
  const state = createState([image, sana], image.id, {
    bootstrap: { mode: 'image' },
  })
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'Video', exact: true }).click()
    await page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1).waitFor({ state: 'visible' })
    await page.locator('#pro-video-source-input').setInputFiles({
      name: 'first-frame.png',
      mimeType: 'image/png',
      buffer: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=', 'base64'),
    })
    await page.locator('.pro-video-source-preview').waitFor({ state: 'visible' })
    await page.getByRole('alert').filter({ hasText: 'Sana Video on this installation supports text-to-video only' }).waitFor({ state: 'visible' })
    const generateButton = page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button')
    assert.equal(await generateButton.isDisabled(), true)
    assert.equal(state.generateRequests.length, 0)
  } finally {
    await context.close()
  }
})

test('a route-blocked model stays selected and offers a usable recovery choice', async () => {
  const healthy = model('healthy-sdxl', 'Healthy SDXL')
  const blocked = model('blocked-sdxl', 'Blocked SDXL', {
    checkpointPathStatus: 'missing',
    routeStatus: 'blocked',
    reason: 'The test checkpoint is missing.',
    suggestedAction: 'Restore the checkpoint and refresh the inventory.',
  })
  const state = createState([healthy], blocked.id, { bootstrap: { blockedModels: [blocked] } })
  const { context, page } = await openStudio(state)
  try {
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    assert.equal(await modelSelect.inputValue(), blocked.id)
    assert.match(await modelSelect.locator('option:checked').innerText(), /Blocked SDXL/)
    const alert = page.getByRole('alert').filter({ hasText: 'The test checkpoint is missing.' })
    await alert.waitFor({ state: 'visible' })
    await alert.getByText('Restore the checkpoint and refresh the inventory.').waitFor({ state: 'visible' })
    assert.equal(await page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').isDisabled(), true)
    await page.getByRole('button', { name: 'Use Healthy SDXL' }).click()
    assert.equal(await modelSelect.inputValue(), 'healthy-sdxl')
    assert.equal(await page.getByRole('alert').count(), 0)
  } finally {
    await context.close()
  }
})

test('missing image support can open the shared model sorter from the generation warning', async () => {
  const blocked = model('flux-dev', 'Flux Dev', {
    architecture: 'flux',
    engineId: 'flux',
    engineLabel: 'Flux',
    status: 'missing-assets',
    checkpointPathStatus: 'present',
    routeStatus: 'blocked',
    reason: 'Flux is missing local support assets.',
    suggestedAction: 'Install Flux conditioning assets.',
  })
  const state = createState([], blocked.id, { bootstrap: { blockedModels: [blocked] } })
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('alert').filter({ hasText: 'Flux is missing local support assets.' }).waitFor({ state: 'visible' })
    await page.getByRole('alert').getByRole('button', { name: 'Find and organize local model files' }).click()
    await page.getByText('Model file sorter', { exact: true }).waitFor({ state: 'visible' })
    await page.waitForFunction(() => document.activeElement?.id === 'model-file-sorter')
    assert.equal(new URL(page.url()).hash, '#settings')
    assert.equal(await page.getByRole('tab', { name: 'System' }).getAttribute('aria-selected'), 'true')
  } finally {
    await context.close()
  }
})

test('a blocked model offers its matching support bundle and reports partial install outcomes', async () => {
  const blocked = model('flux-dev', 'Flux Dev', {
    architecture: 'flux',
    engineId: 'flux',
    engineLabel: 'Flux',
    status: 'missing-assets',
    routeStatus: 'blocked',
    reason: 'Flux is missing its local text encoders and VAE.',
    suggestedAction: 'Install Flux conditioning assets (CLIP-L, T5-XXL, VAE + tokenizers).',
    setupBundleKey: 'flux-components',
  })
  const state = createState([], blocked.id, {
    bootstrap: { blockedModels: [blocked] },
    downloads: {
      catalog: [
        { key: 'flux-clip-l', title: 'Flux CLIP-L', category: 'flux_text_encoder', source: 'huggingface', hfUrl: 'https://example.test/flux-clip-l', destination: 'models/flux/Textencoder', canDownload: false },
        { key: 'flux-t5-fp8', title: 'Flux T5-XXL', category: 'flux_text_encoder', source: 'huggingface', hfUrl: 'https://example.test/flux-t5', destination: 'models/flux/Textencoder', canDownload: false },
        { key: 'flux-ae-vae', title: 'Flux VAE', category: 'vae', source: 'huggingface', hfUrl: 'https://example.test/flux-vae', destination: 'models/flux/VAE', canDownload: false },
        { key: 'flux-clip-tokenizer', title: 'Flux CLIP-L tokenizer files', category: 'flux_tokenizer', source: 'huggingface', hfUrl: 'https://example.test/flux-clip-tokenizer', destination: 'models/flux/tokenizer', canDownload: false },
        { key: 'flux-t5-tokenizer', title: 'Flux T5-XXL tokenizer files', category: 'flux_tokenizer', source: 'huggingface', hfUrl: 'https://example.test/flux-t5-tokenizer', destination: 'models/flux/tokenizer', canDownload: false },
        ...Array.from({ length: 6 }, (_, index) => ({ key: `extra-support-${index + 1}`, title: `Extra support ${index + 1}`, category: 'flux_tokenizer', source: 'huggingface', repoId: index === 5 ? 'example/extra-support-repo' : undefined, filename: `extra-support-${index + 1}.safetensors`, hfUrl: `https://example.test/extra-${index + 1}`, destination: 'models/flux/tokenizer', installed: index === 0, canDownload: false, comingSoon: index === 5 })),
      ],
      bundles: { 'flux-components': ['flux-t5-fp8', 'flux-clip-l', 'flux-ae-vae', 'flux-clip-tokenizer', 'flux-t5-tokenizer'], flux: ['flux-dev-q4km'], video: ['wan-gguf-high-q4km', 'wan-gguf-low-q4km', 'wan-vae-21'] },
      counts: { catalog: 13, installed: 0 },
    },
  })
  const { context, page, requests } = await openStudio(state)
  try {
    await page.getByRole('alert').filter({ hasText: blocked.reason }).getByRole('button', { name: 'Install Flux conditioning assets (CLIP-L, T5-XXL, VAE + tokenizers)' }).click()
    await page.getByText(/flux-components: 0 downloaded, 1 already available, 0 sorted into place, 1 copied from shared folders; unavailable: flux-t5-fp8, deferred while another model operation is active: flux-t5-tokenizer\./).first().waitFor({ state: 'visible' })
    assert.deepEqual(state.bundleInstalls, ['flux-components'])
    assert.equal(requests.filter((item) => item.pathname === '/api/pro/downloads/bundles/flux-components').length, 1)
    assert.deepEqual(state.downloads.bundles['flux-components'], ['flux-t5-fp8', 'flux-clip-l', 'flux-ae-vae', 'flux-clip-tokenizer', 'flux-t5-tokenizer'])
    assert.equal(state.downloads.bundles['flux-components'].includes('flux-dev-q4km'), false)
    await page.getByRole('button', { name: 'Open model inventory' }).click()
    const bundleSelect = page.getByLabel('Model setup bundle')
    await bundleSelect.waitFor({ state: 'visible' })
    assert.match(await bundleSelect.locator('option[value="video"]').innerText(), /Wan 2\.2 video setup/)
    const inventoryText = await page.locator('body').innerText()
    assert.match(inventoryText, /Flux text encoder \| Open source/)
    assert.doesNotMatch(inventoryText, /flux_text_encoder/)
    await page.getByRole('button', { name: 'Show all 11 catalog entries' }).click()
    await page.getByText('Extra support 6', { exact: true }).waitFor({ state: 'visible' })
    await page.getByText('Coming soon · automatic install is not available yet', { exact: true }).waitFor({ state: 'visible' })
    const downloadStats = await page.locator('.pro-download-stat-row').innerText()
    assert.match(downloadStats, /Installed\s+1\s+catalog assets found/)
    await page.getByRole('button', { name: 'Show fewer catalog entries' }).click()
    const catalogSearch = page.getByRole('searchbox', { name: 'Search download catalog' })
    await catalogSearch.fill('models/flux')
    await page.getByText('Showing 11 of 11 matching catalog entries').waitFor({ state: 'visible' })
    assert.equal(await page.getByRole('button', { name: 'Show all 11 catalog entries' }).count(), 0)
    await catalogSearch.fill('example/extra-support-repo')
    await page.getByText('Extra support 6', { exact: true }).waitFor({ state: 'visible' })
    await page.getByText('Showing 1 of 1 matching catalog entries').waitFor({ state: 'visible' })
    await catalogSearch.fill('Extra support 6')
    await page.getByText('Extra support 6', { exact: true }).waitFor({ state: 'visible' })
    await page.getByText('Showing 1 of 1 matching catalog entries').waitFor({ state: 'visible' })
    await catalogSearch.fill('nothing-matches')
    await page.getByText('No catalog entries match this search for the selected engine.').waitFor({ state: 'visible' })
  } finally {
    await context.close()
  }
})

test('installing an image model bundle refreshes its route and loads the selected model', async () => {
  const blocked = model('zimage-local', 'Z-Image', {
    architecture: 'zimage',
    engineId: 'zimage',
    engineLabel: 'Z-Image',
    status: 'missing-assets',
    routeStatus: 'blocked',
    reason: 'Z-Image components are missing.',
    setupBundleKey: 'zimage',
    setupRoute: { routeKey: 'pro.image.zimage', modality: 'image', supportState: 'supported', preflightKey: 'zimage', setupBundleKey: 'zimage' },
  })
  const ready = { ...blocked, status: 'Ready', routeStatus: 'request-eligible', reason: '', checkpointPathStatus: 'present' }
  const state = createState([], blocked.id, { bootstrap: { blockedModels: [blocked] }, initialHash: '#image' })
  state.bundleReadyModel = ready
  const { context, page, requests } = await openStudio(state)
  try {
    await page.getByRole('alert').filter({ hasText: blocked.reason }).getByRole('button', { name: 'Install Z-Image model and components' }).click()
    await page.getByText(/Z-Image and its family support assets are loaded\. Generation has not been verified yet\./).first().waitFor({ state: 'visible' })
    assert.deepEqual(state.bundleInstalls, ['zimage'])
    assert.deepEqual(state.modelLoadRequests, [{ modelId: blocked.id }])
    assert.ok(requests.filter((item) => item.pathname === '/api/pro/bootstrap').length >= 2)
    assert.equal(state.generateRequests.length, 0)
  } finally {
    await context.close()
  }
})

test('standalone Wan TI2V setup installs companions and rechecks the same selected transformer', async () => {
  const blocked = model('wan2.2_ti2v_5b_fp16', 'Wan 2.2 TI2V 5B standalone', {
    filename: 'wan2.2_ti2v_5b_fp16.safetensors',
    architecture: 'wan',
    engineId: 'wan',
    engineLabel: 'Wan Video',
    status: 'missing-assets',
    routeStatus: 'blocked',
    reason: 'Wan TI2V shared components are missing.',
    setupBundleKey: 'wan-ti2v-support',
    setupRoute: {
      routeKey: 'pro.video.wan.ti2v-standalone-safetensors',
      modality: 'video',
      supportState: 'supported',
      preflightKey: 'preflight_wan_pipeline',
      setupBundleKey: 'wan-ti2v-support',
    },
  })
  const ready = { ...blocked, status: 'Ready', routeStatus: 'request-eligible', reason: '' }
  const state = createState([], blocked.id, {
    bootstrap: { blockedModels: [blocked] },
    initialHash: '#video',
    routePrepareReadiness: [false, true],
    bootstrap: { mode: 'video', blockedModels: [blocked] },
    bundleReadyBundleKey: 'wan-ti2v-support',
  })
  state.bundleReadyModel = ready
  const { context, page, requests } = await openStudio(state)
  try {
    await page.getByRole('alert').filter({ hasText: blocked.reason })
      .getByRole('button', { name: 'Install Wan TI2V 5B support assets (UMT5, tokenizer, scheduler + 48-channel VAE)' }).click()
    await page.getByText(/wan-ti2v-support: 0 downloaded, 2 already available/).first().waitFor({ state: 'visible' })
    await page.getByText('Fake route readiness passed.').waitFor({ state: 'visible' })
    assert.deepEqual(state.bundleInstalls, ['wan-ti2v-support'])
    assert.ok(state.routePrepareRequests.length >= 2)
    assert.equal(state.routePrepareRequests.at(-1)?.checkpoint_id, blocked.id)
    assert.equal(requests.filter((item) => item.pathname === '/api/pro/downloads/bundles/wan-ti2v-support').length, 1)
    assert.equal(state.generateRequests.length, 0)
  } finally {
    await context.close()
  }
})

test('installing the selected model bundle from Model Setup also rechecks and loads its route', async () => {
  const blocked = model('zimage-local', 'Z-Image', {
    architecture: 'zimage',
    engineId: 'zimage',
    engineLabel: 'Z-Image',
    status: 'missing-assets',
    routeStatus: 'blocked',
    reason: 'Z-Image components are missing.',
    setupBundleKey: 'zimage',
    setupRoute: { routeKey: 'pro.image.zimage', modality: 'image', supportState: 'supported', preflightKey: 'zimage', setupBundleKey: 'zimage' },
  })
  const ready = { ...blocked, status: 'Ready', routeStatus: 'request-eligible', reason: '', checkpointPathStatus: 'present' }
  const state = createState([], blocked.id, {
    bootstrap: { blockedModels: [blocked] },
    downloads: { catalog: [], bundles: { zimage: ['zimage-model'] }, counts: { catalog: 1, installed: 0 } },
    initialHash: '#image',
  })
  state.bundleReadyModel = ready
  const { context, page, requests } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'Models' }).first().click()
    await page.getByText('Model inventory', { exact: true }).waitFor({ state: 'visible' })
    await page.getByLabel('Model setup bundle').selectOption('zimage')
    await page.getByRole('button', { name: 'Check and install available assets' }).click()
    await page.getByText(/Z-Image and its family support assets are loaded\. Generation has not been verified yet\./).first().waitFor({ state: 'visible' })
    assert.deepEqual(state.bundleInstalls, ['zimage'])
    assert.deepEqual(state.modelLoadRequests, [{ modelId: blocked.id }])
    assert.ok(requests.filter((item) => item.pathname === '/api/pro/bootstrap').length >= 2)
    assert.equal(state.generateRequests.length, 0)
  } finally {
    await context.close()
  }
})

test('an explicit valid model selection submits that same model ID', async () => {
  const first = model('first-sdxl', 'First SDXL')
  const chosen = model('chosen-sdxl', 'Chosen SDXL')
  const state = createState([first, chosen], first.id)
  const { context, page } = await openStudio(state)
  try {
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    await modelSelect.selectOption(chosen.id)
    assert.equal(await modelSelect.inputValue(), chosen.id)
    assert.deepEqual(state.modelLoadRequests, [{ modelId: chosen.id }])
    await page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').click()
    await page.getByRole('button', { name: /Show output settings for A small red apple/ }).waitFor({ state: 'visible' })
    assert.equal(state.generateRequests.length, 1)
    assert.equal(state.generateRequests[0]?.checkpoint_id, chosen.id)
  } finally {
    await context.close()
  }
})

test('bundle setup confirms a shared snapshot copy estimate before import', async () => {
  const blocked = model('zimage-local', 'Z-Image', {
    architecture: 'zimage',
    engineId: 'zimage',
    engineLabel: 'Z-Image',
    status: 'missing-assets',
    routeStatus: 'blocked',
    reason: 'Z-Image components are missing.',
    setupBundleKey: 'zimage',
    setupRoute: { routeKey: 'pro.image.zimage', modality: 'image', supportState: 'supported', preflightKey: 'zimage', setupBundleKey: 'zimage' },
  })
  const ready = { ...blocked, status: 'Ready', routeStatus: 'request-eligible', reason: '', checkpointPathStatus: 'present' }
  const preview = {
    source: 'F:/shared-models/Z-Image-Turbo',
    target: 'F:/AIWF_Studio/models/z-image/Components/Z-Image-Turbo',
    sizeBytes: 8_000_000_000,
    requiredBytes: 8_160_000_000,
    freeBytes: 10_000_000_000,
    enoughSpace: true,
  }
  const state = createState([], blocked.id, {
    bootstrap: { blockedModels: [blocked] },
    initialHash: '#image',
    bundleInstallResponses: {
      zimage: {
        bundles: { zimage: ['zimage-components'] },
        catalog: [],
        counts: { catalog: 1, installed: 0 },
        bundleInstall: {
          key: 'zimage',
          installationStatus: 'confirmation-required',
          readiness: 'refresh-required',
          items: [{ key: 'zimage-components', status: 'shared_snapshot_confirmation_required', sharedSnapshotPreview: preview }],
        },
      },
    },
  })
  state.bundleReadyModel = ready
  const { context, page, requests } = await openStudio(state)
  try {
    page.once('dialog', async (dialog) => {
      assert.match(dialog.message(), /Source: F:\/shared-models\/Z-Image-Turbo/)
      assert.match(dialog.message(), /Destination: F:\/AIWF_Studio\/models\/z-image\/Components\/Z-Image-Turbo/)
      assert.match(dialog.message(), /Required free space: 7\.6 GB/)
      assert.match(dialog.message(), /Route readiness will be checked separately/)
      await dialog.accept()
    })
    await page.getByRole('alert').filter({ hasText: blocked.reason }).getByRole('button', { name: 'Install Z-Image model and components' }).click()
    await page.getByText(/Z-Image and its family support assets are loaded\. Generation has not been verified yet\./).first().waitFor({ state: 'visible' })
    assert.deepEqual(state.sharedSnapshotImports, [{
      key: 'zimage-components',
      query: new URLSearchParams({ confirm: 'true', source: preview.source, size_bytes: String(preview.sizeBytes) }).toString(),
    }])
    assert.deepEqual(state.modelLoadRequests, [{ modelId: blocked.id }])
    assert.equal(requests.filter((item) => item.pathname === '/api/pro/downloads/bundles/zimage').length, 1)
    assert.equal(state.generateRequests.length, 0)
  } finally {
    await context.close()
  }
})

test('declining a pending shared snapshot import keeps bundle incomplete', async () => {
  const blocked = model('zimage-local', 'Z-Image', {
    architecture: 'zimage', engineId: 'zimage', engineLabel: 'Z-Image',
    status: 'missing-assets', routeStatus: 'blocked', reason: 'Z-Image components are missing.',
    setupBundleKey: 'zimage',
    setupRoute: { routeKey: 'pro.image.zimage', modality: 'image', supportState: 'supported', preflightKey: 'zimage', setupBundleKey: 'zimage' },
  })
  const preview = {
    source: 'F:/shared-models/Z-Image-Turbo',
    target: 'F:/AIWF_Studio/models/z-image/Components/Z-Image-Turbo',
    sizeBytes: 1024, requiredBytes: 2048, freeBytes: 4096, enoughSpace: true,
  }
  const state = createState([], blocked.id, {
    bootstrap: { blockedModels: [blocked] },
    initialHash: '#image',
    bundleInstallResponses: {
      zimage: {
        bundles: { zimage: ['zimage-components'] }, catalog: [], counts: { catalog: 1, installed: 0 },
        bundleInstall: {
          key: 'zimage', installationStatus: 'confirmation-required', readiness: 'refresh-required',
          items: [{ key: 'zimage-components', status: 'shared_snapshot_confirmation_required', sharedSnapshotPreview: preview }],
        },
      },
    },
  })
  const { context, page } = await openStudio(state)
  try {
    page.once('dialog', (dialog) => dialog.dismiss())
    await page.getByRole('alert').filter({ hasText: blocked.reason }).getByRole('button', { name: 'Install Z-Image model and components' }).click()
    await page.getByText(/shared snapshot confirmation declined: zimage-components/).first().waitFor({ state: 'visible' })
    assert.deepEqual(state.sharedSnapshotImports, [])
    assert.deepEqual(state.modelLoadRequests, [])
    assert.equal(state.generateRequests.length, 0)
  } finally {
    await context.close()
  }
})

test('changing the Generation defaults model uses the shared model load flow', async () => {
  const first = model('first-sdxl', 'First SDXL')
  const chosen = model('chosen-sdxl', 'Chosen SDXL')
  const state = createState([first, chosen], first.id)
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'Open settings' }).click()
    const defaultModel = page.getByLabel('Default model')
    await defaultModel.waitFor({ state: 'visible' })
    await defaultModel.selectOption(chosen.id)

    await page.locator('strong.pro-generation-status-text').filter({ hasText: 'Chosen SDXL and its family support assets are loaded.' }).waitFor({ state: 'visible' })
    assert.equal(await defaultModel.inputValue(), chosen.id)
    assert.deepEqual(state.modelLoadRequests, [{ modelId: chosen.id }])
  } finally {
    await context.close()
  }
})

test('selecting a dedicated inpaint checkpoint does not call the txt2img model loader', async () => {
  const first = model('first-sdxl', 'First SDXL')
  const inpaint = model('special-inpaint', 'Special Inpaint', { architecture: 'sdxl_inpaint' })
  const state = createState([first, inpaint], first.id)
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'Open settings' }).click()
    const defaultModel = page.getByLabel('Default model')
    await defaultModel.waitFor({ state: 'visible' })
    await defaultModel.selectOption(inpaint.id)
    await page.getByText(/dedicated inpaint pipeline loads when you start an inpaint job/).waitFor({ state: 'visible' })
    assert.deepEqual(state.modelLoadRequests, [])
  } finally {
    await context.close()
  }
})

test('Settings previews a shared-root candidate, asks before copying, then refreshes inventory', async () => {
  const state = createState([model('base-sdxl', 'Base SDXL')], 'base-sdxl')
  state.modelScanAssets = [{
    path: 'F:/shared-models/shared-model.gguf',
    filename: 'shared-model.gguf',
    family: 'flux',
    architecture: 'flux',
    currentSubdir: 'Stable-diffusion',
    recommendedSubdir: 'flux/GGUF',
    placement: 'candidate',
    signals: { filename: 'flux' },
  }]
  const { context, page, requests } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'Open settings' }).click()
    await page.getByRole('tab', { name: 'System & Launch' }).click()
    await page.getByRole('button', { name: 'Scan model roots' }).click()
    await page.getByText('Review model asset placements · 1 total discovered').waitFor({ state: 'visible' })
    page.once('dialog', (dialog) => dialog.accept())
    await page.getByRole('button', { name: 'Place confident files from scan' }).click()
    await page.getByText(/Copied 1 of 1 model file/).waitFor({ state: 'visible' })
    assert.deepEqual(state.placementPreviewRequests, [{ scanId: 'fixture-model-scan', path: 'F:/shared-models/shared-model.gguf' }])
    assert.deepEqual(state.placementApplyRequests, [{ planId: 'placement-plan-1' }])
    assert.equal(requests.filter((request) => request.pathname === '/api/pro/models/scan').length >= 2, true)
  } finally {
    await context.close()
  }
})

test('Settings keeps main-Models reorganize candidates out of the shared-root copy action', async () => {
  const state = createState([model('base-sdxl', 'Base SDXL')], 'base-sdxl')
  state.modelScanAssets = [{
    path: 'F:/models/Stable-diffusion/flux-base.safetensors',
    filename: 'flux-base.safetensors',
    family: 'flux',
    architecture: 'flux',
    currentSubdir: 'Stable-diffusion',
    recommendedSubdir: 'flux/Components',
    placement: 'reorganize-candidate',
    signals: { filename: 'flux' },
  }, {
    path: 'F:/models/unsorted/Qwen-Image',
    filename: 'Qwen-Image',
    family: 'runtime_asset',
    architecture: 'qwen_image',
    currentSubdir: 'unsorted',
    recommendedSubdir: 'qwen-image/Diffusers',
    placement: 'reorganize-check',
    signals: { model_index: 'QwenImagePipeline' },
  }]
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'Open settings' }).click()
    await page.getByRole('tab', { name: 'System & Launch' }).click()
    await page.getByRole('button', { name: 'Scan model roots' }).click()
    await page.getByText('Review model asset placements · 2 total discovered').click()
    await page.getByText('Reorganize candidate in the main Models folder: flux/Components').waitFor({ state: 'visible' })
    await page.getByText('Run Reorganize to check this Diffusers folder; unsupported or incomplete folders stay in place.').waitFor({ state: 'visible' })
    await page.getByText(/Scan checks all configured model and checkpoint roots/).waitFor({ state: 'visible' })
    await page.getByRole('button', { name: 'Place confident files from scan' }).waitFor({ state: 'hidden' })
    assert.deepEqual(state.placementPreviewRequests, [])
  } finally {
    await context.close()
  }
})

test('Settings reports the filename and reason when one candidate preview fails in a mixed batch', async () => {
  const state = createState([model('base-sdxl', 'Base SDXL')], 'base-sdxl')
  state.modelScanAssets = ['ready-model.gguf', 'changed-model.gguf'].map((filename) => ({
    path: `F:/shared-models/${filename}`,
    filename,
    family: 'flux',
    architecture: 'flux',
    currentSubdir: 'Stable-diffusion',
    recommendedSubdir: 'flux/GGUF',
    placement: 'candidate',
    signals: { filename: 'flux' },
  }))
  state.placementPreviewFailures = ['F:/shared-models/changed-model.gguf']
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'Open settings' }).click()
    await page.getByRole('tab', { name: 'System & Launch' }).click()
    await page.getByRole('button', { name: 'Scan model roots' }).click()
    await page.getByText('Review model asset placements · 2 total discovered').waitFor({ state: 'visible' })
    page.once('dialog', (dialog) => dialog.accept())
    await page.getByRole('button', { name: 'Place confident files from scan' }).click()
    await page.getByText(/Copied 1 of 1 model file/).waitFor({ state: 'visible' })
    await page.getByText(/changed-model\.gguf:.*changed after scan/).waitFor({ state: 'visible' })
    assert.equal(state.placementApplyRequests.length, 1)
  } finally {
    await context.close()
  }
})

test('Settings auto-place previews candidates beyond the first inventory page', async () => {
  const state = createState([model('base-sdxl', 'Base SDXL')], 'base-sdxl')
  state.modelScanAssets = Array.from({ length: 60 }, (_, index) => ({
    path: `F:/shared-models/flux-model-${index}.safetensors`,
    filename: `flux-model-${index}.safetensors`,
    family: 'flux',
    architecture: 'flux',
    currentSubdir: 'Stable-diffusion',
    recommendedSubdir: 'flux/UNet',
    placement: 'candidate',
    signals: { filename: 'flux' },
  }))
  const { context, page, requests } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'Open settings' }).click()
    await page.getByRole('tab', { name: 'System & Launch' }).click()
    await page.getByRole('button', { name: 'Scan model roots' }).click()
    await page.getByText('Review model asset placements · 60 total discovered').waitFor({ state: 'visible' })
    page.once('dialog', (dialog) => dialog.accept())
    await page.getByRole('button', { name: 'Place confident files from scan' }).click()
    await page.getByText(/Copied 60 of 60 model file/).waitFor({ state: 'visible' })
    assert.equal(state.placementPreviewRequests.length, 60)
    assert.equal(state.placementApplyRequests.length, 60)
    assert.ok(requests.some((request) => request.pathname === '/api/pro/models/scan/fixture-model-scan' && request.url.includes('limit=250')))
  } finally {
    await context.close()
  }
})

test('Model catalog links directly to the filtered local model sorter', async () => {
  const base = model('base-sdxl', 'Base SDXL')
  const state = createState([base], base.id)
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'Models' }).first().click()
    await page.getByRole('button', { name: 'Find and organize local model files' }).click()
    await page.getByText('Model file sorter', { exact: true }).waitFor({ state: 'visible' })
    await page.getByRole('button', { name: 'Scan model roots' }).waitFor({ state: 'visible' })
    await page.waitForFunction(() => document.activeElement?.id === 'model-file-sorter')
    assert.equal(new URL(page.url()).hash, '#settings')
    assert.equal(await page.getByRole('searchbox', { name: 'Search settings' }).inputValue(), '')
    assert.equal(await page.getByRole('tab', { name: 'System' }).getAttribute('aria-selected'), 'true')
  } finally {
    await context.close()
  }
})

test('switching base model families clears a potentially incompatible custom VAE', async () => {
  const sdxl = model('base-sdxl', 'Base SDXL', { architecture: 'sdxl', engineId: 'sdxl' })
  const sd15 = model('base-sd15', 'Base SD 1.5', { architecture: 'sd15', engineId: 'sd15' })
  const state = createState([sdxl, sd15], sdxl.id, {
    bootstrap: { defaultSettings: { vaeId: 'custom-sdxl-vae' } },
  })
  const { context, page } = await openStudio(state)
  try {
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    await modelSelect.selectOption(sd15.id)
    await page.locator('strong.pro-generation-status-text').filter({ hasText: 'Base SD 1.5 and its family support assets are loaded.' }).waitFor({ state: 'visible' })
    await page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').click()
    await page.getByRole('button', { name: /Show output settings for A small red apple/ }).waitFor({ state: 'visible' })
    assert.equal(state.generateRequests[0]?.vae_id, undefined)
  } finally {
    await context.close()
  }
})

test('image model selection retries one transient model-operation conflict', async () => {
  const first = model('first-sdxl', 'First SDXL')
  const chosen = model('chosen-sdxl', 'Chosen SDXL')
  const state = createState([first, chosen], first.id)
  state.modelLoadFailures = 1
  const { context, page } = await openStudio(state)
  try {
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    await modelSelect.selectOption(chosen.id)
    await page.locator('strong.pro-generation-status-text').filter({ hasText: 'Chosen SDXL and its family support assets are loaded.' }).waitFor({ state: 'visible' })
    assert.deepEqual(state.modelLoadRequests, [{ modelId: chosen.id }, { modelId: chosen.id }])
  } finally {
    await context.close()
  }
})

test('failed image model load blocks generation and exposes a retry action', async () => {
  const first = model('first-sdxl', 'First SDXL')
  const chosen = model('chosen-sdxl', 'Chosen SDXL')
  const state = createState([first, chosen], first.id)
  state.modelLoadTerminalFailures = 1
  const { context, page } = await openStudio(state)
  try {
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    await modelSelect.selectOption(chosen.id)

    const alert = page.getByRole('alert')
    await alert.getByText(/Could not load Chosen SDXL and its family support assets/).waitFor({ state: 'visible' })
    const generateButton = page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button')
    assert.equal(await generateButton.isDisabled(), true)
    await page.locator('.pro-startup-splash').waitFor({ state: 'hidden' })
    await page.screenshot({ path: path.join(tmpdir(), 'aiwf-studio-image-model-load-failed.png'), fullPage: false })

    await page.keyboard.press('Control+k')
    const commandPalette = page.getByRole('dialog', { name: 'Command palette' })
    await commandPalette.getByRole('button', { name: /Run current generation/ }).click()
    assert.equal(state.generateRequests.length, 0, 'command palette must respect the failed image load guard')

    const retryResponse = page.waitForResponse((response) =>
      response.url().includes('/api/pro/models/load')
      && response.request().postDataJSON()?.modelId === chosen.id,
    )
    await alert.getByRole('button', { name: 'Retry selected model load' }).click()
    const response = await retryResponse
    assert.equal(response.status(), 200)
    assert.deepEqual(state.modelLoadRequests, [{ modelId: chosen.id }, { modelId: chosen.id }])
    await alert.waitFor({ state: 'hidden' })
    await page.screenshot({ path: path.join(tmpdir(), 'aiwf-studio-image-model-load-recovered.png'), fullPage: false })
    await page.locator('strong.pro-generation-status-text')
      .filter({ hasText: 'Chosen SDXL and its family support assets are loaded.' }).waitFor({ state: 'visible' })
    assert.equal(await generateButton.isDisabled(), false)
    assert.deepEqual(state.modelLoadRequests, [{ modelId: chosen.id }, { modelId: chosen.id }])
  } finally {
    await context.close()
  }
})

test('Media Foundry uses the shared model switch and hides video-only routes', async () => {
  const sd15 = model('foundry-sd15', 'Foundry SD 1.5', { architecture: 'sd15', engineId: 'sd15' })
  const sdxl = model('foundry-sdxl', 'Foundry SDXL', { architecture: 'sdxl', engineId: 'sdxl' })
  const ltx = model('foundry-ltx', 'Foundry LTX', { architecture: 'ltx', engineId: 'ltx', kind: 'video' })
  const state = createState([sd15, sdxl, ltx], sd15.id, {
    bootstrap: { defaultSettings: { vaeId: 'sd15-custom-vae' } },
  })
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('complementary', { name: 'Subnavigation' }).getByRole('button', { name: 'Foundry' }).click()
    const modelSelect = page.locator('.studio-foundry-inspector select')
    await modelSelect.waitFor({ state: 'visible' })
    const options = await modelSelect.locator('option').allTextContents()
    assert.equal(options.some((option) => option.includes('Foundry LTX')), false)

    const modelLoad = page.waitForRequest((request) =>
      request.url().includes('/api/pro/models/load') && request.postDataJSON()?.modelId === sdxl.id,
    )
    await modelSelect.selectOption(sdxl.id)
    await modelLoad
    await page.locator('strong.pro-generation-status-text')
      .filter({ hasText: 'Foundry SDXL and its family support assets are loaded.' }).waitFor({ state: 'visible' })

    const generateRequest = page.waitForRequest((request) => request.url().includes('/api/pro/generate'))
    await page.getByRole('button', { name: 'Generate Image' }).click()
    const request = await generateRequest
    assert.equal(request.postDataJSON()?.vae_id, undefined, 'family-specific VAE is cleared before generation')
    assert.deepEqual(state.modelLoadRequests, [{ modelId: sdxl.id }])
  } finally {
    await context.close()
  }
})

test('an A-to-B-to-A image selection ignores the stale load and loads the latest choice', async () => {
  const initial = model('initial-sdxl', 'Initial SDXL')
  const older = model('older-sdxl', 'Older SDXL')
  const newer = model('newer-sdxl', 'Newer SDXL')
  const state = createState([initial, older, newer], initial.id)
  let releaseLoad
  state.modelLoadResponseGateId = older.id
  state.modelLoadResponseGate = new Promise((resolve) => { releaseLoad = resolve })
  const { context, page } = await openStudio(state)
  try {
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    const olderRequest = page.waitForRequest((request) =>
      request.url().includes('/api/pro/models/load')
      && request.postDataJSON()?.modelId === older.id,
    )
    await modelSelect.selectOption(older.id)
    await olderRequest

    await modelSelect.selectOption(newer.id)
    await page.waitForTimeout(100)
    assert.deepEqual(state.modelLoadRequests, [{ modelId: older.id }], 'do not contend with an in-progress model load')
    await modelSelect.selectOption(older.id)
    await page.waitForTimeout(100)
    assert.deepEqual(state.modelLoadRequests, [{ modelId: older.id }], 'the pending B selection is superseded by the final A selection')
    const finalOlderRequest = page.waitForRequest((request) =>
      request.url().includes('/api/pro/models/load')
      && request.postDataJSON()?.modelId === older.id,
    )
    releaseLoad()
    await finalOlderRequest
    await page.locator('strong.pro-generation-status-text').filter({ hasText: 'Older SDXL and its family support assets are loaded.' }).waitFor({ state: 'visible' })

    assert.equal(await modelSelect.inputValue(), older.id)
    assert.deepEqual(state.modelLoadRequests, [{ modelId: older.id }, { modelId: older.id }], 'superseded B load must never be requested')
  } finally {
    releaseLoad?.()
    await context.close()
  }
})

test('switching base families restores only a locally confirmed compatible ControlNet choice', async () => {
  const sd15 = model('base-sd15', 'Base SD 1.5', {
    architecture: 'sd15',
    engineId: 'sd15',
    engineLabel: 'Stable Diffusion 1.5',
  })
  const sdxl = model('base-sdxl', 'Base SDXL')
  const state = createState([sd15, sdxl], sd15.id, {
    controlNetModels: [
      { id: 'control_v11p_sd15_canny_fp16', title: 'SD 1.5 Canny', path: 'F:/models/ControlNet/control_v11p_sd15_canny_fp16.safetensors' },
    ],
    bootstrap: {
      defaultSettings: {
        controlNetEnabled: true,
        controlNetModel: 'control_v11p_sd15_canny',
      },
    },
  })
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'ControlNet', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: 'ControlNet' })
    const modelField = dialog.getByPlaceholder('control_v11p_sd15_canny, diffusers folder, or local path')
    const enabled = dialog.getByRole('checkbox', { name: 'ControlNet unit 1' })
    assert.equal(await modelField.inputValue(), 'control_v11p_sd15_canny')
    assert.equal(await enabled.isChecked(), true)
    await dialog.getByText('1 matching local ControlNet model detected.').waitFor({ state: 'visible' })
    assert.equal(await page.locator('#controlnet-model-options-sd15 option').getAttribute('value'), 'control_v11p_sd15_canny_fp16')
    const sd15Option = dialog.getByLabel('ControlNet to install')
    assert.equal(await sd15Option.inputValue(), 'cn15-canny')
    const sd15Setup = dialog.getByRole('button', { name: /Find or install ControlNet v1.1 Canny/ })
    assert.equal(await sd15Setup.isVisible(), true)
    await Promise.all([
      page.waitForResponse((response) => response.url().includes('/api/pro/downloads/catalog/cn15-canny')),
      sd15Setup.click(),
    ])
    assert.equal(state.catalogDownloads.at(-1), 'cn15-canny')
    await modelField.waitFor({ state: 'visible' })
    await page.waitForFunction(() => document.querySelector('[placeholder="control_v11p_sd15_canny, diffusers folder, or local path"]')?.value === 'control_v11p_sd15_canny_fp16')

    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    await page.keyboard.press('Escape')
    await modelSelect.selectOption(sdxl.id)
    await page.waitForFunction((id) => Array.from(document.querySelectorAll('aside[aria-label="Prompt and generation settings"] select')).some((select) => select.value === id), sdxl.id)
    await page.getByRole('button', { name: 'ControlNet', exact: true }).click()
    const sdxlDialog = page.getByRole('dialog', { name: 'ControlNet' })
    const sdxlModelField = sdxlDialog.getByPlaceholder('control_v11p_sd15_canny, diffusers folder, or local path')
    const sdxlEnabled = sdxlDialog.getByRole('checkbox', { name: 'ControlNet unit 1' })
    assert.equal(await sdxlModelField.inputValue(), '')
    assert.equal(await sdxlEnabled.isChecked(), false)
    await sdxlDialog.getByText('No matching local model detected. Choose one to install or import files from a shared folder.').waitFor({ state: 'visible' })
    const sdxlOption = sdxlDialog.getByLabel('ControlNet to install')
    assert.equal(await sdxlOption.inputValue(), 'cnxl-canny')
    const sdxlSetup = sdxlDialog.getByRole('button', { name: /Find or install ControlNet Canny SDXL/ })
    assert.equal(await sdxlSetup.isVisible(), true)
    await Promise.all([
      page.waitForResponse((response) => response.url().includes('/api/pro/downloads/catalog/cnxl-canny')),
      sdxlSetup.click(),
    ])
    assert.equal(state.catalogDownloads.at(-1), 'cnxl-canny')
    await sdxlDialog.getByText('1 matching local ControlNet model detected.').waitFor({ state: 'visible' })
    assert.equal(await page.locator('#controlnet-model-options-sdxl option').getAttribute('value'), 'controlnet-canny-sdxl-1.0')
    assert.equal(await sdxlModelField.inputValue(), 'controlnet-canny-sdxl-1.0')
    await sdxlDialog.getByText('The name matches SDXL. Local file availability is checked before generation.').waitFor({ state: 'visible' })

    await page.keyboard.press('Escape')
    await page.locator('aside[aria-label="Prompt and generation settings"] select').nth(0).selectOption('all')
    await modelSelect.selectOption(sd15.id)
    await page.waitForFunction((id) => Array.from(document.querySelectorAll('aside[aria-label="Prompt and generation settings"] select')).some((select) => select.value === id), sd15.id)
    await page.getByRole('button', { name: 'ControlNet', exact: true }).click()
    const returnedDialog = page.getByRole('dialog', { name: 'ControlNet' })
    const returnedModelField = returnedDialog.getByPlaceholder('control_v11p_sd15_canny, diffusers folder, or local path')
    await page.waitForFunction(() => document.querySelector('[placeholder="control_v11p_sd15_canny, diffusers folder, or local path"]')?.value === 'control_v11p_sd15_canny_fp16')
    assert.equal(await returnedModelField.inputValue(), 'control_v11p_sd15_canny_fp16')
    assert.equal(await returnedDialog.getByRole('checkbox', { name: 'ControlNet unit 1' }).isChecked(), true)

    await page.keyboard.press('Escape')
    await page.locator('aside[aria-label="Prompt and generation settings"] select').nth(0).selectOption('all')
    await modelSelect.selectOption(sdxl.id)
    await page.waitForFunction((id) => Array.from(document.querySelectorAll('aside[aria-label="Prompt and generation settings"] select')).some((select) => select.value === id), sdxl.id)
    state.controlNetModels = state.controlNetModels.filter((item) => !item.id.toLowerCase().includes('sd15'))
    await page.locator('aside[aria-label="Prompt and generation settings"] select').nth(0).selectOption('all')
    await modelSelect.selectOption(sd15.id)
    await page.waitForFunction((id) => Array.from(document.querySelectorAll('aside[aria-label="Prompt and generation settings"] select')).some((select) => select.value === id), sd15.id)
    await page.getByRole('button', { name: 'ControlNet', exact: true }).click()
    const missingDialog = page.getByRole('dialog', { name: 'ControlNet' })
    assert.equal(await missingDialog.getByPlaceholder('control_v11p_sd15_canny, diffusers folder, or local path').inputValue(), '')
    assert.equal(await missingDialog.getByRole('checkbox', { name: 'ControlNet unit 1' }).isChecked(), false)
  } finally {
    await context.close()
  }
})

test('startup clears saved ControlNet state that does not match the selected model', async () => {
  const sdxl = model('base-sdxl', 'Base SDXL')
  const state = createState([sdxl], sdxl.id, {
    bootstrap: {
      defaultSettings: {
        controlNetEnabled: true,
        controlNetModel: 'control_v11p_sd15_canny',
      },
    },
  })
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'ControlNet', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: 'ControlNet' })
    assert.equal(await dialog.getByPlaceholder('control_v11p_sd15_canny, diffusers folder, or local path').inputValue(), '')
    assert.equal(await dialog.getByRole('checkbox', { name: 'ControlNet unit 1' }).isChecked(), false)
  } finally {
    await context.close()
  }
})

test('automatic image to video model changes clear stale ControlNet and Wan sidecars', async () => {
  const sd15 = model('base-sd15', 'Base SD 1.5', {
    architecture: 'sd15',
    engineId: 'sd15',
    engineLabel: 'Stable Diffusion 1.5',
  })
  const wan = model('wan-video', 'Wan Video', {
    architecture: 'wan',
    engineId: 'wan',
    engineLabel: 'Wan',
    kind: 'video',
    mode: 'video',
    status: 'Ready',
    checkpointPathStatus: undefined,
    routeStatus: undefined,
  })
  const state = createState([sd15, wan], sd15.id, {
    bootstrap: {
      defaultSettings: {
        controlNetEnabled: true,
        controlNetModel: 'control_v11p_sd15_canny',
        highNoiseModelId: 'old-wan-high',
        lowNoiseModelId: 'old-wan-low',
        highNoiseLoraId: 'old-wan-high-lora',
        lowNoiseLoraId: 'old-wan-low-lora',
        vaeId: 'old-wan-vae',
        textEncoderPath: 'old-wan-encoder.safetensors',
      },
    },
  })
  const { context, page } = await openStudio(state)
  try {
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    await page.getByRole('button', { name: 'Video', exact: true }).click()
    await page.waitForFunction((id) => Array.from(document.querySelectorAll('aside[aria-label="Prompt and generation settings"] select')).some((select) => select.value === id), wan.id)
    await page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').click()
    await page.getByRole('button', { name: /Show output settings for A small red apple/ }).waitFor({ state: 'visible' })
    assert.equal(state.generateRequests[0]?.checkpoint_id, wan.id)
    for (const field of ['high_noise_model_id', 'low_noise_model_id', 'high_noise_lora_id', 'low_noise_lora_id', 'vae_id', 'text_encoder_path']) {
      assert.equal(state.generateRequests[0]?.[field], undefined, `${field} should not be sent from the previous model family`)
    }
    await page.getByRole('button', { name: 'Image', exact: true }).click()
    await page.waitForFunction((id) => Array.from(document.querySelectorAll('aside[aria-label="Prompt and generation settings"] select')).some((select) => select.value === id), sd15.id)
    await page.getByRole('button', { name: 'ControlNet', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: 'ControlNet' })
    assert.equal(await dialog.getByPlaceholder('control_v11p_sd15_canny, diffusers folder, or local path').inputValue(), '')
    assert.equal(await dialog.getByRole('checkbox', { name: 'ControlNet unit 1' }).isChecked(), false)
    assert.equal(await modelSelect.inputValue(), sd15.id)
  } finally {
    await context.close()
  }
})

test('selecting a named Wan high-noise transformer switches to the dual runtime', async () => {
  const image = model('base-sd15', 'Base SD 1.5', {
    architecture: 'sd15',
    engineId: 'sd15',
    engineLabel: 'Stable Diffusion 1.5',
  })
  const wanHigh = model('GGUF/wan2.2_i2v_high_noise_q4.gguf', 'Wan high noise', {
    architecture: 'wan',
    engineId: 'wan',
    engineLabel: 'Wan',
    kind: 'video',
    mode: 'video',
    status: 'Ready',
    checkpointPathStatus: undefined,
    routeStatus: undefined,
  })
  const state = createState([image, wanHigh], image.id)
  const { context, page } = await openStudio(state)
  try {
    await page.getByRole('button', { name: 'Video', exact: true }).click()
    await page.waitForFunction((id) => Array.from(document.querySelectorAll('aside[aria-label="Prompt and generation settings"] select')).some((select) => select.value === id), wanHigh.id)
    await page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').click()
    await page.getByRole('button', { name: /Show output settings for A small red apple/ }).waitFor({ state: 'visible' })
    assert.equal(state.generateRequests[0]?.wan_runtime_mode, 'high_low')
    assert.equal(state.generateRequests[0]?.high_noise_model_id, undefined)
    assert.equal(state.generateRequests[0]?.low_noise_model_id, undefined)
  } finally {
    await context.close()
  }
})

test('model picker labels distinguish same-name models by family and asset type', async () => {
  const qwen = model('qwen-turbo', 'Turbo', {
    architecture: 'qwen_image',
    engineId: 'qwen',
    engineLabel: 'Qwen Image',
    assetSummary: 'Diffusers folder',
  })
  const krea = model('krea-turbo', 'Turbo', {
    architecture: 'krea2',
    engineId: 'krea2',
    engineLabel: 'Krea 2',
    assetSummary: 'Diffusers folder',
  })
  const fallbackFamily = model('sdxl-no-label', 'Fallback family', {
    architecture: 'sdxl',
    engineId: 'sdxl',
    engineLabel: 'Other',
  })
  const state = createState([qwen, krea, fallbackFamily], qwen.id)
  const { context, page } = await openStudio(state)
  try {
    const options = await page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1).locator('option').allTextContents()
    assert.ok(options.some((label) => label.includes('Turbo (Diffusers folder · Qwen Image)')), options.join('\n'))
    assert.ok(options.some((label) => label.includes('Turbo (Diffusers folder · Krea 2)')), options.join('\n'))
    assert.ok(options.some((label) => label.includes('Fallback family (SDXL)')), options.join('\n'))
  } finally {
    await context.close()
  }
})

test('a ready Krea 2 model remains available in the Pro image picker', async () => {
  const krea = model('krea2-turbo', 'Krea 2 Turbo', {
    architecture: 'krea2',
    engineId: 'krea2',
    engineLabel: 'Krea 2',
  })
  const state = createState([krea], krea.id)
  const { context, page } = await openStudio(state)
  try {
    const modelSelect = page.locator('aside[aria-label="Prompt and generation settings"] select').nth(1)
    assert.equal(await modelSelect.inputValue(), krea.id)
    assert.match(await modelSelect.locator('option:checked').innerText(), /Krea 2 Turbo/)
    await page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').click()
    await page.getByRole('button', { name: /Show output settings for A small red apple/ }).waitFor({ state: 'visible' })
    assert.equal(state.generateRequests.length, 1)
    assert.equal(state.generateRequests[0]?.checkpoint_id, krea.id)
  } finally {
    await context.close()
  }
})

test('the fake success response renders output and history across reload', async () => {
  const state = createState([model('healthy-sdxl', 'Healthy SDXL')], 'healthy-sdxl')
  const { context, page } = await openStudio(state)
  try {
    assert.equal(await page.getByRole('textbox', { name: 'Prompt' }).count(), 1)
    await page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').click()
    await page.getByRole('button', { name: /Show output settings for A small red apple/ }).waitFor({ state: 'visible' })
    assert.match(await page.locator('body').innerText(), /generated successfully in this session/i)
    await page.getByRole('complementary', { name: 'Subnavigation' }).getByRole('button', { name: 'Data' }).click()
    await page.getByText('Dataset and artifact staging').waitFor({ state: 'visible' })
    assert.match(await page.locator('body').innerText(), /Outputs\s+1/)
    const refreshedBootstrap = page.waitForResponse((response) => new URL(response.url()).pathname === '/api/pro/bootstrap')
    await page.reload({ waitUntil: 'domcontentloaded' })
    await refreshedBootstrap
    await page.getByText('Dataset and artifact staging').waitFor({ state: 'visible' })
    await page.waitForFunction(() => /Outputs\s+1/.test(document.body.innerText), undefined, { timeout: 10000 })
    assert.match(await page.locator('body').innerText(), /Outputs\s+1/)
  } finally {
    await context.close()
  }
})

test('the fake backend error remains visible in the rendered shell', async () => {
  const state = createState([model('healthy-sdxl', 'Healthy SDXL')], 'healthy-sdxl', { generationMode: 'error' })
  const { context, page, consoleErrors } = await openStudio(state)
  try {
    await page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').click()
    await page.getByText(/Generation failed:/).first().waitFor({ state: 'visible' })
    assert.equal(state.generateRequests.length, 1)
    assert.ok(consoleErrors.every((message) => /status of 500/.test(message)), consoleErrors.join('\n'))
  } finally {
    await context.close()
  }
})

// --- Studio Flow (unified workspace) -------------------------------------------------------
// A fake same-origin bridge for /api/pro/unified/*. It records request bodies so the
// tests can prove the browser sends IDs and hashes only, never filesystem contents.
function fakeUnifiedBridge() {
  const revision = 'ab'.repeat(32)
  const bridge = { posts: [], project: null }
  const ready = (reason, extra = {}) => ({ available: true, state: 'ready', reason, ...extra })
  bridge.handler = async (route, request, pathname) => {
    const method = request.method()
    const body = method === 'POST' ? request.postDataJSON() : null
    if (method === 'POST') bridge.posts.push({ pathname, body })
    const ok = (json) => route.fulfill({ json: { schema_version: '1', ...json } })
    if (pathname === '/api/pro/unified/status') {
      return ok({
        checked_at: new Date().toISOString(),
        capabilities: {
          dataset_studio: ready('Dataset Studio is running.', { package_listing: true, catalog_studio_outputs: true }),
          retrain: ready('ReTrain accepts text packages.', { training_start_available: false, models: [{ model_id: 'qwen2.5-coder-1.5b', label: 'Qwen2.5-Coder 1.5B', text_capable: true, local_weights_present: true }, { model_id: 'qwen-vl-4b', label: 'Qwen-VL 4B', text_capable: false }] }),
          qwen_chat: ready('Qwen Chat is running.', { models: [{ model_id: 'qwen3-8b', label: 'Qwen 3 8B Chat', family: 'Qwen3', loaded: true }] }),
        },
      })
    }
    if (pathname === '/api/pro/unified/model-families') return ok({ families: [{ artifact: 'ReTrain LoRA/QLoRA adapter', family: 'text LLM', produced_by: 'ReTrain', usable_in: ['ReTrain evaluation'], not_usable_in: ['AIWF Studio image pipelines'] }] })
    if (pathname === '/api/pro/unified/models' && method === 'GET') return ok({ has_key: false, models: [{ model_id: 'qwen2.5-coder-1.5b', label: 'Qwen2.5-Coder 1.5B', family: 'Qwen2.5-Coder', size_b: 1.5, hf_repo: 'Qwen/Qwen2.5-Coder-1.5B-Instruct', present: false, status: 'missing', text_capable: true, download: null, access: { state: 'open', gated: false, has_key: false, download_bytes: 1024, file_count: 2 } }] })
    if (pathname === '/api/pro/unified/models/download' && method === 'POST') return ok({ download: { job_id: 'fixture-download', model_id: body.model_id, repo: 'Qwen/Qwen2.5-Coder-1.5B-Instruct', status: 'completed', total_bytes: 1024, done_bytes: 1024, files_total: 2, files_done: 2, message: 'Fixture complete.', destination_name: 'qwen2.5-coder-1.5b' } })
    if (pathname === '/api/pro/unified/projects' && method === 'GET') return ok({ projects: bridge.project ? [bridge.project] : [] })
    if (pathname === '/api/pro/unified/projects' && method === 'POST') {
      bridge.project = { project_id: 'aiwfp-0123456789abcdef', name: body.name, created_at: new Date().toISOString(), event_counts: {}, events: [] }
      return ok({ project: bridge.project })
    }
    if (pathname === '/api/pro/unified/projects/aiwfp-0123456789abcdef') return ok({ project: bridge.project })
    if (pathname === '/api/pro/unified/datasets/packages') {
      return ok({ packages: [
        { package_name: 'Synthetic demo package (fixture only)', status: 'ready', manifest_sha256: revision, revision_id: `sha256:${revision}`, counts: { train: 40, validation: 10 }, modality: 'text-only' },
        { package_name: 'edited by hand', status: 'invalid', reason: 'Manifest revision hash does not match its contents.' },
      ] })
    }
    if (pathname.endsWith('/catalog-outputs')) {
      bridge.project.events.push({ event_id: 'e1', at: new Date().toISOString(), kind: 'dataset_catalog' })
      return ok({ status: 'cataloged', collection: { id: 3, name: 'AIWF project: Flow test (aiwfp-0123456789abcdef)' }, project_tag: 'aiwf-project:aiwfp-0123456789abcdef', assets: [{ asset_id: 41, relative_path: 'fake-output.png', sha256: 'c'.repeat(64), caption_written: true }], note: 'Image and video assets are cataloged only.' })
    }
    if (pathname === '/api/pro/unified/retrain/import') {
      return ok({ status: 'imported', dataset: { dataset_id: `sha256-${revision}`, package_name: 'Synthetic demo package (fixture only)', manifest_sha256: revision, counts: { train: 40, validation: 10 }, modality: 'text-only', reused: false } })
    }
    if (pathname === '/api/pro/unified/retrain/preflight') {
      return ok({ status: 'preflight', start_enabled: false, execution_requested: false, message: 'Preflight only. This route cannot start training.', result: { dataset_id: `sha256-${revision}`, manifest_sha256: revision, model_id: body.model_id, plan_status: 'warning', gates: [{ gate: 'Base model', state: 'warning', detail: 'pick or download a local model' }, { gate: 'Dataset path', state: 'ready', detail: '<server path>' }], estimate: { fit_state: 'safe', estimated_gb: 2.31, limit_gb: 16 }, dependencies: [{ label: 'PyTorch', available: true }], notes: [], summary: {}, training_args: {} } })
    }
    if (pathname.endsWith('/qwen-context')) return ok({ context: 'Project: Flow test\nProject ID: aiwfp-0123456789abcdef\nReTrain imports: none' })
    if (pathname.endsWith('/qwen-ask')) return ok({ model_id: 'qwen3-8b', answer: 'Forty rows is enough for a smoke run.', context_sent: 'Project: Flow test\nProject ID: aiwfp-0123456789abcdef' })
    return route.fulfill({ status: 404, json: { detail: 'Not Found' } })
  }
  return { bridge, revision }
}

async function openStudioFlow(page) {
  await page.getByRole('complementary', { name: 'Subnavigation' }).getByRole('button', { name: 'Studio Flow' }).click()
  await page.getByRole('heading', { name: 'Studio Flow', level: 1 }).waitFor({ state: 'visible' })
}

test('Studio Flow reports a missing bridge and keeps every cross-app action disabled', async () => {
  const state = createState([model('healthy-sdxl', 'Healthy SDXL')], 'healthy-sdxl')
  state.outputs = [output('Fixture output remains in Studio')]
  const { context, page, requests } = await openStudio(state)
  try {
    await openStudioFlow(page)
    await page.getByText('This AIWF Pro backend has no unified workspace bridge (HTTP 404).').waitFor({ state: 'visible' })
    const before = requests.length
    await page.getByRole('checkbox').check()
    await page.getByText('1 selected', { exact: false }).waitFor({ state: 'visible' })
    assert.equal(requests.slice(before).filter((item) => item.pathname.startsWith('/api/pro/unified/')).length, 0, 'selecting an output must not call the bridge')
    for (const name of [/Catalog .*in Dataset Studio/, 'Import selected revision into ReTrain', 'Run preflight', 'Send with context', 'Create project']) {
      assert.equal(await page.getByRole('button', { name }).isDisabled(), true, `${name} must stay disabled`)
    }
    assert.equal(requests.filter((item) => item.pathname.startsWith('/api/pro/unified/') && item.method !== 'GET').length, 0)
    const text = await page.locator('body').innerText()
    assert.match(text, /none selected/)
    assert.equal(text.includes('\uFFFD'), false, 'no replacement characters may render')
  } finally {
    await context.close()
  }
})

test('Studio Flow runs catalog, explicit import, preflight and Qwen context through the same-origin bridge', async () => {
  const state = createState([model('healthy-sdxl', 'Healthy SDXL')], 'healthy-sdxl')
  state.outputs = [output('Lighthouse at dusk')]
  const { bridge, revision } = fakeUnifiedBridge()
  state.unified = bridge.handler
  const { context, page, requests, consoleErrors } = await openStudio(state)
  try {
    await openStudioFlow(page)
    await page.getByText('Connections checked', { exact: false }).waitFor({ state: 'visible' })
    assert.match(await page.getByLabel('Qwen Chat model').locator('option:checked').innerText(), /Qwen 3 8B Chat \(loaded\)/)

    // shared project identity
    await page.getByLabel('New project name').fill('Flow test')
    await page.getByRole('button', { name: 'Create project' }).click()
    await page.getByText('aiwfp-0123456789abcdef').first().waitFor({ state: 'visible' })

    // step 1: catalog the selected output
    await page.getByRole('checkbox').check()
    await page.getByRole('button', { name: /Catalog 1 in Dataset Studio/ }).click()
    await page.getByText('Cataloged 1 asset in', { exact: false }).waitFor({ state: 'visible' })

    // step 2: the invalid package cannot be chosen; the valid one is chosen explicitly
    assert.equal(await page.getByRole('radio').nth(1).isDisabled(), true)
    assert.equal(await page.getByRole('button', { name: 'Import selected revision into ReTrain' }).isDisabled(), true, 'nothing is imported before an explicit choice')
    await page.getByRole('radio').first().check()
    await page.getByRole('button', { name: 'Import selected revision into ReTrain' }).click()
    await page.getByText('Imported into ReTrain: Synthetic demo package (fixture only)').waitFor({ state: 'visible' })

    // step 3: preflight only; the vision model is not offered for a text package
    assert.equal(await page.getByRole('option', { name: /Qwen-VL/ }).count(), 0)
    await page.getByRole('button', { name: 'Run preflight' }).click()
    await page.getByText('Plan status: warning', { exact: false }).waitFor({ state: 'visible' })
    await page.getByText('pick or download a local model', { exact: false }).waitFor({ state: 'visible' })
    await page.getByText('VRAM estimate: 2.31 GB of 16 GB (safe).').waitFor({ state: 'visible' })

    // step 4: context must be previewed before Send is enabled
    await page.getByLabel('Question for Qwen Chat').fill('Is this enough data?')
    assert.equal(await page.getByRole('button', { name: 'Send with context' }).isDisabled(), true)
    await page.getByRole('button', { name: 'Preview context' }).click()
    await page.getByLabel('Context sent to Qwen Chat').waitFor({ state: 'visible' })
    await page.getByRole('button', { name: 'Send with context' }).click()
    await page.getByText('Forty rows is enough for a smoke run.').waitFor({ state: 'visible' })

    // Train's model-weight dialog uses the same local bridge and downloads only after an explicit click.
    await page.getByRole('button', { name: 'Model weights' }).click()
    const weightsDialog = page.getByRole('dialog', { name: 'Qwen2.5-Coder 1.5B is not on this PC' })
    await weightsDialog.waitFor({ state: 'visible' })
    await weightsDialog.getByRole('button', { name: 'Download 1.0 KB' }).click()
    await weightsDialog.getByText('Download finished. Catalog assets are present; route readiness has not been checked.').waitFor({ state: 'visible' })

    // request bodies carry IDs, hashes and the question only
    const posts = Object.fromEntries(bridge.posts.map((item) => [item.pathname.split('/').slice(-1)[0], item.body]))
    assert.deepEqual(posts.projects, { name: 'Flow test' })
    assert.deepEqual(posts['catalog-outputs'], { output_paths: ['task-temp/fake-output.png'] })
    assert.deepEqual(posts.import, { project_id: 'aiwfp-0123456789abcdef', package_name: 'Synthetic demo package (fixture only)', manifest_sha256: revision })
    assert.deepEqual(posts.preflight, { project_id: 'aiwfp-0123456789abcdef', dataset_id: `sha256-${revision}`, manifest_sha256: revision, model_id: 'qwen2.5-coder-1.5b', settings: { method: 'QLoRA' } })
    assert.deepEqual(posts['qwen-ask'], { model_id: 'qwen3-8b', question: 'Is this enough data?' })
    assert.deepEqual(posts.download, { model_id: 'qwen2.5-coder-1.5b' })
    assert.equal(requests.some((item) => /start|jobs/.test(item.pathname) && item.pathname.includes('unified')), false, 'no training start route exists or is called')
    assert.equal(requests.filter((item) => item.pathname.startsWith('/api/pro/unified/')).every((item) => !/^https?:/.test(item.pathname)), true)
    const text = await page.locator('body').innerText()
    assert.equal(text.includes('\uFFFD'), false, 'no replacement characters may render')
    assert.deepEqual(consoleErrors, [])
    await page.screenshot({ path: path.join(artifactRoot, 'studio-flow-fixture.png'), fullPage: true })
  } finally {
    await context.close()
  }
})

test('an active fake generation can be interrupted', async () => {
  const state = createState([model('healthy-sdxl', 'Healthy SDXL')], 'healthy-sdxl', { generationMode: 'cancel' })
  let resolveStarted
  const started = new Promise((resolve) => { resolveStarted = resolve })
  state.generateStartedResolve = resolveStarted
  const { context, page } = await openStudio(state)
  try {
    await page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').click()
    await started
    await page.locator('aside[aria-label="Prompt and generation settings"] button.pro-generate-button').filter({ hasText: 'Stop' }).click()
    await page.getByText(/Stop requested for active generation/).first().waitFor({ state: 'visible' })
    assert.equal(state.interruptCalls, 1)
  } finally {
    await context.close()
  }
})
