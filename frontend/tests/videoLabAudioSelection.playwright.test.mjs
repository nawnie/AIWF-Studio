import assert from 'node:assert/strict'
import { mkdtemp, rm, symlink, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import test from 'node:test'
import { chromium } from 'playwright'
import { createServer } from 'vite'

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const chromePath = process.env.AIWF_TEST_CHROME_PATH ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe'

test('Video Lab selects, prepares, persists, and sends the chosen soundtrack model', async (t) => {
  const fixtureRoot = await mkdtemp(path.join(tmpdir(), 'aiwf-video-lab-audio-'))
  const browser = await chromium.launch({
    ...(existsSync(chromePath) ? { executablePath: chromePath } : {}),
    args: ['--disable-gpu', '--no-sandbox'],
  })
  const server = await createServer({
    root: fixtureRoot,
    configFile: false,
    cacheDir: path.join(fixtureRoot, '.vite-cache'),
    server: { host: '127.0.0.1', port: 0, strictPort: false, fs: { allow: [fixtureRoot, frontendRoot, path.join(frontendRoot, 'node_modules')] } },
  })
  t.after(async () => {
    await browser.close()
    await server.close()
    const relative = path.relative(tmpdir(), fixtureRoot)
    assert.ok(!path.isAbsolute(relative) && !relative.startsWith('..') && path.basename(fixtureRoot).startsWith('aiwf-video-lab-audio-'))
    await rm(fixtureRoot, { recursive: true, force: true })
  })

  await symlink(path.join(frontendRoot, 'src'), path.join(fixtureRoot, 'src'), 'junction')
  await symlink(path.join(frontendRoot, 'node_modules'), path.join(fixtureRoot, 'node_modules'), 'junction')
  await writeFile(path.join(fixtureRoot, 'index.html'), '<!doctype html><html><body><div id="root"></div><script type="module" src="/main.tsx"></script></body></html>')
  await writeFile(path.join(fixtureRoot, 'main.tsx'), `import React from 'react'
import { createRoot } from 'react-dom/client'
import { VideoLabCard } from './src/App'
createRoot(document.getElementById('root')).render(<VideoLabCard wanModels={[]} onOpenModelSorter={() => { document.body.dataset.modelSorter = 'opened' }} />)`)

  await server.listen()
  const address = server.httpServer.address()
  assert.ok(address && typeof address === 'object')
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
  const requests = []
  let musicGenInstalled = false
  let musicRuntimeReady = false
  let mmaudioLargeInstalled = false
  let failFirstMusicGenLoad = true
  page.on('pageerror', (error) => requests.push({ error: error.message }))
  await page.route('**/api/pro/video-lab/status', (route) => route.fulfill({ json: {
    vsr: { available: false, upscaleAvailable: false, denoiseAvailable: false, sdkRoot: '', modelCount: 0, features: [], help: '' },
    rife: { available: false, checkpoints: [] },
    audio: {
      videoAudioModels: ['mmaudio:small_16k'], defaultModelId: 'mmaudio:small_16k', ready: true, musicGenReady: musicRuntimeReady,
      modelChoices: [
        { label: 'MMAudio small 16k', id: 'mmaudio:small_16k', conditioningMode: 'video-conditioned', available: true, installed: true, installable: true, ready: true },
        { label: 'MusicGen medium', id: 'facebook/musicgen-medium', conditioningMode: 'prompt-only', available: true, installed: musicGenInstalled, installable: true, ready: musicGenInstalled, setupRoute: { routeKey: 'pro.audio.musicgen.medium', modality: 'audio', supportState: 'supported', preflightKey: 'musicgen', setupAction: 'POST /api/pro/audio/setup/musicgen/medium' } },
        { label: 'MMAudio large 44k v2', id: 'mmaudio:large_44k_v2', conditioningMode: 'video-conditioned', available: true, installed: mmaudioLargeInstalled, installable: true, ready: mmaudioLargeInstalled, setupRoute: { routeKey: 'pro.video.audio.mmaudio', modality: 'video', supportState: 'supported', preflightKey: 'mmaudio', setupAction: 'POST /api/pro/audio/setup/mmaudio/{variant}' } },
      ],
    },
    extend: { available: false, note: '' },
  } }))
  await page.route('**/api/pro/audio/setup/musicgen/medium', async (route) => {
    requests.push({ path: new URL(route.request().url()).pathname })
    musicGenInstalled = true
    await route.fulfill({ json: { minimumReady: true, musicDependenciesReady: true, musicReady: true, sfxReady: true, videoAudioReady: true, labReady: true, muxReady: true, defaults: {}, models: {}, components: [] } })
  })
  await page.route('**/api/pro/audio/setup/minimum', async (route) => {
    requests.push({ path: new URL(route.request().url()).pathname })
    musicRuntimeReady = true
    await route.fulfill({ json: { minimumReady: true, musicDependenciesReady: true, musicReady: true, sfxReady: true, videoAudioReady: true, labReady: true, muxReady: true, defaults: {}, models: {}, components: [] } })
  })
  await page.route('**/api/pro/audio/setup/mmaudio/large_44k_v2', async (route) => {
    requests.push({ path: new URL(route.request().url()).pathname })
    mmaudioLargeInstalled = true
    await route.fulfill({ json: { minimumReady: true, musicDependenciesReady: true, musicReady: true, sfxReady: true, videoAudioReady: true, labReady: true, muxReady: true, defaults: {}, models: {}, components: [] } })
  })
  await page.route('**/api/pro/video-lab/prepare-audio', async (route) => {
    const body = route.request().postDataJSON()
    requests.push({ path: new URL(route.request().url()).pathname, body })
    if (body.modelId === 'facebook/musicgen-medium' && failFirstMusicGenLoad) {
      failFirstMusicGenLoad = false
      await route.fulfill({ status: 500, json: { detail: 'Simulated MusicGen load failure.' } })
      return
    }
    await route.fulfill({ json: { kind: body.kind, modelId: body.modelId, ready: true, resident: body.modelId === 'facebook/musicgen-medium', routeStatus: 'prepared' } })
  })
  await page.route('**/api/pro/video-lab/upload', (route) => route.fulfill({ json: {
    path: 'F:/outputs/source.mp4', width: 640, height: 360, fps: 24, frameCount: 24, durationSeconds: 1,
  } }))
  await page.route('**/api/pro/video-lab/run', async (route) => {
    const body = route.request().postDataJSON()
    requests.push({ path: new URL(route.request().url()).pathname, body })
    await route.fulfill({ json: { status: 'complete', outputPath: 'F:/outputs/result.mp4', url: '/outputs/result.mp4', message: 'Added video audio -> result.mp4', probe: {} } })
  })
  await page.addInitScript(() => localStorage.setItem('aiwf.video-lab.audio-model', 'facebook/musicgen-medium'))
  await page.goto(`http://127.0.0.1:${address.port}`)

  await page.getByLabel('Operation').selectOption('audio')
  const soundtrackModel = page.getByLabel('Soundtrack model')
  assert.equal(await soundtrackModel.inputValue(), 'facebook/musicgen-medium')
  await page.getByText('MusicGen creates audio from your text prompt, then adds it to the video.').waitFor({ state: 'visible' })
  await soundtrackModel.selectOption('facebook/musicgen-medium')
  await page.getByRole('button', { name: 'Set up MusicGen medium and Audio runtime' }).click()
  await page.waitForFunction(() => document.body.innerText.includes('Audio model preparation failed:'))
  assert.equal(await page.evaluate(() => localStorage.getItem('aiwf.video-lab.audio-model')), 'facebook/musicgen-medium')
  await page.locator('#pro-video-lab-upload').setInputFiles({ name: 'source.mp4', mimeType: 'video/mp4', buffer: Buffer.from('video') })
  assert.equal(await page.getByRole('button', { name: 'Run', exact: true }).isDisabled(), false)
  assert.equal(requests.some((request) => request.path === '/api/pro/video-lab/run'), false)
  await page.getByRole('button', { name: 'Run', exact: true }).click()
  await page.getByText('Added video audio -> result.mp4').waitFor({ state: 'visible' })
  assert.ok(requests.some((request) => request.path === '/api/pro/video-lab/prepare-audio' && request.body.modelId === 'facebook/musicgen-medium'))
  await soundtrackModel.selectOption('mmaudio:large_44k_v2')
  await page.getByText('MMAudio uses the source video and your prompt to create synchronized audio.').waitFor({ state: 'visible' })
  await page.getByRole('button', { name: 'Install MMAudio large 44k v2' }).click()
  await page.getByText('MMAudio large 44k v2 setup is ready. Its weights load on demand for each render.').waitFor({ state: 'visible' })

  await page.getByLabel('Audio prompt').fill('warm strings and soft percussion')
  await page.getByRole('button', { name: 'Run' }).click()
  await page.getByText('Added video audio -> result.mp4').waitFor({ state: 'visible' })
  await soundtrackModel.selectOption('facebook/musicgen-medium')
  await page.getByText('MusicGen medium setup passed. The runtime loads or reuses its weights when you generate.').waitFor({ state: 'visible' })
  await page.getByRole('button', { name: 'Run', exact: true }).click()
  await page.getByText('Added video audio -> result.mp4').waitFor({ state: 'visible' })

  assert.ok(requests.some((request) => request.path === '/api/pro/video-lab/prepare-audio' && request.body.modelId === 'facebook/musicgen-medium'))
  assert.ok(requests.some((request) => request.path === '/api/pro/audio/setup/musicgen/medium'))
  assert.ok(requests.some((request) => request.path === '/api/pro/audio/setup/mmaudio/large_44k_v2'))
  assert.ok(requests.findIndex((request) => request.path === '/api/pro/audio/setup/minimum') < requests.findIndex((request) => request.path === '/api/pro/audio/setup/musicgen/medium'))
  const generation = requests.find((request) => request.path === '/api/pro/video-lab/run' && request.body.audioModel === 'mmaudio:large_44k_v2')
  assert.equal(generation?.body.audioModel, 'mmaudio:large_44k_v2')
  const finalMusicGenPrepare = requests.filter((request) => request.path === '/api/pro/video-lab/prepare-audio' && request.body.modelId === 'facebook/musicgen-medium').at(-1)
  const finalMusicGenRun = requests.filter((request) => request.path === '/api/pro/video-lab/run').at(-1)
  assert.ok(finalMusicGenPrepare && finalMusicGenRun && requests.indexOf(finalMusicGenPrepare) < requests.indexOf(finalMusicGenRun))
  await page.getByRole('button', { name: 'Find and organize local model files' }).click()
  assert.equal(await page.locator('body').getAttribute('data-model-sorter'), 'opened')
  assert.deepEqual(requests.filter((request) => request.error), [])
})

test('Video Lab installs missing MMAudio runtime and selected variant in one setup action', async (t) => {
  const fixtureRoot = await mkdtemp(path.join(tmpdir(), 'aiwf-video-lab-mmaudio-setup-'))
  const browser = await chromium.launch({
    ...(existsSync(chromePath) ? { executablePath: chromePath } : {}),
    args: ['--disable-gpu', '--no-sandbox'],
  })
  const server = await createServer({
    root: fixtureRoot,
    configFile: false,
    cacheDir: path.join(fixtureRoot, '.vite-cache'),
    server: { host: '127.0.0.1', port: 0, strictPort: false, fs: { allow: [fixtureRoot, frontendRoot, path.join(frontendRoot, 'node_modules')] } },
  })
  t.after(async () => {
    await browser.close()
    await server.close()
    const relative = path.relative(tmpdir(), fixtureRoot)
    assert.ok(!path.isAbsolute(relative) && !relative.startsWith('..') && path.basename(fixtureRoot).startsWith('aiwf-video-lab-mmaudio-setup-'))
    await rm(fixtureRoot, { recursive: true, force: true })
  })

  await symlink(path.join(frontendRoot, 'src'), path.join(fixtureRoot, 'src'), 'junction')
  await symlink(path.join(frontendRoot, 'node_modules'), path.join(fixtureRoot, 'node_modules'), 'junction')
  await writeFile(path.join(fixtureRoot, 'index.html'), '<!doctype html><html><body><div id="root"></div><script type="module" src="/main.tsx"></script></body></html>')
  await writeFile(path.join(fixtureRoot, 'main.tsx'), `import React from 'react'
import { createRoot } from 'react-dom/client'
import { VideoLabCard } from './src/App'
createRoot(document.getElementById('root')).render(<VideoLabCard wanModels={[]} />)`)

  await server.listen()
  const address = server.httpServer.address()
  assert.ok(address && typeof address === 'object')
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
  const requests = []
  let mmaudioRuntimeReady = false
  let largeVariantInstalled = false
  page.on('pageerror', (error) => requests.push({ error: error.message }))
  await page.route('**/api/pro/video-lab/status', (route) => route.fulfill({ json: {
    vsr: { available: false, upscaleAvailable: false, denoiseAvailable: false, sdkRoot: '', modelCount: 0, features: [], help: '' },
    rife: { available: false, checkpoints: [] },
    audio: {
      videoAudioModels: ['mmaudio:large_44k_v2'], defaultModelId: 'mmaudio:large_44k_v2',
      ready: mmaudioRuntimeReady, musicGenReady: true,
      modelChoices: [{
        label: 'MMAudio large 44k v2', id: 'mmaudio:large_44k_v2', conditioningMode: 'video-conditioned',
        available: mmaudioRuntimeReady, unavailableReason: mmaudioRuntimeReady ? '' : 'Runtime unavailable.',
        installed: largeVariantInstalled, installable: mmaudioRuntimeReady, ready: mmaudioRuntimeReady && largeVariantInstalled,
        setupRoute: { routeKey: 'pro.video.audio.mmaudio', modality: 'video', supportState: 'supported', preflightKey: 'mmaudio', setupAction: 'POST /api/pro/audio/setup/mmaudio/{variant}' },
      }],
    },
    extend: { available: false, note: '' },
  } }))
  await page.route('**/api/pro/audio/setup/minimum', async (route) => {
    requests.push({ path: new URL(route.request().url()).pathname })
    mmaudioRuntimeReady = true
    await route.fulfill({ json: { minimumReady: true, musicDependenciesReady: true, musicReady: true, sfxReady: true, videoAudioReady: true, labReady: true, muxReady: true, defaults: {}, models: {}, components: [] } })
  })
  await page.route('**/api/pro/audio/setup/mmaudio/large_44k_v2', async (route) => {
    requests.push({ path: new URL(route.request().url()).pathname })
    largeVariantInstalled = true
    await route.fulfill({ json: { minimumReady: true, musicDependenciesReady: true, musicReady: true, sfxReady: true, videoAudioReady: true, labReady: true, muxReady: true, defaults: {}, models: {}, components: [] } })
  })
  await page.route('**/api/pro/video-lab/prepare-audio', async (route) => {
    const body = route.request().postDataJSON()
    requests.push({ path: new URL(route.request().url()).pathname, body })
    await route.fulfill({ json: { kind: body.kind, modelId: body.modelId, ready: true, resident: null, routeStatus: 'prepared' } })
  })
  await page.goto(`http://127.0.0.1:${address.port}`)
  await page.getByLabel('Operation').selectOption('audio')
  await page.getByRole('button', { name: 'Set up MMAudio large 44k v2 and Audio runtime' }).click()
  await page.getByText('MMAudio large 44k v2 setup is ready. Its weights load on demand for each render.').waitFor({ state: 'visible' })
  assert.ok(requests.findIndex((request) => request.path === '/api/pro/audio/setup/minimum') < requests.findIndex((request) => request.path === '/api/pro/audio/setup/mmaudio/large_44k_v2'))
  assert.ok(requests.some((request) => request.path === '/api/pro/video-lab/prepare-audio' && request.body.modelId === 'mmaudio:large_44k_v2'))
  assert.deepEqual(requests.filter((request) => request.error), [])
})

test('Video Lab blocks soundtrack generation and offers minimum setup when mux tools are missing', async (t) => {
  const fixtureRoot = await mkdtemp(path.join(tmpdir(), 'aiwf-video-lab-mux-missing-'))
  const browser = await chromium.launch({
    ...(existsSync(chromePath) ? { executablePath: chromePath } : {}),
    args: ['--disable-gpu', '--no-sandbox'],
  })
  const server = await createServer({
    root: fixtureRoot,
    configFile: false,
    cacheDir: path.join(fixtureRoot, '.vite-cache'),
    server: { host: '127.0.0.1', port: 0, strictPort: false, fs: { allow: [fixtureRoot, frontendRoot, path.join(frontendRoot, 'node_modules')] } },
  })
  t.after(async () => {
    await browser.close()
    await server.close()
    const relative = path.relative(tmpdir(), fixtureRoot)
    assert.ok(!path.isAbsolute(relative) && !relative.startsWith('..') && path.basename(fixtureRoot).startsWith('aiwf-video-lab-mux-missing-'))
    await rm(fixtureRoot, { recursive: true, force: true })
  })

  await symlink(path.join(frontendRoot, 'src'), path.join(fixtureRoot, 'src'), 'junction')
  await symlink(path.join(frontendRoot, 'node_modules'), path.join(fixtureRoot, 'node_modules'), 'junction')
  await writeFile(path.join(fixtureRoot, 'index.html'), '<!doctype html><html><body><div id="root"></div><script type="module" src="/main.tsx"></script></body></html>')
  await writeFile(path.join(fixtureRoot, 'main.tsx'), `import React from 'react'
import { createRoot } from 'react-dom/client'
import { VideoLabCard } from './src/App'
createRoot(document.getElementById('root')).render(<VideoLabCard wanModels={[]} />)`)

  await server.listen()
  const address = server.httpServer.address()
  assert.ok(address && typeof address === 'object')
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
  const requests = []
  page.on('pageerror', (error) => requests.push({ error: error.message }))
  await page.route('**/api/pro/video-lab/status', (route) => route.fulfill({ json: {
    vsr: { available: false, upscaleAvailable: false, denoiseAvailable: false, sdkRoot: '', modelCount: 0, features: [], help: '' },
    rife: { available: false, checkpoints: [] },
    audio: {
      videoAudioModels: ['mmaudio:small_16k'], defaultModelId: 'mmaudio:small_16k', ready: false, musicGenReady: false,
      modelChoices: [{
        label: 'MMAudio small 16k', id: 'mmaudio:small_16k', conditioningMode: 'video-conditioned',
        available: false, unavailableReason: 'Working FFmpeg and ffprobe are required to mux generated audio into the video.',
        installed: true, installable: false, ready: false,
        setupRoute: { routeKey: 'pro.video.audio.mmaudio', modality: 'video', supportState: 'supported', preflightKey: 'mmaudio', setupAction: 'POST /api/pro/audio/setup/mmaudio/{variant}' },
      }],
    },
    extend: { available: false, note: '' },
  } }))
  await page.route('**/api/pro/audio/setup/minimum', (route) => {
    requests.push({ path: new URL(route.request().url()).pathname })
    return route.fulfill({ json: { minimumReady: false, muxReady: false } })
  })
  await page.route('**/api/pro/video-lab/prepare-audio', (route) => {
    requests.push({ path: new URL(route.request().url()).pathname })
    return route.fulfill({ json: { kind: 'sfx', modelId: 'mmaudio:small_16k', ready: true, resident: null } })
  })
  await page.route('**/api/pro/video-lab/run', (route) => {
    requests.push({ path: new URL(route.request().url()).pathname })
    return route.fulfill({ json: { status: 'complete' } })
  })

  await page.goto(`http://127.0.0.1:${address.port}`)
  await page.getByLabel('Operation').selectOption('audio')
  await page.getByRole('button', { name: 'Set up MMAudio small 16k and Audio runtime' }).waitFor({ state: 'visible' })
  assert.equal(await page.getByRole('button', { name: 'Run', exact: true }).isDisabled(), true)
  assert.equal(requests.some((request) => request.path === '/api/pro/video-lab/prepare-audio'), false)

  await page.getByRole('button', { name: 'Set up MMAudio small 16k and Audio runtime' }).click()
  await page.getByText(/Soundtrack setup failed: The minimum Audio runtime is still incomplete after setup/).waitFor({ state: 'visible' })
  assert.equal(requests.some((request) => request.path === '/api/pro/audio/setup/minimum'), true)
  assert.equal(requests.some((request) => request.path === '/api/pro/video-lab/prepare-audio'), false)
  assert.equal(requests.some((request) => request.path === '/api/pro/video-lab/run'), false)
  assert.deepEqual(requests.filter((request) => request.error), [])
})
