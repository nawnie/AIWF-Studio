import assert from 'node:assert/strict'
import { mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import test from 'node:test'
import { chromium } from 'playwright'
import { createServer } from 'vite'

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const chromePath = process.env.AIWF_TEST_CHROME_PATH ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe'

function audioStatus(installed = false, musicInstalled = false, minimumReady = true) {
  return {
    minimumReady,
    musicDependenciesReady: true,
    runtimeChecksPerformed: false,
    installing: false,
    musicReady: true,
    sfxReady: true,
    videoAudioReady: true,
    labReady: true,
    muxReady: true,
    message: 'Minimum Audio files and dependencies are detected; runtime checks have not run.',
    estimatedDownload: 'Existing files are reused.',
    licenseNotice: 'Test only.',
    // MusicGen and MMAudio are offered only in research mode (they are CC-BY-NC 4.0)
    researchMode: true,
    defaults: { music: 'facebook/musicgen-small', sfx: 'mmaudio:small_16k', videoAudio: 'mmaudio:small_16k' },
    models: {
      music: [
        { label: 'MusicGen small', id: 'facebook/musicgen-small', available: true, installed: true, installable: true, setupRoute: { routeKey: 'pro.audio.musicgen.small', modality: 'audio', supportState: 'supported', preflightKey: 'musicgen', setupAction: 'POST /api/pro/audio/setup/musicgen/small' } },
        { label: 'MusicGen medium', id: 'facebook/musicgen-medium', available: true, installed: musicInstalled, installable: true, setupRoute: { routeKey: 'pro.audio.musicgen.medium', modality: 'audio', supportState: 'supported', preflightKey: 'musicgen', setupAction: 'POST /api/pro/audio/setup/musicgen/medium' } },
      ],
      sfx: [
        { label: 'MMAudio small 16k', id: 'mmaudio:small_16k', available: true, installed: true, installable: true, setupRoute: { routeKey: 'pro.audio.mmaudio.small-16k', modality: 'audio', supportState: 'supported', preflightKey: 'mmaudio', setupAction: 'POST /api/pro/audio/setup/mmaudio/small_16k' } },
        { label: 'MMAudio large 44k v2', id: 'mmaudio:large_44k_v2', available: minimumReady, installed, installable: true, setupRoute: { routeKey: 'pro.audio.mmaudio.large-44k-v2', modality: 'audio', supportState: 'supported', preflightKey: 'mmaudio', setupAction: 'POST /api/pro/audio/setup/mmaudio/large_44k_v2' } },
      ],
      videoAudio: [],
    },
    components: [
      { id: 'mmaudio-large-44k-v2', label: 'MMAudio Large 44 kHz V2', ready: installed, sharedReady: true, path: 'shared', missing: [], error: '' },
    ],
  }
}

test('Audio Studio installs selected MMAudio and MusicGen variants and refreshes readiness', async (t) => {
  const fixtureRoot = await mkdtemp(path.join(tmpdir(), 'aiwf-audio-model-setup-'))
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
    assert.ok(!path.isAbsolute(relative) && !relative.startsWith('..') && path.basename(fixtureRoot).startsWith('aiwf-audio-model-setup-'))
    await rm(fixtureRoot, { recursive: true, force: true })
  })

  await symlink(path.join(frontendRoot, 'src'), path.join(fixtureRoot, 'src'), 'junction')
  await symlink(path.join(frontendRoot, 'node_modules'), path.join(fixtureRoot, 'node_modules'), 'junction')
  await writeFile(path.join(fixtureRoot, 'index.html'), '<!doctype html><html><body><div id="root"></div><script type="module" src="/main.tsx"></script></body></html>')
  await writeFile(path.join(fixtureRoot, 'main.tsx'), `import React from 'react'
import { createRoot } from 'react-dom/client'
import { AudioStudioLayout } from './src/layouts/studio/AudioStudioLayout'
const settings = { prompt: 'soft rain', cfgScale: 3, steps: 25, seed: 11 }
createRoot(document.getElementById('root')).render(<AudioStudioLayout
 settings={settings} runtime={{ state: 'idle', device: 'CPU test' }} recentOutputs={[]}
 selectedModelName="Test image model" statusMessage="Ready" isGenerating={false}
 onSettingsChange={() => {}} onSendToWorkflow={() => {}} onOpenSettings={() => {}}
 onOpenModelSorter={() => { document.body.dataset.modelSorterOpened = 'true' }} />)`)

  await server.listen()
  const address = server.httpServer.address()
  assert.ok(address && typeof address === 'object')
  const page = await browser.newPage({ viewport: { width: 1400, height: 950 } })
  const requests = []
  let failInitialMusicPreparation = true
  let musicResident = false
  page.on('pageerror', (error) => requests.push({ error: error.message }))
  await page.route('**/api/pro/audio/status', (route) => {
    const status = audioStatus(false, false, false)
    for (const choice of status.models.music) {
      choice.routeStatus = 'prepared'
      choice.resident = musicResident
    }
    route.fulfill({ json: status })
  })
  await page.route('**/api/pro/audio/projects', (route) => route.fulfill({ json: { projects: [] } }))
  await page.route('**/api/pro/audio/prepare', async (route) => {
    requests.push({ path: new URL(route.request().url()).pathname, method: route.request().method(), body: route.request().postDataJSON() })
    const body = route.request().postDataJSON()
    if (failInitialMusicPreparation && body.kind === 'music' && body.modelId === 'facebook/musicgen-small') {
      failInitialMusicPreparation = false
      await route.fulfill({ status: 409, json: { detail: 'model operation is already running' } })
      return
    }
    if (body.modelId === 'mmaudio:small_16k') await new Promise((resolve) => setTimeout(resolve, 150))
    if (body.kind === 'music') musicResident = true
    await route.fulfill({ json: { ...body, ready: true, resident: true, routeStatus: 'prepared' } })
  })
  await page.route('**/api/pro/audio/generate', async (route) => {
    musicResident = false
    await route.fulfill({ json: { status: 'completed', url: '', outputPath: '', kind: 'music', modelId: 'facebook/musicgen-medium' } })
  })
  await page.route('**/api/pro/audio/setup/mmaudio/large_44k_v2', async (route) => {
    requests.push({ path: new URL(route.request().url()).pathname, method: route.request().method() })
    await route.fulfill({ json: audioStatus(true) })
  })
  await page.route('**/api/pro/audio/setup/minimum', async (route) => {
    requests.push({ path: new URL(route.request().url()).pathname, method: route.request().method() })
    await route.fulfill({ json: audioStatus(false, false, true) })
  })
  await page.route('**/api/pro/audio/setup/musicgen/medium', async (route) => {
    requests.push({ path: new URL(route.request().url()).pathname, method: route.request().method() })
    await route.fulfill({ json: audioStatus(true, true) })
  })
  await page.goto(`http://127.0.0.1:${address.port}/`, { waitUntil: 'domcontentloaded' })
  const labSetupStatus = page.locator('.studio-audio-setup-components span').filter({ hasText: 'Audio Lab DSP' })
  await page.waitForFunction(() => document.querySelector('.studio-audio-setup-components')?.textContent?.includes('Audio Lab DSP · detected'))
  assert.match(await labSetupStatus.textContent(), /detected/)
  assert.match(await labSetupStatus.getAttribute('title'), /full isolated-engine self-test has not run/)
  const mmaudioStatus = page.locator('.studio-audio-setup-components span').filter({ hasText: 'MMAudio small 16k' })
  await page.waitForFunction(() => document.querySelector('.studio-audio-setup-components')?.textContent?.includes('MMAudio small 16k · detected'))
  assert.match(await mmaudioStatus.getAttribute('title'), /runtime import check has not passed/)
  await page.getByRole('button', { name: 'Find and organize local model files' }).click()
  assert.equal(await page.locator('body').getAttribute('data-model-sorter-opened'), 'true')
  const audioLabEngine = page.locator('.studio-audio-card-list').filter({ hasText: 'Audio engines' }).getByRole('button', { name: /Audio Lab DSP/ })
  assert.match(await audioLabEngine.textContent(), /detected · full check needed/)
  assert.equal(await audioLabEngine.isDisabled(), true)
  assert.match(await audioLabEngine.getAttribute('title'), /Voice Cleanup and Loudness Master in Gradio Audio Lab/)
  assert.match(await audioLabEngine.textContent(), /Gradio Audio Lab/)
  const muxEngine = page.locator('.studio-audio-card-list').filter({ hasText: 'Audio engines' }).getByRole('button', { name: /FFmpeg mux/ })
  assert.equal(await muxEngine.isDisabled(), true)
  assert.match(await muxEngine.getAttribute('title'), /generated audio to video in Video Lab/)
  assert.match(await muxEngine.textContent(), /Video soundtracks · Video Lab/)
  const musicGenerateButton = page.getByRole('button', { name: 'Generate Music' }).first()
  await page.waitForFunction(() => document.body.innerText.includes('Preparing selected audio model…'))
  assert.equal(await musicGenerateButton.isDisabled(), true)
  await page.waitForFunction(() => {
    const button = document.querySelector('.studio-audio-transport button.primary')
    return button instanceof HTMLButtonElement && !button.disabled && !document.body.innerText.includes('Preparing selected audio model…')
  })
  assert.equal(await page.getByRole('button', { name: 'Retry model preparation' }).count(), 0)

  await page.getByLabel('Type').selectOption('sfx')
  await page.getByText('Preparing selected audio model…').waitFor({ state: 'visible' })
  assert.equal(await page.getByLabel('Type').isDisabled(), true)
  assert.equal(await page.locator('.studio-audio-card-list button').filter({ hasText: 'Music Bed' }).isDisabled(), true)
  await page.waitForFunction(() => {
    const status = [...(document.querySelectorAll('.studio-audio-setup-components span') ?? [])]
      .find((node) => node.textContent?.includes('MMAudio small 16k'))
    return status?.textContent?.includes('MMAudio small 16k') && !status.textContent.includes('detected')
  })
  const soundEffectsModel = page.getByLabel('Sound effects model')
  await soundEffectsModel.selectOption('mmaudio:large_44k_v2')
  assert.equal(await page.evaluate(() => localStorage.getItem('aiwf.audio-studio.sfx-model')), 'mmaudio:large_44k_v2')
  const audioEngines = page.locator('.studio-audio-card-list').filter({ hasText: 'Audio engines' })
  const soundEffectsEngine = audioEngines.getByRole('button', { name: /Sound effects generation/ })
  await assert.doesNotReject(() => soundEffectsEngine.getByText('MMAudio large 44k v2').waitFor({ state: 'visible' }))
  assert.match(await soundEffectsEngine.getAttribute('title'), /MMAudio large 44k v2/)
  assert.match(await soundEffectsEngine.getAttribute('title'), /Status reflects that variant's files and dependencies/)
  assert.match(await soundEffectsEngine.textContent(), /setup needed/)
  assert.equal(await soundEffectsEngine.getByText('MMAudio small 16k', { exact: true }).count(), 0)
  const installButton = page.getByRole('button', { name: 'Set up MMAudio large 44k v2 and audio runtime' })
  await installButton.waitFor({ state: 'visible' })
  await page.getByText('Matching MMAudio assets are available in a shared model root. Install copies them into Studio’s audio engine folder.').waitFor({ state: 'visible' })
  const generateButton = page.getByRole('button', { name: 'Generate Sound Effects' }).first()
  assert.equal(await generateButton.isDisabled(), true)

  await installButton.click()
  await page.waitForFunction(() => !document.body.innerText.includes('Setting up MMAudio large 44k v2 and audio runtime…'))
  await page.waitForFunction(() => {
    const button = document.querySelector('.studio-audio-transport button.primary')
    return button instanceof HTMLButtonElement && !button.disabled && !document.body.innerText.includes('Preparing selected audio model…')
  })
  assert.equal(await generateButton.isDisabled(), false)
  await page.getByLabel('Type').selectOption('music')
  await page.getByText('Preparing selected audio model…').waitFor({ state: 'visible' })
  await page.waitForFunction(() => {
    const button = document.querySelector('.studio-audio-transport button.primary')
    return button instanceof HTMLButtonElement && !button.disabled && !document.body.innerText.includes('Preparing selected audio model…')
  })
  assert.equal(await page.getByRole('button', { name: 'Generate Music' }).first().isDisabled(), false)
  const musicModel = page.getByLabel('Music model')
  await musicModel.selectOption('facebook/musicgen-medium')
  assert.equal(await page.evaluate(() => localStorage.getItem('aiwf.audio-studio.music-model')), 'facebook/musicgen-medium')
  const musicEngine = audioEngines.getByRole('button', { name: /Music generation/ })
  await assert.doesNotReject(() => musicEngine.getByText('MusicGen medium').waitFor({ state: 'visible' }))
  assert.match(await musicEngine.getAttribute('title'), /MusicGen medium/)
  assert.match(await musicEngine.textContent(), /setup needed/)
  assert.equal(await musicEngine.getByText('MusicGen small', { exact: true }).count(), 0)
  const musicInstallButton = page.getByRole('button', { name: 'Install MusicGen medium' })
  await musicInstallButton.waitFor({ state: 'visible' })
  await musicInstallButton.click()
  await musicInstallButton.waitFor({ state: 'detached' })
  await page.waitForFunction(() => {
    const button = document.querySelector('.studio-audio-transport button.primary')
    return button instanceof HTMLButtonElement && !button.disabled && !document.body.innerText.includes('Preparing selected audio model…')
  })
  await page.getByRole('button', { name: 'Generate Music' }).first().click()
  await page.getByText('Route prepared · model not loaded.', { exact: false }).waitFor({ state: 'visible' })
  assert.deepEqual(requests, [
    { path: '/api/pro/audio/prepare', method: 'POST', body: { kind: 'music', modelId: 'facebook/musicgen-small' } },
    { path: '/api/pro/audio/prepare', method: 'POST', body: { kind: 'music', modelId: 'facebook/musicgen-small' } },
    { path: '/api/pro/audio/prepare', method: 'POST', body: { kind: 'sfx', modelId: 'mmaudio:small_16k' } },
    { path: '/api/pro/audio/setup/minimum', method: 'POST' },
    { path: '/api/pro/audio/setup/mmaudio/large_44k_v2', method: 'POST' },
    { path: '/api/pro/audio/prepare', method: 'POST', body: { kind: 'sfx', modelId: 'mmaudio:large_44k_v2' } },
    { path: '/api/pro/audio/prepare', method: 'POST', body: { kind: 'music', modelId: 'facebook/musicgen-small' } },
    { path: '/api/pro/audio/setup/musicgen/medium', method: 'POST' },
    { path: '/api/pro/audio/prepare', method: 'POST', body: { kind: 'music', modelId: 'facebook/musicgen-medium' } },
  ])
})
