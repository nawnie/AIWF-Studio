// Audio Studio in commercial-safe mode (the default) and the research-mode switch.
// AIWF Studio offers only commercially licensed audio models by default; MusicGen and MMAudio
// (CC-BY-NC 4.0) appear only after the person turns on research mode, and then carry a
// "non-commercial only" licence line. The API is mocked; no model runs.
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
const commercial = { license: 'MIT', commercial: 'yes', conditions: '', provenance: '', source: 'https://huggingface.co/ACE-Step/Ace-Step1.5' }
const nonCommercial = { license: 'CC-BY-NC-4.0', commercial: 'no', conditions: 'Non-commercial only.', provenance: '', source: 'https://huggingface.co/facebook/musicgen-small' }

// the status the server sends in each mode
function audioStatus(researchMode) {
  return {
    minimumReady: true, musicDependenciesReady: true, runtimeChecksPerformed: false, installing: false,
    musicReady: researchMode, sfxReady: researchMode, videoAudioReady: researchMode, labReady: true, muxReady: true,
    message: 'Audio Lab DSP is ready.',
    estimatedDownload: 'Existing files are reused.',
    researchMode,
    licenseNotice: researchMode
      ? 'Research mode is on: non-commercial models (MusicGen, MMAudio; CC-BY-NC 4.0) are available and labeled.'
      : 'Commercial-safe mode: only audio models whose licences allow commercial use are offered.',
    defaults: researchMode
      ? { music: 'facebook/musicgen-small', sfx: 'mmaudio:small_16k', videoAudio: 'mmaudio:small_16k' }
      : { music: 'acestep:1.5-turbo', sfx: '', videoAudio: '' },
    models: {
      music: [
        { label: 'ACE-Step 1.5 turbo', id: 'acestep:1.5-turbo', available: true, installed: true, installable: true, license: commercial },
        ...(researchMode
          ? [{ label: 'MusicGen small (minimum) · non-commercial (CC-BY-NC-4.0)', id: 'facebook/musicgen-small', available: true, installed: true, installable: true, license: nonCommercial }]
          : []),
      ],
      sfx: [
        { label: 'Licensed SFX', id: 'licensed:sfx', available: true, installed: true, installable: true, license: commercial },
        ...(researchMode
          ? [{ label: 'MMAudio small 16k', id: 'mmaudio:small_16k', available: true, installed: true, installable: true, license: nonCommercial }]
          : []),
      ],
      videoAudio: [],
    },
    components: [],
  }
}

test('Audio Studio is commercial-safe by default and labels research models after opting in', async (t) => {
  const fixtureRoot = await mkdtemp(path.join(tmpdir(), 'aiwf-audio-commercial-'))
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
    assert.ok(!path.isAbsolute(relative) && !relative.startsWith('..') && path.basename(fixtureRoot).startsWith('aiwf-audio-commercial-'))
    await rm(fixtureRoot, { recursive: true, force: true })
  })

  // a tiny page that renders only the Audio workspace
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
 onSettingsChange={() => {}} onSendToWorkflow={() => {}} onOpenSettings={() => {}} onOpenModelSorter={() => {}} />)`)
  await server.listen()
  const address = server.httpServer.address()
  assert.ok(address && typeof address === 'object')

  // the mocked server: research mode starts off and is changed only through its own route
  let researchMode = false
  const switches = []
  const page = await browser.newPage({ viewport: { width: 1400, height: 950 } })
  const pageErrors = []
  page.on('pageerror', (error) => pageErrors.push(error.message))
  await page.route('**/api/pro/audio/status', (route) => route.fulfill({ json: audioStatus(researchMode) }))
  await page.route('**/api/pro/audio/projects', (route) => route.fulfill({ json: { projects: [] } }))
  await page.route('**/api/pro/audio/prepare', (route) => route.fulfill({ json: { ready: true, resident: true, routeStatus: 'prepared' } }))
  await page.route('**/api/pro/audio/research-mode', async (route) => {
    const body = route.request().postDataJSON()
    switches.push(body)
    researchMode = Boolean(body.enabled)
    await route.fulfill({ json: audioStatus(researchMode) })
  })

  await page.goto(`http://127.0.0.1:${address.port}/`, { waitUntil: 'domcontentloaded' })

  // commercial-safe by default: only the commercially licensed engine is offered
  const researchSwitch = page.getByLabel(/Allow non-commercial research models/)
  await researchSwitch.waitFor({ state: 'visible' })
  assert.equal(await researchSwitch.isChecked(), false)
  await page.getByText('Commercial-safe mode: only audio models whose licences allow commercial use are offered.').waitFor({ state: 'visible' })
  const musicModel = page.getByLabel('Music model')
  assert.match(await musicModel.textContent(), /ACE-Step 1\.5 turbo/)
  await page.getByText(/Licence: MIT — commercial use OK/).waitFor({ state: 'visible' })
  assert.equal(await page.getByRole('option', { name: /MusicGen/ }).count(), 0)

  // A commercial-safe saved choice remains valid when research mode is enabled.
  await musicModel.selectOption('acestep:1.5-turbo')
  assert.equal(await page.evaluate(() => localStorage.getItem('aiwf.audio-studio.music-model')), 'acestep:1.5-turbo')

  // opting in: the switch calls the server (it changes only once the server confirms), and
  // MusicGen appears with its licence line
  await researchSwitch.click()
  assert.equal(await page.evaluate(() => localStorage.getItem('aiwf.audio-studio.music-model')), 'acestep:1.5-turbo')
  await musicModel.selectOption('facebook/musicgen-small')
  await page.getByText(/Licence: CC-BY-NC-4.0 — non-commercial only/).waitFor({ state: 'visible' })
  assert.deepEqual(switches, [{ enabled: true }])
  assert.equal(await researchSwitch.isChecked(), true)
  assert.match(await musicModel.textContent(), /MusicGen small \(minimum\) · non-commercial \(CC-BY-NC-4\.0\)/)

  // Select MMAudio in the separately persisted sound-effects kind.
  await page.getByLabel('Type').selectOption('sfx')
  const sfxModel = page.getByLabel('Sound effects model')
  await sfxModel.selectOption('mmaudio:small_16k')
  assert.equal(await page.evaluate(() => localStorage.getItem('aiwf.audio-studio.sfx-model')), 'mmaudio:small_16k')

  // Opting out removes both research-only IDs and falls back to choices still offered by the server.
  await researchSwitch.click()
  await page.waitForFunction(() => !document.body.innerText.includes('Licence: CC-BY-NC-4.0'))
  assert.deepEqual(switches, [{ enabled: true }, { enabled: false }])
  assert.equal(await page.evaluate(() => localStorage.getItem('aiwf.audio-studio.music-model')), null)
  assert.equal(await page.evaluate(() => localStorage.getItem('aiwf.audio-studio.sfx-model')), null)
  await page.getByLabel('Type').selectOption('music')
  assert.equal(await page.getByLabel('Music model').inputValue(), 'acestep:1.5-turbo')
  await page.getByLabel('Type').selectOption('sfx')
  assert.equal(await page.getByLabel('Sound effects model').inputValue(), 'licensed:sfx')
  await page.reload()
  assert.equal(await page.evaluate(() => localStorage.getItem('aiwf.audio-studio.music-model')), null)
  assert.equal(await page.evaluate(() => localStorage.getItem('aiwf.audio-studio.sfx-model')), null)
  assert.deepEqual(pageErrors, [])
})
