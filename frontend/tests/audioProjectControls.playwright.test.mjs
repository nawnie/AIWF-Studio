import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createRequire } from 'node:module'
import { test } from 'node:test'
import { fileURLToPath, pathToFileURL } from 'node:url'

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const studioSource = path.join(frontendRoot, 'src', 'layouts', 'studio')
const canonicalModules = path.join(frontendRoot, 'node_modules')
const require = createRequire(path.join(frontendRoot, 'package.json'))
const vitePath = require.resolve('vite')
const reactPluginPath = require.resolve('@vitejs/plugin-react')
const { createServer } = await import(pathToFileURL(vitePath))
const reactPlugin = (await import(pathToFileURL(reactPluginPath))).default
const { chromium } = require('playwright')

test('audio project controls honor cancel, save, repeated load, and missing-audio errors', async (t) => {
  const tempBase = path.resolve(process.env.AUDIO_PROJECT_TEST_TMP || os.tmpdir())
  fs.mkdirSync(tempBase, { recursive: true })
  const fixtureRoot = fs.mkdtempSync(path.join(tempBase, 'aiwf-audio-project-ui-'))
  const sourceLink = path.join(fixtureRoot, 'src', 'layouts', 'studio')
  fs.mkdirSync(path.dirname(sourceLink), { recursive: true })
  fs.symlinkSync(studioSource, sourceLink, 'junction')
  fs.symlinkSync(canonicalModules, path.join(fixtureRoot, 'node_modules'), 'junction')
  fs.writeFileSync(path.join(fixtureRoot, 'index.html'), `<!doctype html>
<html lang="en"><head><meta charset="UTF-8"><link rel="icon" href="data:,"></head>
<body><div id="root"></div><script type="module" src="/main.tsx"></script></body></html>`)
  fs.writeFileSync(path.join(fixtureRoot, 'main.tsx'), `import React from 'react'
import { createRoot } from 'react-dom/client'
import { AudioProjectControls } from './src/layouts/studio/AudioProjectControls'
window.audioProjectEvents = []
createRoot(document.getElementById('root')).render(<AudioProjectControls projects={[
  { project_id: '11111111111111111111111111111111', name: 'Ambient sketch', updated_at: '2026-10-03T13:00:00Z', has_audio: true },
  { project_id: '22222222222222222222222222222222', name: 'Missing reference', updated_at: '2026-10-03T12:00:00Z', has_audio: false, audio_missing: true },
]} onRefresh={() => window.audioProjectEvents.push({ type: 'refresh' })}
onSave={(payload) => window.audioProjectEvents.push({ type: 'save', payload })}
onLoad={(id) => { window.audioProjectEvents.push({ type: 'load', payload: id }); if (id.startsWith('2')) throw new Error('Audio project refers to missing audio.') }} />)`)

  const server = await createServer({
    configFile: false,
    root: fixtureRoot,
    cacheDir: path.join(fixtureRoot, '.vite-cache'),
    plugins: [reactPlugin()],
    resolve: { dedupe: ['react', 'react-dom'] },
    server: {
      host: '127.0.0.1',
      port: 0,
      strictPort: false,
      fs: { allow: [fixtureRoot, studioSource, canonicalModules] },
    },
  })
  let browser
  t.after(async () => {
    await browser?.close()
    await server.close()
    const rel = path.relative(tempBase, fixtureRoot)
    if (path.isAbsolute(rel) || rel.startsWith('..') || !path.basename(fixtureRoot).startsWith('aiwf-audio-project-ui-')) {
      throw new Error('Refusing to remove an audio UI fixture outside its generated temporary directory.')
    }
    fs.rmSync(fixtureRoot, { recursive: true, force: true })
  })

  await server.listen()
  const address = server.httpServer.address()
  assert.ok(address && typeof address === 'object')
  browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROME_PATH || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    args: ['--disable-gpu', '--no-sandbox'],
  })
  const page = await browser.newPage({ viewport: { width: 920, height: 640 } })
  const browserErrors = []
  page.on('pageerror', (error) => browserErrors.push(error.message))
  page.on('console', (message) => {
    if (message.type() === 'error') browserErrors.push(message.text())
  })
  await page.goto(`http://127.0.0.1:${address.port}/`, { waitUntil: 'networkidle' })

  const saveTrigger = page.getByRole('button', { name: 'Save Project' })
  await saveTrigger.click()
  await page.getByLabel('Project name').fill('Cancelled title')
  await page.keyboard.press('Shift+Tab')
  assert.equal(await page.getByRole('dialog').getByRole('button', { name: 'Save' }).evaluate((node) => node === document.activeElement), true)
  await page.keyboard.press('Tab')
  assert.equal(await page.getByLabel('Project name').evaluate((node) => node === document.activeElement), true)
  await page.keyboard.press('Escape')
  assert.equal(await page.getByRole('dialog').count(), 0)
  assert.equal(await saveTrigger.evaluate((node) => node === document.activeElement), true)
  assert.deepEqual(await page.evaluate(() => window.audioProjectEvents), [])

  await page.getByRole('button', { name: 'Save Project' }).click()
  await page.getByLabel('Project name').fill('Rendered project')
  await page.getByRole('dialog').getByRole('button', { name: 'Save' }).click()
  assert.deepEqual(await page.evaluate(() => window.audioProjectEvents), [
    { type: 'save', payload: { name: 'Rendered project' } },
  ])

  for (let attempt = 0; attempt < 2; attempt += 1) {
    await page.getByRole('button', { name: 'Load', exact: true }).click()
    await page.getByRole('dialog').getByRole('combobox').selectOption('11111111111111111111111111111111')
    await page.getByRole('dialog').getByRole('button', { name: 'Load' }).click()
  }
  assert.deepEqual(await page.evaluate(() => window.audioProjectEvents), [
    { type: 'save', payload: { name: 'Rendered project' } },
    { type: 'refresh' },
    { type: 'load', payload: '11111111111111111111111111111111' },
    { type: 'refresh' },
    { type: 'load', payload: '11111111111111111111111111111111' },
  ])

  await page.getByRole('button', { name: 'Load', exact: true }).click()
  await page.getByRole('dialog').getByRole('combobox').selectOption('22222222222222222222222222222222')
  await page.getByRole('dialog').getByRole('button', { name: 'Load' }).click()
  await page.getByRole('alert').getByText('Audio project refers to missing audio.').waitFor()
  assert.equal(await page.getByRole('dialog').count(), 1)
  await page.getByRole('button', { name: 'Cancel' }).click()

  await page.getByRole('button', { name: 'Save Project' }).click()
  if (process.env.AUDIO_PROJECT_SCREENSHOT) {
    await page.screenshot({ path: process.env.AUDIO_PROJECT_SCREENSHOT, fullPage: true })
  }
  await page.getByRole('button', { name: 'Cancel' }).click()
  assert.deepEqual(browserErrors, [])
})
