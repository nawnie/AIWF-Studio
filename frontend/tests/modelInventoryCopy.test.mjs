import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const sourcePath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../src/App.tsx')
const source = await readFile(sourcePath, 'utf8')
const apiSource = await readFile(path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../src/api.ts'), 'utf8')

test('catalog-installed count does not claim route readiness', () => {
  assert.match(source, /StatTile label="Installed" value=\{`\$\{downloadSummary\.installed\}`\} hint="catalog assets found"/)
  assert.doesNotMatch(source, /StatTile label="Installed"[^\n]*hint="ready locally"/)
})

test('model cards do not call unspecified file details a local asset', () => {
  assert.match(source, /'Asset details unavailable'/)
  assert.doesNotMatch(source, /'Local asset'/)
})

test('Settings exposes a nonmoving scan of configured model roots', () => {
  assert.match(source, /Scan model roots/)
  assert.match(source, /Scan checks all configured model and checkpoint roots without moving files/)
  assert.match(source, /Covered by another configured root/)
  assert.match(source, /Folder not found/)
  assert.match(source, /Partially scanned/)
  assert.match(source, /familyCounts/)
  assert.match(apiSource, /requestJson\(`\/api\/pro\/models\/scan\?limit=/)
})

test('shared-root sorting previews safe candidates and keeps originals', () => {
  assert.match(source, /Place confident files from scan/)
  assert.match(source, /offset < scan\.matchedCount; offset \+= 250/)
  assert.match(source, /limit: 250/)
  assert.doesNotMatch(source, /slice\(0, 25\)/)
  assert.match(source, /revalidate every file before copying/)
  assert.match(apiSource, /previewSharedModelPlacement/)
  assert.match(apiSource, /applySharedModelPlacement/)
  assert.match(apiSource, /sourcePreserved/)
})

test('manual review rows explain why automatic placement was withheld', () => {
  assert.match(source, /Manual review: \$\{asset\.placementReason \|\| 'automatic placement is not safe'\}/)
})

test('supported routes without an installer show their route-specific setup guidance', () => {
  assert.match(source, /supportState === 'supported-when-folder-installed'/)
  assert.match(source, /selectedModel\.setupRoute\.limitation/)
})
