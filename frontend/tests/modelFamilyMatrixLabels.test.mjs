import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const sourcePath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../src/layouts/studio/ModelFamilyMatrixLayout.tsx')
const source = await readFile(sourcePath, 'utf8')

test('unsupported routes are shown as blocked and included in the gap filter', () => {
  assert.match(source, /normalized\.includes\('unsupported'\)\) return 'blocked'/)
  assert.match(source, /text\.includes\('unsupported'\)/)
})

test('only explicitly supported or smoke-backed precision states receive the ready style', () => {
  assert.match(source, /normalized === 'supported' \|\| normalized\.includes\('supported-smoked'\)\) return 'ready'/)
  assert.match(source, /return 'neutral'/)
})

test('family support and readiness enums use clear user-facing labels', () => {
  assert.match(source, /Code-supported families/)
  assert.match(source, /family support status; local asset readiness is listed below/)
  assert.match(source, /'supported-plus-sidecars': 'Supported with required sidecars'/)
  assert.match(source, /'supported-when-folder-installed': 'Supported when a complete model folder is installed'/)
  assert.match(source, /'metadata-only': 'Metadata found · runtime unverified'/)
  assert.match(source, /return labels\[status\] \?\? humanizeStatus\(status\)/)
  assert.match(source, /Other status: \$\{humanizeStatus\(status\)\}/)
  assert.match(source, /supportStatusLabel\(family\.status\)/)
  assert.match(source, /supportStatusLabel\(precision\.status\)/)
  assert.match(source, /supportStatusLabel\(route\.status\)/)
  assert.match(source, /normalized\.includes\('partial'\)[\s\S]*normalized\.includes\('gated'\)[\s\S]*return 'warning'/)
  assert.ok(source.indexOf("normalized.includes('gated')") < source.indexOf("normalized.includes('supported')"))
})

test('a family with blockers beyond the shared example cap is not reported blocker-free', () => {
  assert.match(source, /function localBlockerTotal\(family: ModelFamily\)/)
  assert.match(source, /selectedBlockerTotal === 0 \? <li><span>No local blockers found for this family\.<\/span><\/li>/)
  assert.match(source, /selectedBlockerTotal > 0 \? <li><span>\{selectedBlockerTotal\} local blocker records exist; details are outside the current example limit\.<\/span><\/li>/)
})
