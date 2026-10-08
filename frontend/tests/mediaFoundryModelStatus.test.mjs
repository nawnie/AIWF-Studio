import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

import { formatStudioModelAvailability } from '../src/layouts/studio/modelLabels.ts'

const sourcePath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../src/layouts/studio/MediaFoundryImageLayout.tsx')
const source = await readFile(sourcePath, 'utf8')

test('Media Foundry model rows show checkpoint and route readiness from Pro payloads', () => {
  assert.match(source, /formatStudioModelAvailability/)
  assert.match(source, /formatStudioModelFamily\(model\), formatStudioModelAvailability\(model\)/)
  assert.doesNotMatch(source, /formatStudioModelStatus\(model\.status\)/)

  assert.equal(formatStudioModelAvailability({
    id: 'local-checkpoint',
    name: 'Local checkpoint',
    checkpointPathStatus: 'present',
    routeStatus: 'request-eligible',
  }), 'Selectable · generation unverified')
  assert.equal(formatStudioModelAvailability({
    id: 'missing-checkpoint',
    name: 'Missing checkpoint',
    checkpointPathStatus: 'missing',
    routeStatus: 'blocked',
  }), 'Model files missing')
})

test('Media Foundry routes model changes through the shared loader and mode-filtered choices', () => {
  assert.match(source, /onChange=\{\(event\) => onModelSelect\?\.\(event\.target\.value\)\}/)
  assert.match(source, /\(selectableModels \?\? bootstrap\.models\)/)
  assert.doesNotMatch(source, /onChange=\{\(event\) => onSettingsChange\(\(current\) => \(\{ \.\.\.current, modelId: event\.target\.value \}\)\)\}/)
})
