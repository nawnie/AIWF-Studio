import assert from 'node:assert/strict'
import test from 'node:test'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

import { formatAgentModelLabel } from '../src/layouts/studio/agentModelLabel.ts'

test('Agentic Chat labels the selected Ollama model, independent of the Pro model', () => {
  const ollamaModels = [
    { id: 'llama-local', name: 'Llama 3.1 8B' },
    { id: 'qwen-local', name: 'Qwen 2.5 14B' },
  ]

  assert.equal(formatAgentModelLabel('qwen-local', ollamaModels), 'Qwen 2.5 14B')
})

test('Agentic Chat header uses its own Ollama selection label', async () => {
  const sourcePath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../src/layouts/studio/AgenticChatLayout.tsx')
  const source = await readFile(sourcePath, 'utf8')

  assert.match(source, /formatAgentModelLabel\(selectedModel, models\)/)
  assert.match(source, /\{runtime\.state\} · \{selectedAgentModelLabel\} · \{statusMessage\}/)
  assert.doesNotMatch(source, /runtime\.state · \{selectedModelName\}/)
})

test('Ollama tag listing is described as available models, not loaded models', async () => {
  const sourcePath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../src/layouts/studio/AgenticChatLayout.tsx')
  const source = await readFile(sourcePath, 'utf8')

  assert.match(source, /Found \$\{nextModels\.length\} Ollama model\(s\)\./)
  assert.match(source, /No Ollama models available/)
  assert.doesNotMatch(source, /Loaded \$\{nextModels\.length\} Ollama model/)
  assert.doesNotMatch(source, /No Ollama model loaded/)
})

test('Pro shell passes family-aware selection labels to persistent model surfaces', async () => {
  const sourcePath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../src/App.tsx')
  const source = await readFile(sourcePath, 'utf8')

  assert.match(source, /const selectedModelLabel = selectedModel[\s\S]{0,150}formatStudioModelLabel\(selectedModel, bootstrap\.models\)/)
  assert.match(source, /selectedModelName=\{selectedModelLabel\}/)
  assert.match(source, /selectedModelName: selectedModelLabel/)
  assert.doesNotMatch(source, /selectedModelName=\{selectedModel\?\.name/)
})

test('model card asset summaries are compared without casing mismatches', async () => {
  const sourcePath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../src/App.tsx')
  const source = await readFile(sourcePath, 'utf8')

  assert.match(source, /model\.name\.toLowerCase\(\)\.includes\(model\.assetSummary\.toLowerCase\(\)\)/)
})
