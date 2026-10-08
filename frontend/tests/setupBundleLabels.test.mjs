import assert from 'node:assert/strict'
import test from 'node:test'

import { formatSetupBundleLabel, SETUP_BUNDLE_LABELS } from '../src/layouts/studio/setupBundleLabels.ts'

test('known video and image setup bundles have route-specific labels', () => {
  assert.equal(formatSetupBundleLabel('flux-components'), 'Flux conditioning assets (CLIP-L, T5-XXL, VAE + tokenizers)')
  assert.equal(formatSetupBundleLabel('wan-14b-components'), 'Wan 2.2 A14B high/low support (UMT5, tokenizer, scheduler + Wan 2.1 VAE)')
  assert.equal(formatSetupBundleLabel('wan-ti2v-diffusers'), 'Wan 2.2 TI2V 5B Diffusers snapshot (large download)')
  assert.equal(formatSetupBundleLabel('wan-ti2v-support'), 'Wan TI2V 5B support assets (UMT5, tokenizer, scheduler + 48-channel VAE)')
  assert.equal(formatSetupBundleLabel('sana-video-720p'), 'Sana Video 2B 720p pipeline')
  assert.equal(formatSetupBundleLabel('sana-sprint-16b'), 'Sana Sprint 1.6B image model')
  assert.equal(formatSetupBundleLabel('flux-kontext'), 'Flux Kontext image pipeline')
  assert.equal(formatSetupBundleLabel('flux2-4b-base'), 'Flux.2 Klein Base 4B Diffusers pipeline')
  assert.equal(formatSetupBundleLabel('ltx-2b'), 'LTX Video 0.9.5 2B text-to-video + T5-XXL fp16 (~16 GB)')
  assert.equal(formatSetupBundleLabel('ltx23-one-stage'), 'LTX 2.3 one-stage BF16 model assets (22B; blocked on Windows)')
  assert.equal(formatSetupBundleLabel('ltx23-one-stage-fp8'), 'LTX 2.3 one-stage FP8 model and Gemma text encoder (supported on Windows)')
})

test('every explicit setup bundle label is non-empty and formatting is stable', () => {
  assert.ok(Object.keys(SETUP_BUNDLE_LABELS).length > 0)
  for (const [key, label] of Object.entries(SETUP_BUNDLE_LABELS)) {
    assert.ok(label.trim(), `${key} must have a label`)
    assert.equal(formatSetupBundleLabel(key), label)
  }
  assert.equal(formatSetupBundleLabel('controlnet-sd15'), 'ControlNet SD 1.5')
})
