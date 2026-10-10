import assert from 'node:assert/strict'
import test from 'node:test'
import { audioProjectAttribution } from '../src/layouts/studio/audioProjectAttribution.ts'

const musicGenLicense = {
  model_id: 'facebook/musicgen-small',
  checked: '2026-10-08',
  family: 'MusicGen (Meta)',
  license: 'CC-BY-NC-4.0',
  commercial: 'no',
  conditions: 'Non-commercial use only.',
  provenance: 'Licensed music data.',
  source: 'https://huggingface.co/facebook/musicgen-small',
}

test('project attribution follows the generated model after the selector changes', () => {
  const generated = {
    modelId: 'facebook/musicgen-small',
    license: musicGenLicense,
  }

  const attribution = audioProjectAttribution(generated, 'acestep:1.5-turbo')

  assert.equal(attribution.modelId, 'facebook/musicgen-small')
  assert.deepEqual(attribution.license, musicGenLicense)
  assert.match(attribution.licenseNotice, /CC-BY-NC-4\.0/)
  assert.match(attribution.licenseNotice, /commercial use: no/)
  assert.match(attribution.licenseNotice, /checked: 2026-10-08/)
  assert.match(attribution.licenseNotice, /source: https:\/\/huggingface\.co\/facebook\/musicgen-small/)
})

test('legacy artifacts without a record remain unattributed instead of borrowing the selected model', () => {
  const attribution = audioProjectAttribution({ modelId: 'facebook/musicgen-small' }, 'acestep:1.5-turbo')

  assert.equal(attribution.modelId, 'facebook/musicgen-small')
  assert.equal(attribution.license, null)
  assert.equal(attribution.licenseNotice, null)
})

test('a mismatched structured record fails closed', () => {
  assert.throws(
    () => audioProjectAttribution({ modelId: 'facebook/musicgen-small', license: { ...musicGenLicense, model_id: 'acestep:1.5-turbo' } }, 'acestep:1.5-turbo'),
    /does not match its generating model/,
  )
})
