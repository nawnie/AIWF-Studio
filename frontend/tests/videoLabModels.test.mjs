import assert from 'node:assert/strict'
import test from 'node:test'

import { videoLabExtendModels } from '../src/layouts/studio/videoLabModels.ts'

const models = [
  { id: 'Wan-AI/Wan2.2-TI2V-5B-Diffusers', name: 'Wan 2.2 TI2V 5B', engineId: 'wan' },
  { id: 'wan2.2_i2v_a14b', name: 'Wan 2.2 I2V A14B high/low', engineId: 'wan' },
  { id: 'wan2.2_t2v_5b', name: 'Wan 2.2 T2V 5B', engineId: 'wan' },
  { id: 'not-wan-ti2v-5b', name: 'Wan TI2V 5B', engineId: 'unknown' },
  { id: 'wan-ti2v-5b-missing', name: 'Wan TI2V 5B missing', engineId: 'wan', checkpointPathStatus: 'missing' },
  { id: 'wan-ti2v-5b-blocked', name: 'Wan TI2V 5B blocked', engineId: 'wan', routeStatus: 'blocked' },
  { id: 'wan-ti2v-5b-route-runtime', name: 'Wan TI2V 5B route runtime blocked', engineId: 'wan', routeStatus: 'blocked_runtime' },
  { id: 'wan-ti2v-5b-blocked-runtime', name: 'Wan TI2V 5B blocked runtime', engineId: 'wan', status: 'blocked_runtime' },
  { id: 'wan-ti2v-5b-runtime', name: 'Wan TI2V 5B runtime blocked', engineId: 'wan', status: 'broken_runtime' },
  { id: 'wan-ti2v-5b-needs-snapshot', name: 'Wan TI2V 5B needs snapshot', engineId: 'wan', status: 'needs snapshot' },
]

test('Video Lab Extend choices include only Wan TI2V 5B models', () => {
  assert.deepEqual(videoLabExtendModels(models).map((model) => model.id), [
    'Wan-AI/Wan2.2-TI2V-5B-Diffusers',
  ])
})
