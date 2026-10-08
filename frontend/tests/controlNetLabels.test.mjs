import assert from 'node:assert/strict'
import test from 'node:test'

import { formatControlNetCompatibilityLabel } from '../src/layouts/studio/controlNetLabels.ts'

test('ControlNet label does not imply local readiness from family compatibility alone', () => {
  assert.equal(
    formatControlNetCompatibilityLabel({ supported: true, controlNetFamily: null }),
    'ControlNet compatibility unverified',
  )
  assert.equal(
    formatControlNetCompatibilityLabel({ supported: true, controlNetFamily: 'sdxl' }),
    'ControlNet family matches',
  )
  assert.equal(
    formatControlNetCompatibilityLabel({ supported: false, controlNetFamily: null }),
    'ControlNet unavailable',
  )
})
