import assert from 'node:assert/strict'
import test from 'node:test'

import { resolveLoopbackApiBase } from '../src/apiBasePolicy.ts'

test('uses same-origin requests when no API base is configured', () => {
  assert.equal(resolveLoopbackApiBase(undefined), '')
  assert.equal(resolveLoopbackApiBase('  '), '')
})

test('accepts explicit loopback API bases', () => {
  assert.equal(resolveLoopbackApiBase('http://127.0.0.1:7861/'), 'http://127.0.0.1:7861')
  assert.equal(resolveLoopbackApiBase('http://localhost:7861'), 'http://localhost:7861')
  assert.equal(resolveLoopbackApiBase('http://[::1]:7861/'), 'http://[::1]:7861')
})

test('rejects non-loopback and credential-bearing API bases', () => {
  assert.throws(() => resolveLoopbackApiBase('https://studio.example.com'), /restricted to/)
  assert.throws(() => resolveLoopbackApiBase('http://192.168.1.50:7861'), /restricted to/)
  assert.throws(() => resolveLoopbackApiBase('http://user:secret@localhost:7861'), /cannot contain credentials/)
})
