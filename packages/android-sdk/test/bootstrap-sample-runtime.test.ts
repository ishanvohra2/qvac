import test from 'node:test'
import assert from 'node:assert/strict'
import { normalizeDigest } from '../scripts/bootstrap-sample-runtime.ts'

test('normalizeDigest strips the sha256 prefix and lowercases the value', () => {
  assert.equal(normalizeDigest('sha256:ABCDEF'), 'abcdef')
  assert.equal(normalizeDigest('SHA256:ABCDEF'), 'abcdef')
  assert.equal(normalizeDigest('ABCDEF'), 'abcdef')
})
