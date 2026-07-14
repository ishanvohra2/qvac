import test from 'node:test'
import assert from 'node:assert/strict'
import { isBarePrebuild, normalizeSharedObjectBaseName } from '../scripts/sync-runtime-addons.ts'

test('normalizeSharedObjectBaseName rewrites scoped package names', () => {
  assert.equal(normalizeSharedObjectBaseName('@qvac/llm-llamacpp'), 'qvac__llm-llamacpp')
  assert.equal(normalizeSharedObjectBaseName('bare-fs'), 'bare-fs')
})

test('isBarePrebuild only matches regular files ending in .bare', () => {
  assert.equal(isBarePrebuild({ isFile: () => true, name: 'addon.bare' }), true)
  assert.equal(isBarePrebuild({ isFile: () => true, name: 'addon.so' }), false)
  assert.equal(isBarePrebuild({ isFile: () => false, name: 'nested.bare' }), false)
})
