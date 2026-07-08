# Android Codegen Contract

This folder defines a deterministic contract between `@qvac/sdk` and an Android-native SDK artifact.

## Source of truth

- `manifest.source.json` is the only hand-authored input.
- `scripts/android/generate.ts` combines:
  - SDK version + dependency graph from `package.json`
  - addon and engine presence from `models/registry/models.ts`
  - Android policy from `manifest.source.json`

## Generated outputs

All generated outputs are written to `android/generated/`:

- `qvac-sdk-manifest.json`
  - Canonical generated contract used by downstream tooling.
  - Contains sdk version, Android metadata, runtime metadata, dependency list, and capabilities.
- `capabilities.json`
  - Addon capability matrix for Android runtime guards (`androidSupported` + `fallbackBehavior`).
- `models-catalog.json`
  - Generated mirror of `models/registry/models.ts` constants for Android consumption.
  - Contains `name`, `src`, `modelId`, `engine`, `addon`, and registry metadata.
- `api-contract.json`
  - Auto-discovered SDK API operations from `schemas/*.ts` request/response schema pairs.
  - Used to keep Android API surface generation in sync with new SDK APIs.
- `libs.versions.toml`
  - Generated Gradle version catalog entries from SDK dependency policy.
- `GeneratedQvacSdkInfo.kt`
  - Generated Kotlin constants used by Android build/runtime code.
- `GeneratedQvacApi.kt`
  - Generated Kotlin request/response wrappers and API interface stubs.
- `addon-manifest.json`
  - Minimal addon policy manifest for packaging and autolinking.

## Commands

Run from `packages/sdk`:

- `bun run android:sync` to regenerate outputs.
- `bun run android:check-sync` to fail if generated files are stale.

## Version policy

- The generated contract is lockstep with `@qvac/sdk` version.
- When SDK version changes, `android:sync` regenerates contract files with the new version.
