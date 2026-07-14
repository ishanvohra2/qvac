# QVAC Android SDK Scaffold

`@qvac/android-sdk` adapts the QVAC SDK for native Android with a Bare Kit worklet runtime.
It solves the parity problem between `@qvac/sdk` and Android by syncing generated Kotlin/Gradle contract files and consuming generated JSON assets directly from `packages/sdk/android/generated`.
It also solves native runtime packaging pain by bootstrapping pinned Bare Kit artifacts and syncing required addon `.so` files automatically.
The result is a reproducible Android integration path that tracks SDK releases in lockstep.

## Setup guide

Run all commands from `packages/android-sdk`:

1. Install JS dependencies:
   - `bun install`
2. Sync generated Android contracts from `packages/sdk`:
   - `bun run android:sync-contract`
   - This also regenerates:
     - `sample-app/src/main/js/app.js` contract dispatch sections
     - `sample-app/src/main/java/io/tether/qvac/sample/BareQvacBridge.kt` generated client methods
3. Bootstrap pinned Bare Kit runtime artifacts:
   - `bun run sample:bootstrap-runtime`
4. Verify runtime bootstrap state (optional but recommended):
   - `bun run sample:check-runtime`
5. Build sample app (runs runtime check + link + addon sync + pack):
   - `./gradlew :sample-app:assembleDebug`

## Sync contract from SDK

Run from `packages/android-sdk`:

- `bun run android:sync-contract` to copy generated Kotlin/Gradle files from `packages/sdk`.
- `bun run android:check-contract` to fail when local files are stale.
- `bun run android:generate-bindings` to regenerate sample bridge/worklet contract bindings.
- `bun run android:check-bindings` to verify generated sample bindings are current.

## Contract files consumed

- `gradle/qvac-sdk.versions.toml`
- `src/main/java/io/tether/qvac/sdk/generated/GeneratedQvacSdkInfo.kt`
- `src/main/java/io/tether/qvac/sdk/generated/api/GeneratedQvacApi.kt`
- `../sdk/android/generated/qvac-sdk-manifest.json`
- `../sdk/android/generated/capabilities.json`
- `../sdk/android/generated/models-catalog.json`
- `../sdk/android/generated/api-contract.json`
- `../sdk/android/generated/addon-manifest.json`

This keeps Android dependency metadata and capability policy lockstep with `@qvac/sdk`.

## Android build integration

- The library module reads `../sdk/android/generated/qvac-sdk-manifest.json` for:
  - `group`
  - `version`
  - `namespace`
  - `minSdk`
  - `compileSdk`
- `preBuild` depends on:
  - `checkContractSync` (runs `bun run android:check-contract`)
  - `validateAddonPolicy` (ensures `addon-manifest.json` and `capabilities.json` are consistent)

Useful Gradle task:

- `./gradlew printQvacAndroidInfo`

## Sample application

A local Android app module is included at `sample-app/` to smoke-test SDK wiring.

- Runtime: Bare Kit worklet + `@qvac/bare-sdk` (real `loadModel` / `completion` / `unloadModel` path)
- Launcher activity: `sample-app/src/main/java/io/tether/qvac/sample/LauncherActivity.kt`
- Model test activities:
  - `sample-app/src/main/java/io/tether/qvac/sample/MainActivity.kt` (LLM)
  - `sample-app/src/main/java/io/tether/qvac/sample/TtsActivity.kt` (TTS)
  - `sample-app/src/main/java/io/tether/qvac/sample/WhisperActivity.kt` (Whisper)
  - `sample-app/src/main/java/io/tether/qvac/sample/TranslateActivity.kt` (Translation)
- ABI support: `arm64-v8a` only (current addon prebuild coverage is Android ARM64)
- x86/x86_64 emulators are not currently supported by this POC

Build sample debug APK:

- `./gradlew :sample-app:assembleDebug`

## SDK usage

The sample app demonstrates the Android integration pattern used by host apps:

1. Start the Bare worklet runtime once per screen/session via `BareQvacBridge.start()`.
2. Load a model with explicit plugin model type:
   - LLM: `llamacpp-completion`
   - TTS: `tts-ggml`
   - Whisper: `whispercpp-transcription`
   - Translation: `nmtcpp-translation`
3. Invoke task-specific operations (`completion`, `textToSpeech`, `transcribe`, `translate`).
4. Unload the active model and stop the bridge when the Activity is destroyed.

### Generated API contract status

`GeneratedQvacApi.kt` contains schema-derived contract metadata and typed wrappers only; it does not define the transport. The sample app still uses a separate ad-hoc JSON IPC protocol to talk to the worklet bundle.
The sample app IPC bridge (`BareQvacBridge.kt` + `app.js`) routes inference calls through generated contract operations (`pluginInvoke`, `pluginInvokeStream`, `heartbeat`, lifecycle/model-registry methods).
`loadModel` remains an explicit bootstrap action because model loading lives outside the currently generated Android API contract.

### Kotlin usage pattern

From `sample-app/src/main/java/io/tether/qvac/sample/BareQvacBridge.kt`, the app uses
JSON request/response calls over IPC to the worklet bundle:

- Load:
  - `bridge.loadModel(modelSrc, modelType, modelConfig)`
- LLM:
  - `bridge.streamCompletion(prompt)`
- TTS:
  - `bridge.textToSpeech(text)`
- Whisper:
  - `bridge.transcribe(audioPath, prompt)`
- Translate:
  - `bridge.translate(text)`
- Teardown:
  - `bridge.unloadModel()`
  - `bridge.stop()`

### Worklet plugin wiring

The worklet entry file (`sample-app/src/main/js/app.js`) registers plugins with
`@qvac/bare-sdk/plugins` and dispatches actions from Android IPC:

- `llmPlugin`
- `ttsPlugin`
- `whisperPlugin`
- `nmtPlugin`

When adding support for another model family, register the plugin in `app.js`,
add a corresponding action handler, and expose a typed bridge method in
`BareQvacBridge.kt`.

## Runtime bootstrap security and platform notes

- `sample:bootstrap-runtime` pins Bare Kit by tag (`v2.3.0`) and verifies `prebuilds.zip` SHA-256 before extraction.
- Extraction is cross-platform:
  - macOS/Linux: `unzip`
  - Windows: PowerShell `Expand-Archive`
- The sample build assumes `bun install` has already been run in `packages/android-sdk`; Gradle now fails fast with a clear `node_modules` error when dependencies are missing.
