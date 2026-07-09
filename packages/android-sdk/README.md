# QVAC Android SDK Scaffold

`@qvac/android-sdk` adapts the QVAC SDK for native Android with a Bare Kit worklet runtime.
It solves the parity problem between `@qvac/sdk` and Android by syncing generated API/contracts into Android-native assets and Kotlin sources.
It also solves native runtime packaging pain by bootstrapping pinned Bare Kit artifacts and syncing required addon `.so` files automatically.
The result is a reproducible Android integration path that tracks SDK releases in lockstep.

## Setup guide

Run all commands from `packages/android-sdk`:

1. Install JS dependencies:
   - `bun install`
2. Sync generated Android contracts from `packages/sdk`:
   - `bun run android:sync-contract`
3. Bootstrap pinned Bare Kit runtime artifacts:
   - `bun run sample:bootstrap-runtime`
4. Verify runtime bootstrap state (optional but recommended):
   - `bun run sample:check-runtime`
5. Build sample app (runs runtime check + link + addon sync + pack):
   - `./gradlew :sample-app:assembleDebug`

## Sync contract from SDK

Run from `packages/android-sdk`:

- `bun run android:sync-contract` to copy generated files from `packages/sdk`.
- `bun run android:check-contract` to fail when local files are stale.

## Contract files consumed

- `gradle/qvac-sdk.versions.toml`
- `src/main/java/io/tether/qvac/sdk/generated/GeneratedQvacSdkInfo.kt`
- `src/main/java/io/tether/qvac/sdk/generated/api/GeneratedQvacApi.kt`
- `src/main/assets/qvac-sdk-manifest.json`
- `src/main/assets/capabilities.json`
- `src/main/assets/models-catalog.json`
- `src/main/assets/api-contract.json`
- `src/main/assets/addon-manifest.json`

This keeps Android dependency metadata and capability policy lockstep with `@qvac/sdk`.

## Android build integration

- The library module reads `src/main/assets/qvac-sdk-manifest.json` for:
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
