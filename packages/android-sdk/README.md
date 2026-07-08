# QVAC Android SDK Scaffold

This package consumes generated contract artifacts from `packages/sdk/android/generated` and places them in Android-standard locations.

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
- Entry activity: `sample-app/src/main/java/io/tether/qvac/sample/MainActivity.kt`

### One-time setup

From `packages/android-sdk`:

1. Install JS tooling + bare-sdk dependencies:
   - `bun install`
2. Download Bare Kit prebuild:
   - `gh release download --repo holepunchto/bare-kit v2.3.0 --pattern prebuilds.zip --dir sample-app`
3. Extract and place Android runtime artifacts:
   - unzip `sample-app/prebuilds.zip`
   - move `android/bare-kit` to `sample-app/libs/bare-kit`
4. Build (Gradle runs `bare-link` + `bare-pack` during `preBuild`):
   - `./gradlew :sample-app:assembleDebug`

Build sample debug APK:

- `./gradlew :sample-app:assembleDebug`
