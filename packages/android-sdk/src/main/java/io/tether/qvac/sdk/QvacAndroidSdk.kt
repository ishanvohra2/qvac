package io.tether.qvac.sdk

import io.tether.qvac.sdk.generated.GeneratedQvacSdkInfo

object QvacAndroidSdk {
  fun version(): String {
    return GeneratedQvacSdkInfo.VERSION
  }

  fun coordinates(): String {
    return "${GeneratedQvacSdkInfo.GROUP_ID}:${GeneratedQvacSdkInfo.ARTIFACT_ID}:${GeneratedQvacSdkInfo.VERSION}"
  }

  fun bundledAssetNames(): List<String> {
    return listOf(
      "qvac-sdk-manifest.json",
      "capabilities.json",
      "models-catalog.json",
      "api-contract.json",
      "addon-manifest.json"
    )
  }
}
