import groovy.json.JsonSlurper
import java.io.File

plugins {
  id("com.android.library")
  id("org.jetbrains.kotlin.android")
}

typealias JsonObject = Map<String, Any>

fun readJsonObject(file: File): JsonObject {
  if (!file.exists()) {
    throw GradleException("Required JSON file not found: ${file.path}")
  }
  @Suppress("UNCHECKED_CAST")
  return JsonSlurper().parse(file) as JsonObject
}

fun readJsonArray(file: File): List<JsonObject> {
  if (!file.exists()) {
    throw GradleException("Required JSON file not found: ${file.path}")
  }
  @Suppress("UNCHECKED_CAST")
  return JsonSlurper().parse(file) as List<JsonObject>
}

fun stringValue(map: JsonObject, key: String): String {
  val value = map[key] ?: throw GradleException("Missing key '$key' in JSON object")
  return value.toString()
}

fun intValue(map: JsonObject, key: String): Int {
  val value = map[key] ?: throw GradleException("Missing key '$key' in JSON object")
  return when (value) {
    is Number -> value.toInt()
    is String -> value.toInt()
    else -> throw GradleException("Expected numeric key '$key', got ${value::class}")
  }
}

fun readUtf8(file: File): String {
  if (!file.exists()) {
    throw GradleException("Required file not found: ${file.path}")
  }
  return file.readText(Charsets.UTF_8)
}

val sdkGeneratedDir = file("../sdk/android/generated")
val syncEntriesFile = file("scripts/sync-contract-entries.json")
val qvacManifestFile = File(sdkGeneratedDir, "qvac-sdk-manifest.json")
val qvacManifest = readJsonObject(qvacManifestFile)
@Suppress("UNCHECKED_CAST")
val androidConfig = qvacManifest["android"] as? JsonObject
  ?: throw GradleException("Missing 'android' object in ${qvacManifestFile.path}")
@Suppress("UNCHECKED_CAST")
val sdkConfig = qvacManifest["sdk"] as? JsonObject
  ?: throw GradleException("Missing 'sdk' object in ${qvacManifestFile.path}")

group = stringValue(androidConfig, "groupId")
version = stringValue(sdkConfig, "version")

android {
  namespace = stringValue(androidConfig, "namespace")
  compileSdk = intValue(androidConfig, "compileSdk")

  defaultConfig {
    minSdk = intValue(androidConfig, "minSdk")
    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    consumerProguardFiles("consumer-rules.pro")
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }

  kotlinOptions {
    jvmTarget = "17"
  }

  sourceSets {
    getByName("main") {
      assets.srcDir(sdkGeneratedDir)
    }
  }
}

dependencies {
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")
}

tasks.register("checkContractSync") {
  group = "verification"
  description = "Checks whether Android SDK contract files are synced from packages/sdk"
  doLast {
    if (!syncEntriesFile.exists()) {
      throw GradleException("Missing sync entries file: ${syncEntriesFile.path}")
    }
    @Suppress("UNCHECKED_CAST")
    val syncPairs = (JsonSlurper().parse(syncEntriesFile) as List<Map<String, String>>).map { entry ->
      val sourceRelative = entry["sourceRelativePath"]
        ?: throw GradleException("sync-contract entry missing sourceRelativePath")
      val destinationRelative = entry["destinationRelativePath"]
        ?: throw GradleException("sync-contract entry missing destinationRelativePath")
      sourceRelative to destinationRelative
    }

    val driftedFiles = mutableListOf<String>()
    for ((sourceRelative, destinationRelative) in syncPairs) {
      val sourceFile = File(sdkGeneratedDir, sourceRelative)
      val destinationFile = file(destinationRelative)
      val sourceContent = readUtf8(sourceFile)
      val destinationContent = if (destinationFile.exists()) {
        destinationFile.readText(Charsets.UTF_8)
      } else {
        null
      }

      if (destinationContent != sourceContent) {
        driftedFiles.add(destinationRelative)
      }
    }

    val sdkManifest = readJsonObject(File(sdkGeneratedDir, "qvac-sdk-manifest.json"))
    @Suppress("UNCHECKED_CAST")
    val sdkSection = sdkManifest["sdk"] as? JsonObject
      ?: throw GradleException("Missing 'sdk' object in ${sdkGeneratedDir.path}/qvac-sdk-manifest.json")
    val expectedVersion = stringValue(sdkSection, "version")
    val androidPackageJson = readJsonObject(file("package.json"))
    val currentVersion = stringValue(androidPackageJson, "version")
    if (currentVersion != expectedVersion) {
      driftedFiles.add("package.json (version=${currentVersion}, expected=${expectedVersion})")
    }

    if (driftedFiles.isNotEmpty()) {
      throw GradleException(
        "Android SDK contract is out of sync. Run `bun run android:sync-contract` from packages/android-sdk.\n" +
          "Drifted paths:\n- ${driftedFiles.joinToString("\n- ")}"
      )
    }

    println("Contract sync validation passed")
  }
}

tasks.register("validateAddonPolicy") {
  group = "verification"
  description = "Validates addon manifest and capabilities consistency"

  doLast {
    val capabilities = readJsonArray(File(sdkGeneratedDir, "capabilities.json"))
    val addonManifest = readJsonObject(File(sdkGeneratedDir, "addon-manifest.json"))
    @Suppress("UNCHECKED_CAST")
    val addonEntries = addonManifest["addons"] as? List<JsonObject>
      ?: throw GradleException("Missing 'addons' array in addon-manifest.json")

    val capabilitiesByAddon = capabilities.associateBy { stringValue(it, "addon") }
    val addonByName = addonEntries.associateBy { stringValue(it, "addon") }

    val missingCapabilities = addonByName.keys.filterNot { capabilitiesByAddon.containsKey(it) }
    if (missingCapabilities.isNotEmpty()) {
      throw GradleException(
        "Addon manifest contains addons not present in capabilities: ${missingCapabilities.joinToString(", ")}"
      )
    }

    val missingAddons = capabilitiesByAddon.keys.filterNot { addonByName.containsKey(it) }
    if (missingAddons.isNotEmpty()) {
      throw GradleException(
        "Capabilities contains addons not present in addon-manifest: ${missingAddons.joinToString(", ")}"
      )
    }

    println("Addon policy validation passed (${addonByName.size} addons)")
  }
}

tasks.register("printQvacAndroidInfo") {
  group = "help"
  description = "Prints Android SDK coordinates from synced manifest"
  doLast {
    println("group=$group")
    println("version=$version")
    println("namespace=${stringValue(androidConfig, "namespace")}")
    println("minSdk=${intValue(androidConfig, "minSdk")}")
    println("compileSdk=${intValue(androidConfig, "compileSdk")}")
  }
}

tasks.matching { it.name == "preBuild" }.configureEach {
  dependsOn("checkContractSync")
  dependsOn("validateAddonPolicy")
}
