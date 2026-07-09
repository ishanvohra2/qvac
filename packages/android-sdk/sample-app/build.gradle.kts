import org.gradle.api.GradleException

plugins {
  id("com.android.application")
  id("org.jetbrains.kotlin.android")
}

android {
  namespace = "io.tether.qvac.sample"
  compileSdk = 35
  ndkVersion = "27.2.12479018"

  defaultConfig {
    applicationId = "io.tether.qvac.sample"
    minSdk = 26
    targetSdk = 35
    versionCode = 1
    versionName = "1.0.0"
    ndk {
      abiFilters += "arm64-v8a"
    }
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      proguardFiles(
        getDefaultProguardFile("proguard-android-optimize.txt"),
        "proguard-rules.pro"
      )
    }
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
      jniLibs.srcDirs("src/main/addons", "libs/bare-kit/jni")
    }
  }

  packaging {
    jniLibs {
      pickFirsts += "**/libc++_shared.so"
    }
  }
}

dependencies {
  implementation(project(":"))
  api(fileTree(mapOf("dir" to "libs", "include" to listOf("bare-kit/classes.jar"))))
  implementation("androidx.appcompat:appcompat:1.7.0")
  implementation("com.google.android.material:material:1.12.0")
  implementation("androidx.core:core-ktx:1.15.0")
  implementation("androidx.constraintlayout:constraintlayout:2.2.0")
  implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
  implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}

tasks.register<Exec>("link") {
  workingDir = file("..")
  commandLine(
    "node_modules/.bin/bare-link",
    "--preset",
    "android",
    "--out",
    "sample-app/src/main/addons",
    "sample-app/src/main/js/app.js"
  )
}

tasks.register("checkBareKitRuntimeBootstrap") {
  val requiredFiles = listOf(
    "libs/bare-kit/classes.jar",
    "libs/bare-kit/jni/arm64-v8a/libbare-kit.so",
    "libs/bare-kit/jni/arm64-v8a/libc++_shared.so",
    "libs/bare-kit/.bootstrap-metadata.json"
  )
  doLast {
    requiredFiles.forEach { relativePath ->
      val file = file(relativePath)
      if (!file.exists()) {
        throw GradleException(
          "Missing Bare Kit runtime artifact: ${file.path}. Run `bun run sample:bootstrap-runtime` from packages/android-sdk."
        )
      }
    }
  }
}

tasks.register<Exec>("copyMissingRuntimeAddons") {
  workingDir = file("..")
  commandLine(
    "node_modules/.bin/tsx",
    "scripts/sync-runtime-addons.ts"
  )
}

tasks.register<Exec>("pack") {
  workingDir = file("..")
  commandLine(
    "node_modules/.bin/bare-pack",
    "--preset",
    "android",
    "--out",
    "sample-app/src/main/assets/app.bundle",
    "sample-app/src/main/js/app.js"
  )
}

tasks.named("preBuild").configure {
  dependsOn("checkBareKitRuntimeBootstrap")
  dependsOn("link")
  dependsOn("copyMissingRuntimeAddons")
  dependsOn("pack")
}

// Android Studio/IntelliJ may probe this legacy task during import.
tasks.register("prepareKotlinBuildScriptModel") {
  group = "ide"
  description = "Compatibility task for IDE Kotlin build script model import"
}
