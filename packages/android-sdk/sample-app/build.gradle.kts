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

tasks.register("copyMissingRuntimeAddons") {
  val addonOutputDir = file("src/main/addons/arm64-v8a")
  val addonPackages = listOf(
    file("../node_modules/rabin-native"),
    file("../node_modules/@qvac/bare-sdk/node_modules/bare-signals")
  )

  doLast {
    addonOutputDir.mkdirs()
    addonPackages.forEach { packageDir ->
      val packageJson = packageDir.resolve("package.json")
      if (!packageJson.exists()) {
        throw GradleException("Missing runtime addon package.json at ${packageJson.path}")
      }

      val packageJsonText = packageJson.readText()
      val packageName = "\"name\"\\s*:\\s*\"([^\"]+)\"".toRegex()
        .find(packageJsonText)
        ?.groupValues
        ?.get(1)
        ?: throw GradleException("Unable to read package name from ${packageJson.path}")
      val packageVersion = "\"version\"\\s*:\\s*\"([^\"]+)\"".toRegex()
        .find(packageJsonText)
        ?.groupValues
        ?.get(1)
        ?: throw GradleException("Unable to read package version from ${packageJson.path}")
      val bareBinary = packageDir
        .resolve("prebuilds/android-arm64")
        .listFiles()
        ?.firstOrNull { it.isFile && it.extension == "bare" }
        ?: throw GradleException("Missing android-arm64 .bare prebuild in ${packageDir.path}")

      val soBaseName = if (packageName.startsWith("@")) {
        packageName.removePrefix("@").replace("/", "__")
      } else {
        packageName
      }
      bareBinary.copyTo(
        addonOutputDir.resolve("lib${soBaseName}.${packageVersion}.so"),
        overwrite = true
      )
    }
  }
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
  dependsOn("link")
  dependsOn("copyMissingRuntimeAddons")
  dependsOn("pack")
}

// Android Studio/IntelliJ may probe this legacy task during import.
tasks.register("prepareKotlinBuildScriptModel") {
  group = "ide"
  description = "Compatibility task for IDE Kotlin build script model import"
}
