package io.tether.qvac.sample

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import io.tether.qvac.sdk.QvacAndroidSdk
import io.tether.qvac.sdk.QvacModelCatalog
import io.tether.qvac.sdk.generated.api.QvacGeneratedApiContract
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import org.json.JSONObject

class MainActivity : AppCompatActivity() {
  private val integrationPrompts = listOf(
    "Reply with exactly one word: READY",
    "Name one prime number less than ten.",
    "Complete this sentence in one short line: Android SDK parity means"
  )
  private var streamJob: Job? = null
  private lateinit var modelInput: EditText
  private lateinit var promptInput: EditText
  private lateinit var loadedModelText: TextView
  private lateinit var statusText: TextView
  private lateinit var bareLogsText: TextView
  private lateinit var responseText: TextView
  private lateinit var bridge: BareQvacBridge
  private val bareLogBuffer = StringBuilder()

  private val llmClient by lazy { QvacLlmClient(bridge) }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)

    val sdkInfoView = findViewById<TextView>(R.id.sdkInfo)
    val apiContractInfoView = findViewById<TextView>(R.id.apiContractInfo)
    val runtimeHealthInfoView = findViewById<TextView>(R.id.runtimeHealthInfo)
    modelInput = findViewById(R.id.modelInput)
    promptInput = findViewById(R.id.promptInput)
    loadedModelText = findViewById(R.id.loadedModelText)
    statusText = findViewById(R.id.statusText)
    bareLogsText = findViewById(R.id.bareLogsText)
    responseText = findViewById(R.id.responseText)
    val loadModelButton = findViewById<Button>(R.id.loadModelButton)
    val unloadModelButton = findViewById<Button>(R.id.unloadModelButton)
    val streamButton = findViewById<Button>(R.id.streamButton)
    val cancelStreamButton = findViewById<Button>(R.id.cancelStreamButton)
    val healthCheckButton = findViewById<Button>(R.id.healthCheckButton)
    val runPromptSetButton = findViewById<Button>(R.id.runPromptSetButton)
    bridge = BareQvacBridge(this)
    bridge.setEventListener { event ->
      if (event.optString("type") == "log") {
        appendBareLog(renderBareLog(event))
      }
    }
    try {
      bridge.start()
      statusText.text = "Status: worklet started"
      lifecycleScope.launch {
        runtimeHealthInfoView.text = renderHealth(bridge.healthCheck())
      }
    } catch (error: Exception) {
      statusText.text = "Status: failed to start worklet (${error.message})"
      runtimeHealthInfoView.text = "Runtime health: failed (${error.message})"
    }

    val details = buildString {
      appendLine("QVAC Android SDK integration check")
      appendLine()
      appendLine("version: ${QvacAndroidSdk.version()}")
      appendLine("coordinates: ${QvacAndroidSdk.coordinates()}")
      appendLine("runtime mode: bare-kit worklet + @qvac/bare-sdk")
      appendLine("assets: ${QvacAndroidSdk.bundledAssetNames().joinToString(", ")}")
      val llmDefault = QvacModelCatalog.findByName(this@MainActivity, "LLAMA_3_2_1B_INST_Q4_0")
      if (llmDefault != null) {
        appendLine("default constant: ${llmDefault.name} -> ${llmDefault.modelId}")
      }
    }
    sdkInfoView.text = details
    apiContractInfoView.text = buildString {
      append("Generated API operations (")
      append(QvacGeneratedApiContract.operations.size)
      append("): ")
      append(QvacGeneratedApiContract.operations.joinToString(", "))
    }
    renderLoadedModel()

    loadModelButton.setOnClickListener {
      handleLoadModel()
    }
    unloadModelButton.setOnClickListener {
      handleUnloadModel()
    }
    streamButton.setOnClickListener {
      handleStreamResponse()
    }
    cancelStreamButton.setOnClickListener {
      handleCancelStream()
    }
    healthCheckButton.setOnClickListener {
      handleHealthCheck(runtimeHealthInfoView)
    }
    runPromptSetButton.setOnClickListener {
      handlePromptSet()
    }
  }

  override fun onDestroy() {
    streamJob?.cancel()
    bridge.stop()
    super.onDestroy()
  }

  private fun handleLoadModel() {
    val modelId = modelInput.text?.toString()?.trim().orEmpty()
    if (modelId.isEmpty()) {
      statusText.text = "Status: enter a model id first"
      return
    }

    lifecycleScope.launch {
      try {
        statusText.text = "Status: loading model..."
        llmClient.loadModel(modelId)
        renderLoadedModel()
        statusText.text = "Status: model '$modelId' loaded"
      } catch (error: Exception) {
        Log.i("MainActivity", "handleLoadModel: ${error.message}")
        statusText.text = "Status: load failed (${error.message})"
      }
    }
  }

  private fun handleUnloadModel() {
    streamJob?.cancel()
    lifecycleScope.launch {
      try {
        statusText.text = "Status: unloading model..."
        llmClient.unloadModel()
        renderLoadedModel()
        responseText.text = ""
        statusText.text = "Status: model unloaded"
      } catch (error: Exception) {
        statusText.text = "Status: unload failed (${error.message})"
      }
    }
  }

  private fun handleStreamResponse() {
    if (llmClient.currentLoadedModelId() == null) {
      statusText.text = "Status: load a model before streaming"
      return
    }
    val prompt = promptInput.text?.toString()?.trim().orEmpty()
    if (prompt.isEmpty()) {
      statusText.text = "Status: prompt cannot be empty"
      return
    }

    streamJob?.cancel()
    responseText.text = ""
    streamJob = lifecycleScope.launch {
      try {
        statusText.text = "Status: streaming..."
        llmClient.streamCompletion(prompt).collect { token ->
          responseText.append(token)
        }
        statusText.text = "Status: stream complete"
      } catch (error: Exception) {
        statusText.text = "Status: stream failed (${error.message})"
      }
    }
  }

  private fun handleCancelStream() {
    streamJob?.cancel()
    streamJob = null
    statusText.text = "Status: stream cancelled"
  }

  private fun handleHealthCheck(runtimeHealthInfoView: TextView) {
    lifecycleScope.launch {
      try {
        runtimeHealthInfoView.text = renderHealth(bridge.healthCheck())
      } catch (error: Exception) {
        runtimeHealthInfoView.text = "Runtime health: failed (${error.message})"
      }
    }
  }

  private fun handlePromptSet() {
    if (llmClient.currentLoadedModelId() == null) {
      statusText.text = "Status: load a model before running prompt set"
      return
    }

    streamJob?.cancel()
    responseText.text = ""
    streamJob = lifecycleScope.launch {
      val report = StringBuilder()
      try {
        statusText.text = "Status: running integration prompt set..."
        integrationPrompts.forEachIndexed { index, prompt ->
          val response = StringBuilder()
          llmClient.streamCompletion(prompt).collect { token ->
            response.append(token)
            responseText.text = buildString {
              append(report.toString())
              append("[")
              append(index + 1)
              append("/")
              append(integrationPrompts.size)
              append("] ")
              append(prompt)
              append("\n")
              append(response.toString())
            }
          }

          report.append("[")
          report.append(index + 1)
          report.append("/")
          report.append(integrationPrompts.size)
          report.append("] ")
          report.append(prompt)
          report.append("\n")
          report.append(response.toString().trim())
          report.append("\n\n")
          responseText.text = report.toString()
        }
        statusText.text = "Status: prompt set complete"
      } catch (error: Exception) {
        statusText.text = "Status: prompt set failed (${error.message})"
      }
    }
  }

  private fun renderLoadedModel() {
    val loaded = llmClient.currentLoadedModelId()
    loadedModelText.text = if (loaded == null) {
      "Loaded model: none"
    } else {
      "Loaded model: $loaded"
    }
  }

  private fun renderHealth(health: JSONObject): String {
    val runtime = health.optString("runtime", "unknown")
    val plugin = health.optString("plugin", "unknown")
    val loaded = health.optString("loadedModelId", "none").ifBlank { "none" }
    return "Runtime health: ok (runtime=$runtime, plugin=$plugin, loadedModel=$loaded)"
  }

  private fun renderBareLog(event: JSONObject): String {
    val message = event.optString("message", "log")
    val requestId = if (event.has("requestId") && !event.isNull("requestId")) {
      event.optInt("requestId").toString()
    } else {
      "-"
    }
    val action = event.optString("action", "").ifBlank { "-" }
    val error = event.optString("error", "")
    return if (error.isNotBlank()) {
      "request=$requestId action=$action $message error=$error"
    } else {
      "request=$requestId action=$action $message"
    }
  }

  private fun appendBareLog(line: String) {
    runOnUiThread {
      if (bareLogBuffer.isNotEmpty()) {
        bareLogBuffer.append('\n')
      }
      bareLogBuffer.append(line)
      val maxChars = 4000
      if (bareLogBuffer.length > maxChars) {
        bareLogBuffer.delete(0, bareLogBuffer.length - maxChars)
      }
      bareLogsText.text = bareLogBuffer.toString()
    }
  }
}
