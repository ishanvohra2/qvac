package io.tether.qvac.sample

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import io.tether.qvac.sdk.QvacModelCatalog
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
  private var streamJob: Job? = null
  private lateinit var messageInput: EditText
  private lateinit var loadedModelText: TextView
  private lateinit var statusText: TextView
  private lateinit var chatTranscriptText: TextView
  private lateinit var bridge: BareQvacBridge
  private val transcriptBuffer = StringBuilder()
  private var isModelLoaded = false

  private val llmClient by lazy { QvacLlmClient(bridge) }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)

    messageInput = findViewById(R.id.messageInput)
    loadedModelText = findViewById(R.id.loadedModelText)
    statusText = findViewById(R.id.statusText)
    chatTranscriptText = findViewById(R.id.chatTranscriptText)
    val sendButton = findViewById<Button>(R.id.sendButton)

    bridge = BareQvacBridge(this)

    try {
      bridge.start()
      statusText.text = "Status: loading LLM model..."
      autoLoadModel(sendButton)
    } catch (error: Exception) {
      statusText.text = "Status: failed to start worklet (${error.message})"
      sendButton.isEnabled = false
    }

    sendButton.isEnabled = false
    renderLoadedModel()
    sendButton.setOnClickListener { handleSend(sendButton) }
  }

  override fun onDestroy() {
    streamJob?.cancel()
    bridge.stop()
    super.onDestroy()
  }

  private fun autoLoadModel(sendButton: Button) {
    val modelName = resolveDefaultLlmModel()
    lifecycleScope.launch {
      try {
        llmClient.loadModel(modelName)
        isModelLoaded = true
        renderLoadedModel()
        statusText.text = "Status: model loaded and ready"
        sendButton.isEnabled = true
      } catch (error: Exception) {
        statusText.text = "Status: model load failed (${error.message})"
        sendButton.isEnabled = false
      }
    }
  }

  private fun handleSend(sendButton: Button) {
    if (!isModelLoaded) {
      statusText.text = "Status: wait for model to load"
      return
    }
    val prompt = messageInput.text?.toString()?.trim().orEmpty()
    if (prompt.isEmpty()) return

    streamJob?.cancel()
    appendChatLine("You: $prompt")
    appendChatLine("Assistant: ")
    messageInput.text?.clear()
    sendButton.isEnabled = false

    streamJob = lifecycleScope.launch {
      try {
        statusText.text = "Status: streaming..."
        llmClient.streamCompletion(prompt).collect { token ->
          appendAssistantToken(token)
        }
        appendChatLine("")
        statusText.text = "Status: ready"
      } catch (error: Exception) {
        statusText.text = "Status: stream failed (${error.message})"
      } finally {
        sendButton.isEnabled = isModelLoaded
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

  private fun resolveDefaultLlmModel(): String {
    val model = QvacModelCatalog.findByName(this, "LLAMA_3_2_1B_INST_Q4_0")
    return model?.name ?: "LLAMA_3_2_1B_INST_Q4_0"
  }

  private fun appendChatLine(line: String) {
    if (transcriptBuffer.isNotEmpty()) {
      transcriptBuffer.append('\n')
    }
    transcriptBuffer.append(line)
    chatTranscriptText.text = transcriptBuffer.toString()
  }

  private fun appendAssistantToken(token: String) {
    runOnUiThread {
      transcriptBuffer.append(token)
      chatTranscriptText.text = transcriptBuffer.toString()
    }
  }
}
