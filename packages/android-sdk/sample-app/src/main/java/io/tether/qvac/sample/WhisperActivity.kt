package io.tether.qvac.sample

import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import io.tether.qvac.sdk.QvacModelCatalog
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

class WhisperActivity : AppCompatActivity() {
  private lateinit var statusText: TextView
  private lateinit var outputText: TextView
  private lateinit var loadedModelText: TextView
  private lateinit var selectedAudioText: TextView
  private lateinit var selectAudioButton: Button
  private lateinit var bridge: BareQvacBridge
  private var loadedModelId: String? = null

  private val openAudioPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
    if (uri != null) {
      transcribeSelectedAudio(uri)
    }
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_whisper)

    statusText = findViewById(R.id.statusText)
    outputText = findViewById(R.id.outputText)
    loadedModelText = findViewById(R.id.loadedModelText)
    selectedAudioText = findViewById(R.id.selectedAudioText)
    selectAudioButton = findViewById(R.id.selectAudioButton)

    bridge = BareQvacBridge(this)
    bridge.start()
    renderLoadedModel()
    selectAudioButton.isEnabled = false

    selectAudioButton.setOnClickListener {
      openAudioPicker.launch(arrayOf("audio/wav", "audio/x-wav"))
    }
    autoLoadWhisperModel()
  }

  override fun onDestroy() {
    bridge.stop()
    super.onDestroy()
  }

  private fun autoLoadWhisperModel() {
    lifecycleScope.launch {
      try {
        val modelSrc = withContext(Dispatchers.IO) { resolveWhisperModel() }
        statusText.text = "Status: loading Whisper model..."
        loadedModelId = bridge.loadModel(
          modelSrc = modelSrc,
          modelType = "whispercpp-transcription",
          modelConfig = JSONObject()
            .put("audio_format", "f32le")
            .put("language", "en")
            .put("translate", false)
        )
        renderLoadedModel()
        selectAudioButton.isEnabled = true
        statusText.text = "Status: model loaded - select a WAV file"
      } catch (error: Exception) {
        selectAudioButton.isEnabled = false
        statusText.text = "Status: load failed (${error.message})"
      }
    }
  }

  private fun transcribeSelectedAudio(uri: Uri) {
    if (loadedModelId == null) {
      statusText.text = "Status: wait for model to load"
      return
    }
    lifecycleScope.launch {
      try {
        val copiedFile = copyUriToCache(uri)
        selectedAudioText.text = copiedFile.absolutePath
        statusText.text = "Status: transcribing..."
        val text = bridge.transcribe(copiedFile.absolutePath, "Transcribe clearly with punctuation.")
        outputText.text = text
        statusText.text = "Status: transcription complete"
      } catch (error: Exception) {
        statusText.text = "Status: transcription failed (${error.message})"
      }
    }
  }

  private fun copyUriToCache(uri: Uri): File {
    val target = File(cacheDir, "whisper-input-${System.currentTimeMillis()}.wav")
    contentResolver.openInputStream(uri).use { input ->
      if (input == null) {
        throw IllegalStateException("Could not open selected audio file")
      }
      target.outputStream().use { output ->
        input.copyTo(output)
      }
    }
    return target
  }

  private fun resolveWhisperModel(): String {
    val model = QvacModelCatalog.findByName(applicationContext, "WHISPER_TINY")
    if (model != null) {
      return model.name
    }
    return QvacModelCatalog.findByEngine(applicationContext, "whispercpp-transcription").firstOrNull()?.name
      ?: "WHISPER_TINY"
  }

  private fun renderLoadedModel() {
    val loaded = loadedModelId ?: "none"
    loadedModelText.text = "Loaded model: $loaded"
  }
}

