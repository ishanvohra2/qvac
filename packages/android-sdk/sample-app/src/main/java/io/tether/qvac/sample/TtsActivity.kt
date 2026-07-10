package io.tether.qvac.sample

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Bundle
import android.util.Base64
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import io.tether.qvac.sdk.QvacModelCatalog
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

class TtsActivity : AppCompatActivity() {
  private val defaultSupertonicOutputSampleRateHz = 24000
  private lateinit var textInput: EditText
  private lateinit var languageSpinner: Spinner
  private lateinit var statusText: TextView
  private lateinit var outputText: TextView
  private lateinit var loadedModelText: TextView
  private lateinit var speakButton: Button
  private lateinit var bridge: BareQvacBridge
  private var loadedModelId: String? = null
  private var audioTrack: AudioTrack? = null
  private val supertonicLanguages = listOf(
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es",
    "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv",
    "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi"
  )

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_tts)

    textInput = findViewById(R.id.textInput)
    languageSpinner = findViewById(R.id.languageSpinner)
    statusText = findViewById(R.id.statusText)
    outputText = findViewById(R.id.outputText)
    loadedModelText = findViewById(R.id.loadedModelText)
    val loadButton = findViewById<Button>(R.id.loadModelButton)
    speakButton = findViewById(R.id.speakButton)

    languageSpinner.adapter = ArrayAdapter(
      this,
      android.R.layout.simple_spinner_dropdown_item,
      supertonicLanguages
    )
    languageSpinner.setSelection(supertonicLanguages.indexOf("en"))

    bridge = BareQvacBridge(this)
    bridge.start()
    renderLoadedModel()
    speakButton.isEnabled = false

    loadButton.setOnClickListener { handleLoadModel() }
    speakButton.setOnClickListener { handleTextToSpeech() }
  }

  override fun onDestroy() {
    audioTrack?.stop()
    audioTrack?.release()
    audioTrack = null
    bridge.stop()
    super.onDestroy()
  }

  private fun handleLoadModel() {
    val language = languageSpinner.selectedItem?.toString() ?: "en"
    lifecycleScope.launch {
      try {
        val modelSrc = withContext(Dispatchers.IO) { resolveSupertonicModel() }
        speakButton.isEnabled = false
        statusText.text = "Status: loading Supertonic model..."
        loadedModelId = bridge.loadModel(
          modelSrc = modelSrc,
          modelType = "tts-ggml",
          modelConfig = JSONObject()
            .put("ttsEngine", "supertonic")
            .put("language", language)
            .put("voice", "F1")
            .put("ttsSpeed", 1.05)
            .put("ttsNumInferenceSteps", 5),
          ttsSampleRateHintHz = defaultSupertonicOutputSampleRateHz
        )
        renderLoadedModel()
        speakButton.isEnabled = true
        statusText.text = "Status: model loaded"
      } catch (error: Exception) {
        speakButton.isEnabled = false
        statusText.text = "Status: load failed (${error.message})"
      }
    }
  }

  private fun handleTextToSpeech() {
    if (loadedModelId == null) {
      statusText.text = "Status: load a model before synthesis"
      return
    }
    val text = textInput.text?.toString()?.trim().orEmpty()
    if (text.isEmpty()) {
      statusText.text = "Status: text cannot be empty"
      return
    }
    lifecycleScope.launch {
      try {
        speakButton.isEnabled = false
        statusText.text = "Status: synthesizing..."
        val result = bridge.textToSpeech(text)
        playPcmAudio(result.pcmBase64, result.sampleRate)
        outputText.text = "Audio samples generated: ${result.sampleCount} at ${result.sampleRate} Hz"
        statusText.text = "Status: playback complete"
      } catch (error: Exception) {
        statusText.text = "Status: synthesis failed (${error.message})"
      } finally {
        speakButton.isEnabled = loadedModelId != null
      }
    }
  }

  private fun playPcmAudio(pcmBase64: String, sampleRate: Int) {
    if (pcmBase64.isBlank()) return
    val pcmBytes = Base64.decode(pcmBase64, Base64.DEFAULT)
    val shorts = ShortArray(pcmBytes.size / 2)
    val shortBuffer = ByteBuffer.wrap(pcmBytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
    shortBuffer.get(shorts)

    audioTrack?.stop()
    audioTrack?.release()

    val minBufferSize = AudioTrack.getMinBufferSize(
      sampleRate,
      AudioFormat.CHANNEL_OUT_MONO,
      AudioFormat.ENCODING_PCM_16BIT
    )
    val track = AudioTrack.Builder()
      .setAudioAttributes(
        AudioAttributes.Builder()
          .setUsage(AudioAttributes.USAGE_MEDIA)
          .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
          .build()
      )
      .setAudioFormat(
        AudioFormat.Builder()
          .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
          .setSampleRate(sampleRate)
          .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
          .build()
      )
      .setTransferMode(AudioTrack.MODE_STATIC)
      .setBufferSizeInBytes(maxOf(minBufferSize, pcmBytes.size))
      .build()
    track.write(shorts, 0, shorts.size)
    track.play()
    audioTrack = track
  }

  private fun resolveSupertonicModel(): String {
    val model = QvacModelCatalog.findByName(applicationContext, "TTS_MULTILINGUAL_SUPERTONIC3_Q8_0")
    if (model != null) {
      return model.name
    }
    return "TTS_MULTILINGUAL_SUPERTONIC3_Q8_0"
  }

  private fun renderLoadedModel() {
    val loaded = loadedModelId ?: "none"
    loadedModelText.text = "Loaded model: $loaded"
  }
}

