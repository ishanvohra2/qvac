package io.tether.qvac.sample

import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import io.tether.qvac.sdk.QvacModelCatalog
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

class TranslateActivity : AppCompatActivity() {
  private lateinit var textInput: EditText
  private lateinit var outputLanguageSpinner: Spinner
  private lateinit var statusText: TextView
  private lateinit var outputText: TextView
  private lateinit var loadedModelText: TextView
  private lateinit var translateButton: Button
  private lateinit var bridge: BareQvacBridge
  private var loadedModelId: String? = null
  private val targetLanguages = listOf("fr", "es", "de", "it", "pt", "nl")

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_translate)

    textInput = findViewById(R.id.textInput)
    outputLanguageSpinner = findViewById(R.id.outputLanguageSpinner)
    statusText = findViewById(R.id.statusText)
    outputText = findViewById(R.id.outputText)
    loadedModelText = findViewById(R.id.loadedModelText)
    val loadButton = findViewById<Button>(R.id.loadModelButton)
    translateButton = findViewById(R.id.translateButton)

    outputLanguageSpinner.adapter = ArrayAdapter(
      this,
      android.R.layout.simple_spinner_dropdown_item,
      targetLanguages
    )
    outputLanguageSpinner.setSelection(targetLanguages.indexOf("fr"))
    textInput.setText("This sample validates the Android translation flow.")

    bridge = BareQvacBridge(this)
    bridge.start()
    renderLoadedModel()
    translateButton.isEnabled = false

    loadButton.setOnClickListener { handleLoadModel() }
    translateButton.setOnClickListener { handleTranslation() }
  }

  override fun onDestroy() {
    bridge.stop()
    super.onDestroy()
  }

  private fun handleLoadModel() {
    val outputLanguage = outputLanguageSpinner.selectedItem?.toString() ?: "fr"
    lifecycleScope.launch {
      try {
        val modelSrc = withContext(Dispatchers.IO) { resolveTranslationModel(outputLanguage) }
        if (modelSrc == null) {
          statusText.text = "Status: no English->$outputLanguage Bergamot model in catalog"
          return@launch
        }
        translateButton.isEnabled = false
        statusText.text = "Status: loading translation model..."
        loadedModelId = bridge.loadModel(
          modelSrc = modelSrc,
          modelType = "nmtcpp-translation",
          modelConfig = JSONObject()
            .put("engine", "Bergamot")
            .put("from", "en")
            .put("to", outputLanguage)
        )
        renderLoadedModel()
        translateButton.isEnabled = true
        statusText.text = "Status: model loaded"
      } catch (error: Exception) {
        translateButton.isEnabled = false
        statusText.text = "Status: load failed (${error.message})"
      }
    }
  }

  private fun handleTranslation() {
    if (loadedModelId == null) {
      statusText.text = "Status: load a model before translation"
      return
    }
    val sourceText = textInput.text?.toString()?.trim().orEmpty()
    if (sourceText.isEmpty()) {
      statusText.text = "Status: source text cannot be empty"
      return
    }
    lifecycleScope.launch {
      try {
        translateButton.isEnabled = false
        statusText.text = "Status: translating..."
        val translated = bridge.translate(sourceText)
        outputText.text = translated
        statusText.text = "Status: translation complete"
      } catch (error: Exception) {
        statusText.text = "Status: translation failed (${error.message})"
      } finally {
        translateButton.isEnabled = loadedModelId != null
      }
    }
  }

  private fun renderLoadedModel() {
    val loaded = loadedModelId ?: "none"
    loadedModelText.text = "Loaded model: $loaded"
  }

  private fun resolveTranslationModel(outputLanguage: String): String? {
    val constant = "BERGAMOT_EN_${outputLanguage.uppercase()}"
    val model = QvacModelCatalog.findByName(applicationContext, constant)
    if (model != null) {
      return model.name
    }
    return null
  }
}

