package io.tether.qvac.sample

import kotlinx.coroutines.flow.Flow

class QvacLlmClient(private val bridge: BareQvacBridge) {
  private var loadedModelId: String? = null

  suspend fun loadModel(modelId: String) {
    loadedModelId = bridge.loadModel(
      modelSrc = modelId,
      modelType = "llamacpp-completion"
    )
  }

  suspend fun unloadModel() {
    bridge.unloadModel()
    loadedModelId = null
  }

  fun currentLoadedModelId(): String? {
    return loadedModelId
  }

  fun streamCompletion(prompt: String): Flow<String> {
    if (loadedModelId == null) {
      throw IllegalStateException("No loaded model")
    }
    return bridge.streamCompletion(prompt)
  }
}
