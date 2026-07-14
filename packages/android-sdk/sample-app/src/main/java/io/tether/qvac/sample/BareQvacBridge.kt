package io.tether.qvac.sample

import android.content.Context
import io.tether.qvac.sdk.generated.api.*
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.suspendCancellableCoroutine
import org.json.JSONObject
import to.holepunch.bare.kit.IPC
import to.holepunch.bare.kit.Worklet

data class TtsAudioResult(
  val sampleCount: Int,
  val sampleRate: Int,
  val pcmBase64: String
)

class BareQvacBridge(private val context: Context) {
  private var worklet: Worklet? = null
  private var ipc: IPC? = null
  private val nextId = AtomicInteger(1)
  private val messageBuffer = StringBuilder()
  private val handlers = ConcurrentHashMap<Int, (JSONObject) -> Unit>()
  private var eventListener: ((JSONObject) -> Unit)? = null
  private var ttsSampleRateHintHz: Int? = null
  private var activeModelId: String? = null

  fun start() {
    if (worklet != null) return
    val qvacHomeDir = File(context.filesDir, "qvac-home")
    if (!qvacHomeDir.exists()) {
      qvacHomeDir.mkdirs()
    }
    File(qvacHomeDir, ".qvac").mkdirs()

    val runtime = Worklet(null)
    runtime.start(
      "/app.bundle",
      context.assets.open("app.bundle"),
      arrayOf(
        "android-bare-kit",
        "app.bundle",
        JSONObject()
          .put("HOME_DIR", qvacHomeDir.absolutePath)
          .toString()
      )
    )
    val runtimeIpc = IPC(runtime)
    runtimeIpc.readable { drainReadable(runtimeIpc) }
    worklet = runtime
    ipc = runtimeIpc
  }

  fun stop() {
    handlers.clear()
    eventListener = null
    ttsSampleRateHintHz = null
    activeModelId = null
    ipc = null
    worklet?.terminate()
    worklet = null
  }

  fun setEventListener(listener: ((JSONObject) -> Unit)?) {
    eventListener = listener
  }

  suspend fun loadModel(
    modelSrc: String,
    modelType: String = "llamacpp-completion",
    modelConfig: JSONObject = JSONObject(),
    ttsSampleRateHintHz: Int? = null
  ): String {
    return suspendCancellableCoroutine { continuation ->
      val payload = JSONObject()
        .put("modelSrc", modelSrc)
        .put("modelType", modelType)
        .put("modelConfig", modelConfig)
      var requestId = -1
      requestId = sendRequest("loadModel", payload) { message ->
        val success = message.optBoolean("success", false)
        if (!success) {
          handlers.remove(requestId)
          continuation.resumeWithException(
            IllegalStateException(extractErrorSummary(message, "loadModel failed"))
          )
          return@sendRequest
        }
        val modelId = message.getString("modelId")
        activeModelId = modelId
        handlers.remove(requestId)
        continuation.resume(modelId)
      }
      if (modelType == "tts-ggml") {
        this.ttsSampleRateHintHz = when {
          ttsSampleRateHintHz != null && ttsSampleRateHintHz > 0 -> ttsSampleRateHintHz
          else -> {
            val configuredRateHz = modelConfig.optInt("outputSampleRate", 0)
            if (configuredRateHz > 0) configuredRateHz else null
          }
        }
      } else {
        this.ttsSampleRateHintHz = null
      }
      continuation.invokeOnCancellation {
        handlers.remove(requestId)
      }
    }
  }

  suspend fun textToSpeech(text: String): TtsAudioResult {
    val modelId = requireActiveModelId("text to speech")
    val params = JSONObject()
      .put("type", "textToSpeech")
      .put("modelId", modelId)
      .put("text", text)
      .put("inputType", "text")
      .put("stream", true)
    val chunks = generatedClient.pluginInvokeStream(
      PluginInvokeStreamRequest(
        JSONObject()
          .put("type", "pluginInvokeStream")
          .put("modelId", modelId)
          .put("handler", "textToSpeech")
          .put("params", params)
      )
    )
    val merged = mutableListOf<Int>()
    var response = JSONObject()
    chunks.collect { event ->
      val result = event.payload.optJSONObject("result") ?: return@collect
      response = result
      val buffer = result.optJSONArray("buffer") ?: return@collect
      for (index in 0 until buffer.length()) {
        merged += buffer.optInt(index)
      }
    }
    val sampleRateHz = when {
      response.has("sampleRate") -> response.optInt("sampleRate", 0)
      ttsSampleRateHintHz != null -> ttsSampleRateHintHz ?: 0
      else -> 0
    }
    if (sampleRateHz <= 0) {
      throw IllegalStateException(
        "text to speech response missing sampleRate; set outputSampleRate in modelConfig"
      )
    }
    return TtsAudioResult(
      sampleCount = if (merged.isEmpty()) response.optInt("sampleCount", 0) else merged.size,
      sampleRate = sampleRateHz,
      pcmBase64 = encodePcm16LeBase64(merged)
    )
  }

  suspend fun transcribe(audioPath: String, prompt: String?): String {
    val modelId = requireActiveModelId("transcription")
    val params = JSONObject()
      .put("type", "transcribe")
      .put("modelId", modelId)
      .put(
        "audioChunk",
        JSONObject()
          .put("type", "filePath")
          .put("value", audioPath)
      )
    if (!prompt.isNullOrBlank()) {
      params.put("prompt", prompt)
    }
    val chunks = generatedClient.pluginInvokeStream(
      PluginInvokeStreamRequest(
        JSONObject()
          .put("type", "pluginInvokeStream")
          .put("modelId", modelId)
          .put("handler", "transcribe")
          .put("params", params)
      )
    )
    var text = ""
    chunks.collect { event ->
      val result = event.payload.optJSONObject("result") ?: return@collect
      val partial = result.optString("text", "")
      if (partial.isNotEmpty()) {
        text = partial
      }
    }
    return text
  }

  suspend fun translate(text: String): String {
    val modelId = requireActiveModelId("translation")
    val params = JSONObject()
      .put("type", "translate")
      .put("modelId", modelId)
      .put("text", text)
      .put("modelType", "nmtcpp-translation")
      .put("stream", true)
    val chunks = generatedClient.pluginInvokeStream(
      PluginInvokeStreamRequest(
        JSONObject()
          .put("type", "pluginInvokeStream")
          .put("modelId", modelId)
          .put("handler", "translate")
          .put("params", params)
      )
    )
    val textBuilder = StringBuilder()
    chunks.collect { event ->
      val result = event.payload.optJSONObject("result") ?: return@collect
      val token = result.optString("token", "")
      if (token.isNotEmpty()) {
        textBuilder.append(token)
      }
    }
    return textBuilder.toString()
  }

  suspend fun unloadModel() {
    val modelId = activeModelId ?: return
    val response = generatedClient.unloadModel(
      UnloadModelRequest(
        JSONObject()
          .put("type", "unloadModel")
          .put("modelId", modelId)
          .put("clearStorage", false)
      )
    ).payload
    val success = response.optBoolean("success", false)
    if (!success) {
      throw IllegalStateException(extractErrorSummary(response, "unloadModel failed"))
    }
    ttsSampleRateHintHz = null
    activeModelId = null
  }

  fun streamCompletion(prompt: String): Flow<String> = callbackFlow {
    val modelId = activeModelId
    if (modelId == null) {
      close(IllegalStateException("No model loaded"))
      return@callbackFlow
    }
    val requestPayload = JSONObject()
      .put("type", "pluginInvokeStream")
      .put("modelId", modelId)
      .put("handler", "completionStream")
      .put(
        "params",
        JSONObject()
          .put("type", "completionStream")
          .put("modelId", modelId)
          .put(
            "history",
            org.json.JSONArray().put(
              JSONObject()
                .put("role", "user")
                .put("content", prompt)
            )
          )
          .put("stream", true)
      )
    val requestId = sendRequest("pluginInvokeStream", requestPayload) { message ->
      val type = message.optString("type")
      if (type == "error") {
        handlers.remove(message.optInt("id", -1))
        close(IllegalStateException(extractErrorSummary(message, "stream failed")))
        return@sendRequest
      }
      if (type != "pluginInvokeStream") return@sendRequest
      if (message.optBoolean("done", false)) {
        close()
        return@sendRequest
      }
      val result = message.optJSONObject("result") ?: return@sendRequest
      val events = result.optJSONArray("events") ?: return@sendRequest
      for (index in 0 until events.length()) {
        val event = events.optJSONObject(index) ?: continue
        if (event.optString("type") == "contentDelta") {
          trySend(event.optString("text", ""))
        }
      }
    }

    awaitClose {
      handlers.remove(requestId)
    }
  }

  suspend fun healthCheck(): JSONObject {
    return generatedClient.heartbeat(HeartbeatRequest(JSONObject().put("type", "heartbeat"))).payload
  }

  private val generatedClient: QvacGeneratedApiClient = object : QvacGeneratedApiClient {
    // <generated-contract-client:start>
    override fun batchCompletionStream(request: BatchCompletionStreamRequest): Flow<BatchCompletionStreamStreamEvent> =
      invokeContractStream("batchCompletionStream", request.payload).map { payload ->
        BatchCompletionStreamStreamEvent(payload)
      }

    override suspend fun bciTranscribe(request: BciTranscribeRequest): BciTranscribeResponse =
      BciTranscribeResponse(invokeContract("bciTranscribe", request.payload))

    override fun bciTranscribeStream(request: BciTranscribeStreamRequest): Flow<BciTranscribeStreamStreamEvent> =
      invokeContractStream("bciTranscribeStream", request.payload).map { payload ->
        BciTranscribeStreamStreamEvent(payload)
      }

    override suspend fun cancel(request: CancelRequest): CancelResponse =
      CancelResponse(invokeContract("cancel", request.payload))

    override suspend fun classify(request: ClassifyRequest): ClassifyResponse =
      ClassifyResponse(invokeContract("classify", request.payload))

    override fun completionStream(request: CompletionStreamRequest): Flow<CompletionStreamStreamEvent> =
      invokeContractStream("completionStream", request.payload).map { payload ->
        CompletionStreamStreamEvent(payload)
      }

    override suspend fun deleteCache(request: DeleteCacheRequest): DeleteCacheResponse =
      DeleteCacheResponse(invokeContract("deleteCache", request.payload))

    override fun diffusionStream(request: DiffusionStreamRequest): Flow<DiffusionStreamStreamEvent> =
      invokeContractStream("diffusionStream", request.payload).map { payload ->
        DiffusionStreamStreamEvent(payload)
      }

    override suspend fun downloadAsset(request: DownloadAssetRequest): DownloadAssetResponse =
      DownloadAssetResponse(invokeContract("downloadAsset", request.payload))

    override suspend fun embed(request: EmbedRequest): EmbedResponse =
      EmbedResponse(invokeContract("embed", request.payload))

    override suspend fun finetune(request: FinetuneRequest): FinetuneResponse =
      FinetuneResponse(invokeContract("finetune", request.payload))

    override suspend fun getLoadedModelInfo(request: GetLoadedModelInfoRequest): GetLoadedModelInfoResponse =
      GetLoadedModelInfoResponse(invokeContract("getLoadedModelInfo", request.payload))

    override suspend fun getModelInfo(request: GetModelInfoRequest): GetModelInfoResponse =
      GetModelInfoResponse(invokeContract("getModelInfo", request.payload))

    override suspend fun heartbeat(request: HeartbeatRequest): HeartbeatResponse =
      HeartbeatResponse(invokeContract("heartbeat", request.payload))

    override suspend fun loadModel(request: LoadModelRequest): LoadModelResponse =
      LoadModelResponse(invokeContract("loadModel", request.payload))

    override fun loggingStream(request: LoggingStreamRequest): Flow<LoggingStreamStreamEvent> =
      invokeContractStream("loggingStream", request.payload).map { payload ->
        LoggingStreamStreamEvent(payload)
      }

    override suspend fun modelRegistryGetModel(request: ModelRegistryGetModelRequest): ModelRegistryGetModelResponse =
      ModelRegistryGetModelResponse(invokeContract("modelRegistryGetModel", request.payload))

    override suspend fun modelRegistryList(request: ModelRegistryListRequest): ModelRegistryListResponse =
      ModelRegistryListResponse(invokeContract("modelRegistryList", request.payload))

    override suspend fun modelRegistrySearch(request: ModelRegistrySearchRequest): ModelRegistrySearchResponse =
      ModelRegistrySearchResponse(invokeContract("modelRegistrySearch", request.payload))

    override fun ocrStream(request: OcrStreamRequest): Flow<OcrStreamStreamEvent> =
      invokeContractStream("ocrStream", request.payload).map { payload ->
        OcrStreamStreamEvent(payload)
      }

    override suspend fun pluginInvoke(request: PluginInvokeRequest): PluginInvokeResponse =
      PluginInvokeResponse(invokeContract("pluginInvoke", request.payload))

    override fun pluginInvokeStream(request: PluginInvokeStreamRequest): Flow<PluginInvokeStreamStreamEvent> =
      invokeContractStream("pluginInvokeStream", request.payload).map { payload ->
        PluginInvokeStreamStreamEvent(payload)
      }

    override suspend fun provide(request: ProvideRequest): ProvideResponse =
      ProvideResponse(invokeContract("provide", request.payload))

    override suspend fun rag(request: RagRequest): RagResponse =
      RagResponse(invokeContract("rag", request.payload))

    override suspend fun resume(request: ResumeRequest): ResumeResponse =
      ResumeResponse(invokeContract("resume", request.payload))

    override suspend fun state(request: StateRequest): StateResponse =
      StateResponse(invokeContract("state", request.payload))

    override suspend fun stopProvide(request: StopProvideRequest): StopProvideResponse =
      StopProvideResponse(invokeContract("stopProvide", request.payload))

    override suspend fun suspendOperation(request: SuspendRequest): SuspendResponse =
      SuspendResponse(invokeContract("suspend", request.payload))

    override suspend fun textToSpeech(request: TextToSpeechRequest): TextToSpeechResponse =
      TextToSpeechResponse(invokeContract("textToSpeech", request.payload))

    override fun textToSpeechStream(request: TextToSpeechStreamRequest): Flow<TextToSpeechStreamStreamEvent> =
      invokeContractStream("textToSpeechStream", request.payload).map { payload ->
        TextToSpeechStreamStreamEvent(payload)
      }

    override suspend fun transcribe(request: TranscribeRequest): TranscribeResponse =
      TranscribeResponse(invokeContract("transcribe", request.payload))

    override fun transcribeStream(request: TranscribeStreamRequest): Flow<TranscribeStreamStreamEvent> =
      invokeContractStream("transcribeStream", request.payload).map { payload ->
        TranscribeStreamStreamEvent(payload)
      }

    override suspend fun translate(request: TranslateRequest): TranslateResponse =
      TranslateResponse(invokeContract("translate", request.payload))

    override suspend fun unloadModel(request: UnloadModelRequest): UnloadModelResponse =
      UnloadModelResponse(invokeContract("unloadModel", request.payload))

    override fun upscaleStream(request: UpscaleStreamRequest): Flow<UpscaleStreamStreamEvent> =
      invokeContractStream("upscaleStream", request.payload).map { payload ->
        UpscaleStreamStreamEvent(payload)
      }

    override fun videoStream(request: VideoStreamRequest): Flow<VideoStreamStreamEvent> =
      invokeContractStream("videoStream", request.payload).map { payload ->
        VideoStreamStreamEvent(payload)
      }

    override suspend fun vlaHparams(request: VlaHparamsRequest): VlaHparamsResponse =
      VlaHparamsResponse(invokeContract("vlaHparams", request.payload))

    override suspend fun vlaRun(request: VlaRunRequest): VlaRunResponse =
      VlaRunResponse(invokeContract("vlaRun", request.payload))
    // <generated-contract-client:end>
  }

  private suspend fun invokeContract(operation: String, payload: JSONObject): JSONObject {
    return suspendCancellableCoroutine { continuation ->
      var requestId = -1
      requestId = sendRequest(operation, payload) { message ->
        val type = message.optString("type")
        if (type == "error") {
          handlers.remove(requestId)
          continuation.resumeWithException(
            IllegalStateException(extractErrorSummary(message, "$operation failed"))
          )
          return@sendRequest
        }
        if (type == operation) {
          handlers.remove(requestId)
          continuation.resume(extractPayloadObject(message))
        }
      }
      continuation.invokeOnCancellation {
        handlers.remove(requestId)
      }
    }
  }

  private fun invokeContractStream(operation: String, payload: JSONObject): Flow<JSONObject> {
    return callbackFlow {
      val requestId = sendRequest(operation, payload) { message ->
        val type = message.optString("type")
        if (type == "error") {
          handlers.remove(message.optInt("id", -1))
          close(IllegalStateException(extractErrorSummary(message, "$operation failed")))
          return@sendRequest
        }
        if (type != operation) return@sendRequest
        if (message.optBoolean("done", false)) {
          close()
          return@sendRequest
        }
        trySend(extractPayloadObject(message))
      }
      awaitClose {
        handlers.remove(requestId)
      }
    }
  }

  private fun requireActiveModelId(action: String): String {
    return activeModelId ?: throw IllegalStateException("No model loaded for $action")
  }

  private fun encodePcm16LeBase64(samples: List<Int>): String {
    val bytes = ByteBuffer
      .allocate(samples.size * 2)
      .order(ByteOrder.LITTLE_ENDIAN)
      .apply {
        for (sample in samples) {
          putShort(sample.toShort())
        }
      }
      .array()
    return android.util.Base64.encodeToString(bytes, android.util.Base64.NO_WRAP)
  }

  private fun extractPayloadObject(message: JSONObject): JSONObject {
    return if (message.has("payload") && !message.isNull("payload")) {
      message.optJSONObject("payload") ?: JSONObject()
    } else {
      message
    }
  }

  private fun sendRequest(action: String, payload: JSONObject, handler: (JSONObject) -> Unit): Int {
    val requestId = nextId.getAndIncrement()
    handlers[requestId] = handler
    sendOneWay(action, requestId, payload)
    return requestId
  }

  private fun sendOneWay(action: String, requestId: Int, payload: JSONObject) {
    val channel = ipc ?: throw IllegalStateException("IPC is not initialized")
    val message = JSONObject(payload.toString())
      .put("id", requestId)
      .put("action", action)
      .toString()
    val bytes = "$message\n".toByteArray(StandardCharsets.UTF_8)
    channel.write(ByteBuffer.wrap(bytes))
  }

  private fun drainReadable(channel: IPC) {
    while (true) {
      val incoming = channel.read() ?: break
      val bytes = ByteArray(incoming.remaining())
      incoming.get(bytes)
      messageBuffer.append(String(bytes, StandardCharsets.UTF_8))
      consumeBufferedMessages()
    }
  }

  private fun consumeBufferedMessages() {
    while (true) {
      val newlineIndex = messageBuffer.indexOf("\n")
      if (newlineIndex < 0) return
      val raw = messageBuffer.substring(0, newlineIndex).trim()
      messageBuffer.delete(0, newlineIndex + 1)
      if (raw.isEmpty()) continue

      val message = JSONObject(raw)
      if (message.optString("type") == "log") {
        eventListener?.invoke(message)
      }
      if (message.has("id") && !message.isNull("id")) {
        val requestId = message.getInt("id")
        handlers[requestId]?.invoke(message)
      }
    }
  }

  private fun extractErrorSummary(message: JSONObject, fallback: String): String {
    val errorMessage = message.optString("error", fallback)
    val errorCode = message.optString("errorCode", "")
    val errorName = message.optString("errorName", "")
    val parts = mutableListOf<String>()
    if (errorCode.isNotBlank()) {
      parts += "code=$errorCode"
    }
    if (errorName.isNotBlank()) {
      parts += "name=$errorName"
    }
    parts += "message=$errorMessage"
    return parts.joinToString(" ")
  }
}
