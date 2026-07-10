package io.tether.qvac.sample

import android.content.Context
import java.io.File
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
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
      val requestId = sendRequest("loadModel", payload) { message ->
        val success = message.optBoolean("success", false)
        if (!success) {
          continuation.resumeWithException(
            IllegalStateException(extractErrorSummary(message, "loadModel failed"))
          )
          return@sendRequest
        }
        continuation.resume(message.getString("modelId"))
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
    return suspendCancellableCoroutine { continuation ->
      val payload = JSONObject().put("text", text)
      val sampleRateHintHz = ttsSampleRateHintHz
      if (sampleRateHintHz != null) {
        payload.put("sampleRate", sampleRateHintHz)
      }
      val requestId = sendRequest("textToSpeech", payload) { message ->
        val type = message.optString("type")
        if (type == "error") {
          continuation.resumeWithException(
            IllegalStateException(extractErrorSummary(message, "text to speech failed"))
          )
          return@sendRequest
        }
        val sampleRateHz = when {
          message.has("sampleRate") -> message.optInt("sampleRate", 0)
          ttsSampleRateHintHz != null -> ttsSampleRateHintHz ?: 0
          else -> 0
        }
        if (sampleRateHz <= 0) {
          continuation.resumeWithException(
            IllegalStateException(
              "text to speech response missing sampleRate; set outputSampleRate in modelConfig"
            )
          )
          return@sendRequest
        }
        continuation.resume(
          TtsAudioResult(
            sampleCount = message.optInt("sampleCount", 0),
            sampleRate = sampleRateHz,
            pcmBase64 = message.optString("pcmBase64", "")
          )
        )
      }
      continuation.invokeOnCancellation {
        handlers.remove(requestId)
      }
    }
  }

  suspend fun transcribe(audioPath: String, prompt: String?): String {
    return suspendCancellableCoroutine { continuation ->
      val payload = JSONObject().put("audioChunk", audioPath)
      if (!prompt.isNullOrBlank()) {
        payload.put("prompt", prompt)
      }
      val requestId = sendRequest("transcribe", payload) { message ->
        val type = message.optString("type")
        if (type == "error") {
          continuation.resumeWithException(
            IllegalStateException(extractErrorSummary(message, "transcription failed"))
          )
          return@sendRequest
        }
        continuation.resume(message.optString("text", ""))
      }
      continuation.invokeOnCancellation {
        handlers.remove(requestId)
      }
    }
  }

  suspend fun translate(text: String): String {
    return suspendCancellableCoroutine { continuation ->
      val requestId = sendRequest("translate", JSONObject().put("text", text)) { message ->
        val type = message.optString("type")
        if (type == "error") {
          continuation.resumeWithException(
            IllegalStateException(extractErrorSummary(message, "translation failed"))
          )
          return@sendRequest
        }
        continuation.resume(message.optString("text", ""))
      }
      continuation.invokeOnCancellation {
        handlers.remove(requestId)
      }
    }
  }

  suspend fun unloadModel() {
    return suspendCancellableCoroutine { continuation ->
      val requestId = sendRequest("unloadModel", JSONObject()) { message ->
        val success = message.optBoolean("success", false)
        if (!success) {
          continuation.resumeWithException(
            IllegalStateException(extractErrorSummary(message, "unloadModel failed"))
          )
          return@sendRequest
        }
        ttsSampleRateHintHz = null
        continuation.resume(Unit)
      }
      continuation.invokeOnCancellation {
        handlers.remove(requestId)
      }
    }
  }

  fun streamCompletion(prompt: String): Flow<String> = callbackFlow {
    val requestId = sendRequest("completionStream", JSONObject().put("prompt", prompt)) { message ->
      when (message.optString("type")) {
        "token" -> trySend(message.optString("token", ""))
        "done" -> close()
        "error" -> close(IllegalStateException(extractErrorSummary(message, "stream failed")))
      }
    }

    awaitClose {
      handlers.remove(requestId)
      sendOneWay("cancelStream", requestId, JSONObject())
    }
  }

  suspend fun healthCheck(): JSONObject {
    return suspendCancellableCoroutine { continuation ->
      val requestId = sendRequest("health", JSONObject()) { message ->
        val success = message.optBoolean("success", false)
        if (!success) {
          continuation.resumeWithException(
            IllegalStateException(extractErrorSummary(message, "health check failed"))
          )
          return@sendRequest
        }
        continuation.resume(message)
      }
      continuation.invokeOnCancellation {
        handlers.remove(requestId)
      }
    }
  }

  private fun sendRequest(action: String, payload: JSONObject, handler: (JSONObject) -> Unit): Int {
    val requestId = nextId.getAndIncrement()
    handlers[requestId] = { message ->
      val type = message.optString("type")
      handler(message)
      if (
        type == "loadModelResult" ||
        type == "unloadModelResult" ||
        type == "done" ||
        type == "error" ||
        type == "translationResult" ||
        type == "transcriptionResult" ||
        type == "textToSpeechResult" ||
        type == "healthResult"
      ) {
        handlers.remove(requestId)
      }
    }
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
