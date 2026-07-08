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

class BareQvacBridge(private val context: Context) {
  private var worklet: Worklet? = null
  private var ipc: IPC? = null
  private val nextId = AtomicInteger(1)
  private val messageBuffer = StringBuilder()
  private val handlers = ConcurrentHashMap<Int, (JSONObject) -> Unit>()
  private var eventListener: ((JSONObject) -> Unit)? = null

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
    ipc = null
    worklet?.terminate()
    worklet = null
  }

  fun setEventListener(listener: ((JSONObject) -> Unit)?) {
    eventListener = listener
  }

  suspend fun loadModel(modelSrc: String): String {
    return suspendCancellableCoroutine { continuation ->
      val requestId = sendRequest("loadModel", JSONObject().put("modelSrc", modelSrc)) { message ->
        val success = message.optBoolean("success", false)
        if (!success) {
          continuation.resumeWithException(
            IllegalStateException(message.optString("error", "loadModel failed"))
          )
          return@sendRequest
        }
        continuation.resume(message.getString("modelId"))
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
            IllegalStateException(message.optString("error", "unloadModel failed"))
          )
          return@sendRequest
        }
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
        "error" -> close(IllegalStateException(message.optString("error", "stream failed")))
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
            IllegalStateException(message.optString("error", "health check failed"))
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
      if (type == "loadModelResult" || type == "unloadModelResult" || type == "done" || type == "error") {
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
}
