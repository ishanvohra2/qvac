package io.tether.qvac.sdk

import android.content.Context
import org.json.JSONArray

data class QvacModelConstant(
  val name: String,
  val src: String,
  val modelId: String,
  val registryPath: String,
  val registrySource: String,
  val addon: String,
  val engine: String,
  val quantization: String,
  val params: String
)

object QvacModelCatalog {
  fun load(context: Context): List<QvacModelConstant> {
    val raw = context.assets.open("models-catalog.json").bufferedReader().use { it.readText() }
    val array = JSONArray(raw)
    val constants = mutableListOf<QvacModelConstant>()
    for (index in 0 until array.length()) {
      val entry = array.getJSONObject(index)
      constants.add(
        QvacModelConstant(
          name = entry.getString("name"),
          src = entry.getString("src"),
          modelId = entry.getString("modelId"),
          registryPath = entry.getString("registryPath"),
          registrySource = entry.getString("registrySource"),
          addon = entry.getString("addon"),
          engine = entry.getString("engine"),
          quantization = entry.optString("quantization", ""),
          params = entry.optString("params", "")
        )
      )
    }
    return constants
  }

  fun findByName(context: Context, constantName: String): QvacModelConstant? {
    return load(context).firstOrNull { it.name == constantName }
  }

  fun findByEngine(context: Context, engine: String): List<QvacModelConstant> {
    return load(context).filter { it.engine == engine }
  }
}
