package io.tether.qvac.sample

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

class LauncherActivity : AppCompatActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_launcher)

    findViewById<Button>(R.id.openLlmButton).setOnClickListener {
      startActivity(Intent(this, MainActivity::class.java))
    }
    findViewById<Button>(R.id.openTtsButton).setOnClickListener {
      startActivity(Intent(this, TtsActivity::class.java))
    }
    findViewById<Button>(R.id.openWhisperButton).setOnClickListener {
      startActivity(Intent(this, WhisperActivity::class.java))
    }
    findViewById<Button>(R.id.openTranslateButton).setOnClickListener {
      startActivity(Intent(this, TranslateActivity::class.java))
    }
  }
}

