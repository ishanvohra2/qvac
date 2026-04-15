import { detectOne } from "@qvac/langdetect-text";
import {
  loadModel,
  textToSpeech,
  unloadModel,
  type ModelProgressUpdate,
  TTS_SUPERTONIC2_OFFICIAL_TEXT_ENCODER_SUPERTONE_FP32,
  TTS_SUPERTONIC2_OFFICIAL_DURATION_PREDICTOR_SUPERTONE_FP32,
  TTS_SUPERTONIC2_OFFICIAL_VECTOR_ESTIMATOR_SUPERTONE_FP32,
  TTS_SUPERTONIC2_OFFICIAL_VOCODER_SUPERTONE_FP32,
  TTS_SUPERTONIC2_OFFICIAL_UNICODE_INDEXER_SUPERTONE_FP32,
  TTS_SUPERTONIC2_OFFICIAL_TTS_CONFIG_SUPERTONE,
  TTS_SUPERTONIC2_OFFICIAL_VOICE_STYLE_SUPERTONE,
} from "@qvac/sdk";
import {
  createWav,
  playAudio,
  int16ArrayToBuffer,
  createWavHeader,
} from "./utils";

const SUPERTONIC_SAMPLE_RATE = 44100;

// Supertonic (Supertone) UI languages — en, ko, es, pt, fr (@qvac/tts-onnx)
type SupertonicLanguageCode = "en" | "ko" | "es" | "pt" | "fr";

function isSupertonicLanguageCode(code: string): code is SupertonicLanguageCode {
  return (
    code === "en" ||
    code === "ko" ||
    code === "es" ||
    code === "pt" ||
    code === "fr"
  );
}

function supertonicLanguageFromDetection(
  isoCode: string,
): SupertonicLanguageCode {
  if (isoCode === "und" || !isSupertonicLanguageCode(isoCode)) {
    return "en";
  }
  return isoCode;
}

function supertonicMultilingualFor(
  ttsLanguage: SupertonicLanguageCode,
  usedUnsupportedOrUndFallback: boolean,
): boolean {
  if (usedUnsupportedOrUndFallback) {
    return true;
  }
  return ttsLanguage !== "en";
}

// Change this string to try different detections (e.g. Korean, Portuguese, Spanish).
const TEXT_TO_SPEAK =
  "Bonjour, ceci est une courte phrase en français avant la synthèse vocale.";

try {
  const detected = detectOne(TEXT_TO_SPEAK);
  console.log("Detected language:", detected);

  const ttsLanguage = supertonicLanguageFromDetection(detected.code);
  const usedUnsupportedOrUndFallback =
    detected.code === "und" || !isSupertonicLanguageCode(detected.code);
  const supertonicMultilingual = supertonicMultilingualFor(
    ttsLanguage,
    usedUnsupportedOrUndFallback,
  );

  if (usedUnsupportedOrUndFallback) {
    console.log(
      `No Supertonic language slot for ISO "${detected.code}" (en/ko/es/pt/fr) — using "${ttsLanguage}" with multilingual=${supertonicMultilingual}.`,
    );
  }

  const modelId = await loadModel({
    modelSrc: TTS_SUPERTONIC2_OFFICIAL_TEXT_ENCODER_SUPERTONE_FP32.src,
    modelType: "tts",
    modelConfig: {
      ttsEngine: "supertonic",
      language: ttsLanguage,
      speed: 1.05,
      numInferenceSteps: 5,
      supertonicMultilingual,
      ttsTextEncoderSrc: TTS_SUPERTONIC2_OFFICIAL_TEXT_ENCODER_SUPERTONE_FP32.src,
      ttsDurationPredictorSrc:
        TTS_SUPERTONIC2_OFFICIAL_DURATION_PREDICTOR_SUPERTONE_FP32.src,
      ttsVectorEstimatorSrc:
        TTS_SUPERTONIC2_OFFICIAL_VECTOR_ESTIMATOR_SUPERTONE_FP32.src,
      ttsVocoderSrc: TTS_SUPERTONIC2_OFFICIAL_VOCODER_SUPERTONE_FP32.src,
      ttsUnicodeIndexerSrc:
        TTS_SUPERTONIC2_OFFICIAL_UNICODE_INDEXER_SUPERTONE_FP32.src,
      ttsTtsConfigSrc: TTS_SUPERTONIC2_OFFICIAL_TTS_CONFIG_SUPERTONE.src,
      ttsVoiceStyleSrc: TTS_SUPERTONIC2_OFFICIAL_VOICE_STYLE_SUPERTONE.src,
    },
    onProgress: (progress: ModelProgressUpdate) => {
      console.log(progress);
    },
  });

  console.log(`Model loaded: ${modelId}`);

  console.log("🎵 Text-to-Speech (non-streaming)...");
  const result = textToSpeech({
    modelId,
    text: TEXT_TO_SPEAK,
    inputType: "text",
    stream: false,
  });

  const audioBuffer = await result.buffer;
  console.log(`TTS complete. Total samples: ${audioBuffer.length}`);

  createWav(
    audioBuffer,
    SUPERTONIC_SAMPLE_RATE,
    "supertonic-langdetect-output.wav",
  );

  const audioData = int16ArrayToBuffer(audioBuffer);
  const wavBuffer = Buffer.concat([
    createWavHeader(audioData.length, SUPERTONIC_SAMPLE_RATE),
    audioData,
  ]);
  playAudio(wavBuffer);

  await unloadModel({ modelId });
  console.log("Model unloaded");
  process.exit(0);
} catch (error) {
  console.error("❌ Error:", error);
  process.exit(1);
}
