https://github.com/user-attachments/assets/8a020777-36bf-4a11-a041-09d5de372849

# 🎙️ Voice-to-Voice Translator

Speak in one language, hear it translated in another. A three-stage speech
translation pipeline: **speech recognition → machine translation → speech
synthesis**, built with a locally-run Whisper model (no per-request API
cost) and an automatic text-to-speech failover so the app keeps working even
if the primary TTS engine is rate-limited.

## 🚀 Live Demo

**[Try it here](https://huggingface.co/spaces/mubashrawaqar123/voice-to-voice-translator)**


## 🧠 How it works

1. **Speech-to-text** — your recorded or uploaded audio is transcribed
   locally using [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
   (the `base` model, run on CPU with int8 quantization to stay lightweight
   enough for free-tier hosting).
2. **Translation** 

— the transcript is translated into the selected target
   language via Google Translate (`deep-translator`).
3. **Speech synthesis** — the translated text is converted back to speech
   using gTTS. If gTTS fails (rate limiting, network issues), the app
   automatically falls back to Microsoft Edge's neural TTS engine, so the
   user still gets working audio without seeing an error.

## 🛠️ Tech stack

- **Interface:** Gradio
- **ASR:** faster-whisper (local, open-source, no API key required)
- **Translation:** Google Translate via `deep-translator`
- **Speech synthesis:** gTTS (primary) with Edge TTS (automatic failover)

## 🧭 Technical decisions

- **Why local Whisper instead of the OpenAI Whisper API?** The API costs
  ~$0.006/minute of audio. For a portfolio demo that isn't running production
  traffic, that's a needless ongoing cost and an extra dependency (API key
  management, billing). Running Whisper locally on the `base` model keeps
  the app fully free to operate, at a small cost to transcription accuracy
  and speed compared to larger models — an acceptable trade-off for this
  use case.
- **Why keep the gTTS → Edge TTS failover?** gTTS is a free, unofficial
  wrapper around Google's TTS endpoint and can be rate-limited under load.
  Rather than let that surface as a user-facing error, the app catches the
  failure and transparently retries with a second engine.
- **Known limitation:** transcription accuracy depends on audio clarity —
  background noise or heavy accents can reduce Whisper's accuracy, especially
  at the `base` model size. A larger model (`small`/`medium`) would improve
  this at the cost of slower inference and higher memory use.

## 📂 Project structure

- `app.py` — pipeline logic (ASR, translation, TTS with failover) and Gradio UI
- `requirements.txt` — dependencies for local and cloud deployment

## 🔧 Local setup

```bash
git clone https://github.com/YOUR_USERNAME/voice-to-voice-translator.git
cd voice-to-voice-translator
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The first run will download the Whisper `base` model (~150MB) automatically —
this only happens once.

---

*Built as a demonstration of a multi-stage speech AI pipeline (ASR → MT →
TTS) with attention to reliability (automatic TTS failover) and cost-aware
architecture (local inference over paid API calls).*
