"""
Voice-to-Voice Translator
--------------------------
A three-stage speech translation pipeline:

    1. ASR   (Speech -> Text)          via faster-whisper (local, free, open-source)
    2. MT    (Text -> Translated Text) via Google Translate (deep-translator)
    3. TTS   (Translated Text -> Speech) via gTTS, with automatic failover to
             Microsoft Edge Neural TTS if gTTS fails or is rate-limited.

Design notes (see README for the full write-up):
- Whisper runs locally via faster-whisper so the demo has no per-request API
  cost and no external ASR dependency.
- Speech input can be in ANY language Whisper supports (auto-detected) and
  translated to any of the target languages below -- including English, so
  the app genuinely lives up to "variety of languages in, variety out"
  rather than assuming the speaker is a non-English speaker.
- The TTS failover logic is intentionally kept from the earlier version of
  this project, since it's a genuinely useful reliability pattern.
"""

import os
import time
import asyncio
import tempfile

import gradio as gr
from deep_translator import GoogleTranslator
from gtts import gTTS
import edge_tts
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
WHISPER_LOCAL_PATH = os.environ.get("WHISPER_LOCAL_PATH", "./whisper-model-base").strip()

# (deep_translator code, edge-tts voice). English included so the pipeline
# works in both directions -- e.g. speak Urdu -> hear English, or speak
# English -> hear Japanese.
LANGUAGE_MAP = {
    "English": ("en", "en-US-AriaNeural"),
    "Spanish": ("es", "es-ES-AlvaroNeural"),
    "French": ("fr", "fr-FR-EloiseNeural"),
    "German": ("de", "de-DE-KillianNeural"),
    "Hindi": ("hi", "hi-IN-MadhurNeural"),
    "Japanese": ("ja", "ja-JP-NanamiNeural"),
    "Chinese": ("zh-CN", "zh-CN-XiaoxiaoNeural"),
    "Arabic": ("ar", "ar-SA-ZariyahNeural"),
    "Italian": ("it", "it-IT-DiegoNeural"),
    "Korean": ("ko", "ko-KR-SunHiNeural"),
    "Urdu": ("ur", "ur-PK-AsadNeural"),
}

# ---------------------------------------------------------------------------
# Model loading (once, at startup -- not per-request)
# ---------------------------------------------------------------------------

_model_source = WHISPER_LOCAL_PATH if WHISPER_LOCAL_PATH else WHISPER_MODEL_SIZE
print(f"Loading Whisper model from '{_model_source}'... this happens once at startup.")
asr_model = WhisperModel(_model_source, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
print("Whisper model loaded.")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str) -> str:
    """Stage 1: Speech -> Text using local Whisper (any supported source language)."""
    segments, _info = asr_model.transcribe(audio_path, beam_size=1)
    transcript = " ".join(segment.text.strip() for segment in segments)
    return transcript.strip()


def translate_text(text: str, lang_code: str) -> str:
    """Stage 2: Text -> Translated text."""
    return GoogleTranslator(source="auto", target=lang_code).translate(text)


async def synthesize_speech(text: str, lang_code: str, voice_code: str, output_path: str) -> str:
    """Stage 3: Translated text -> Speech, with automatic failover.

    Tries gTTS first. If it fails (rate limiting, network issues, unsupported
    language combination), silently falls back to Edge TTS so the user still
    gets working audio.
    """
    try:
        tts = gTTS(text=text, lang=lang_code)
        tts.save(output_path)
    except Exception:
        communicate = edge_tts.Communicate(text, voice_code)
        await communicate.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline(audio_path, target_lang):
    """Runs the full ASR -> MT -> TTS pipeline and returns UI-ready outputs."""
    if audio_path is None:
        return (
            gr.update(value="Please record or upload some audio first."),
            gr.update(value=""),
            None,
        )

    if target_lang not in LANGUAGE_MAP:
        return (
            gr.update(value="Please choose a target language."),
            gr.update(value=""),
            None,
        )

    lang_code, voice_code = LANGUAGE_MAP[target_lang]

    t0 = time.time()
    try:
        transcript = transcribe_audio(audio_path)
    except Exception as e:
        return gr.update(value=f"Transcription failed: {e}"), gr.update(value=""), None
    print(f"[timing] ASR: {time.time() - t0:.1f}s")

    if not transcript:
        return (
            gr.update(value="Couldn't detect clear speech in that audio. Try again with less background noise."),
            gr.update(value=""),
            None,
        )

    t1 = time.time()
    try:
        translated = translate_text(transcript, lang_code)
    except Exception as e:
        return gr.update(value=f"Translation failed: {e}"), gr.update(value=transcript), None
    print(f"[timing] Translation: {time.time() - t1:.1f}s")

    t2 = time.time()
    output_path = os.path.join(tempfile.gettempdir(), "translated_audio.mp3")
    try:
        asyncio.run(synthesize_speech(translated, lang_code, voice_code, output_path))
    except Exception as e:
        return (
            gr.update(value=f"Speech synthesis failed: {e}"),
            gr.update(value=f"{transcript}\n\n\u2192 {translated}"),
            None,
        )
    print(f"[timing] TTS: {time.time() - t2:.1f}s")

    original_and_translation = f"You said:\n{transcript}\n\nTranslation:\n{translated}"
    return gr.update(value=original_and_translation), gr.update(value=translated), output_path


# ---------------------------------------------------------------------------
# Visual design
# ---------------------------------------------------------------------------
# Token system:
#   Ink       #10231C  -- near-black with a green cast, body text
#   Canvas    #F6F4EE  -- warm paper background (not the AI-cliche cream+terracotta;
#                          paired with deep green + brass instead)
#   Evergreen #14532D  -- primary accent, brand color, buttons
#   Brass     #B08D57  -- secondary accent, used sparingly (badges, active states)
#   Line      #DAD5C6  -- hairline borders / dividers
#   Card      #FFFFFF  -- panel background
#
# Signature element: a "language chips" strip -- the set of supported
# languages rendered as small pill tokens, echoing the idea of many tongues
# feeding into one pipeline. Numbered steps (1/2/3) are used because the
# flow genuinely is sequential -- record, choose, translate.

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    font=(gr.themes.GoogleFont("Manrope"), "ui-sans-serif", "system-ui", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"),
).set(
    body_background_fill="#F6F4EE",
    body_background_fill_dark="#10231C",
    block_background_fill="#FFFFFF",
    block_border_color="#DAD5C6",
    block_radius="16px",
    button_primary_background_fill="#14532D",
    button_primary_background_fill_hover="#1C6B3A",
    button_primary_text_color="#FFFFFF",
    block_label_text_color="#10231C",
    block_title_text_weight="600",
)

CUSTOM_CSS = """
#hero {
    text-align: center;
    padding: 1.75rem 1rem 0.5rem 1rem;
}
#hero h1 {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #10231C;
    margin-bottom: 0.35rem;
}
#hero p {
    color: #5B5647;
    font-size: 1.02rem;
    max-width: 640px;
    margin: 0 auto;
}
#lang-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    justify-content: center;
    margin: 1rem 0 1.75rem 0;
}
.lang-chip {
    font-size: 0.78rem;
    font-weight: 600;
    color: #14532D;
    background: #E7F0E9;
    border: 1px solid #C7DBCB;
    border-radius: 999px;
    padding: 0.25rem 0.7rem;
}
.step-label {
    font-weight: 700;
    color: #B08D57;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.15rem;
}
#footer-note {
    text-align: center;
    color: #8A8471;
    font-size: 0.82rem;
    margin-top: 1.5rem;
    padding-bottom: 1rem;
}
footer {visibility: hidden}
"""

LANGUAGE_CHIPS_HTML = (
    '<div id="lang-strip">'
    + "".join(f'<span class="lang-chip">{name}</span>' for name in LANGUAGE_MAP.keys())
    + "</div>"
)

with gr.Blocks(title="Voice-to-Voice Translator") as demo:
    gr.HTML(
        """
        <div id="hero">
            <h1>Voice-to-Voice Translator</h1>
            <p>Speak in any language. Hear it back translated. A local Whisper model
            transcribes, Google Translate converts the text, and dual-engine speech
            synthesis reads it back to you.</p>
        </div>
        """
    )
    gr.HTML(LANGUAGE_CHIPS_HTML)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320):
            gr.Markdown('<div class="step-label">Step 1</div>')
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Speak or upload audio",
            )
            gr.Markdown('<div class="step-label">Step 2</div>')
            target_select = gr.Dropdown(
                choices=list(LANGUAGE_MAP.keys()),
                value="Spanish",
                label="Target language",
            )
            gr.Markdown('<div class="step-label">Step 3</div>')
            translate_btn = gr.Button("Translate", variant="primary", size="lg")
            status_note = gr.Markdown("", elem_id="status-note")

        with gr.Column(scale=1, min_width=320):
            transcript_box = gr.Textbox(
                label="Transcript & translation",
                lines=7,
                interactive=False,
            )
            translated_text_box = gr.Textbox(
                label="Translated text (plain)",
                lines=2,
                interactive=False,
            )
            audio_output = gr.Audio(label="Translated speech", type="filepath")

    gr.HTML(
        """
        <div id="footer-note">
            Pipeline: local Whisper ASR &rarr; Google Translate &rarr; gTTS
            (falls back to Microsoft Edge Neural TTS automatically if needed).
        </div>
        """
    )

    def _on_click(audio_path, target_lang):
        return run_pipeline(audio_path, target_lang)

    translate_btn.click(
        fn=_on_click,
        inputs=[audio_input, target_select],
        outputs=[transcript_box, translated_text_box, audio_output],
    )

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS)
