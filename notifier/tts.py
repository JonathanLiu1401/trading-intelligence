"""
TTS pipeline:
  1. Claude Sonnet 4.6 writes a DualAssets-style narration script from the market data
  2. Kokoro (local ONNX) voices it (am_adam — deep confident American male)
  3. ffplay / aplay plays the audio locally
"""
import os
import re
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

# ── Kokoro config ────────────────────────────────────────────────────────────
KOKORO_MODEL  = Path(os.environ.get("KOKORO_MODEL",  "/home/zeph/whisper-server/kokoro-v1.0.onnx"))
KOKORO_VOICES = Path(os.environ.get("KOKORO_VOICES", "/home/zeph/whisper-server/voices-v1.0.bin"))
KOKORO_VOICE  = "am_adam"   # deep confident American male
KOKORO_SPEED  = 1.1         # slightly faster — DualAssets pacing

MAX_SCRIPT_CHARS = 1400     # ~90-100 seconds of speech at 1.1x

# ── Claude config ─────────────────────────────────────────────────────────────
SONNET_MODEL = "claude-sonnet-4-6"

NARRATION_PROMPT_ALERT = """You are a confident financial markets narrator in the style of DualAssets — fast-paced, engaging, energetic American male voice. An urgent market event just happened.

Write a 30-45 second spoken narration (about 120-150 words). Rules:
- Start with "Breaking." or "Alright guys, this just dropped —"
- Speak conversationally, like you're live on air
- State exactly what happened and the number (EPS, price move, %, etc.)
- Explain WHY it matters for the portfolio in plain English
- End with one actionable line: what to watch next
- NO markdown, NO emojis, NO asterisks — pure spoken text only

Market event:
{event_text}

Write ONLY the spoken narration, nothing else."""

NARRATION_PROMPT_BRIEFING = """You are a confident financial markets narrator in the style of DualAssets — clear, engaging, well-paced American male voice. You're delivering the daily market briefing.

Write a 90-120 second spoken narration (about 250-300 words). Rules:
- Open with energy: "Alright, let's get into it —" or "What's moving in markets today —"
- Cover in order: overall market direction, key macro driver, memory/semis sector, top portfolio name
- Use real numbers from the data provided
- Speak conversationally — no bullet points, no headers, flowing sentences
- Build narrative: connect each section with transitions ("Now, what's driving this is...", "Over in the semis space...", "For the portfolio...")
- Close with: "That's the read. Keep watching [ticker/level]. Stay sharp."
- NO markdown, NO emojis, NO asterisks — pure spoken text only

Market data and top stories:
{briefing_text}

Write ONLY the spoken narration, nothing else."""


def _generate_script(text: str, is_alert: bool) -> str:
    """Use Claude Sonnet to write a spoken narration script."""
    if not shutil.which("claude"):
        return re.sub(r"[*_`~>#\[\]|━=─]", "", text)[:MAX_SCRIPT_CHARS]

    prompt = (NARRATION_PROMPT_ALERT if is_alert else NARRATION_PROMPT_BRIEFING).format(
        event_text=text[:1500], briefing_text=text[:2000]
    )

    try:
        result = subprocess.run(
            ["claude", "--model", SONNET_MODEL, "--print",
             "--permission-mode", "bypassPermissions", prompt],
            capture_output=True, text=True, timeout=45,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()[:MAX_SCRIPT_CHARS]
    except Exception as e:
        print(f"[tts] Script generation error: {e}")

    return re.sub(r"[*_`~>#\[\]|━=─]", "", text)[:MAX_SCRIPT_CHARS]


def _speak_kokoro(script: str):
    """Generate audio with local Kokoro ONNX and play it via ffplay/aplay."""
    if not KOKORO_MODEL.exists() or not KOKORO_VOICES.exists():
        print(f"[tts] Kokoro model not found at {KOKORO_MODEL} — skipping TTS")
        return

    try:
        from kokoro_onnx import Kokoro
        import numpy as np

        kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
        samples, sample_rate = kokoro.create(
            script,
            voice=KOKORO_VOICE,
            speed=KOKORO_SPEED,
            lang="en-us",
        )

        # Write WAV to temp file and play
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name

        # kokoro returns float32 in [-1, 1]; convert to int16 for WAV
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())

        # prefer ffplay, fallback to aplay
        if shutil.which("ffplay"):
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp],
                timeout=300,
            )
        elif shutil.which("aplay"):
            subprocess.run(["aplay", "-q", tmp], timeout=300)
        else:
            print(f"[tts] No audio player found — WAV saved to {tmp}")
            return

        Path(tmp).unlink(missing_ok=True)
        print("[tts] Kokoro playback complete")

    except Exception as e:
        print(f"[tts] Kokoro error: {e}")


def speak(message: str, is_alert: bool = False):
    """Generate narration script then speak it. Runs synchronously (call from thread)."""
    print(f"[tts] Generating {'alert' if is_alert else 'briefing'} narration script...")
    script = _generate_script(message, is_alert=is_alert)
    print(f"[tts] Script ({len(script)} chars): {script[:120]}...")
    _speak_kokoro(script)


def speak_async(message: str, is_alert: bool = False):
    """Fire-and-forget TTS in a background thread."""
    threading.Thread(target=speak, args=(message, is_alert), daemon=True).start()


if __name__ == "__main__":
    speak("Alright guys, this is a test of the Kokoro TTS system. Markets are looking strong today. Stay sharp.", is_alert=False)
