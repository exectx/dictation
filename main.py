#!/usr/bin/env python3
"""
Hold a designated key (ACTIVATION_KEY) → start recording from the default macOS input device.
Release the key → stop, transcribe, and type the result.

Dependencies:
    pip install pynput sounddevice numpy parakeet-mlx

macOS: give the terminal (or your Python app bundle) **Microphone** and
       **Input Monitoring** (for pynput to control keyboard) access.
"""

from parakeet_mlx import from_pretrained, audio

from pathlib import Path
from threading import Event, Lock

import numpy as np
import sounddevice as sd
from pynput import keyboard
import mlx.core as mx
from parakeet_mlx.audio import get_logmel
from parakeet_mlx.audio import load_audio
import Quartz


# --- User-Definable Hotkey ---
# Define the key to trigger recording.
# Press this key to start, release to stop.
#
# Examples:
#   For the F12 key: ACTIVATION_KEY = keyboard.Key.f12
#   For the right Shift key: ACTIVATION_KEY = keyboard.Key.shift_r
#
# **Regarding the "fn" (Globe) key:**
#   It's not supported by the pynput library.
#
ACTIVATION_KEY = keyboard.Key.shift_r

MAX_CHUNK = 20  # CGEvent truncates beyond ~20 UTF-16 units
# --- Initialize Parakeet Model ---
# Moved model initialization here to ensure it's done early.
print("🎙️ Initializing transcription model...")
try:
    # model = from_pretrained("mlx-community/parakeet-ctc-0.6b")
    model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    ""
    print("✅ Transcription model initialized.")
    # Warm up the model using the provided example WAV to prime transcription internals
    print("🧪 Warming up transcription model using warmup.wav...")
    try:
        warmup_file = Path(__file__).parent / "warmup.wav"
        model.transcribe(str(warmup_file))
        print("✅ Model warm-up complete.")
    except Exception as e:
        print(f"⚠️ Model warm-up failed: {e}")
except Exception as e:
    print(f"❌ Error initializing transcription model: {e}")
    print("Please ensure parakeet-mlx is installed and models can be downloaded.")
    exit(1)


# ---------- audio backend ----------------------------------------------------
# Record at model's target sample rate to avoid resampling
RATE = model.preprocessor_config.sample_rate  # Hz
CHANNELS = 1  # mono
_dtype = "int16"  # 16-bit PCM (easy for WAV)

_frames: list[np.ndarray] = []
_lock = Lock()
_stream: sd.InputStream | None = None
_recording = Event()  # thread-safe “are we recording?” flag


def _audio_cb(indata, *_):
    with _lock:
        _frames.append(indata.copy())


def _start_rec():
    global _stream
    if _recording.is_set():
        return
    _frames.clear()
    print("🎤 Attempting to start recording...")
    try:
        _stream = sd.InputStream(
            samplerate=RATE,
            channels=CHANNELS,
            dtype=_dtype,
            callback=_audio_cb,
            blocksize=0,
        )
        _stream.start()
        _recording.set()
        print("🔴 Microphone listening… (Release key to stop)")
    except Exception as e:
        print(f"❌ Error starting audio stream: {e}")
        if _stream:
            _stream.close()
        _stream = None


def _stop_rec():
    global _stream
    if not _recording.is_set():
        return

    print("🎤 Stopping recording...")
    if _stream:
        try:
            _stream.stop()
            _stream.close()
        except Exception as e:
            print(f"⚠️ Error stopping audio stream: {e}")
        finally:
            _stream = None
    else:  # Should not happen if _recording.is_set() but as a safeguard
        _recording.clear()
        print("⁉️ Stream was not active but recording flag was set.")
        return

    with _lock:
        if not _frames:
            _recording.clear()
            print("🎙️ No audio recorded.")
            return
        audio_data = np.concatenate(_frames)

    _recording.clear()

    print("🗣️ Transcribing audio...")
    audio = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    # Recorded at target sample rate; skipping resampling
    audio = mx.array(audio)
    mel = get_logmel(audio, model.preprocessor_config)
    alignments = model.generate(mel)
    transcribed_text = "".join([seg.text for seg in alignments]).strip()
    print(f'💬 Transcription: "{transcribed_text}"')
    if transcribed_text:
        type_fast(transcribed_text)
        print("✅ Typing complete.")
    else:
        print("⌨️ No text transcribed to type.")


# ---------- hot-key state machine for single key activation -----------------
_key_pressed = False  # Simpler flag for single key


def on_press(key):
    global _key_pressed
    if key == ACTIVATION_KEY:
        if not _key_pressed:  # Start only on the first press event
            _key_pressed = True
            if not _recording.is_set():
                _start_rec()


def on_release(key):
    global _key_pressed
    if key == ACTIVATION_KEY:
        if _key_pressed:  # Stop only if it was the key we track
            _key_pressed = False
            if _recording.is_set():
                _stop_rec()


def _key_down(chunk: str) -> None:
    """
    Post a single key-down CGEvent carrying `chunk` as its Unicode payload.
    No key-up counterpart is sent.
    """
    evt = Quartz.CGEventCreateKeyboardEvent(  # type: ignore[attr-defined]
        None, 0, True
    )  # isKeyDown = True
    Quartz.CGEventKeyboardSetUnicodeString(evt, len(chunk), chunk)  # type: ignore[attr-defined]
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)  # type: ignore[attr-defined]
    # PyObjC releases 'evt' automatically; no CFRelease() needed.


# pynput controller.type is slow, outputting chunked UnicodeString is >50x faster
def type_fast(text: str) -> None:
    """Up to 20 code-points per key-down event."""
    for i in range(0, len(text), MAX_CHUNK):
        _key_down(text[i : i + MAX_CHUNK])


if __name__ == "__main__":
    print("---")
    print("macOS Permissions Reminder:")
    print("1. Microphone access (System Settings > Privacy & Security > Microphone)")
    print(
        "2. Input Monitoring access (System Settings > Privacy & Security > Input Monitoring)"
    )
    print("   (Required for pynput to type and listen to global hotkeys)")
    print("---")
    print(
        f"🚀 Script ready. Hold the '{str(ACTIVATION_KEY).replace('Key.', '')}' key to record audio."
    )
    print("   Release the key to stop recording, save, transcribe, and type.")
    print("   Press Ctrl+C in this terminal to quit the script.")
    print("---")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        listener.join()
    except KeyboardInterrupt:
        print("\n🛑 Exiting program via Ctrl+C.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
    finally:
        if listener.is_alive():
            listener.stop()
        if _recording.is_set():  # Ensure cleanup if exited while recording
            print("🔴 Stopping active recording due to exit...")
            if _stream:
                _stream.stop()
                _stream.close()
            _recording.clear()
            print("Cleanup complete.")
        print("Program terminated.")

model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")
