#!/usr/bin/env python3
"""
Rearchitected hold-to-record -> transcribe -> type script

Key improvements (see notes below):
 - Encapsulated recording/transcription in a `Recorder` class
 - Clear, thread-safe start() and stop() semantics using Lock/Event
 - Non-blocking audio capture (sounddevice InputStream callback) + single-shot processing worker
 - Robust error handling and state introspection
 - Optional minimum-duration guard to avoid empty/very-short transcriptions
 - Cleaner shutdown handling (joins background workers)

Usage: same as original — hold ACTIVATION_KEY to record, release to stop and automatically transcribe+type.

Dependencies:
    pip install pynput sounddevice numpy parakeet-mlx

macOS permissions: Microphone and Input Monitoring access required.
"""

from pathlib import Path
from threading import Event, Lock, Thread
from typing import Optional, List

import numpy as np
import sounddevice as sd
from pynput import keyboard
import mlx.core as mx
from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import get_logmel
import Quartz
import time


# --- User-Definable Hotkey ---
ACTIVATION_KEY = keyboard.Key.shift_r
MAX_CHUNK = 20  # max Unicode code points per CGEvent


def type_fast(text: str) -> None:
    """Very fast Unicode typing via CoreGraphics keyboard events.

    Sends only key-down events with chunked Unicode payloads.
    This mirrors the original approach for speed.
    """
    for i in range(0, len(text), MAX_CHUNK):
        chunk = text[i : i + MAX_CHUNK]
        evt = Quartz.CGEventCreateKeyboardEvent(  # pyright: ignore[reportAttributeAccessIssue]
            None, 0, True
        )
        Quartz.CGEventKeyboardSetUnicodeString(  # pyright: ignore[reportAttributeAccessIssue]
            evt, len(chunk), chunk
        )
        Quartz.CGEventPost(  # pyright: ignore[reportAttributeAccessIssue]
            Quartz.kCGHIDEventTap, evt  # pyright: ignore[reportAttributeAccessIssue]
        )


class Recorder:
    """Encapsulates recording state and processing.

    start(): begin listening to the default input device (non-blocking).
    stop(): stop listening, snapshot captured audio and schedule/perform transcription + typing.
    shutdown(): force-clean resources and wait for processing workers to finish.
    """

    def __init__(
        self,
        model,
        rate: Optional[int] = None,
        channels: int = 1,
        dtype: str = "int16",
        min_duration_sec: float = 0.08,
    ) -> None:
        self.model = model
        self.rate = rate or int(model.preprocessor_config.sample_rate)
        self.channels = channels
        self.dtype = dtype
        self.min_duration_sec = min_duration_sec

        # internal state
        self._frames: List[np.ndarray] = []
        self._frames_lock = Lock()
        self._stream: Optional[sd.InputStream] = None
        self._is_recording = Event()
        self._processing_lock = Lock()  # ensure one transcription runs at once
        self._workers: List[Thread] = []

    # ---------------- audio callback ----------------
    def _audio_callback(self, indata, frames, time_info, status):
        # status is a CallbackFlags object; log it if present
        if status:
            print(f"⚠️ Input stream status: {status}")
        # copy the incoming chunk and append under lock
        with self._frames_lock:
            self._frames.append(indata.copy())

    # ---------------- start / stop ----------------
    def start(self) -> None:
        """Start the input stream and begin collecting frames.

        Safe to call repeatedly; will no-op if already recording.
        """
        if self._is_recording.is_set():
            print("🔁 Already recording — start() ignored.")
            return

        print("🎤 Starting recording...")
        with self._frames_lock:
            self._frames.clear()

        try:
            self._stream = sd.InputStream(
                samplerate=self.rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback,
                blocksize=0,
            )
            self._stream.start()
            self._is_recording.set()
            print("🔴 Microphone listening… (release key to stop)")
        except Exception as e:
            # ensure no half-open stream remains
            print(f"❌ Failed to start audio stream: {e}")
            try:
                if self._stream is not None:
                    self._stream.close()
            except Exception:
                pass
            self._stream = None

    def stop(self) -> None:
        """Stop recording and kick off transcription.

        This returns immediately after scheduling the transcription worker.
        The actual transcription/typing runs in a background thread so the
        key-listener remains responsive.
        """
        if not self._is_recording.is_set():
            print("⏹️ Not currently recording — stop() ignored.")
            return

        print("🎤 Stopping recording and snapshotting audio...")

        # Stop and close the stream first (best-effort)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                print(f"⚠️ Error while stopping audio stream: {e}")
            finally:
                self._stream = None

        # Snapshot collected frames atomically
        with self._frames_lock:
            if not self._frames:
                self._is_recording.clear()
                print("🎙️ No audio frames were captured.")
                return
            audio_chunks = self._frames.copy()
            self._frames.clear()

        self._is_recording.clear()

        # Concatenate into a single 1-D array
        try:
            audio_data = np.concatenate(audio_chunks, axis=0)
            # If stereo or shaped (N,1), flatten to (N,) mono samples
            audio_data = np.squeeze(audio_data)
        except Exception as e:
            print(f"❌ Failed to concatenate audio frames: {e}")
            return

        duration = audio_data.shape[0] / float(self.rate)
        if duration < self.min_duration_sec:
            print(
                f"🔇 Captured audio too short ({duration:.3f}s); skipping transcription."
            )
            return

        # Launch a worker thread to process without blocking the keyboard listener
        worker = Thread(target=self._process_and_type, args=(audio_data,))
        worker.daemon = True
        worker.start()
        self._workers.append(worker)
        print("🗳️ Audio handed off to transcription worker.")

    # ---------------- processing ----------------
    def _process_and_type(self, audio_data: np.ndarray) -> None:
        """Normalize, run model inference, and emit typed text.

        This method acquires a processing lock so only one transcription runs at a time.
        """
        # Ensure only one transcription runs concurrently
        if not self._processing_lock.acquire(blocking=False):
            print("⏳ A transcription is already in progress — queuing skipped.")
            return

        try:
            print("🗣️ Transcribing audio...")
            # normalize to float32 in [-1, 1]
            # audio_data might already be float depending on sounddevice, but we handle int types too
            if np.issubdtype(audio_data.dtype, np.integer):
                norm = np.iinfo(audio_data.dtype).max
                audio_f32 = audio_data.astype(np.float32) / float(norm)
            else:
                audio_f32 = audio_data.astype(np.float32)

            # flatten to 1-D
            audio_f32 = audio_f32.flatten()

            audio_mx = mx.array(audio_f32)
            mel = get_logmel(audio_mx, self.model.preprocessor_config)

            alignments = self.model.generate(mel)
            transcribed_text = "".join([seg.text for seg in alignments]).strip()

            print(f'💬 Transcription result: "{transcribed_text}"')

            if transcribed_text:
                try:
                    type_fast(transcribed_text)
                    print("✅ Typing complete.")
                except Exception as e:
                    print(f"⚠️ Typing failed: {e}")
            else:
                print("⌨️ No text to type after transcription.")

        except Exception as e:
            print(f"❌ Error during transcription/typing: {e}")
        finally:
            self._processing_lock.release()

    # ---------------- shutdown ----------------
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Clean up resources. Optionally wait for worker threads to finish.

        - wait=True: join background workers (optional timeout)
        """
        if self._is_recording.is_set():
            print("🔴 Shutdown requested while recording — stopping first.")
            try:
                if self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
            except Exception:
                pass
            self._is_recording.clear()

        if wait and self._workers:
            print("⏳ Waiting for background workers to finish...")
            start = time.time()
            for w in list(self._workers):
                remaining = None
                if timeout is not None:
                    elapsed = time.time() - start
                    remaining = max(0, timeout - elapsed)
                w.join(remaining)
            print("🧹 All workers joined (or timed out).")


# ----------------- MAIN - wiring to pynput -----------------
if __name__ == "__main__":
    print("🎙️ Initializing transcription model...")
    try:
        model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
        print("✅ Model initialized.")
    except Exception as e:
        print(f"❌ Could not initialize model: {e}")
        raise

    # warm up (best-effort)
    try:
        warmup_file = Path(__file__).parent / "warmup.wav"
        if warmup_file.exists():
            model.transcribe(str(warmup_file))
            print("🧪 Model warm-up complete.")
        else:
            print("🧪 Warm-up file not found.")
    except Exception as e:
        print(f"⚠️ Warm-up failed: {e}")
    recorder = Recorder(model)

    # simplified key-state handling; encapsulated in a HotkeyHandler to avoid nonlocal/global issues
    class HotkeyHandler:
        def __init__(self, recorder: Recorder, activation_key):
            self.recorder = recorder
            self.activation_key = activation_key
            self._key_pressed = False

        def on_press(self, key):
            if key == self.activation_key and not self._key_pressed:
                self._key_pressed = True
                self.recorder.start()

        def on_release(self, key):
            if key == self.activation_key and self._key_pressed:
                self._key_pressed = False
                self.recorder.stop()

    handler = HotkeyHandler(recorder, ACTIVATION_KEY)
    listener = keyboard.Listener(
        on_press=handler.on_press, on_release=handler.on_release
    )
    listener.start()

    try:
        listener.join()
    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt received — exiting.")
    finally:
        if listener.is_alive():
            listener.stop()
        recorder.shutdown(wait=True, timeout=5.0)
        print("Program terminated.")
