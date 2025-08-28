"""Speech-to-text hold-to-record prototype.

Flow:
    1. Hold activation key to begin recording (non-blocking input stream)
    2. Release key -> audio snapshot -> background transcription -> fast Unicode typing

Design highlights:
    * Recorder class isolates capture / processing
    * Thread-safe lifecycle (Locks + Events)
    * Single concurrent transcription (processing lock)
    * Minimum duration guard to skip accidental taps
    * Graceful shutdown waiting for workers

Logging improvements:
    * Rich colored console output (no emojis) by default
    * Optional --timestamps for wall-clock time prefix
    * Optional --no-color to fall back to plain logging (for file capture / piping)
    * Debug mode adds detailed audio and model timing information
"""

from pathlib import Path
from threading import Event, Lock, Thread
from typing import Optional, List
import argparse
import logging
import sys
import time

import numpy as np
import sounddevice as sd
from pynput import keyboard
import mlx.core as mx
from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import get_logmel
import Quartz
from rich.logging import RichHandler

# Rich console globals (set during logging configuration when color enabled)
RICH_ENABLED = False
RICH_CONSOLE = None

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------


def _configure_logging(
    *,
    debug: bool = False,
    timestamps: bool = False,
    no_color: bool = False,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure root + app logger.

    Parameters
    ----------
    debug : bool
        Enable DEBUG level.
    timestamps : bool
        Include time in log output (useful for performance review).
    no_color : bool
        Use plain logging formatter (disables Rich). Useful when redirecting
        to files or for minimal environments.
    """
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()

    # Remove pre-existing handlers so re-running in REPL doesn't duplicate output
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    handlers: List[logging.Handler] = []

    if no_color:
        fmt_parts = []
        if timestamps:
            fmt_parts.append("%(asctime)s")
        fmt_parts.append("%(levelname)s")
        fmt_parts.append("%(name)s")
        fmt_parts.append("%(message)s")
        formatter = logging.Formatter(" | ".join(fmt_parts), "%H:%M:%S")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)
    else:
        # Rich handles coloring + layout; we inject time optionally by adjusting RichHandler config
        rich_handler = RichHandler(
            rich_tracebacks=True,
            show_time=timestamps,
            show_path=False,
            markup=False,
            log_time_format="%H:%M:%S",
        )
        # Base format kept minimal; Rich supplies level + time columns
        formatter = logging.Formatter("%(message)s")
        rich_handler.setFormatter(formatter)
        handlers.append(rich_handler)
        global RICH_ENABLED, RICH_CONSOLE
        RICH_ENABLED = True
        RICH_CONSOLE = rich_handler.console

    if log_file:
        file_fmt_parts = ["%(asctime)s", "%(levelname)s", "%(name)s", "%(message)s"]
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(" | ".join(file_fmt_parts), "%Y-%m-%d %H:%M:%S")
        )
        handlers.append(file_handler)

    for h in handlers:
        root.addHandler(h)
    root.setLevel(level)

    logger = logging.getLogger("dictation")
    logger.setLevel(level)
    logger.debug(
        "Logger initialized (debug=%s, timestamps=%s, color=%s, log_file=%s)",
        debug,
        timestamps,
        not no_color,
        log_file,
    )
    return logger


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
        dtype: str = "int16",
        min_duration_sec: float = 0.04,
        device: Optional[int] = None,
    ) -> None:
        self.model = model
        # Always use the model's required sample rate, ignore device capabilities
        self.rate = int(model.preprocessor_config.sample_rate)
        # Always use mono audio since this model only supports 1 channel
        self.channels = 1
        self.dtype = dtype
        self.min_duration_sec = min_duration_sec
        self.device = device
        self._record_start_ts: Optional[float] = None

        # internal state
        self._frames: List[np.ndarray] = []
        self._frames_lock = Lock()
        self._stream: Optional[sd.InputStream] = None
        self._is_recording = Event()
        self._processing_lock = Lock()  # ensure one transcription runs at once
        self._workers: List[Thread] = []

    # (timer/live display removed)

    # ---------------- audio callback ----------------
    def _audio_callback(self, indata, frames, time_info, status):
        # status is a CallbackFlags object; log it if present
        if status:
            logging.getLogger("dictation").warning("Input stream status: %s", status)
        # copy the incoming chunk and append under lock
        with self._frames_lock:
            self._frames.append(indata.copy())

    # ---------------- start / stop ----------------
    def start(self) -> None:
        """Start the input stream and begin collecting frames.

        Safe to call repeatedly; will no-op if already recording.
        """
        logger = logging.getLogger("dictation")
        if self._is_recording.is_set():
            logger.debug("start() called while already recording; ignoring.")
            return

        with self._frames_lock:
            self._frames.clear()

        try:
            logger.debug("Starting audio stream with model sample rate: %s", self.rate)
            self._stream = sd.InputStream(
                samplerate=self.rate,
                channels=self.channels,
                dtype=self.dtype,
                device=self.device,
                callback=self._audio_callback,
                blocksize=0,
            )
            self._stream.start()
            self._record_start_ts = time.time()
            self._is_recording.set()
            logger.info("Microphone listening (release key to stop)")
        except Exception as e:
            # ensure no half-open stream remains
            logger.error("Failed to start audio stream: %s", e)
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
        logger = logging.getLogger("dictation")
        if not self._is_recording.is_set():
            logger.debug("stop() called while not recording; ignoring.")
            return

        # (suppressed stopping recording log)

        # Stop and close the stream first (best-effort)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning("Error while stopping audio stream: %s", e)
            finally:
                self._stream = None

        # Snapshot collected frames atomically
        with self._frames_lock:
            if not self._frames:
                # stop timer and report wall time even if no frames
                wall_elapsed = 0.0
                if self._record_start_ts is not None:
                    wall_elapsed = time.time() - self._record_start_ts
                self._is_recording.clear()
                logger.info("Recorded %.2fs of audio (no data captured)", wall_elapsed)
                logger.info("No audio frames were captured.")
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
            logger.error("Failed to concatenate audio frames: %s", e)
            return

        duration = audio_data.shape[0] / float(self.rate)
        if self._record_start_ts is not None:
            wall = time.time() - self._record_start_ts
        else:
            wall = duration
        if duration < self.min_duration_sec:
            logger.info("Recorded %.2fs of audio (too short; skipped)", duration)
            logger.info(
                "Captured audio too short (%.3fs); skipping transcription.", duration
            )
            return

        # Final duration line
        logger.info("Recorded %.2fs of audio", duration)

        # Launch a worker thread to process without blocking the keyboard listener
        worker = Thread(target=self._process_and_type, args=(audio_data,))
        worker.daemon = True
        worker.start()
        self._workers.append(worker)
        logger.debug(
            "Audio handed off to transcription worker thread id=%s", worker.ident
        )

    # ---------------- processing ----------------
    def _process_and_type(self, audio_data: np.ndarray) -> None:
        """Normalize, run model inference, and emit typed text.

        This method acquires a processing lock so only one transcription runs at a time.
        """
        # Ensure only one transcription runs concurrently
        logger = logging.getLogger("dictation")
        if not self._processing_lock.acquire(blocking=False):
            logger.warning("A transcription is already in progress; new audio dropped.")
            return

        try:
            # logger.info("Transcribing audio...")
            # normalize to float32 in [-1, 1]
            # audio_data might already be float depending on sounddevice, but we handle int types too
            t0 = time.time()
            if np.issubdtype(audio_data.dtype, np.integer):
                norm = np.iinfo(audio_data.dtype).max
                audio_f32 = audio_data.astype(np.float32) / float(norm)
                logger.debug(
                    "Converted integer samples -> float32 with norm=%s (range ~[%.3f, %.3f])",
                    norm,
                    float(audio_f32.min()) if audio_f32.size else 0.0,
                    float(audio_f32.max()) if audio_f32.size else 0.0,
                )
            else:
                audio_f32 = audio_data.astype(np.float32)
                logger.debug(
                    "Samples already float; min=%.3f max=%.3f",
                    float(audio_f32.min()) if audio_f32.size else 0.0,
                    float(audio_f32.max()) if audio_f32.size else 0.0,
                )

            # flatten to 1-D
            audio_f32 = audio_f32.flatten()
            audio_mx = mx.array(audio_f32)
            mel = get_logmel(audio_mx, self.model.preprocessor_config)
            t_feat = time.time()
            alignments = self.model.generate(mel)
            t_model = time.time()
            transcribed_text = "".join([seg.text for seg in alignments]).strip()

            logger.debug(
                "Timings: preprocess=%.3fs model=%.3fs total=%.3fs",
                t_feat - t0,
                t_model - t_feat,
                t_model - t0,
            )

            total_time = time.time() - t0
            logger.info("Transcribed in %.2fs", total_time)
            # logger.info('Transcription result: "%s"', transcribed_text)

            if transcribed_text:
                try:
                    type_fast(transcribed_text)
                    logger.info('Typed: "%s"', transcribed_text)
                except Exception as e:
                    logger.warning("Typing failed: %s", e)
            else:
                logger.info("No text produced by transcription.")

        except Exception as e:
            logger.exception("Error during transcription/typing: %s", e)
        finally:
            self._processing_lock.release()

    # ---------------- shutdown ----------------
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Clean up resources. Optionally wait for worker threads to finish.

        - wait=True: join background workers (optional timeout)
        """
        logger = logging.getLogger("dictation")
        if self._is_recording.is_set():
            logger.info("Shutdown requested while recording — stopping first.")
            try:
                if self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
            except Exception:
                pass
            self._is_recording.clear()

        if wait and self._workers:
            logger.info("Waiting for background workers to finish...")
            start = time.time()
            for w in list(self._workers):
                remaining = None
                if timeout is not None:
                    elapsed = time.time() - start
                    remaining = max(0, timeout - elapsed)
                w.join(remaining)
            logger.debug("All workers joined (or timeout reached).")


# ----------------- MAIN - wiring to pynput -----------------


def list_audio_devices():
    """List available audio input devices."""
    devices = sd.query_devices()
    input_devices = []
    for i in range(len(devices)):
        device = devices[i]  # type: ignore
        if device.get("max_input_channels", 0) > 0:
            input_devices.append(
                (i, device.get("name", "Unknown"), device.get("max_input_channels", 0))
            )
    return input_devices


def validate_device(device_id: int) -> bool:
    """Check if device ID is valid for input."""
    try:
        devices = sd.query_devices()
        if 0 <= device_id < len(devices):
            device = devices[device_id]  # type: ignore
            return device.get("max_input_channels", 0) > 0
        return False
    except Exception:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hold-to-record dictation prototype")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--timestamps", action="store_true", help="Show timestamps in log output"
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored Rich logging"
    )
    parser.add_argument(
        "--log-file", help="Append logs to a file (always plain format)", default=None
    )
    parser.add_argument(
        "--device",
        type=int,
        help="Audio input device ID (use --list-devices to see available devices)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    args = parser.parse_args()

    # Handle device listing
    if args.list_devices:
        print("Available audio input devices:")
        devices = list_audio_devices()
        if not devices:
            print("No input devices found.")
        else:
            for device_id, name, channels in devices:
                print(f"  {device_id}: {name} ({channels} channels)")
        sys.exit(0)

    # Validate selected device
    selected_device = None
    if args.device is not None:
        if validate_device(args.device):
            selected_device = args.device
        else:
            print(
                f"Error: Device ID {args.device} is not valid or has no input channels."
            )
            print("Use --list-devices to see available devices.")
            sys.exit(1)

    logger = _configure_logging(
        debug=args.debug,
        timestamps=args.timestamps,
        no_color=args.no_color,
        log_file=args.log_file,
    )
    logger.info("Initializing transcription model...")
    try:
        model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
        logger.info("Model initialized.")
    except Exception as e:
        logger.exception("Could not initialize model: %s", e)
        raise

    # warm up (best-effort)
    try:
        warmup_file = Path(__file__).parent / "warmup.wav"
        if warmup_file.exists():
            model.transcribe(str(warmup_file))
            logger.info("Model warm-up complete.")
        else:
            logger.warning("Warm-up file not found (skipped).")
    except Exception as e:
        logger.warning("Warm-up failed: %s", e)
    recorder = Recorder(model, device=selected_device)

    # Log input device once after initialization
    try:
        device_to_use = selected_device
        if device_to_use is None:
            device_to_use, _ = sd.default.device

        if device_to_use is not None:
            devices = sd.query_devices()
            if 0 <= device_to_use < len(devices):
                dev = devices[device_to_use]  # type: ignore
                device_source = "selected" if selected_device is not None else "default"
                logger.info(
                    "Input device (%s): %s (id=%s, sr=%s, channels=%s)",
                    device_source,
                    dev.get("name", "?"),
                    device_to_use,
                    recorder.rate,  # Show the actual rate that will be used (model's rate)
                    recorder.channels,  # Show the actual channels used (always 1 for this model)
                )
            else:
                logger.info("Input device id=%s (out of range)", device_to_use)
        else:
            logger.info("No input device available.")
    except Exception as e:
        logger.warning("Could not query input device: %s", e)

    # Usage hint
    try:
        hotkey_label = getattr(ACTIVATION_KEY, "name", str(ACTIVATION_KEY))
    except Exception:
        hotkey_label = str(ACTIVATION_KEY)
    logger.info(
        "Hold %s to record; release to transcribe & type (Esc to exit).", hotkey_label
    )

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
        logger.info("KeyboardInterrupt received — exiting.")
    finally:
        if listener.is_alive():
            listener.stop()
        recorder.shutdown(wait=True, timeout=5.0)
        logger.info("Program terminated.")
