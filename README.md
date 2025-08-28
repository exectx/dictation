Dictation POC
=============

Hold-to-record speech-to-text prototype that transcribes audio locally using a Parakeet MLX model and rapidly injects the resulting Unicode text via macOS CoreGraphics events.

Features
--------
* Press and hold Right Shift (configurable) to record; release to transcribe & type.
* Non-blocking keyboard listener + background transcription.
* Single concurrent transcription (drops overlapping recordings gracefully).
* Rich (optional) color logging with debug timing, frame counts, and preprocessing/model breakdown.
* Plain logger mode for piping to files / CI.

Installation
------------
This project uses a modern Python (>=3.11). Install dependencies (uv / pip):

```
uv sync
# or
pip install -e .
```

Usage
-----
```
python main.py                    # normal run (info level, colored logs)
python main.py --debug            # verbose diagnostics & timings
python main.py --timestamps       # add HH:MM:SS to log records
python main.py --no-color         # disable Rich; plain text logs
python main.py --log-file run.log # also append plain formatted logs to run.log
```

While running, hold Right Shift to capture audio. Release to trigger transcription. The recognized text is injected immediately at the current cursor location.

Logging Notes
-------------
* Rich output omits path for cleanliness; enable timestamps for performance tuning.
* Debug mode reports preprocessing & model inference timings plus sample normalization stats.
* Use `--no-color` when redirecting output, or `--log-file` to simultaneously keep colored console output and a plain file.

Safety / Edge Cases
-------------------
* Very short taps (<40ms default) are ignored to avoid stray noise.
* Only one transcription runs at a time; overlapping captures while one is processing are dropped with a warning.
* On shutdown (Ctrl+C) the program waits (up to 5s) for background workers.

Customization
-------------
Change activation key inside `main.py` (`ACTIVATION_KEY`). Minimum duration and data type are constructor parameters of `Recorder`.

License
-------
Prototype code for experimentation. Add your licensing terms here.

