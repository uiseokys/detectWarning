# detectWarning

Start with person and face detection before adding risk analysis.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Use the webcam:

```bash
python3 app/main.py --source 0
```

Enable microphone speech-to-text while the video is running:

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR
```

Use the recommended balance for warning detection:

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR --stt-model base --stt-phrase-seconds 1.8 --stt-beam-size 1
```

Use a larger Whisper model for better accuracy:

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR --stt-model small --stt-phrase-seconds 2.8 --stt-beam-size 4 --stt-best-of 4
```

Use low-latency STT for alert-style detection:

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR --stt-model tiny --stt-phrase-seconds 1.5 --stt-beam-size 1
```

Use a local video file:

```bash
python3 app/main.py --source /path/to/video.mp4
```

Quit with `q` or `Esc`.

The app draws:

- Green boxes for detected people with tracking IDs like `Person 1`
- Blue boxes for detected faces
- STT status and the most recent recognized speech from the microphone
- A live risk score that combines recent speech keywords, audio intensity, and visible people or faces

## Risk Score

- The risk score is a heuristic 0-100 value, not a safety-certified classifier.
- It becomes more sensitive when emergency phrases like `살려줘`, `도와줘`, `하지마`, or `불이야` are recognized.
- Loud audio and visible people or faces increase the score further to make warning behavior more immediate.

## Notes

- STT uses the microphone even when the video source is a local file.
- Install the new speech dependencies with `pip install -r requirements.txt`.
- STT now uses local Whisper inference through `faster-whisper`.
- The first model load may download Whisper weights, so internet access can be required once during setup.
- The default STT settings now aim for a warning-detection balance: `base` model with a short phrase window and fast decoding.
- Larger models like `small` or `medium` are usually more accurate, but they need more CPU or GPU resources.
- `tiny` is faster, but `base` is usually a better balance when you still want usable Korean recognition quality.
- On macOS, STT runs in a separate process to avoid FFmpeg library conflicts between OpenCV and Whisper dependencies.
