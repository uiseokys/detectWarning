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

Use a larger Whisper model for better accuracy:

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR --stt-model small
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

## Notes

- STT uses the microphone even when the video source is a local file.
- Install the new speech dependencies with `pip install -r requirements.txt`.
- STT now uses local Whisper inference through `faster-whisper`.
- The first model load may download Whisper weights, so internet access can be required once during setup.
- Larger models like `small` or `medium` are usually more accurate, but they need more CPU or GPU resources.
