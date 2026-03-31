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

Use a local video file:

```bash
python3 app/main.py --source /path/to/video.mp4
```

Quit with `q` or `Esc`.

The app draws:

- Green boxes for detected people with tracking IDs like `Person 1`
- Blue boxes for detected faces
