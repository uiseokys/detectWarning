from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from fastapi.responses import FileResponse, Response

try:
    import av
except Exception:  # pragma: no cover - optional runtime dependency
    av = None


ANALYSIS_FRAME_BUFFER_SECONDS = 30.0
ANALYSIS_FRAME_BUFFER_MAX_FRAMES = 1200
EVENT_CLIP_MAX_FPS = 20.0
EVENT_CLIP_MIN_PLAYBACK_FPS = 15.0


@dataclass
class AnalysisFrame:
    timestamp: float
    jpeg: bytes


def sanitize_event_id(value: object) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value or "").strip())[:80]
    return safe if safe.startswith("evt_") and len(safe) >= 8 else ""


def build_event_clip_dir() -> Path:
    return Path("training_data") / "action_pipeline_aihub" / "realtime_event_clips"


def event_clip_path(event_id: str, clip_dir: Path | None = None) -> Path:
    safe_event_id = sanitize_event_id(event_id)
    if not safe_event_id:
        raise ValueError("Invalid event_id")
    return (clip_dir or build_event_clip_dir()) / f"{safe_event_id}.mp4"


def build_event_clip_url(public_base_url: str, port: int, event_id: str) -> str:
    base = str(public_base_url or "").strip().rstrip("/")
    if not base:
        base = f"http://127.0.0.1:{int(port or 8001)}"
    return f"{base}/api/events/{quote(sanitize_event_id(event_id), safe='')}/clip.mp4"


def build_event_preview_clip_url(public_base_url: str, port: int, event_id: str) -> str:
    preview_event_id = f"{sanitize_event_id(event_id)}-preview"
    return build_event_clip_url(public_base_url, port, preview_event_id)


def append_analysis_frame(
    buffer: deque,
    jpeg_bytes: bytes,
    timestamp: float,
    *,
    max_age_seconds: float = ANALYSIS_FRAME_BUFFER_SECONDS,
    max_frames: int = ANALYSIS_FRAME_BUFFER_MAX_FRAMES,
) -> None:
    if not jpeg_bytes:
        return
    buffer.append(AnalysisFrame(timestamp=float(timestamp), jpeg=bytes(jpeg_bytes)))
    while buffer and (
        timestamp - float(buffer[0].timestamp) > max_age_seconds
        or len(buffer) > max_frames
    ):
        buffer.popleft()


def select_event_clip_frames(
    frames: list[AnalysisFrame],
    start_at: float,
    end_at: float,
    *,
    max_fps: float = EVENT_CLIP_MAX_FPS,
) -> list[AnalysisFrame]:
    selected = [frame for frame in frames if start_at <= frame.timestamp <= end_at]
    if len(selected) <= 2:
        return selected
    min_interval = 1.0 / max(float(max_fps or EVENT_CLIP_MAX_FPS), 1.0)
    thinned: list[AnalysisFrame] = []
    last_ts = -1e9
    for frame in selected:
        if not thinned or frame.timestamp - last_ts >= min_interval:
            thinned.append(frame)
            last_ts = frame.timestamp
    if selected[-1] is not thinned[-1]:
        thinned.append(selected[-1])
    return thinned


def estimate_clip_fps(frames: list[AnalysisFrame], fallback: float = 12.0) -> float:
    if len(frames) < 2:
        return float(fallback)
    duration = max(0.001, float(frames[-1].timestamp) - float(frames[0].timestamp))
    return max(1.0, min(EVENT_CLIP_MAX_FPS, (len(frames) - 1) / duration))


def stable_clip_playback_fps(
    frames: list[AnalysisFrame],
    *,
    max_fps: float = EVENT_CLIP_MAX_FPS,
    fallback: float = EVENT_CLIP_MIN_PLAYBACK_FPS,
) -> float:
    limit = max(1.0, float(max_fps or EVENT_CLIP_MAX_FPS))
    minimum = min(limit, max(1.0, float(fallback or EVENT_CLIP_MIN_PLAYBACK_FPS)))
    estimated = estimate_clip_fps(frames, fallback=minimum)
    return max(1.0, min(limit, max(minimum, estimated)))


def decode_jpeg_frame(jpeg_bytes: bytes):
    array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def resample_decoded_clip_frames(
    frames: list[tuple[float, np.ndarray]],
    *,
    fps: float,
) -> list[np.ndarray]:
    if len(frames) <= 1:
        return [image for _, image in frames]
    playback_fps = max(1.0, float(fps or EVENT_CLIP_MIN_PLAYBACK_FPS))
    start_time = float(frames[0][0])
    end_time = float(frames[-1][0])
    duration = max(0.0, end_time - start_time)
    if duration <= 0:
        return [image for _, image in frames]

    frame_count = max(len(frames), int(round(duration * playback_fps)) + 1)
    output: list[np.ndarray] = []
    source_index = 0
    last_index = len(frames) - 1
    for output_index in range(frame_count):
        target_time = start_time + output_index / playback_fps
        while source_index < last_index and float(frames[source_index + 1][0]) <= target_time:
            source_index += 1
        output.append(frames[source_index][1])
    return output


def write_event_clip_mp4_with_pyav(
    frames: list[np.ndarray],
    output_path: Path,
    *,
    fps: float,
    target_size: tuple[int, int],
) -> bool:
    if av is None or not frames:
        return False
    output_path.unlink(missing_ok=True)
    width, height = target_size
    try:
        with av.open(str(output_path), mode="w") as container:
            stream = container.add_stream(
                "libx264",
                rate=max(1, int(round(float(fps or EVENT_CLIP_MIN_PLAYBACK_FPS)))),
            )
            stream.width = int(width)
            stream.height = int(height)
            stream.pix_fmt = "yuv420p"
            for image in frames:
                frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception:
        output_path.unlink(missing_ok=True)
        return False
    return output_path.exists() and output_path.stat().st_size > 0


def build_event_clip_mp4(
    frames: list[AnalysisFrame],
    output_path: Path,
    *,
    max_fps: float = EVENT_CLIP_MAX_FPS,
) -> bool:
    if not frames:
        return False
    selected = select_event_clip_frames(frames, frames[0].timestamp, frames[-1].timestamp, max_fps=max_fps)
    decoded_frames: list[tuple[float, np.ndarray]] = []
    target_size: tuple[int, int] | None = None
    for item in selected:
        image = decode_jpeg_frame(item.jpeg)
        if image is None:
            continue
        if target_size is None:
            target_size = (int(image.shape[1]), int(image.shape[0]))
        elif (int(image.shape[1]), int(image.shape[0])) != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        decoded_frames.append((float(item.timestamp), image))
    if not decoded_frames or target_size is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.stem}.tmp.mp4")
    fps = stable_clip_playback_fps(
        [AnalysisFrame(timestamp=timestamp, jpeg=b"") for timestamp, _ in decoded_frames],
        max_fps=max_fps,
    )
    playback_frames = resample_decoded_clip_frames(decoded_frames, fps=fps)
    if not write_event_clip_mp4_with_pyav(playback_frames, temp_path, fps=fps, target_size=target_size):
        return False
    temp_path.replace(output_path)
    return True


def parse_http_range(range_header: str, file_size: int) -> tuple[int, int] | None:
    value = str(range_header or "").strip()
    if not value:
        return None
    if not value.lower().startswith("bytes="):
        raise ValueError("Unsupported range unit")
    range_value = value.split("=", 1)[1].split(",", 1)[0].strip()
    if "-" not in range_value:
        raise ValueError("Invalid range")
    start_text, end_text = range_value.split("-", 1)
    if start_text == "":
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise ValueError("Invalid suffix range")
        start = max(file_size - suffix_length, 0)
        end = file_size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
        if start == file_size and end_text == "" and file_size > 0:
            start = file_size - 1
            end = file_size - 1
    if start < 0 or end < start or start >= file_size:
        raise ValueError("Range not satisfiable")
    return start, min(end, file_size - 1)


def guess_video_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".m4v"}:
        return "video/mp4"
    if suffix == ".mov":
        return "video/quicktime"
    if suffix == ".webm":
        return "video/webm"
    if suffix == ".avi":
        return "video/x-msvideo"
    if suffix == ".mkv":
        return "video/x-matroska"
    return "application/octet-stream"


def build_file_range_response(path: Path, range_header: str | None = None, media_type: str | None = None) -> Response:
    file_size = path.stat().st_size
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store",
    }
    resolved_media_type = media_type or "application/octet-stream"
    byte_range = parse_http_range(range_header or "", file_size)
    if byte_range is None:
        headers["Content-Length"] = str(file_size)
        return FileResponse(path, media_type=resolved_media_type, headers=headers)
    start, end = byte_range
    length = end - start + 1
    with path.open("rb") as handle:
        handle.seek(start)
        content = handle.read(length)
    headers.update(
        {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(length),
        }
    )
    return Response(content=content, status_code=206, media_type=resolved_media_type, headers=headers)


def build_mp4_range_response(path: Path, range_header: str | None = None) -> Response:
    return build_file_range_response(path, range_header, media_type="video/mp4")
