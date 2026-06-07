from __future__ import annotations

import asyncio
from fractions import Fraction
import io
import time
from collections.abc import Callable

try:
    from aiortc import RTCPeerConnection, RTCRtpSender, RTCSessionDescription, VideoStreamTrack
    from av import VideoFrame
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - optional runtime feature
    RTCPeerConnection = None  # type: ignore[assignment]
    RTCRtpSender = None  # type: ignore[assignment]
    RTCSessionDescription = None  # type: ignore[assignment]
    VideoStreamTrack = object  # type: ignore[assignment,misc]
    VideoFrame = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]


WEBRTC_AVAILABLE = (
    RTCPeerConnection is not None
    and RTCSessionDescription is not None
    and VideoFrame is not None
    and Image is not None
)


async def wait_for_ice_gathering_complete(pc, timeout: float = 2.0) -> None:
    if getattr(pc, "iceGatheringState", "complete") == "complete":
        return
    complete = asyncio.Event()

    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange() -> None:
        if getattr(pc, "iceGatheringState", "") == "complete":
            complete.set()

    if getattr(pc, "iceGatheringState", "") == "complete":
        return
    try:
        await asyncio.wait_for(complete.wait(), timeout=max(0.1, float(timeout or 2.0)))
    except TimeoutError:
        return


def fit_size_to_max_edge(width: int, height: int, max_edge: int) -> tuple[int, int]:
    width = max(1, int(width or 1))
    height = max(1, int(height or 1))
    max_edge = max(0, int(max_edge or 0))
    longest = max(width, height)
    if max_edge <= 0 or longest <= max_edge:
        return width, height
    ratio = max_edge / float(longest)
    return max(1, int(width * ratio)), max(1, int(height * ratio))


def ordered_video_codecs(codecs: list[object], preferred_names: tuple[str, ...] = ("H264", "VP8")) -> list[object]:
    preferred: list[object] = []
    fallback: list[object] = []
    seen: set[int] = set()
    normalized_preferences = tuple(name.upper() for name in preferred_names)
    for name in normalized_preferences:
        for codec in codecs:
            mime_type = str(getattr(codec, "mimeType", "") or "")
            codec_name = mime_type.split("/")[-1].upper()
            if codec_name == name and id(codec) not in seen:
                preferred.append(codec)
                seen.add(id(codec))
    for codec in codecs:
        if id(codec) not in seen:
            fallback.append(codec)
    return preferred + fallback


class LatestJpegVideoTrack(VideoStreamTrack):  # type: ignore[misc]
    def __init__(
        self,
        get_frame: Callable[[], object | None],
        *,
        fps: float = 12.0,
        max_width: int = 0,
        stale_after_seconds: float = 5.0,
        label: str = "detectWarning WebRTC",
    ) -> None:
        super().__init__()
        self._get_frame = get_frame
        self._fps = max(1.0, min(float(fps or 12.0), 60.0))
        self._max_width = max(0, min(int(max_width or 0), 3840))
        self._stale_after_seconds = max(0.5, float(stale_after_seconds or 5.0))
        self._label = label
        self._last_emit_at = 0.0
        self._last_good_frame: object | None = None
        self._last_good_at = 0.0
        self._cached_source_id: int | None = None
        self._cached_source_kind = ""
        self._cached_rgb_frame = None
        self._cached_image = None
        self._cached_placeholder = None
        self._timestamp = 0
        self._time_base = Fraction(1, 90000)
        self._timestamp_step = max(1, int(90000 / self._fps))

    async def recv(self):
        interval = 1.0 / self._fps
        elapsed = time.monotonic() - self._last_emit_at
        if elapsed < interval:
            await asyncio.sleep(interval - elapsed)
        self._last_emit_at = time.monotonic()

        pts, time_base = self._next_timestamp()
        source_frame = self._read_latest_frame()
        if source_frame is not None:
            frame = await asyncio.to_thread(self._frame_from_source, source_frame)
        else:
            frame = await asyncio.to_thread(self._placeholder_frame)
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def _next_timestamp(self) -> tuple[int, Fraction]:
        pts = self._timestamp
        self._timestamp += self._timestamp_step
        return pts, self._time_base

    def _read_latest_frame(self) -> object | None:
        try:
            frame = self._get_frame()
        except Exception:
            frame = None
        now = time.monotonic()
        if frame is not None:
            self._last_good_frame = frame
            self._last_good_at = now
            return frame
        if self._last_good_frame is not None and now - self._last_good_at <= self._stale_after_seconds:
            return self._last_good_frame
        return None

    def _frame_from_source(self, source_frame: object):
        if isinstance(source_frame, (bytes, bytearray, memoryview)):
            return self._frame_from_jpeg(bytes(source_frame))
        return self._frame_from_ndarray(source_frame)

    def _frame_from_jpeg(self, jpeg_bytes: bytes):
        if not WEBRTC_AVAILABLE:
            raise RuntimeError("WebRTC dependencies are not installed.")
        source_id = id(jpeg_bytes)
        if self._cached_source_id == source_id and self._cached_source_kind == "jpeg" and self._cached_image is not None:
            image = self._cached_image
        else:
            image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            target = fit_size_to_max_edge(image.width, image.height, self._max_width)
            if target != image.size:
                image = image.resize(target, Image.Resampling.BILINEAR)
            self._cached_source_id = source_id
            self._cached_source_kind = "jpeg"
            self._cached_image = image
            self._cached_rgb_frame = None
        return VideoFrame.from_image(image)

    def _frame_from_ndarray(self, frame_array: object):
        if not WEBRTC_AVAILABLE:
            raise RuntimeError("WebRTC dependencies are not installed.")
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("numpy is required for ndarray WebRTC frames.") from exc
        source_id = id(frame_array)
        if self._cached_source_id == source_id and self._cached_source_kind == "ndarray" and self._cached_rgb_frame is not None:
            return VideoFrame.from_ndarray(self._cached_rgb_frame, format="rgb24")
        array = np.asarray(frame_array)
        if array.ndim != 3 or array.shape[2] < 3:
            raise RuntimeError("WebRTC ndarray frame must be HxWx3.")
        target = fit_size_to_max_edge(array.shape[1], array.shape[0], self._max_width)
        if target != (array.shape[1], array.shape[0]):
            try:
                import cv2

                array = cv2.resize(array, target, interpolation=cv2.INTER_LINEAR)
            except Exception:
                pass
        rgb = np.ascontiguousarray(array[:, :, :3][:, :, ::-1])
        self._cached_source_id = source_id
        self._cached_source_kind = "ndarray"
        self._cached_rgb_frame = rgb
        self._cached_image = None
        return VideoFrame.from_ndarray(rgb, format="rgb24")

    def _placeholder_frame(self):
        if not WEBRTC_AVAILABLE:
            raise RuntimeError("WebRTC dependencies are not installed.")
        if self._cached_placeholder is not None:
            return VideoFrame.from_image(self._cached_placeholder)
        image = Image.new("RGB", (960, 540), (15, 23, 42))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 960, 540), fill=(15, 23, 42))
        draw.text((36, 38), self._label, fill=(226, 232, 240))
        draw.text((36, 78), "waiting for inference frame", fill=(148, 163, 184))
        self._cached_placeholder = image
        return VideoFrame.from_image(image)


class WebRTCConnectionManager:
    def __init__(self) -> None:
        self._connections: set[RTCPeerConnection] = set() if RTCPeerConnection is not None else set()
        self._last_offer_at = 0.0
        self._last_success_at = 0.0
        self._last_error_at = 0.0
        self._last_error = ""
        self._last_closed_at = 0.0

    @property
    def active_count(self) -> int:
        self._discard_inactive_connections()
        return len(self._connections)

    def snapshot(self) -> dict:
        self._discard_inactive_connections()
        now = time.time()
        return {
            "available": WEBRTC_AVAILABLE,
            "active": len(self._connections),
            "lastOfferAgeSeconds": round(now - self._last_offer_at, 2) if self._last_offer_at else None,
            "lastSuccessAgeSeconds": round(now - self._last_success_at, 2) if self._last_success_at else None,
            "lastErrorAgeSeconds": round(now - self._last_error_at, 2) if self._last_error_at else None,
            "lastClosedAgeSeconds": round(now - self._last_closed_at, 2) if self._last_closed_at else None,
            "lastError": self._last_error,
        }

    def _discard_inactive_connections(self) -> None:
        stale = [
            pc
            for pc in self._connections
            if getattr(pc, "connectionState", "") in {"failed", "closed", "disconnected"}
            or getattr(pc, "iceConnectionState", "") in {"failed", "closed", "disconnected"}
        ]
        for pc in stale:
            self._connections.discard(pc)

    async def create_answer(self, *, sdp: str, offer_type: str, track: LatestJpegVideoTrack) -> dict[str, str]:
        if not WEBRTC_AVAILABLE:
            raise RuntimeError("aiortc/Pillow dependencies are required for WebRTC streaming.")
        self._last_offer_at = time.time()
        self._discard_inactive_connections()
        pc = RTCPeerConnection()
        self._connections.add(pc)
        sender = pc.addTrack(track)
        self._prefer_stable_video_codecs(pc, sender)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if pc.connectionState in {"failed", "closed", "disconnected"}:
                self._last_closed_at = time.time()
                if pc.connectionState == "failed":
                    self._last_error_at = self._last_closed_at
                    self._last_error = "peer_connection_failed"
                await self.close(pc)

        try:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=offer_type))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await wait_for_ice_gathering_complete(pc)
        except Exception as exc:
            self._last_error_at = time.time()
            self._last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            await self.close(pc)
            raise
        self._last_success_at = time.time()
        self._last_error = ""
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }

    def _prefer_stable_video_codecs(self, pc, sender) -> None:
        if RTCRtpSender is None:
            return
        try:
            capabilities = RTCRtpSender.getCapabilities("video")
            codecs = ordered_video_codecs(list(getattr(capabilities, "codecs", []) or ()))
            if not codecs:
                return
            for transceiver in pc.getTransceivers():
                if getattr(transceiver, "sender", None) is sender:
                    transceiver.setCodecPreferences(codecs)
                    return
        except Exception:
            return

    async def close(self, pc) -> None:
        if pc in self._connections:
            self._connections.discard(pc)
            self._last_closed_at = time.time()
        try:
            await pc.close()
        except Exception:
            pass

    async def close_all(self) -> None:
        connections = list(self._connections)
        self._connections.clear()
        for pc in connections:
            try:
                await pc.close()
            except Exception:
                pass

