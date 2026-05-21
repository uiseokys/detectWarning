from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import platform
import queue
import secrets
import shutil
import socket
import string
import subprocess
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter, sleep
from types import SimpleNamespace
from urllib.parse import quote
import wave

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import Body
from fastapi.responses import HTMLResponse, Response
from fastapi import FastAPI, HTTPException, Query, Request

from action_realtime import RealtimeActionRecognizer
from detector import FaceDetector, POSE_CONNECTIONS, PersonDetector
from person_classifier import PersonPresenceFilter
from risk_analyzer import RiskAnalyzer
from tracker import PersonTracker

try:
    import psutil
except Exception:
    psutil = None


@dataclass
class ClientSession:
    client_id: str
    tracker: PersonTracker
    person_filter: PersonPresenceFilter
    risk_analyzer: RiskAnalyzer
    last_seen: float
    latest_frame_jpeg: bytes | None = None
    latest_people: list[dict] | None = None
    latest_tracked_people: list[dict] | None = None
    latest_meta: dict | None = None
    last_detection_log_at: float = 0.0
    last_backend_event_at: float = 0.0
    last_backend_event_signature: str = ""
    frame_index: int = 0


@dataclass
class AudioJob:
    client_id: str
    wav_bytes: bytes


@dataclass
class ServerSpeechResult:
    status: str
    transcript: str = ""
    audio_level: float = 0.0


PAIRING_CODE_ALPHABET = string.ascii_uppercase + string.digits
SUPPORTED_ACTION_LABELS = {"violence", "collapse", "loitering"}


def env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def load_local_env_files() -> list[str]:
    loaded: list[str] = []
    env_paths = [
        Path(".env"),
        Path("clova.env"),
        Path("training_data") / "action_pipeline_aihub" / "clova.env",
    ]
    for env_path in env_paths:
        if not env_path.exists():
            continue
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lstrip("\ufeff")
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        loaded.append(str(env_path))
    return loaded


def generate_server_instance_id(host: str | None = None, node: int | None = None) -> str:
    raw_host = str(host if host is not None else socket.gethostname()).strip().lower()
    raw_node = str(node if node is not None else uuid.getnode()).strip()
    seed = f"{raw_host}:{raw_node}".encode("utf-8")
    return hashlib.sha1(seed).hexdigest()[:32]


def normalize_connection_code(code: object) -> str:
    return "".join(ch for ch in str(code or "").upper() if ch.isalnum())[:12]


def sanitize_client_id(value: object) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value or "").strip())[:48]
    return safe if len(safe) >= 3 else f"client-{secrets.token_hex(3).upper()}"


def build_client_cctv_code(server_code: str, client_id: str) -> str:
    seed = f"{normalize_connection_code(server_code)[:12]}:{sanitize_client_id(client_id)}".encode("utf-8")
    return base64.b32encode(hashlib.sha1(seed).digest()).decode("ascii").rstrip("=")[:6]


def build_server_identity_path() -> Path:
    return Path("training_data") / "action_pipeline_aihub" / "server_identity.json"


def load_server_identity(path: Path | None = None) -> dict:
    identity_path = path or build_server_identity_path()
    try:
        with identity_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def save_server_identity(payload: dict, path: Path | None = None) -> None:
    identity_path = path or build_server_identity_path()
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = identity_path.with_name(f".{identity_path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    temp_path.replace(identity_path)


def load_or_create_server_identity(path: Path | None = None) -> dict:
    identity_path = path or build_server_identity_path()
    payload = load_server_identity(identity_path)
    payload.pop("pairing_code", None)
    server_instance_id = str(payload.get("server_instance_id") or "").strip()
    if len(server_instance_id) < 16:
        server_instance_id = generate_server_instance_id()
        payload["server_instance_id"] = server_instance_id
    if not isinstance(payload.get("cctv"), dict):
        payload["cctv"] = {}
    if not isinstance(payload.get("camera_cctvs"), dict):
        payload["camera_cctvs"] = {}
    save_server_identity(payload, identity_path)
    return payload


def save_server_cctv_settings(settings: dict, path: Path | None = None) -> None:
    identity_path = path or build_server_identity_path()
    payload = load_or_create_server_identity(identity_path)
    payload["cctv"] = dict(settings)
    save_server_identity(payload, identity_path)


class SystemMonitor:
    def __init__(self, args, audio_job_queue: queue.Queue) -> None:
        self.args = args
        self.audio_job_queue = audio_job_queue
        self._lock = threading.Lock()
        self._snapshot = self._collect_snapshot()
        if psutil is not None:
            psutil.cpu_percent(interval=None)

    def start(self) -> None:
        threading.Thread(target=self._run, name="system-monitor", daemon=True).start()

    def get_snapshot(self) -> dict:
        with self._lock:
            snapshot = dict(self._snapshot)
        snapshot["audio_queue_size"] = self.audio_job_queue.qsize()
        return snapshot

    def _run(self) -> None:
        while True:
            snapshot = self._collect_snapshot()
            with self._lock:
                self._snapshot = snapshot
            threading.Event().wait(1.0)

    def _collect_snapshot(self) -> dict:
        snapshot = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "stt_provider": self.args.stt_provider,
            "yolo_device": self.args.yolo_device,
            "stt_device": self.args.stt_device,
            "stt_compute_type": self.args.stt_compute_type,
            "stt_beam_size": self.args.stt_beam_size,
            "stt_best_of": self.args.stt_best_of,
            "audio_queue_size": self.audio_job_queue.qsize(),
            "cpu_percent": None,
            "memory_percent": None,
            "memory_used_gb": None,
            "memory_total_gb": None,
            "gpu_name": "",
            "gpu_utilization_percent": None,
            "gpu_memory_percent": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_temperature_c": None,
            "gpu_power_watts": None,
            "gpu_status": "unavailable",
        }

        if psutil is not None:
            try:
                memory = psutil.virtual_memory()
                snapshot["cpu_percent"] = round(psutil.cpu_percent(interval=None), 1)
                snapshot["memory_percent"] = round(memory.percent, 1)
                snapshot["memory_used_gb"] = round(memory.used / (1024 ** 3), 1)
                snapshot["memory_total_gb"] = round(memory.total / (1024 ** 3), 1)
            except Exception:
                pass

        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            snapshot["gpu_status"] = "nvidia-smi not found"
            return snapshot

        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=True,
            )
            first_line = result.stdout.strip().splitlines()[0]
            parts = [part.strip() for part in first_line.split(",")]
            if len(parts) >= 7:
                snapshot["gpu_name"] = parts[0]
                snapshot["gpu_utilization_percent"] = _to_float(parts[1])
                snapshot["gpu_memory_percent"] = _to_float(parts[2])
                snapshot["gpu_memory_used_mb"] = _to_float(parts[3])
                snapshot["gpu_memory_total_mb"] = _to_float(parts[4])
                snapshot["gpu_temperature_c"] = _to_float(parts[5])
                snapshot["gpu_power_watts"] = _to_float(parts[6])
                snapshot["gpu_status"] = "ok"
        except Exception as exc:
            snapshot["gpu_status"] = f"error: {exc}"

        return snapshot


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def normalize_audio_for_stt(audio: np.ndarray, *, target_rms: float = 0.075, max_gain: float = 8.0) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.98:
        audio = audio / max(peak, 1e-6) * 0.96
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    if 0.0 < rms < target_rms:
        audio = audio * min(target_rms / max(rms, 1e-6), max_gain)
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def decode_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        sample_rate = int(wav_file.getframerate() or 16000)
        channels = int(wav_file.getnchannels() or 1)
        frames = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1 and audio.size:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32, copy=False), sample_rate


def encode_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate or 16000))
        wav_file.writeframes(pcm.tobytes())
    return output.getvalue()


def clova_language_code(language: str) -> str:
    normalized = normalize_language(language)
    if normalized == "en":
        return "Eng"
    if normalized == "ja":
        return "Jpn"
    if normalized == "zh":
        return "Chn"
    return "Kor"


class ClovaCsrClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        language: str,
        timeout_seconds: float,
    ) -> None:
        self.client_id = client_id.strip()
        self.client_secret = client_secret.strip()
        self.language = clova_language_code(language)
        self.timeout_seconds = max(float(timeout_seconds), 1.0)
        self.enabled = bool(self.client_id and self.client_secret)
        self.last_error = ""

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if not self.enabled:
            raise RuntimeError("CLOVA credentials are not configured.")
        payload = encode_wav_bytes(audio, sample_rate)
        if len(payload) > 3 * 1024 * 1024:
            raise RuntimeError("CLOVA CSR request is larger than 3MB.")
        headers = {
            "Content-Type": "application/octet-stream",
            "x-ncp-apigw-api-key-id": self.client_id,
            "x-ncp-apigw-api-key": self.client_secret,
        }
        last_error = ""
        for attempt in range(2):
            try:
                response = requests.post(
                    f"https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang={self.language}",
                    data=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt == 0:
                    last_error = f"HTTP {response.status_code}"
                    sleep(0.12)
                    continue
                if response.status_code >= 400:
                    raise RuntimeError(f"CLOVA CSR failed: HTTP {response.status_code} {response.text[:160]}")
                payload_json = response.json()
                self.last_error = ""
                return str(payload_json.get("text", "")).strip()
            except requests.Timeout as exc:
                last_error = f"timeout: {exc}"
                if attempt == 0:
                    continue
                raise RuntimeError(f"CLOVA CSR timeout: {exc}") from exc
            except requests.RequestException as exc:
                last_error = f"network: {exc}"
                if attempt == 0:
                    sleep(0.12)
                    continue
                raise RuntimeError(f"CLOVA CSR network error: {exc}") from exc
        self.last_error = last_error
        return ""


class ServerSpeechRecognizer:
    def __init__(
        self,
        model_size: str,
        compute_type: str,
        language: str,
        device: str,
        beam_size: int,
        best_of: int,
        no_speech_threshold: float,
        provider: str,
        clova_client_id: str,
        clova_client_secret: str,
        clova_timeout_seconds: float,
    ) -> None:
        self.provider = provider.strip().lower()
        self.language = normalize_language(language)
        self.device = device
        self.beam_size = beam_size
        self.best_of = best_of
        self.no_speech_threshold = no_speech_threshold
        self.last_provider = "none"
        self.clova = ClovaCsrClient(
            client_id=clova_client_id,
            client_secret=clova_client_secret,
            language=language,
            timeout_seconds=clova_timeout_seconds,
        )
        self.use_clova = self.provider in {"clova", "clova-whisper"}
        self.use_whisper = self.provider in {"whisper", "clova-whisper"} or not self.use_clova
        self.initial_prompt = (
            "?쒓뎅??CCTV ?꾪뿕?곹솴 媛먯? ?뚯꽦?낅땲?? "
            "二쇱슂 ?쒗쁽: ?대젮二쇱꽭?? ?꾩?二쇱꽭?? 寃쎌같 遺덈윭二쇱꽭?? ?좉퀬?댁＜?몄슂, "
            "?섏? 留덉꽭?? 洹몃쭔?? 硫덉떠, ?ㅼ? 留? ?먮?吏 留? ?볦븘以? "
            "?뚮━吏 留? ?뚭퀬 媛吏 留? ?⑹튂, 移? 二쎌뿬踰꾨┫嫄곗빞, 媛留????? "
            "二쎄퀬 ?띕떎, ?먯궡, ?꾪뿕?댁슂."
        )
        self.model = None
        if self.use_whisper:
            try:
                from faster_whisper import WhisperModel
            except Exception as exc:
                raise RuntimeError(
                    "faster-whisper瑜?遺덈윭?ㅼ? 紐삵뻽?듬땲?? `pip install -r requirements.txt`瑜??뺤씤??二쇱꽭??"
                ) from exc
            self.model = WhisperModel(
                model_size_or_path=model_size,
                device=device,
                compute_type=compute_type,
            )
        self._lock = threading.Lock()

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> tuple[str, float]:
        audio, sample_rate = decode_wav_bytes(wav_bytes)
        if len(audio) == 0:
            self.last_provider = "empty_audio"
            return "", 0.0

        audio_level = float(np.sqrt(np.mean(np.square(audio))))
        duration_seconds = len(audio) / float(sample_rate or 16000)
        if duration_seconds < 0.45:
            self.last_provider = "audio_too_short"
            return "", audio_level
        if audio_level < 0.0015:
            self.last_provider = "audio_too_quiet"
            return "", audio_level
        audio = normalize_audio_for_stt(audio)
        if self.use_clova and self.clova.enabled:
            try:
                transcript = self.clova.transcribe(audio, sample_rate)
                if transcript:
                    self.last_provider = "clova"
                    return transcript, audio_level
                self.last_provider = "clova_empty"
            except Exception:
                self.last_provider = "clova_error"
                if not self.use_whisper:
                    raise
        elif self.use_clova and not self.clova.enabled:
            self.last_provider = "clova_missing_credentials"
        if self.model is None:
            if not self.last_provider:
                self.last_provider = "none"
            return "", audio_level
        with self._lock:
            try:
                segments, _info = self.model.transcribe(
                    audio,
                    language=self.language,
                    vad_filter=True,
                    vad_parameters={
                        "threshold": 0.35,
                        "min_speech_duration_ms": 160,
                        "min_silence_duration_ms": 350,
                        "speech_pad_ms": 220,
                    },
                    beam_size=self.beam_size,
                    best_of=self.best_of,
                    no_speech_threshold=self.no_speech_threshold,
                    condition_on_previous_text=False,
                    initial_prompt=self.initial_prompt,
                    temperature=0.0,
                )
            except Exception:
                segments, _info = self.model.transcribe(
                    audio,
                    language=self.language,
                    vad_filter=False,
                    beam_size=self.beam_size,
                    best_of=self.best_of,
                    no_speech_threshold=self.no_speech_threshold,
                    condition_on_previous_text=False,
                    initial_prompt=self.initial_prompt,
                    temperature=0.0,
                )
        transcript = " ".join(
            segment.text.strip() for segment in segments if segment.text.strip()
        ).strip()
        self.last_provider = "whisper"
        return transcript, audio_level


def normalize_language(language: str) -> str:
    normalized = language.strip()
    if "-" in normalized:
        normalized = normalized.split("-", 1)[0]
    return normalized.lower() or "ko"


def enqueue_latest_audio_job(audio_job_queue: queue.Queue, job: AudioJob) -> bool:
    """Queue speech chunks in order, dropping only the oldest chunk when overloaded."""
    try:
        with audio_job_queue.mutex:
            if audio_job_queue._qsize() >= audio_job_queue.maxsize:
                audio_job_queue._get()
                audio_job_queue.unfinished_tasks = max(0, audio_job_queue.unfinished_tasks - 1)
            audio_job_queue._put(job)
            audio_job_queue.unfinished_tasks += 1
            audio_job_queue.not_empty.notify()
            audio_job_queue.not_full.notify()
        return True
    except Exception:
        try:
            audio_job_queue.put_nowait(job)
            return True
        except queue.Full:
            return False


def build_realtime_event_log_path() -> Path:
    return Path("training_data") / "action_pipeline_aihub" / "realtime_detection_events.jsonl"


def should_log_detection_event(session: ClientSession, risk_score: int, transcript: str, action_label: str) -> bool:
    now = monotonic()
    important = bool(transcript) or risk_score >= 20 or (action_label and action_label != "normal")
    if not important:
        return False
    if risk_score >= 45 or transcript:
        min_interval = 1.0
    else:
        min_interval = 3.0
    if now - session.last_detection_log_at < min_interval:
        return False
    session.last_detection_log_at = now
    return True


def write_detection_event_log(payload: dict) -> None:
    path = build_realtime_event_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        pass



def parse_detection_event_datetime(event: dict) -> datetime:
    value = str(event.get("timestamp") or event.get("occurred_at") or event.get("occurredAt") or "").strip()
    if value:
        try:
            normalized = value.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized).astimezone(timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def detection_event_is_risk(event: dict) -> bool:
    try:
        score = int(event.get("risk_score", event.get("riskScore", 0)) or 0)
    except Exception:
        score = 0
    level = str(event.get("risk_level", event.get("riskLevel", "")) or "").upper()
    action_label = str(event.get("action_label", "") or "").lower()
    categories = event.get("risk_categories", []) or []
    if action_label in SUPPORTED_ACTION_LABELS:
        return True
    if level in {"ELEVATED", "MEDIUM", "HIGH", "CRITICAL", "WARNING", "DANGER"}:
        return True
    return score >= 20 or bool(categories)


def detection_event_class(event: dict) -> str:
    action_label = str(event.get("action_label", "") or "").lower()
    if action_label == "collapse":
        return "fall"
    if action_label in SUPPORTED_ACTION_LABELS:
        return action_label
    categories = event.get("risk_categories", []) or []
    if isinstance(categories, list):
        for category in categories:
            text = str(category or "").lower()
            if "fall" in text or "collapse" in text:
                return "fall"
            if "violence" in text:
                return "violence"
            if "loiter" in text:
                return "loitering"
    source = str(event.get("source", "") or "").lower()
    if source == "audio":
        return "audio_risk"
    return "abnormal"


def detection_stats_points(counter: Counter) -> list[dict]:
    if not counter:
        return []
    peak = max(counter.values())
    return [
        {"label": label, "count": int(count), "isPeak": int(count) == peak}
        for label, count in sorted(counter.items())
    ]


def event_int(event: dict, *names: str, default: int = 0) -> int:
    for name in names:
        if name in event:
            try:
                return int(event.get(name) or 0)
            except Exception:
                return default
    return default


def event_bool(event: dict, *names: str) -> bool:
    for name in names:
        value = event.get(name)
        if isinstance(value, bool):
            return value
        if str(value).strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def build_detection_effect_summary(events: list[dict]) -> dict:
    total = len(events)
    if total <= 0:
        return {
            "totalEvents": 0,
            "audioLiftedEvents": 0,
            "audioOnlySuspicionEvents": 0,
            "videoOnlyDangerEvents": 0,
            "averageVideoOnlyScore": 0.0,
            "averageRiskScore": 0.0,
            "averageAudioVideoGain": 0.0,
            "maxAudioVideoGain": 0,
            "stageCounts": {},
            "audioLiftRate": 0.0,
        }

    stage_counts: Counter = Counter()
    audio_lifted = 0
    audio_only_suspicion = 0
    video_only_danger = 0
    total_video_only = 0
    total_risk = 0
    total_gain = 0
    max_gain = 0

    for event in events:
        stage = str(event.get("risk_stage", event.get("riskStage", "")) or "")
        signal_source = str(event.get("risk_signal_source", event.get("riskSignalSource", "")) or "")
        gain = event_int(event, "risk_audio_video_gain", "audioVideoGain")
        video_only = event_int(event, "risk_video_only_score", "videoOnlyScore", "risk_video_score", "videoScore")
        score = event_int(event, "risk_score", "riskScore")
        lifted = event_bool(event, "risk_audio_lifted", "audioLiftedRisk") or gain > 0

        if stage:
            stage_counts[stage] += 1
        if lifted:
            audio_lifted += 1
        if stage == "위험 의심" and signal_source == "audio":
            audio_only_suspicion += 1
        if stage == "위험" and signal_source == "video":
            video_only_danger += 1
        total_video_only += video_only
        total_risk += score
        total_gain += gain
        max_gain = max(max_gain, gain)

    return {
        "totalEvents": total,
        "audioLiftedEvents": audio_lifted,
        "audioOnlySuspicionEvents": audio_only_suspicion,
        "videoOnlyDangerEvents": video_only_danger,
        "averageVideoOnlyScore": round(total_video_only / total, 1),
        "averageRiskScore": round(total_risk / total, 1),
        "averageAudioVideoGain": round(total_gain / total, 1),
        "maxAudioVideoGain": max_gain,
        "stageCounts": dict(stage_counts),
        "audioLiftRate": round(audio_lifted / total * 100.0, 1),
    }


def build_detection_stats(client_id: str = "") -> dict:
    by_day = Counter()
    by_week = Counter()
    by_month = Counter()
    by_class = Counter()
    risk_events: list[dict] = []
    target_client_id = str(client_id or "").strip()
    path = build_realtime_event_log_path()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if target_client_id and str(event.get("client_id", "") or "") != target_client_id:
                        continue
                    if not detection_event_is_risk(event):
                        continue
                    occurred_at = parse_detection_event_datetime(event)
                    by_day[occurred_at.strftime("%Y-%m-%d")] += 1
                    by_week[occurred_at.strftime("%G-W%V")] += 1
                    by_month[occurred_at.strftime("%Y-%m")] += 1
                    by_class[detection_event_class(event)] += 1
                    risk_events.append(event)
        except Exception:
            pass
    return {
        "daily": detection_stats_points(by_day),
        "weekly": detection_stats_points(by_week),
        "monthly": detection_stats_points(by_month),
        "byClass": dict(by_class),
        "voiceLift": build_detection_effect_summary(risk_events),
        "source": "detectWarning",
    }


def build_app_backend_url(base_url: str, path: str) -> str:
    base = str(base_url or "").rstrip("/")
    suffix = str(path or "").strip() or "/"
    if not suffix.startswith("/"):
        suffix = "/" + suffix
    return base + suffix


def build_client_frame_url(public_base_url: str, client_id: str) -> str:
    base = str(public_base_url or "").strip().rstrip("/")
    if not base:
        return ""
    safe_client_id = quote(sanitize_client_id(client_id), safe="")
    return f"{base}/api/client/{safe_client_id}/frame.jpg"


def infer_public_base_url(args) -> str:
    configured = env_first("INFERENCE_PUBLIC_BASE_URL", "DETECTWARNING_PUBLIC_BASE_URL", default="")
    if configured:
        return configured.strip().rstrip("/")
    explicit = str(getattr(args, "public_base_url", "") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    host = str(getattr(args, "host", "") or "").strip()
    port = int(getattr(args, "port", 8001) or 8001)
    if host and host not in {"0.0.0.0", "::", "127.0.0.1", "localhost"}:
        return f"http://{host}:{port}"
    hostname = socket.gethostname().strip()
    return f"http://{hostname}:{port}" if hostname else ""


def post_app_backend_json(args, path: str, payload: dict) -> bool:
    if bool(getattr(args, "disable_app_backend_sync", False)):
        return False
    base_url = str(getattr(args, "app_backend_base_url", "") or "").strip()
    token = str(getattr(args, "app_backend_token", "") or "").strip()
    if not base_url or not token:
        return False
    try:
        response = requests.post(
            build_app_backend_url(base_url, path),
            json=payload,
            headers={
                "X-Inference-Token": token,
                "Content-Type": "application/json",
            },
            timeout=float(getattr(args, "app_backend_timeout_seconds", 2.5) or 2.5),
        )
        return 200 <= response.status_code < 300
    except Exception:
        return False


def post_app_backend_json_async(args, path: str, payload: dict) -> None:
    if bool(getattr(args, "disable_app_backend_sync", False)):
        return
    thread = threading.Thread(
        target=post_app_backend_json,
        args=(args, path, payload),
        daemon=True,
        name="app-backend-post",
    )
    thread.start()


def map_backend_risk_level(level: str, score: int) -> str:
    if score >= 75:
        return "danger"
    if score >= 45:
        return "warning"
    return "normal"


def map_backend_risk_class(action_label: str, categories: list[str], audio_risk: bool) -> str:
    label = str(action_label or "").lower()
    if label == "collapse":
        return "fall"
    if label == "violence":
        return "violence"
    if label == "loitering":
        return "loitering"
    if label and label not in {"normal", "unknown"}:
        return "abnormal"
    if audio_risk:
        return "audio_risk"
    for category in categories or []:
        text = str(category).lower()
        if "collapse" in text or "fall" in text or "?곕윭" in text:
            return "fall"
        if "violence" in text or "??뻾" in text:
            return "violence"
    return "abnormal"


def classify_user_risk_category(text: str) -> str | None:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return None

    collapse_tokens = (
        "collapse",
        "fall",
        "fallen",
        "lying",
        "쓰러",
        "넘어",
        "실신",
        "기절",
        "장시간",
    )
    violence_tokens = (
        "violence",
        "fight",
        "assault",
        "hit",
        "punch",
        "kick",
        "폭행",
        "폭력",
        "몸싸움",
        "밀치",
        "위협",
    )
    loitering_tokens = (
        "loitering",
        "intrusion",
        "wander",
        "배회",
        "침입",
        "서성",
        "주변을 맴",
    )
    audio_tokens = (
        "audio",
        "speech",
        "voice",
        "clova",
        "matched",
        "발화",
        "음성",
        "비명",
        "도와",
        "살려",
        "구해",
    )

    if any(token in normalized for token in collapse_tokens):
        return "쓰러짐 의심"
    if any(token in normalized for token in violence_tokens):
        return "폭력/몸싸움 의심"
    if any(token in normalized for token in loitering_tokens):
        return "배회/침입 의심"
    if any(token in normalized for token in audio_tokens):
        return "위험 음성"
    if normalized.startswith("action:abnormal") or normalized == "abnormal":
        return "이상행동 의심"
    return None


def action_danger_category(label: str) -> str:
    mapping = {
        "violence": "폭력/몸싸움 위험",
        "collapse": "쓰러짐 위험",
        "loitering": "배회/침입 위험",
    }
    return mapping.get(str(label or "").lower(), "위험")


def unique_limited(values: list[str], limit: int = 4) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def build_user_risk_presentation(risk, action_result=None) -> dict:
    action_label = str(getattr(action_result, "label", "") or "")
    action_label = action_label if action_label in SUPPORTED_ACTION_LABELS else ""
    score = int(getattr(risk, "score", 0) or 0)
    audio_score = int(getattr(risk, "audio_score", 0) or 0)
    video_score = int(getattr(risk, "video_score", 0) or 0)
    audio_gain = int(getattr(risk, "audio_video_gain", 0) or 0)
    audio_confirmed_class = str(getattr(risk, "audio_confirmed_class", "") or "")
    match_quality = str(getattr(risk, "speech_match_quality", "") or "").lower()
    raw_categories = [str(value) for value in getattr(risk, "categories", []) or []]
    raw_context = [str(value) for value in getattr(risk, "context_flags", []) or []]

    audio_risk = audio_score >= 20 or match_quality in {"weak", "medium", "strong", "critical"}
    audio_suspicion = audio_score >= 45 or (
        score >= 45 and match_quality in {"medium", "strong", "critical"}
    )
    has_loud_audio = any("loud" in value.lower() or "고성" in value or "큰 소리" in value for value in raw_context + raw_categories)
    audio_lifted = audio_gain >= 10 or (audio_confirmed_class in SUPPORTED_ACTION_LABELS)

    if action_label and (video_score >= 35 or score >= 45):
        category = action_danger_category(action_label)
        stage = "위험"
        source = "audio_video" if audio_risk or audio_lifted else "video"
        reasons = [f"영상에서 {category} 감지"]
        if audio_risk or audio_lifted:
            reasons.append("음성 신호로 위험 판단 강화")
        return {
            "stage": stage,
            "signal_source": source,
            "audio_lifted": bool(audio_risk or audio_lifted),
            "categories": [category],
            "reasons": unique_limited(reasons, limit=4),
        }

    if audio_suspicion and not action_label:
        return {
            "stage": "위험 의심",
            "signal_source": "audio",
            "audio_lifted": True,
            "categories": ["음성 기반 위험 의심"],
            "reasons": ["영상에서는 확정하지 못했지만 음성 위험 신호 감지"],
        }

    if score >= 20 or audio_score > 0 or has_loud_audio:
        category = "큰 소리 감지" if audio_score > 0 or has_loud_audio else "불명확한 이상 신호"
        return {
            "stage": "주의",
            "signal_source": "audio" if audio_score > 0 or has_loud_audio else "mixed",
            "audio_lifted": False,
            "categories": [category],
            "reasons": ["위험으로 확정하기에는 부족한 신호 감지"],
        }

    raw_values: list[str] = []
    if action_label in SUPPORTED_ACTION_LABELS:
        raw_values.append(f"action:{action_label}")
    raw_values.extend(str(value) for value in getattr(risk, "categories", []) or [])
    raw_values.extend(str(value) for value in getattr(risk, "context_flags", []) or [])
    raw_values.extend(str(value) for value in getattr(risk, "reasons", []) or [])
    if is_audio_risk_signal_detected(risk):
        raw_values.append("audio:risk")

    categories = unique_limited(
        [category for category in (classify_user_risk_category(value) for value in raw_values) if category],
        limit=4,
    )
    if not categories and int(getattr(risk, "score", 0) or 0) >= 45:
        categories = ["위험 신호"]

    reasons: list[str] = []
    for category in categories:
        if category == "위험 음성":
            reasons.append("음성에서 위험 신호 감지")
        elif category == "위험 신호":
            reasons.append("복합 위험 신호 감지")
        else:
            reasons.append(f"영상에서 {category}")
    if int(getattr(risk, "audio_score", 0) or 0) > 0 and int(getattr(risk, "video_score", 0) or 0) > 0:
        reasons.append("영상+음성 신호 동시 감지")

    return {
        "stage": "정상" if not categories else "주의",
        "signal_source": "none" if not categories else "mixed",
        "audio_lifted": False,
        "categories": unique_limited(categories, limit=4),
        "reasons": unique_limited(reasons, limit=4),
    }


def build_backend_video_reason(risk, action_result) -> str:
    presentation = build_user_risk_presentation(risk, action_result)
    reasons = presentation["reasons"] or presentation["categories"]
    return ", ".join(reasons[:4]) or "위험 신호 감지"


def build_backend_risk_evidence(risk, action_result, presentation: dict) -> list[dict]:
    evidence: list[dict] = []
    action_label = str(getattr(action_result, "label", "") or "")
    if action_label in SUPPORTED_ACTION_LABELS:
        evidence.append(
            {
                "source": "video",
                "type": "action_class",
                "label": action_danger_category(action_label),
                "confidence": round(float(getattr(action_result, "confidence", 0.0) or 0.0), 4),
            }
        )
    if bool(presentation.get("audio_lifted")) or int(getattr(risk, "audio_score", 0) or 0) >= 20:
        evidence.append(
            {
                "source": "audio",
                "type": "risk_signal",
                "label": "음성 위험 신호",
                "score": int(getattr(risk, "audio_score", 0) or 0),
            }
        )
    if int(getattr(risk, "audio_video_gain", 0) or 0) > 0:
        evidence.append(
            {
                "source": "fusion",
                "type": "audio_video_gain",
                "label": "음성 결합으로 위험도 상승",
                "score": int(getattr(risk, "audio_video_gain", 0) or 0),
            }
        )
    return evidence[:4]


def build_backend_cctv_payload(
    cctv_code: str,
    *,
    name: str = "",
    location: str = "",
    status: str = "?뺤긽",
    latest_risk_score: int = 0,
    stream_url: str | None = None,
    inference_stream_url: str | None = None,
    client_id: str | None = None,
    inference_source_url: str | None = None,
) -> dict:
    code = normalize_connection_code(cctv_code)[:6]
    return {
        "code": code,
        "name": str(name or f"detectWarning CCTV {code}"),
        "location": str(location or socket.gethostname()),
        "status": str(status or "?뺤긽"),
        "latestRiskScore": int(max(0, min(100, latest_risk_score))),
        "streamUrl": stream_url or None,
        "inferenceStreamUrl": inference_stream_url or None,
        "clientId": str(client_id or "") or None,
        "inferenceSourceUrl": str(inference_source_url or "") or None,
    }


def is_audio_risk_signal_detected(risk) -> bool:
    audio_score = int(getattr(risk, "audio_score", 0) or 0)
    match_quality = str(getattr(risk, "speech_match_quality", "") or "").lower()
    return audio_score >= 20 or match_quality in {"weak", "medium", "strong", "critical"}


def build_backend_event_payload(cctv_code: str, risk, action_result=None) -> dict:
    categories = list(getattr(risk, "categories", []) or [])
    action_label = str(getattr(action_result, "label", "") or "")
    audio_risk = is_audio_risk_signal_detected(risk)
    score = int(getattr(risk, "score", 0) or 0)
    presentation = build_user_risk_presentation(risk, action_result)
    return {
        "cctvCode": normalize_connection_code(cctv_code)[:6],
        "riskLevel": map_backend_risk_level(str(getattr(risk, "level", "") or ""), score),
        "riskClass": map_backend_risk_class(action_label, categories, audio_risk),
        "riskScore": score,
        "videoReason": ", ".join((presentation["reasons"] or presentation["categories"])[:4]) or "위험 신호 감지",
        "audioRiskSignalDetected": bool(audio_risk),
        "riskStage": str(presentation.get("stage") or "정상"),
        "riskCategories": list(presentation.get("categories") or []),
        "riskReasons": list(presentation.get("reasons") or []),
        "riskSignalSource": str(presentation.get("signal_source") or "none"),
        "audioLiftedRisk": bool(presentation.get("audio_lifted", False)),
        "videoOnlyScore": int(getattr(risk, "video_only_score", getattr(risk, "video_score", 0)) or 0),
        "videoScore": int(getattr(risk, "video_score", 0) or 0),
        "audioScore": int(getattr(risk, "audio_score", 0) or 0),
        "audioVideoGain": int(getattr(risk, "audio_video_gain", 0) or 0),
        "audioConfirmedClass": str(getattr(risk, "audio_confirmed_class", "") or ""),
        "riskEvidence": build_backend_risk_evidence(risk, action_result, presentation),
        "snapshotUrl": None,
        "clipUrl": None,
    }


def should_send_backend_event(session: ClientSession, payload: dict) -> bool:
    score = int(payload.get("riskScore", 0) or 0)
    level = str(payload.get("riskLevel", "") or "")
    if score < 45:
        return False

    now = monotonic()
    signature = "|".join(
        [
            level,
            str(payload.get("riskStage", "")),
            str(payload.get("riskClass", "")),
            str(score // 10),
            str(bool(payload.get("audioRiskSignalDetected", False))),
        ]
    )
    if signature == session.last_backend_event_signature and now - session.last_backend_event_at < 10.0:
        return False
    if now - session.last_backend_event_at < 4.0 and score < 75:
        return False

    session.last_backend_event_at = now
    session.last_backend_event_signature = signature
    return True


def build_adaptive_upload_hint(
    latency_ms: float,
    risk_score: int,
    action_label: str,
    active_client_count: int = 1,
) -> dict:
    active_clients = max(int(active_client_count or 1), 1)
    risk_active = risk_score >= 45 or (action_label and action_label != "normal")
    if latency_ms >= 1200:
        width = 640 if active_clients >= 2 else 720
        return {"max_fps": 3.0, "frame_width": width, "reason": "server_latency_overloaded"}
    if active_clients >= 3:
        if risk_active:
            return {"max_fps": 12.0, "frame_width": 840, "reason": "risk_active_multi_camera"}
        if latency_ms >= 700:
            return {"max_fps": 4.0, "frame_width": 640, "reason": "server_latency_high_multi_camera"}
        return {"max_fps": 8.0, "frame_width": 720, "reason": "balanced_multi_camera"}
    if active_clients >= 2:
        if risk_active:
            return {"max_fps": 16.0, "frame_width": 960, "reason": "risk_active_multi_camera"}
        if latency_ms >= 900:
            return {"max_fps": 4.0, "frame_width": 720, "reason": "server_latency_high_multi_camera"}
        if latency_ms >= 550:
            return {"max_fps": 8.0, "frame_width": 840, "reason": "server_latency_medium_multi_camera"}
        return {"max_fps": 14.0, "frame_width": 840, "reason": "balanced_multi_camera"}
    if risk_active:
        return {"max_fps": 24.0, "frame_width": 1120, "reason": "risk_active"}
    if latency_ms >= 900:
        return {"max_fps": 5.0, "frame_width": 840, "reason": "server_latency_high"}
    if latency_ms >= 550:
        return {"max_fps": 8.0, "frame_width": 960, "reason": "server_latency_medium"}
    return {"max_fps": 20.0, "frame_width": 960, "reason": "balanced"}


def draw_server_overlay(frame, client_id: str, faces, latency_ms: float, people_count: int) -> None:
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
        cv2.putText(
            frame,
            "Face",
            (x, max(y - 10, 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 200, 0),
            2,
        )

    cv2.putText(
        frame,
        f"client: {client_id} | people: {people_count} | latency: {latency_ms:.0f}ms",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )


def draw_action_overlay(frame, action_result) -> None:
    if action_result is None or not getattr(action_result, "available", False):
        status = getattr(action_result, "status", "unavailable") if action_result is not None else "unavailable"
        reason = getattr(action_result, "reason", "") if action_result is not None else ""
        text = f"Action AI: {status}"
        if reason:
            text += f" ({reason[:34]})"
        color = (180, 180, 180)
    else:
        label = localize_action_label(str(action_result.label))
        confidence = float(action_result.confidence or 0.0)
        abnormal = float(action_result.abnormal_score or 0.0)
        text = f"Action AI: {label} {confidence:.2f} | abnormal {abnormal:.2f}"
        color = (40, 220, 120) if action_result.label == "normal" else (0, 120, 255)
    cv2.putText(frame, text, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)


def localize_action_label(label: str) -> str:
    labels = {
        "normal": "normal",
        "violence": "violence",
        "collapse": "collapse",
        "loitering": "loitering",
        "abnormal": "abnormal",
    }
    return labels.get(label, label)


def draw_pose_overlay(frame, keypoints) -> None:
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue
        start = keypoints[start_idx]
        end = keypoints[end_idx]
        if start.get("confidence", 0.0) < 0.35 or end.get("confidence", 0.0) < 0.35:
            continue
        cv2.line(
            frame,
            (int(start["x"]), int(start["y"])),
            (int(end["x"]), int(end["y"])),
            (60, 200, 255),
            2,
        )

    for point in keypoints:
        if point.get("confidence", 0.0) < 0.35:
            continue
        cv2.circle(frame, (int(point["x"]), int(point["y"])), 4, (0, 120, 255), -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="?먭꺽 ?곸긽 異붾줎 ?쒕쾭瑜??ㅽ뻾?⑸땲??")
    parser.add_argument("--host", default="0.0.0.0", help="?쒕쾭 諛붿씤??二쇱냼")
    parser.add_argument("--port", type=int, default=8000, help="?쒕쾭 ?ы듃")
    parser.add_argument(
        "--public-base-url",
        default=os.environ.get("INFERENCE_PUBLIC_BASE_URL", ""),
        help="Public base URL for backend CCTV stream links. Example: http://DESKTOP-PHTP8KM:8001",
    )
    parser.add_argument(
        "--person-score-threshold",
        type=float,
        default=0.25,
        help="Minimum confidence for person detection.",
    )
    parser.add_argument(
        "--person-imgsz",
        type=int,
        default=640,
        help="YOLO ?낅젰 ?ш린",
    )
    parser.add_argument(
        "--person-detect-interval",
        type=int,
        default=2,
        help="YOLO person detection interval in frames. 1 runs every frame; 2 reuses tracking every other frame.",
    )
    parser.add_argument(
        "--enable-face-detection",
        action="store_true",
        help="Enable lightweight face counting overlay. Disabled by default for higher FPS.",
    )
    parser.add_argument(
        "--client-session-ttl",
        type=float,
        default=30.0,
        help="?대씪?댁뼵??異붿쟻 ?곹깭瑜??좎???理쒕? ?좏쑕 ?쒓컙(珥?",
    )
    parser.add_argument(
        "--show-windows",
        action="store_true",
        help="?섏떊?????移대찓???꾨젅?꾩쓣 ?곗뒪?ы깙 OpenCV 李쎌뿉 ?쒖떆?⑸땲??",
    )
    parser.add_argument(
        "--stt-provider",
        default="clova",
        choices=("whisper", "clova", "clova-whisper"),
        help="Speech-to-text provider. clova-whisper tries CLOVA first and falls back to Whisper.",
    )
    parser.add_argument(
        "--clova-client-id",
        default=env_first("NCLOUD_CLOVA_CLIENT_ID", "CLOVA_CLIENT_ID", "CLOVA_API_KEY_ID"),
        help="Naver Cloud CLOVA CSR API key ID. Defaults to NCLOUD_CLOVA_CLIENT_ID/CLOVA_CLIENT_ID.",
    )
    parser.add_argument(
        "--clova-client-secret",
        default=env_first("NCLOUD_CLOVA_CLIENT_SECRET", "CLOVA_CLIENT_SECRET", "CLOVA_API_KEY"),
        help="Naver Cloud CLOVA CSR API key. Defaults to NCLOUD_CLOVA_CLIENT_SECRET/CLOVA_CLIENT_SECRET.",
    )
    parser.add_argument(
        "--clova-timeout-seconds",
        type=float,
        default=3.5,
        help="CLOVA CSR request timeout before Whisper fallback.",
    )
    parser.add_argument(
        "--stt-model",
        default="medium",
        help="?쒕쾭 STT??Whisper 紐⑤뜽 ?ш린",
    )
    parser.add_argument(
        "--stt-compute-type",
        default="int8",
        help="Whisper compute type for server STT.",
    )
    parser.add_argument(
        "--stt-beam-size",
        type=int,
        default=3,
        help="?쒕쾭 STT beam size. ?댁닔濡?蹂댄넻 ???뺥솗?섏?留??먮젮吏묐땲??",
    )
    parser.add_argument(
        "--stt-best-of",
        type=int,
        default=3,
        help="?쒕쾭 STT best_of. ?댁닔濡?蹂댄넻 ???뺥솗?섏?留??먮젮吏묐땲??",
    )
    parser.add_argument(
        "--stt-no-speech-threshold",
        type=float,
        default=0.55,
        help="??텧?섎줉 ??留롮? ?ㅻ뵒?ㅻ? ?뚯꽦?쇰줈 媛꾩＜?⑸땲??",
    )
    parser.add_argument(
        "--stt-language",
        default="ko-KR",
        help="?쒕쾭 STT ?몄뼱 肄붾뱶. ?? ko-KR",
    )
    parser.add_argument(
        "--person-debug",
        action="store_true",
        help="Minimum confidence for person detection.",
    )
    parser.add_argument(
        "--yolo-device",
        default="cuda:0",
        help="YOLO 異붾줎 ?μ튂. ?? cuda:0, cpu",
    )
    parser.add_argument(
        "--stt-device",
        default="cuda",
        help="Whisper 異붾줎 ?μ튂. ?? cuda, cpu",
    )
    parser.add_argument(
        "--action-artifacts-dir",
        default="training_data/action_pipeline_aihub/artifacts",
        help="?숈뒿??action/RGB-I3D 紐⑤뜽 artifacts ?붾젆?곕━",
    )
    parser.add_argument(
        "--action-rgb-model",
        default="i3d_r50",
        choices=("i3d_r50", "r3d_18", "mc3_18", "r2plus1d_18"),
        help="?ㅼ떆媛?RGB feature 異붿텧 紐⑤뜽",
    )
    parser.add_argument(
        "--disable-action-model",
        action="store_true",
        help="?숈뒿??action 紐⑤뜽 ?곌껐???뺣땲??",
    )
    parser.add_argument(
        "--action-clip-seconds",
        type=float,
        default=4.0,
        help="?ㅼ떆媛?action ?먮떒???ъ슜??理쒓렐 clip 湲몄씠(珥?",
    )
    parser.add_argument(
        "--action-interval-seconds",
        type=float,
        default=2.0,
        help="action 紐⑤뜽 異붾줎 理쒖냼 媛꾧꺽(珥?",
    )
    parser.add_argument(
        "--action-normal-threshold",
        type=float,
        default=0.78,
        help="?ㅼ떆媛?action ?먮떒?먯꽌 normal濡??④만 理쒖냼 abnormal threshold. ?믪쓣?섎줉 ?ㅽ깘??以꾩뼱??땲??",
    )
    parser.add_argument(
        "--action-min-confidence",
        type=float,
        default=0.45,
        help="?댁긽?됰룞 ?쇰꺼???쒖떆?섍린 ?꾪븳 理쒖냼 遺꾨쪟 confidence.",
    )
    parser.add_argument(
        "--action-collapse-static-motion-threshold",
        type=float,
        default=4.0,
        help="collapse ?ㅽ깘 ?꾪솕???뺤? ?곹깭 motion threshold.",
    )
    parser.add_argument(
        "--action-collapse-static-abnormal-threshold",
        type=float,
        default=0.90,
        help="?뺤? ?곹깭?먯꽌 collapse濡??몄젙??理쒖냼 abnormal score.",
    )
    parser.add_argument(
        "--app-backend-base-url",
        default=os.environ.get("APP_BACKEND_BASE_URL", "https://desktop-phtp8km.tailc597eb.ts.net"),
        help="App backend base URL for CCTV sync and risk event delivery.",
    )
    parser.add_argument(
        "--app-backend-token",
        default=os.environ.get("INFERENCE_BACKEND_TOKEN", "dev-inference-token"),
        help="Inference API token. Prefer INFERENCE_BACKEND_TOKEN in production.",
    )
    parser.add_argument(
        "--app-backend-cctv-sync-path",
        default=os.environ.get("APP_BACKEND_CCTV_SYNC_PATH", "/inference/cctvs"),
        help="Best-effort CCTV registration/sync path on the app backend.",
    )
    parser.add_argument(
        "--app-backend-cctv-name",
        default=os.environ.get("APP_BACKEND_CCTV_NAME", ""),
        help="CCTV display name sent to the app backend.",
    )
    parser.add_argument(
        "--app-backend-cctv-location",
        default=os.environ.get("APP_BACKEND_CCTV_LOCATION", ""),
        help="CCTV location text sent to the app backend.",
    )
    parser.add_argument(
        "--app-backend-cctv-status",
        default=os.environ.get("APP_BACKEND_CCTV_STATUS", "normal"),
        help="CCTV status text sent to the app backend.",
    )
    parser.add_argument(
        "--app-backend-stream-url",
        default=os.environ.get("APP_BACKEND_STREAM_URL", ""),
        help="Optional streamUrl sent to the app backend.",
    )
    parser.add_argument(
        "--app-backend-inference-stream-url",
        default=os.environ.get("APP_BACKEND_INFERENCE_STREAM_URL", ""),
        help="Optional inferenceStreamUrl sent to the app backend.",
    )
    parser.add_argument(
        "--app-backend-event-path",
        default=os.environ.get("APP_BACKEND_EVENT_PATH", "/inference/events"),
        help="Risk event ingestion path on the app backend.",
    )
    parser.add_argument(
        "--app-backend-timeout-seconds",
        type=float,
        default=float(os.environ.get("APP_BACKEND_TIMEOUT_SECONDS", "2.5")),
        help="Timeout for app backend POST requests.",
    )
    parser.add_argument(
        "--disable-app-backend-sync",
        action="store_true",
        help="Disable app backend CCTV sync and event delivery.",
    )
    return parser.parse_args()


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="detectWarning Inference Server")
    person_detector = PersonDetector(
        score_threshold=args.person_score_threshold,
        resize_width=args.person_imgsz,
        device=args.yolo_device,
    )
    action_recognizer = RealtimeActionRecognizer(
        artifacts_dir=args.action_artifacts_dir,
        enabled=not bool(args.disable_action_model),
        device=args.yolo_device,
        rgb_model=args.action_rgb_model,
        clip_seconds=args.action_clip_seconds,
        min_interval_seconds=args.action_interval_seconds,
        normal_threshold=args.action_normal_threshold,
        min_action_confidence=args.action_min_confidence,
        collapse_static_motion_threshold=args.action_collapse_static_motion_threshold,
        collapse_static_abnormal_threshold=args.action_collapse_static_abnormal_threshold,
    )
    face_detector = FaceDetector() if args.enable_face_detection else None
    speech_recognizer = ServerSpeechRecognizer(
        model_size=args.stt_model,
        compute_type=args.stt_compute_type,
        language=args.stt_language,
        device=args.stt_device,
        beam_size=args.stt_beam_size,
        best_of=args.stt_best_of,
        no_speech_threshold=args.stt_no_speech_threshold,
        provider=args.stt_provider,
        clova_client_id=args.clova_client_id,
        clova_client_secret=args.clova_client_secret,
        clova_timeout_seconds=args.clova_timeout_seconds,
    )
    sessions: dict[str, ClientSession] = {}
    session_lock = threading.Lock()
    cctv_settings_lock = threading.Lock()
    server_identity = load_or_create_server_identity()
    server_instance_id = str(server_identity.get("server_instance_id", ""))
    public_base_url = infer_public_base_url(args)
    stored_cctv_settings = server_identity.get("cctv", {}) if isinstance(server_identity.get("cctv"), dict) else {}
    cctv_settings = {
        "name": str(args.app_backend_cctv_name or stored_cctv_settings.get("name") or ""),
        "location": str(args.app_backend_cctv_location or stored_cctv_settings.get("location") or ""),
        "status": str(args.app_backend_cctv_status or stored_cctv_settings.get("status") or "?뺤긽"),
        "stream_url": str(args.app_backend_stream_url or stored_cctv_settings.get("stream_url") or ""),
        "inference_stream_url": str(
            args.app_backend_inference_stream_url or stored_cctv_settings.get("inference_stream_url") or ""
        ),
    }
    stored_camera_cctvs = (
        server_identity.get("camera_cctvs", {}) if isinstance(server_identity.get("camera_cctvs"), dict) else {}
    )
    camera_cctv_settings: dict[str, dict] = {
        sanitize_client_id(client_id): dict(settings)
        for client_id, settings in stored_camera_cctvs.items()
        if isinstance(settings, dict)
    }
    audio_job_queue: queue.Queue[AudioJob] = queue.Queue(maxsize=32)
    system_monitor = SystemMonitor(args, audio_job_queue)
    system_monitor.start()

    def latest_server_risk_score() -> int:
        with session_lock:
            scores = [
                int((session.latest_meta or {}).get("risk_score", 0) or 0)
                for session in sessions.values()
                if monotonic() - session.last_seen <= args.client_session_ttl
            ]
        return max(scores, default=0)

    def active_client_count() -> int:
        now = monotonic()
        with session_lock:
            return sum(1 for session in sessions.values() if now - session.last_seen <= args.client_session_ttl)

    def multi_client_min_action_interval(count: int) -> float:
        if count >= 3:
            return 2.0
        if count >= 2:
            return 1.5
        return 1.0

    def multi_client_person_detect_interval(base_interval: int, count: int) -> int:
        if count >= 3:
            return max(base_interval, 3)
        if count >= 2:
            return max(base_interval, 2)
        return max(base_interval, 1)

    def effective_cctv_status(configured_status: str) -> str:
        with session_lock:
            has_recent_frame = any(
                session.latest_frame_jpeg is not None and monotonic() - session.last_seen <= args.client_session_ttl
                for session in sessions.values()
            )
        if not has_recent_frame:
            return "?ㅽ봽?쇱씤"
        return str(configured_status or "?뺤긽")

    def effective_client_cctv_status(client_id: str, configured_status: str) -> str:
        with session_lock:
            session = sessions.get(client_id)
            has_recent_frame = (
                session is not None
                and session.latest_frame_jpeg is not None
                and monotonic() - session.last_seen <= args.client_session_ttl
            )
        if not has_recent_frame:
            return "오프라인"
        return str(configured_status or "정상")

    def default_client_cctv_settings(client_id: str) -> dict:
        with cctv_settings_lock:
            defaults = dict(cctv_settings)
        display_id = sanitize_client_id(client_id)
        base_name = str(defaults.get("name") or "detectWarning CCTV").strip()
        return {
            "name": f"{base_name} - {display_id}",
            "location": str(defaults.get("location") or ""),
            "status": str(defaults.get("status") or "정상"),
            "stream_url": str(defaults.get("stream_url") or ""),
            "inference_stream_url": str(defaults.get("inference_stream_url") or ""),
        }

    def persist_camera_cctv_settings() -> None:
        with cctv_settings_lock:
            server_settings = dict(cctv_settings)
            camera_settings = {key: dict(value) for key, value in camera_cctv_settings.items()}
        payload = load_or_create_server_identity()
        payload["cctv"] = server_settings
        payload["camera_cctvs"] = camera_settings
        save_server_identity(payload)

    def current_client_cctv_settings(client_id: str, latest_risk_score: int | None = None) -> dict:
        safe_client_id = sanitize_client_id(client_id)
        code = build_client_cctv_code(server_instance_id, safe_client_id)
        settings = dict(default_client_cctv_settings(safe_client_id))
        with cctv_settings_lock:
            settings.update(dict(camera_cctv_settings.get(safe_client_id, {})))
        risk_score = latest_server_risk_score() if latest_risk_score is None else int(latest_risk_score or 0)
        inferred_frame_url = build_client_frame_url(public_base_url, safe_client_id)
        inference_stream_url = str(settings.get("inference_stream_url") or inferred_frame_url or "")
        stream_url = str(settings.get("stream_url") or inference_stream_url or "")
        payload = build_backend_cctv_payload(
            code,
            name=settings.get("name", ""),
            location=settings.get("location", ""),
            status=effective_client_cctv_status(client_id, settings.get("status", "정상")),
            latest_risk_score=risk_score,
            stream_url=stream_url,
            inference_stream_url=inference_stream_url,
            client_id=safe_client_id,
            inference_source_url=public_base_url,
        )
        return {
            "code": payload["code"],
            "name": payload["name"],
            "location": payload["location"],
            "status": payload["status"],
            "configuredStatus": str(settings.get("status", "정상") or "정상"),
            "latestRiskScore": payload["latestRiskScore"],
            "streamUrl": payload["streamUrl"],
            "inferenceStreamUrl": payload["inferenceStreamUrl"],
        }

    def sync_client_cctv_with_backend(client_id: str, latest_risk_score: int | None = None) -> None:
        cctv = current_client_cctv_settings(client_id, latest_risk_score=latest_risk_score)
        post_app_backend_json_async(
            args,
            args.app_backend_cctv_sync_path,
            build_backend_cctv_payload(
                cctv["code"],
                name=cctv.get("name", ""),
                location=cctv.get("location", ""),
                status=cctv.get("status", "정상"),
                latest_risk_score=int(cctv.get("latestRiskScore", 0) or 0),
                stream_url=cctv.get("streamUrl", ""),
                inference_stream_url=cctv.get("inferenceStreamUrl", ""),
                client_id=client_id,
                inference_source_url=public_base_url,
            ),
        )

    def remember_client_cctv_meta(session: ClientSession, client_id: str, latest_risk_score: int | None = None) -> dict:
        cctv = current_client_cctv_settings(client_id, latest_risk_score=latest_risk_score)
        if session.latest_meta is None:
            session.latest_meta = {}
        session.latest_meta.update(
            {
                "cctv_code": cctv["code"],
                "cctv_name": cctv["name"],
                "cctv_location": cctv["location"],
                "cctv_status": cctv["status"],
            }
        )
        return cctv

    def current_cctv_settings() -> dict:
        with cctv_settings_lock:
            settings = dict(cctv_settings)
        risk_score = latest_server_risk_score()
        return {
            "code": "",
            "name": str(settings.get("name", "") or ""),
            "location": str(settings.get("location", "") or ""),
            "status": effective_cctv_status(settings.get("status", "?뺤긽")),
            "configuredStatus": str(settings.get("status", "정상") or "정상"),
            "latestRiskScore": risk_score,
            "streamUrl": str(settings.get("stream_url", "") or ""),
            "inferenceStreamUrl": str(settings.get("inference_stream_url", "") or ""),
        }

    def send_backend_event_if_needed(session: ClientSession, risk, action_result=None) -> None:
        cctv_code = str((session.latest_meta or {}).get("cctv_code") or "")
        if not cctv_code and session.client_id:
            cctv_code = build_client_cctv_code(server_instance_id, session.client_id)
        payload = build_backend_event_payload(cctv_code, risk, action_result)
        if should_send_backend_event(session, payload):
            post_app_backend_json_async(args, args.app_backend_event_path, payload)

    def get_session(client_id: str) -> ClientSession:
        now = monotonic()
        created = False
        with session_lock:
            expired = [
                key
                for key, session in sessions.items()
                if now - session.last_seen > args.client_session_ttl
            ]
            for key in expired:
                del sessions[key]

            session = sessions.get(client_id)
            if session is None:
                session = ClientSession(
                    client_id=client_id,
                    tracker=PersonTracker(),
                    person_filter=PersonPresenceFilter(debug=args.person_debug),
                    risk_analyzer=RiskAnalyzer(),
                    last_seen=now,
                )
                sessions[client_id] = session
                created = True
            else:
                session.last_seen = now
        if created:
            remember_client_cctv_meta(session, client_id)
            sync_client_cctv_with_backend(client_id)
        return session

    def list_client_summaries() -> list[dict]:
        with session_lock:
            items = []
            for client_id, session in sessions.items():
                meta = session.latest_meta or {}
                items.append(
                    {
                        "client_id": client_id,
                        "last_seen_seconds": round(monotonic() - session.last_seen, 1),
                        "people_count": int(meta.get("people_count", 0)),
                        "face_count": int(meta.get("face_count", 0)),
                        "face_detection_enabled": bool(meta.get("face_detection_enabled", False)),
                        "latency_ms": float(meta.get("latency_ms", 0.0)),
                        "speech_status": str(meta.get("speech_status", "idle")),
                        "speech_provider": str(meta.get("speech_provider", "")),
                        "risk_score": int(meta.get("risk_score", 0)),
                        "risk_audio_score": int(meta.get("risk_audio_score", 0)),
                        "risk_video_score": int(meta.get("risk_video_score", 0)),
                        "risk_raw_score": int(meta.get("risk_raw_score", 0)),
                        "risk_video_only_score": int(meta.get("risk_video_only_score", meta.get("risk_video_score", 0))),
                        "risk_audio_video_gain": int(meta.get("risk_audio_video_gain", 0)),
                        "risk_audio_confirmed_class": str(meta.get("risk_audio_confirmed_class", "")),
                        "risk_stage": str(meta.get("risk_stage", "정상")),
                        "risk_signal_source": str(meta.get("risk_signal_source", "none")),
                        "risk_audio_lifted": bool(meta.get("risk_audio_lifted", False)),
                        "risk_level": str(meta.get("risk_level", "LOW")),
                        "risk_level_label": localize_risk_level(str(meta.get("risk_level", "LOW"))),
                        "risk_categories": list(meta.get("risk_categories", [])),
                        "action_label": str(meta.get("action_label", "unknown")),
                        "action_label_label": str(meta.get("action_label_label", localize_action_label(str(meta.get("action_label", "unknown"))))),
                        "action_confidence": float(meta.get("action_confidence", 0.0)),
                        "action_abnormal_score": float(meta.get("action_abnormal_score", 0.0)),
                        "action_available": bool(meta.get("action_available", False)),
                        "client_type": str(meta.get("client_type", "uploader")),
                        "device_name": str(meta.get("device_name", "")),
                        "paired": bool(meta.get("paired", False)),
                        "cctv_code": str(meta.get("cctv_code", "")),
                        "cctv_name": str(meta.get("cctv_name", "")),
                        "cctv_status": str(meta.get("cctv_status", "")),
                        "has_frame": session.latest_frame_jpeg is not None,
                    }
                )
            return sorted(items, key=lambda item: item["client_id"])

    def render_dashboard() -> str:
        return """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>detectWarning Dashboard</title>
  <style>
    :root {
      --bg-top: #f8fbff;
      --bg-bottom: #ecf2f8;
      --panel: rgba(255, 255, 255, 0.88);
      --panel-strong: rgba(255, 255, 255, 0.96);
      --panel-soft: rgba(248, 251, 255, 0.82);
      --ink: #0f172a;
      --muted: #64748b;
      --line: rgba(148, 163, 184, 0.22);
      --line-strong: rgba(148, 163, 184, 0.35);
      --accent: #2563eb;
      --accent-soft: rgba(37, 99, 235, 0.12);
      --success: #059669;
      --success-soft: rgba(5, 150, 105, 0.12);
      --warn: #d97706;
      --warn-soft: rgba(217, 119, 6, 0.12);
      --danger: #dc2626;
      --danger-soft: rgba(220, 38, 38, 0.12);
      --shadow-lg: 0 24px 60px rgba(15, 23, 42, 0.10);
      --shadow-md: 0 12px 28px rgba(15, 23, 42, 0.08);
      --shadow-sm: 0 8px 18px rgba(15, 23, 42, 0.06);
      --radius-xl: 28px;
      --radius-lg: 22px;
      --radius-md: 18px;
      --radius-sm: 14px;
    }
    * {
      box-sizing: border-box;
    }
    html, body {
      min-height: 100%;
    }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "SF Pro Display", "Pretendard", "SUIT", "Apple SD Gothic Neo", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.10), transparent 24%),
        linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
    }
    .wrap {
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px 28px 36px;
    }
    .hero {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 24px;
    }
    .hero-copy {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.66);
      border: 1px solid rgba(37, 99, 235, 0.14);
      color: #1d4ed8;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      backdrop-filter: blur(12px);
    }
    .eyebrow::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 0 6px rgba(37, 99, 235, 0.10);
    }
    .hero h1 {
      margin: 0;
      font-size: clamp(34px, 4vw, 48px);
      line-height: 1.02;
      letter-spacing: -0.04em;
      font-weight: 800;
    }
    .hero p {
      margin: 0;
      max-width: 760px;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.65;
    }
    .hero-summary {
      min-width: 240px;
      padding: 16px 18px;
      border-radius: 20px;
      background: linear-gradient(145deg, rgba(255, 255, 255, 0.92), rgba(244, 248, 255, 0.82));
      border: 1px solid rgba(148, 163, 184, 0.22);
      box-shadow: var(--shadow-md);
      backdrop-filter: blur(14px);
    }
    .hero-summary-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .hero-summary-value {
      font-size: 18px;
      font-weight: 700;
      line-height: 1.4;
    }
    .system-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }
    .system-card,
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow-sm);
      backdrop-filter: blur(16px);
    }
    .system-card {
      padding: 18px;
      min-height: 132px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .system-card-wide {
      grid-column: span 2;
    }
    .system-title {
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .system-value {
      font-size: 15px;
      font-weight: 700;
      line-height: 1.6;
      white-space: pre-line;
      color: var(--ink);
    }
    .cctv-settings {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) 112px auto;
      gap: 8px;
      align-items: center;
    }
    .cctv-input {
      width: 100%;
      min-width: 0;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.82);
      color: var(--ink);
      font-size: 14px;
      font-weight: 700;
      outline: none;
    }
    .cctv-input:focus {
      border-color: rgba(37, 99, 235, 0.48);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.10);
    }
    .cctv-button {
      padding: 10px 14px;
      border-radius: 10px;
      border: 1px solid rgba(37, 99, 235, 0.28);
      background: #2563eb;
      color: #fff;
      font-size: 13px;
      font-weight: 800;
      cursor: pointer;
    }
    .cctv-button:disabled {
      cursor: default;
      opacity: 0.58;
    }
    .cctv-status {
      grid-column: 1 / -1;
      min-height: 18px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .dashboard-grid {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      overflow: hidden;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 20px 22px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.72), rgba(255, 255, 255, 0.52));
    }
    .panel-head-copy {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .panel-kicker {
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .panel-title {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }
    .panel-note {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: var(--panel-soft);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .panel-note::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 0 5px rgba(5, 150, 105, 0.10);
    }
    .clients {
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      max-height: 76vh;
      overflow: auto;
    }
    .clients::-webkit-scrollbar {
      width: 10px;
    }
    .clients::-webkit-scrollbar-thumb {
      background: rgba(148, 163, 184, 0.26);
      border-radius: 999px;
    }
    .client-card {
      padding: 16px;
      border-radius: 18px;
      border: 1px solid rgba(148, 163, 184, 0.16);
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(246, 249, 253, 0.88));
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
      cursor: pointer;
      transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }
    .client-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 14px 28px rgba(15, 23, 42, 0.08);
      border-color: rgba(37, 99, 235, 0.24);
    }
    .client-card.active {
      border-color: rgba(37, 99, 235, 0.46);
      box-shadow:
        inset 0 0 0 1px rgba(37, 99, 235, 0.16),
        0 18px 34px rgba(37, 99, 235, 0.12);
      background: linear-gradient(180deg, rgba(255, 255, 255, 1), rgba(239, 246, 255, 0.96));
    }
    .client-top {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .client-name {
      font-size: 15px;
      font-weight: 700;
      line-height: 1.45;
      word-break: break-all;
    }
    .client-id-line {
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      line-height: 1.45;
      word-break: break-all;
    }
    .client-badges {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 6px;
    }
    .mini-badge,
    .status-pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      min-height: 32px;
      padding: 7px 12px;
      border-radius: 999px;
      border: 1px solid transparent;
      font-size: 12px;
      font-weight: 700;
      line-height: 1;
      white-space: nowrap;
    }
    .mini-badge::before,
    .status-pill::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: currentColor;
      opacity: 0.85;
    }
    .tone-neutral {
      color: #475569;
      background: rgba(148, 163, 184, 0.10);
      border-color: rgba(148, 163, 184, 0.18);
    }
    .tone-good {
      color: var(--success);
      background: var(--success-soft);
      border-color: rgba(5, 150, 105, 0.18);
    }
    .tone-warn {
      color: var(--warn);
      background: var(--warn-soft);
      border-color: rgba(217, 119, 6, 0.18);
    }
    .tone-danger {
      color: var(--danger);
      background: var(--danger-soft);
      border-color: rgba(220, 38, 38, 0.18);
    }
    .tone-accent {
      color: var(--accent);
      background: var(--accent-soft);
      border-color: rgba(37, 99, 235, 0.18);
    }
    .client-stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }
    .client-stat {
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(248, 250, 252, 0.9);
      border: 1px solid rgba(148, 163, 184, 0.12);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .client-stat strong {
      display: block;
      color: var(--ink);
      font-size: 15px;
      margin-top: 2px;
    }
    .client-transcript,
    .client-categories {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.6;
    }
    .client-transcript {
      min-height: 38px;
      margin-bottom: 8px;
    }
    .client-empty {
      padding: 18px;
      border-radius: 16px;
      border: 1px dashed rgba(148, 163, 184, 0.28);
      color: var(--muted);
      text-align: center;
      font-size: 14px;
      background: rgba(255, 255, 255, 0.56);
    }
    .viewer {
      padding: 20px 22px 22px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }
    .metric-card {
      min-height: 98px;
      padding: 14px 16px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 252, 0.84));
      border: 1px solid rgba(148, 163, 184, 0.14);
      box-shadow: 0 10px 22px rgba(15, 23, 42, 0.04);
      display: flex;
      flex-direction: column;
      gap: 8px;
      justify-content: space-between;
    }
    .metric-label {
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .metric-value {
      font-size: 20px;
      font-weight: 800;
      letter-spacing: -0.03em;
      line-height: 1.3;
      color: var(--ink);
    }
    .metric-value.metric-compact {
      font-size: 17px;
      font-weight: 700;
    }
    .metric-value.status-pill {
      width: fit-content;
      max-width: 100%;
      font-size: 13px;
      font-weight: 700;
    }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .detail-card {
      min-height: 106px;
      padding: 16px 18px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid rgba(148, 163, 184, 0.14);
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
    }
    .detail-card-wide {
      grid-column: 1 / -1;
    }
    .client-cctv-settings {
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) 120px auto;
    }
    .detail-label {
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .detail-value {
      color: var(--ink);
      font-size: 15px;
      font-weight: 700;
      line-height: 1.65;
      word-break: keep-all;
    }
    .gain-bar {
      height: 9px;
      overflow: hidden;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.18);
      margin-top: 10px;
    }
    .gain-bar-fill {
      height: 100%;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--accent), var(--success));
      transition: width 220ms ease;
    }
    .screen-shell {
      padding: 16px;
      border-radius: 24px;
      background:
        radial-gradient(circle at top right, rgba(37, 99, 235, 0.10), transparent 28%),
        linear-gradient(180deg, rgba(255, 255, 255, 0.9), rgba(243, 247, 252, 0.82));
      border: 1px solid rgba(148, 163, 184, 0.16);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6), var(--shadow-sm);
    }
    .screen-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .screen-title {
      font-size: 16px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .screen-subtitle {
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }
    .screen {
      width: 100%;
      aspect-ratio: 16 / 9;
      border-radius: 20px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.96));
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), var(--shadow-md);
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    .screen::before {
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(transparent, rgba(15, 23, 42, 0.08)),
        radial-gradient(circle at top left, rgba(96, 165, 250, 0.14), transparent 24%);
      pointer-events: none;
    }
    .screen img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #111827;
      position: relative;
      z-index: 1;
    }
    .screen-empty {
      position: relative;
      z-index: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      text-align: center;
      padding: 24px;
      color: rgba(226, 232, 240, 0.90);
    }
    .screen-empty-icon {
      width: 72px;
      height: 72px;
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.12);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 28px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }
    .screen-empty-title {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .screen-empty-copy {
      font-size: 14px;
      line-height: 1.7;
      color: rgba(226, 232, 240, 0.72);
      max-width: 420px;
    }
    @media (max-width: 1280px) {
      .metric-grid,
      .detail-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
    @media (max-width: 980px) {
      .dashboard-grid {
        grid-template-columns: 1fr;
      }
      .system-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
    @media (max-width: 720px) {
      .wrap {
        padding: 18px;
      }
      .hero {
        flex-direction: column;
        align-items: stretch;
      }
      .system-grid,
      .metric-grid,
      .detail-grid {
        grid-template-columns: 1fr;
      }
      .system-card-wide {
        grid-column: auto;
      }
      .cctv-settings {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <div class="hero-copy">
        <span class="eyebrow">AI Monitoring System</span>
        <h1>detectWarning 실시간 모니터링 대시보드</h1>
        <p>카메라 영상과 음성 위험 신호를 추론 서버가 분석하고, 현재 CCTV 상태와 위험도를 한 화면에서 확인할 수 있습니다.</p>
      </div>
      <div class="hero-summary">
        <div class="hero-summary-label">Presentation Ready</div>
        <div class="hero-summary-value">실시간 분석 화면, 위험도, 시스템 상태를 한 번에 확인</div>
      </div>
    </header>

    <section class="system-grid">
      <article class="system-card">
        <div class="system-title">Desktop</div>
        <div class="system-value" id="desktopHost">-</div>
      </article>
      <article class="system-card">
        <div class="system-title">GPU</div>
        <div class="system-value" id="desktopGpu">-</div>
      </article>
      <article class="system-card">
        <div class="system-title">CPU / RAM</div>
        <div class="system-value" id="desktopCpuRam">-</div>
      </article>
      <article class="system-card">
        <div class="system-title">Runtime</div>
        <div class="system-value" id="desktopRuntime">-</div>
      </article>
      <article class="system-card">
        <div class="system-title">Camera Codes</div>
        <div class="system-value" id="cameraCodes">카메라별 코드 사용</div>
      </article>
    </section>

    <section class="dashboard-grid">
      <aside class="panel">
        <div class="panel-head">
          <div class="panel-head-copy">
            <div class="panel-kicker">Clients</div>
            <h2 class="panel-title">연결된 클라이언트</h2>
          </div>
          <div class="panel-note">자동 갱신</div>
        </div>
        <div id="clients" class="clients"></div>
      </aside>

      <main class="panel">
        <div class="panel-head">
          <div class="panel-head-copy">
            <div class="panel-kicker">Live View</div>
            <h2 class="panel-title">실시간 분석 화면</h2>
          </div>
          <div class="panel-note">AI 분석 중</div>
        </div>
        <div class="viewer">
          <section class="metric-grid">
            <article class="metric-card">
              <div class="metric-label">클라이언트</div>
              <div class="metric-value metric-compact" id="clientName">-</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">사람 수</div>
              <div class="metric-value" id="peopleCount">0</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">Face Detect</div>
              <div class="metric-value" id="faceCount">OFF</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">지연 시간</div>
              <div class="metric-value" id="latency">0ms</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">최근 수신</div>
              <div class="metric-value metric-compact" id="lastSeen">-</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">음성 인식 상태</div>
              <div class="metric-value status-pill tone-neutral" id="speechStatus">대기</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">오디오 레벨</div>
              <div class="metric-value metric-compact" id="audioLevel">0.000</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">위험도</div>
              <div class="metric-value status-pill tone-neutral" id="riskLevel">0/100 | 낮음</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">판단 단계</div>
              <div class="metric-value status-pill tone-neutral" id="riskStage">정상</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">Video-only</div>
              <div class="metric-value metric-compact" id="riskVideoScore">0/100</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">Audio-only</div>
              <div class="metric-value metric-compact" id="riskAudioScore">0/100</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">Fusion raw</div>
              <div class="metric-value metric-compact" id="riskRawScore">0/100</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">Audio gain</div>
              <div class="metric-value metric-compact" id="riskAudioVideoGain">+0</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">Action AI</div>
              <div class="metric-value status-pill tone-neutral" id="actionLabel">-</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">Abnormal score</div>
              <div class="metric-value metric-compact" id="actionAbnormalScore">0.000</div>
            </article>
          </section>

          <section class="detail-grid">
            <article class="detail-card detail-card-wide">
              <div class="detail-label">선택 카메라 CCTV</div>
              <div class="detail-value" id="clientCctvCode">-</div>
              <div class="cctv-settings client-cctv-settings">
                <input class="cctv-input" id="clientCctvNameInput" maxlength="80" placeholder="카메라별 CCTV 이름" />
                <input class="cctv-input" id="clientCctvLocationInput" maxlength="160" placeholder="설치 위치" />
                <select class="cctv-input" id="clientCctvStatusInput">
                  <option value="정상">정상</option>
                  <option value="오프라인">오프라인</option>
                </select>
                <button class="cctv-button" id="clientCctvSave" type="button">Save</button>
                <div class="cctv-status" id="clientCctvStatus"></div>
              </div>
            </article>
            <article class="detail-card">
              <div class="detail-label">인식 내용</div>
              <div class="detail-value" id="transcript">-</div>
            </article>
            <article class="detail-card">
              <div class="detail-label">위험 카테고리</div>
              <div class="detail-value" id="riskCategories">없음</div>
            </article>
            <article class="detail-card">
              <div class="detail-label">위험 신호</div>
              <div class="detail-value" id="riskReasons">없음</div>
            </article>
            <article class="detail-card">
              <div class="detail-label">음성 기여 요약</div>
              <div class="detail-value" id="voiceLiftSummary">기록 없음</div>
              <div class="gain-bar"><div class="gain-bar-fill" id="voiceLiftBar"></div></div>
            </article>
            <article class="detail-card">
              <div class="detail-label">Speech match</div>
              <div class="detail-value" id="riskMatchQuality">-</div>
            </article>
            <article class="detail-card">
              <div class="detail-label">Action probabilities</div>
              <div class="detail-value" id="actionProbabilities">-</div>
            </article>
          </section>

          <section class="screen-shell">
            <div class="screen-head">
              <div class="screen-title">실시간 분석 프레임</div>
              <div class="screen-subtitle">Pose / Action / Risk Overlay</div>
            </div>
            <div class="screen" id="screen">
              <div class="screen-empty">
                <div class="screen-empty-icon">AI</div>
                <div class="screen-empty-title">분석 화면을 준비하는 중입니다</div>
                <div class="screen-empty-copy">카메라 클라이언트가 연결되면 실시간 영상과 분석 오버레이가 표시됩니다.</div>
              </div>
            </div>
          </section>
        </div>
      </main>
    </section>
  </div>
  <script>
    let selectedClientId = null;
    let lastClientCctvSettings = null;

    function escapeHtml(value) {
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    function speechTone(status) {
      if (status === 'recognized' || status === 'clova') return 'tone-good';
      if (status === 'processing' || status === 'clova_empty' || status === 'listening' || status === 'audio_too_short' || status === 'audio_too_quiet') return 'tone-warn';
      if (status === 'error' || status === 'clova_error' || status === 'clova_missing_credentials' || status === 'none') return 'tone-danger';
      return 'tone-neutral';
    }

    function riskTone(level) {
      if (level === 'CRITICAL') return 'tone-danger';
      if (level === 'HIGH') return 'tone-danger';
      if (level === 'MEDIUM') return 'tone-warn';
      if (level === 'ELEVATED') return 'tone-accent';
      return 'tone-neutral';
    }

    function screenPlaceholder(title, copy) {
      return `
        <div class="screen-empty">
          <div class="screen-empty-icon">AI</div>
          <div class="screen-empty-title">${escapeHtml(title)}</div>
          <div class="screen-empty-copy">${escapeHtml(copy)}</div>
        </div>
      `;
    }

    let selectedClientHasFrame = false;

    function ensureScreenImage() {
      const screen = document.getElementById('screen');
      let image = screen.querySelector('img');
      if (!image) {
        screen.innerHTML = '';
        image = document.createElement('img');
        image.alt = '遺꾩꽍 ?붾㈃';
        screen.appendChild(image);
      }
      return image;
    }

    function setScreenPlaceholder(title, copy) {
      document.getElementById('screen').innerHTML = screenPlaceholder(title, copy);
    }

    function setClientCctvStatus(message, tone = 'neutral') {
      const status = document.getElementById('clientCctvStatus');
      if (!status) return;
      status.textContent = message || '';
      status.style.color = tone === 'error' ? '#dc2626' : (tone === 'ok' ? '#059669' : '');
    }

    function updateClientCctvInputs(cctv) {
      lastClientCctvSettings = cctv || null;
      const code = document.getElementById('clientCctvCode');
      const nameInput = document.getElementById('clientCctvNameInput');
      const locationInput = document.getElementById('clientCctvLocationInput');
      const statusInput = document.getElementById('clientCctvStatusInput');
      if (code) {
        code.textContent = cctv ? `연결 코드 ${cctv.code || '------'} | ${cctv.status || '-'}` : '-';
      }
      if (nameInput && document.activeElement !== nameInput) {
        nameInput.value = cctv ? cctv.name || '' : '';
      }
      if (locationInput && document.activeElement !== locationInput) {
        locationInput.value = cctv ? cctv.location || '' : '';
      }
      if (statusInput && cctv) {
        statusInput.value = cctv.configuredStatus || cctv.status || '정상';
      }
    }

    async function saveClientCctvSettings() {
      if (!selectedClientId) return;
      const nameInput = document.getElementById('clientCctvNameInput');
      const locationInput = document.getElementById('clientCctvLocationInput');
      const statusInput = document.getElementById('clientCctvStatusInput');
      const button = document.getElementById('clientCctvSave');
      if (!nameInput || !button) return;
      button.disabled = true;
      setClientCctvStatus('Saving...');
      try {
        const response = await fetch(`/api/client/${encodeURIComponent(selectedClientId)}/cctv/settings`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            name: nameInput.value.trim(),
            location: locationInput ? locationInput.value.trim() : '',
            status: statusInput ? statusInput.value : '정상',
            streamUrl: lastClientCctvSettings ? lastClientCctvSettings.streamUrl : null,
            inferenceStreamUrl: lastClientCctvSettings ? lastClientCctvSettings.inferenceStreamUrl : null,
          }),
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        updateClientCctvInputs(data.cctv || null);
        setClientCctvStatus('Saved and synced', 'ok');
        refreshClients();
      } catch (error) {
        setClientCctvStatus('Save failed', 'error');
      } finally {
        button.disabled = false;
      }
    }

    async function refreshSystem() {
      if (document.hidden) {
        return;
      }
      const response = await fetch('/api/system');
      if (!response.ok) {
        return;
      }
      const data = await response.json();
      document.getElementById('desktopHost').textContent = `${data.hostname}\n${data.platform}`;
      const gpuLine = data.gpu_status === 'ok'
        ? `${data.gpu_name}\nGPU ${data.gpu_utilization_percent ?? 0}% | VRAM ${Math.round(data.gpu_memory_used_mb ?? 0)}/${Math.round(data.gpu_memory_total_mb ?? 0)} MB\n온도 ${data.gpu_temperature_c ?? '-'}C | 전력 ${data.gpu_power_watts ?? '-'}W`
        : `GPU 정보 없음\n${data.gpu_status}`;
      document.getElementById('desktopGpu').textContent = gpuLine;
      const cpuRamLine =
        `CPU ${data.cpu_percent ?? 0}%\nRAM ${data.memory_percent ?? 0}% (${data.memory_used_gb ?? 0}/${data.memory_total_gb ?? 0} GB)`;
      document.getElementById('desktopCpuRam').textContent = cpuRamLine;
      const runtimeLine =
        `활성 카메라 ${data.active_client_count || 0}대${data.multi_camera_mode ? ' | 멀티 카메라 절약 모드' : ''}\n오디오 큐 ${data.audio_queue_size}\nYOLO ${data.yolo_device}\nSTT ${data.stt_device} / ${data.stt_compute_type}\nbeam ${data.stt_beam_size} | best_of ${data.stt_best_of}`;
      document.getElementById('desktopRuntime').textContent = runtimeLine;
      const cameraCodesElement = document.getElementById('cameraCodes');
      if (cameraCodesElement) {
        cameraCodesElement.textContent = `각 카메라별 고유 코드 사용\n활성 카메라 ${data.active_client_count || 0}대`;
      }
    }

    async function refreshClients() {
      if (document.hidden) {
        return;
      }
      const response = await fetch('/api/clients');
      const clients = await response.json();
      const container = document.getElementById('clients');
      const previousSelectedClientId = selectedClientId;
      container.innerHTML = '';

      if (!clients.length) {
        container.innerHTML = '<div class="client-empty">아직 연결된 클라이언트가 없습니다.</div>';
        selectedClientHasFrame = false;
        setScreenPlaceholder('클라이언트를 기다리는 중입니다', '카메라 업로더를 실행하면 실시간 분석 화면이 표시됩니다.');
        updateMeta(null);
        return;
      }

      if (!selectedClientId || !clients.some(client => client.client_id === selectedClientId)) {
        selectedClientId = clients[0].client_id;
      }

      for (const client of clients) {
        const item = document.createElement('div');
        item.className = 'client-card' + (client.client_id === selectedClientId ? ' active' : '');
        const displayName = client.cctv_name || client.device_name || client.client_id;
        item.innerHTML = `
          <div class="client-top">
            <div>
              <div class="client-name">${escapeHtml(displayName)}</div>
              <div class="client-id-line">${escapeHtml(client.client_id)} · ${escapeHtml(client.cctv_code || '------')}</div>
            </div>
            <div class="client-badges">
              <span class="mini-badge ${speechTone(client.speech_status)}">${escapeHtml(client.speech_status)}</span>
              <span class="mini-badge ${riskTone(client.risk_level)}">${escapeHtml(client.risk_level_label)}</span>
            </div>
          </div>
          <div class="client-stats">
            <div class="client-stat">사람<strong>${client.people_count}</strong></div>
            <div class="client-stat">Face<strong>${client.face_detection_enabled ? client.face_count : 'OFF'}</strong></div>
            <div class="client-stat">Action<strong>${escapeHtml(client.action_label_label || '-')}</strong></div>
            <div class="client-stat">Abnormal<strong>${Number(client.action_abnormal_score || 0).toFixed(2)}</strong></div>
            <div class="client-stat">최근 수신<strong>${client.last_seen_seconds}초 전</strong></div>
            <div class="client-stat">지연<strong>${client.latency_ms.toFixed(1)}ms</strong></div>
            <div class="client-stat">CCTV<strong>${escapeHtml(client.cctv_status || '-')}</strong></div>
          </div>
          <div class="client-transcript">STT 원문은 앱 백엔드 조회 목록에 표시하지 않습니다.</div>
          <div class="client-categories">카테고리 ${escapeHtml(client.risk_categories && client.risk_categories.length ? client.risk_categories.join(', ') : '없음')}</div>
        `;
        item.onclick = () => {
          selectedClientId = client.client_id;
          selectedClientHasFrame = Boolean(client.has_frame);
          refreshClients();
          refreshSelectedClient();
        };
        container.appendChild(item);
      }

      const selectedSummary = clients.find((client) => client.client_id === selectedClientId);
      selectedClientHasFrame = Boolean(selectedSummary && selectedSummary.has_frame);
      if (previousSelectedClientId !== selectedClientId) {
        refreshSelectedClient();
      }
    }

    async function refreshSelectedClient() {
      if (document.hidden) {
        return;
      }
      if (!selectedClientId) {
        return;
      }
      const response = await fetch(`/api/client/${encodeURIComponent(selectedClientId)}`);
      if (!response.ok) {
        return;
      }
      const data = await response.json();
      selectedClientHasFrame = Boolean(data.has_frame);
      updateMeta(data);
      if (!selectedClientHasFrame) {
        setScreenPlaceholder('프레임을 기다리는 중입니다', '선택한 클라이언트에서 아직 수신된 프레임이 없습니다.');
        return;
      }
      refreshSelectedFrame();
    }

    function refreshSelectedFrame() {
      if (document.hidden) {
        return;
      }
      if (!selectedClientId) {
        return;
      }
      if (!selectedClientHasFrame) {
        return;
      }
      const image = ensureScreenImage();
      image.src = `/api/client/${encodeURIComponent(selectedClientId)}/frame.jpg?ts=${Date.now()}`;
    }

    async function refreshStats() {
      if (document.hidden) {
        return;
      }
      const suffix = selectedClientId ? `?client_id=${encodeURIComponent(selectedClientId)}` : '';
      try {
        const response = await fetch(`/api/stats${suffix}`);
        if (!response.ok) return;
        updateVoiceLiftSummary(await response.json());
      } catch (_error) {
        updateVoiceLiftSummary(null);
      }
    }

    function updateVoiceLiftSummary(stats) {
      const target = document.getElementById('voiceLiftSummary');
      const bar = document.getElementById('voiceLiftBar');
      if (!target || !bar) return;
      const summary = stats && stats.voiceLift ? stats.voiceLift : null;
      if (!summary || !summary.totalEvents) {
        target.textContent = '기록 없음';
        bar.style.width = '0%';
        return;
      }
      const rate = Number(summary.audioLiftRate || 0);
      const gain = Number(summary.averageAudioVideoGain || 0);
      target.textContent = `음성 보강 ${summary.audioLiftedEvents}/${summary.totalEvents}건 (${rate.toFixed(1)}%), 평균 +${gain.toFixed(1)}점`;
      bar.style.width = `${Math.max(0, Math.min(100, rate))}%`;
    }

    function updateMeta(data) {
      document.getElementById('clientName').textContent = data ? ((data.cctv && data.cctv.name) || data.device_name || data.client_id) : '-';
      updateClientCctvInputs(data ? data.cctv || null : null);
      document.getElementById('peopleCount').textContent = `${data ? data.people_count : 0}`;
      document.getElementById('faceCount').textContent = data && data.face_detection_enabled ? `${data.face_count}` : 'OFF';
      document.getElementById('latency').textContent = `${data ? data.latency_ms.toFixed(1) : 0}ms`;
      document.getElementById('lastSeen').textContent = data ? `${data.last_seen_seconds.toFixed(1)}초 전` : '-';

      const speechStatus = document.getElementById('speechStatus');
      speechStatus.textContent = data ? data.speech_status_label : '대기';
      speechStatus.className = `metric-value status-pill ${speechTone(data ? data.speech_status : 'idle')}`;

      document.getElementById('audioLevel').textContent = data ? data.audio_level.toFixed(3) : '0.000';

      const riskLevel = document.getElementById('riskLevel');
      riskLevel.textContent = `${data ? data.risk_score : 0}/100 | ${data ? data.risk_level_label : '낮음'}`;
      riskLevel.className = `metric-value status-pill ${riskTone(data ? data.risk_level : 'LOW')}`;
      const riskStage = document.getElementById('riskStage');
      const stageText = data && data.risk_stage ? data.risk_stage : '정상';
      riskStage.textContent = stageText;
      riskStage.className = `metric-value status-pill ${stageText === '위험' ? 'tone-danger' : stageText === '위험 의심' ? 'tone-warn' : stageText === '주의' ? 'tone-accent' : 'tone-good'}`;
      document.getElementById('riskVideoScore').textContent = `${data ? data.risk_video_score || 0 : 0}/100`;
      document.getElementById('riskAudioScore').textContent = `${data ? data.risk_audio_score || 0 : 0}/100`;
      document.getElementById('riskRawScore').textContent = `${data ? data.risk_raw_score || 0 : 0}/100`;
      const audioGain = data ? data.risk_audio_video_gain || 0 : 0;
      const confirmedClass = data && data.risk_audio_confirmed_class ? ` ${data.risk_audio_confirmed_class}` : '';
      document.getElementById('riskAudioVideoGain').textContent = `+${audioGain}${confirmedClass}`;

      const actionLabel = document.getElementById('actionLabel');
      const actionText = data && data.action_available
        ? `${data.action_label_label || data.action_label} ${(data.action_confidence || 0).toFixed(2)}`
        : (data && data.action_status ? data.action_status : '-');
      actionLabel.textContent = actionText;
      actionLabel.className = `metric-value status-pill ${data && data.action_label !== 'normal' && data.action_available ? 'tone-danger' : 'tone-good'}`;
      document.getElementById('actionAbnormalScore').textContent = data ? `${(data.action_abnormal_score || 0).toFixed(3)} / ${(data.action_threshold || 0).toFixed(3)}` : '0.000';

      document.getElementById('transcript').textContent = data && data.transcript ? data.transcript : '-';
      document.getElementById('riskCategories').textContent = data && data.risk_categories && data.risk_categories.length ? data.risk_categories.join(', ') : '없음';
      document.getElementById('riskReasons').textContent = data && data.risk_reasons && data.risk_reasons.length ? data.risk_reasons.join(', ') : '없음';
      document.getElementById('riskMatchQuality').textContent = data && data.risk_match_quality ? data.risk_match_quality : '-';
      const probabilities = data && data.action_probabilities ? data.action_probabilities : {};
      const probabilityText = Object.keys(probabilities).length
        ? Object.entries(probabilities).map(([label, value]) => `${label}: ${Number(value || 0).toFixed(3)}`).join(' | ')
        : (data && data.action_reason ? data.action_reason : '-');
      document.getElementById('actionProbabilities').textContent = probabilityText;
    }

    const params = new URLSearchParams(window.location.search);
    const requestedClientId = params.get('client_id');
    if (requestedClientId) {
      selectedClientId = requestedClientId;
    }

    const clientCctvSaveButton = document.getElementById('clientCctvSave');
    const clientCctvNameInput = document.getElementById('clientCctvNameInput');
    const clientCctvLocationInput = document.getElementById('clientCctvLocationInput');
    if (clientCctvSaveButton) {
      clientCctvSaveButton.addEventListener('click', saveClientCctvSettings);
    }
    for (const editable of [clientCctvNameInput, clientCctvLocationInput]) {
      if (!editable) continue;
      editable.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
          saveClientCctvSettings();
        }
      });
    }

    refreshSystem();
    refreshClients();
    refreshStats();
    setInterval(refreshSystem, 2000);
    setInterval(refreshClients, 1000);
    setInterval(refreshSelectedClient, 500);
    setInterval(refreshSelectedFrame, 500);
    setInterval(refreshStats, 3000);
  </script>
</body>
</html>"""

    @app.get("/health")
    def health() -> dict:
        snapshot = system_monitor.get_snapshot()
        snapshot.update(
            {
                "status": "ok",
                "sessions": len(sessions),
                "action_model_enabled": bool(action_recognizer.enabled),
                "action_model_status": action_recognizer.latest.status,
                "action_model_reason": action_recognizer.latest.reason,
            }
        )
        return snapshot

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return render_dashboard()

    @app.get("/api/clients")
    def api_clients() -> list[dict]:
        return list_client_summaries()

    @app.get("/api/stats")
    def api_stats(code: str = "", client_id: str = "") -> dict:
        return build_detection_stats(client_id=client_id)

    @app.get("/api/system")
    def api_system() -> dict:
        snapshot = system_monitor.get_snapshot()
        snapshot["sessions"] = len(sessions)
        snapshot["active_client_count"] = active_client_count()
        snapshot["multi_camera_mode"] = snapshot["active_client_count"] >= 2
        snapshot["action_model_enabled"] = bool(action_recognizer.enabled)
        snapshot["action_model_status"] = action_recognizer.latest.status
        snapshot["action_model_reason"] = action_recognizer.latest.reason
        snapshot["clova_credentials_configured"] = bool(speech_recognizer.clova.enabled)
        snapshot["app_backend_enabled"] = bool(
            str(getattr(args, "app_backend_base_url", "") or "").strip()
            and str(getattr(args, "app_backend_token", "") or "").strip()
            and not bool(getattr(args, "disable_app_backend_sync", False))
        )
        snapshot["app_backend_base_url"] = str(getattr(args, "app_backend_base_url", "") or "")
        snapshot["public_base_url"] = public_base_url
        snapshot["pairing_mode"] = "client_cctv_codes"
        snapshot["cctv"] = current_cctv_settings()
        return snapshot

    @app.get("/api/pairing")
    def api_pairing() -> dict:
        return {
            "mode": "client_cctv_codes",
            "message": "Legacy server pairing code is disabled. Use each client's cctv.code instead.",
            "clients": list_client_summaries(),
            "cctv": current_cctv_settings(),
        }

    @app.get("/api/cctv/settings")
    def api_cctv_settings() -> dict:
        return current_cctv_settings()

    @app.post("/api/cctv/settings")
    def api_cctv_settings_update(payload: dict = Body(...)) -> dict:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON body is required.")

        def clean_text(value: object, limit: int = 120) -> str:
            return str(value or "").strip()[:limit]

        with cctv_settings_lock:
            updated = dict(cctv_settings)
            if "name" in payload:
                updated["name"] = clean_text(payload.get("name"), 80)
            if "location" in payload:
                updated["location"] = clean_text(payload.get("location"), 160)
            if "status" in payload:
                updated["status"] = clean_text(payload.get("status") or "?뺤긽", 40)
            if "streamUrl" in payload or "stream_url" in payload:
                updated["stream_url"] = clean_text(payload.get("streamUrl", payload.get("stream_url")), 512)
            if "inferenceStreamUrl" in payload or "inference_stream_url" in payload:
                updated["inference_stream_url"] = clean_text(
                    payload.get("inferenceStreamUrl", payload.get("inference_stream_url")),
                    512,
                )
            cctv_settings.update(updated)
            stored_settings = dict(cctv_settings)
        save_server_cctv_settings(stored_settings)
        with session_lock:
            active_client_ids = list(sessions.keys())
        for active_client_id in active_client_ids:
            with session_lock:
                active_session = sessions.get(active_client_id)
            if active_session is not None:
                remember_client_cctv_meta(active_session, active_client_id)
            sync_client_cctv_with_backend(active_client_id)
        return {
            "ok": True,
            "cctv": current_cctv_settings(),
        }

    @app.get("/api/client/{client_id}/cctv/settings")
    def api_client_cctv_settings(client_id: str) -> dict:
        safe_client_id = sanitize_client_id(client_id)
        return {
            "client_id": safe_client_id,
            "cctv": current_client_cctv_settings(safe_client_id),
        }

    @app.post("/api/client/{client_id}/cctv/settings")
    def api_client_cctv_settings_update(client_id: str, payload: dict = Body(...)) -> dict:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON body is required.")

        def clean_text(value: object, limit: int = 120) -> str:
            return str(value or "").strip()[:limit]

        safe_client_id = sanitize_client_id(client_id)
        default_settings = default_client_cctv_settings(safe_client_id)
        with cctv_settings_lock:
            updated = dict(camera_cctv_settings.get(safe_client_id, default_settings))
            if "name" in payload:
                updated["name"] = clean_text(payload.get("name"), 80)
            if "location" in payload:
                updated["location"] = clean_text(payload.get("location"), 160)
            if "status" in payload:
                updated["status"] = clean_text(payload.get("status") or "정상", 40)
            if "streamUrl" in payload or "stream_url" in payload:
                updated["stream_url"] = clean_text(payload.get("streamUrl", payload.get("stream_url")), 512)
            if "inferenceStreamUrl" in payload or "inference_stream_url" in payload:
                updated["inference_stream_url"] = clean_text(
                    payload.get("inferenceStreamUrl", payload.get("inference_stream_url")),
                    512,
                )
            camera_cctv_settings[safe_client_id] = updated
        persist_camera_cctv_settings()
        with session_lock:
            session = sessions.get(safe_client_id)
        if session is not None:
            remember_client_cctv_meta(session, safe_client_id)
        sync_client_cctv_with_backend(safe_client_id)
        return {
            "ok": True,
            "client_id": safe_client_id,
            "cctv": current_client_cctv_settings(safe_client_id),
        }

    @app.post("/api/pairing/regenerate")
    def api_pairing_regenerate() -> dict:
        raise HTTPException(
            status_code=410,
            detail="Legacy server pairing code is disabled. Each camera/client now owns its own CCTV code.",
        )

    @app.post("/api/pair/register")
    def api_pair_register(payload: dict = Body(...)) -> dict:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON body is required.")

        raw_client_id = str(payload.get("client_id") or "").strip()
        if raw_client_id:
            safe_client_id = sanitize_client_id(raw_client_id)
        else:
            safe_client_id = f"ios-{secrets.token_hex(3).upper()}"
        if len(safe_client_id) < 3:
            safe_client_id = f"ios-{secrets.token_hex(3).upper()}"

        device_name = str(payload.get("device_name") or payload.get("name") or "iOS App").strip()[:80]
        session = get_session(safe_client_id)
        if session.latest_meta is None:
            session.latest_meta = {}
        session.latest_meta.update(
            {
                "client_type": "ios_app",
                "device_name": device_name,
                "paired": True,
                "paired_at_monotonic": round(monotonic(), 3),
            }
        )
        cctv = remember_client_cctv_meta(session, safe_client_id)
        sync_client_cctv_with_backend(safe_client_id)
        return {
            "ok": True,
            "client_id": safe_client_id,
            "device_name": device_name,
            "cctv": cctv,
            "upload": {
                "frame_url": f"/analyze/frame?client_id={safe_client_id}&lite=1",
                "audio_url": f"/analyze/audio?client_id={safe_client_id}",
            },
        }

    @app.get("/api/client/{client_id}")
    def api_client(client_id: str) -> dict:
        with session_lock:
            session = sessions.get(client_id)
            if session is None:
                raise HTTPException(status_code=404, detail="?대씪?댁뼵?몃? 李얠쓣 ???놁뒿?덈떎.")
            meta = session.latest_meta or {}
            result = {
                "client_id": client_id,
                "people_count": int(meta.get("people_count", 0)),
                "face_count": int(meta.get("face_count", 0)),
                "face_detection_enabled": bool(meta.get("face_detection_enabled", False)),
                "latency_ms": float(meta.get("latency_ms", 0.0)),
                "speech_status": str(meta.get("speech_status", "idle")),
                "speech_status_label": localize_speech_status(str(meta.get("speech_status", "idle"))),
                "speech_provider": str(meta.get("speech_provider", "")),
                "transcript": str(meta.get("transcript", "")),
                "audio_level": float(meta.get("audio_level", 0.0)),
                "risk_score": int(meta.get("risk_score", 0)),
                "risk_audio_score": int(meta.get("risk_audio_score", 0)),
                "risk_video_score": int(meta.get("risk_video_score", 0)),
                "risk_raw_score": int(meta.get("risk_raw_score", 0)),
                "risk_video_only_score": int(meta.get("risk_video_only_score", meta.get("risk_video_score", 0))),
                "risk_audio_video_gain": int(meta.get("risk_audio_video_gain", 0)),
                "risk_audio_confirmed_class": str(meta.get("risk_audio_confirmed_class", "")),
                "risk_stage": str(meta.get("risk_stage", "정상")),
                "risk_signal_source": str(meta.get("risk_signal_source", "none")),
                "risk_audio_lifted": bool(meta.get("risk_audio_lifted", False)),
                "risk_match_quality": str(meta.get("risk_match_quality", "")),
                "risk_level": str(meta.get("risk_level", "LOW")),
                "risk_level_label": localize_risk_level(str(meta.get("risk_level", "LOW"))),
                "risk_categories": list(meta.get("risk_categories", [])),
                "risk_reasons": list(meta.get("risk_reasons", [])),
                "action_status": str(meta.get("action_status", "unknown")),
                "action_label": str(meta.get("action_label", "unknown")),
                "action_label_label": str(meta.get("action_label_label", localize_action_label(str(meta.get("action_label", "unknown"))))),
                "action_confidence": float(meta.get("action_confidence", 0.0)),
                "action_abnormal_score": float(meta.get("action_abnormal_score", 0.0)),
                "action_threshold": float(meta.get("action_threshold", 0.0)),
                "action_available": bool(meta.get("action_available", False)),
                "action_reason": str(meta.get("action_reason", "")),
                "action_probabilities": dict(meta.get("action_probabilities", {}) or {}),
                "client_type": str(meta.get("client_type", "uploader")),
                "device_name": str(meta.get("device_name", "")),
                "last_seen_seconds": monotonic() - session.last_seen,
                "has_frame": session.latest_frame_jpeg is not None,
            }
        result["cctv"] = current_client_cctv_settings(client_id, int(result.get("risk_score", 0) or 0))
        return result

    @app.get("/api/client/{client_id}/frame.jpg")
    def api_client_frame(client_id: str) -> Response:
        with session_lock:
            session = sessions.get(client_id)
            if session is None:
                raise HTTPException(status_code=404, detail="?대씪?댁뼵?몃? 李얠쓣 ???놁뒿?덈떎.")
            frame_bytes = session.latest_frame_jpeg
        if not frame_bytes:
            return Response(status_code=204, headers={"Cache-Control": "no-store"})
        return Response(
            content=frame_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/analyze/audio")
    def analyze_audio(
        audio_bytes: bytes = Body(..., media_type="audio/wav"),
        client_id: str = Query(..., min_length=3, description="Per-client tracking identifier"),
    ) -> dict:
        started_at = perf_counter()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="鍮??ㅻ뵒???붿껌?낅땲??")

        session = get_session(client_id)
        latency_ms = (perf_counter() - started_at) * 1000.0
        if session.latest_meta is None:
            session.latest_meta = {}
        session.latest_meta.update({"speech_status": "processing"})
        queued = enqueue_latest_audio_job(audio_job_queue, AudioJob(client_id=client_id, wav_bytes=audio_bytes))
        if not queued:
            queued = False
            session.latest_meta.update(
                {
                    "speech_status": "error",
                    "speech_error": "?ㅻ뵒??泥섎━ ?湲곗뿴??媛??李쇱뒿?덈떎.",
                }
            )
        return {
            "client_id": client_id,
            "speech_status": "queued" if queued else "error",
            "transcript": "",
            "audio_level": 0.0,
            "latency_ms": round(latency_ms, 1),
            "queued": queued,
        }

    @app.post("/analyze/frame")
    def analyze_frame(
        image_bytes: bytes = Body(..., media_type="image/jpeg"),
        client_id: str = Query(..., min_length=3, description="Per-client tracking identifier"),
        lite: bool = Query(False, description="Return compact response for uploaders."),
    ) -> dict:
        started_at = perf_counter()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="鍮??대?吏 ?붿껌?낅땲??")

        np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="JPEG ?대?吏瑜??붿퐫?⑺븯吏 紐삵뻽?듬땲??")

        session = get_session(client_id)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous_risk_score = int((session.latest_meta or {}).get("risk_score", 0) or 0)
        session.frame_index += 1
        current_active_clients = active_client_count()
        base_detect_interval = max(int(args.person_detect_interval), 1)
        detect_interval = multi_client_person_detect_interval(base_detect_interval, current_active_clients)
        force_person_detect = previous_risk_score >= 35 or session.latest_tracked_people is None
        should_detect_person = force_person_detect or ((session.frame_index - 1) % detect_interval == 0)
        if should_detect_person:
            people = person_detector.detect(frame)
            tracked_people = session.tracker.update(people)
            session.latest_tracked_people = tracked_people
        else:
            tracked_people = session.latest_tracked_people or []
        faces = face_detector.detect(frame) if face_detector is not None else []
        evaluated_people = session.person_filter.evaluate(
            tracked_people=tracked_people,
            faces=faces,
            gray_frame=gray_frame,
            frame_shape=frame.shape,
        )
        confirmed_people = [
            person
            for person in evaluated_people
            if person.get("person_state") in {"full_body_person", "upper_body_person"}
        ]
        if previous_risk_score >= 35 or float((session.latest_meta or {}).get("audio_level", 0.0) or 0.0) >= 0.12:
            action_recognizer.min_interval_seconds = min(action_recognizer.min_interval_seconds, 1.0)
        elif not confirmed_people:
            action_recognizer.min_interval_seconds = max(action_recognizer.min_interval_seconds, 2.5)
        action_recognizer.min_interval_seconds = max(
            action_recognizer.min_interval_seconds,
            multi_client_min_action_interval(current_active_clients),
        )
        action_result = action_recognizer.update(frame, confirmed_people)
        latency_ms = (perf_counter() - started_at) * 1000.0
        annotated = frame.copy()
        session.person_filter.draw_debug_overlay(annotated, evaluated_people, draw_pose_overlay)
        draw_server_overlay(annotated, client_id, faces, latency_ms, len(confirmed_people))
        draw_action_overlay(annotated, action_result)
        success, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            session.latest_frame_jpeg = encoded.tobytes()
        session.latest_people = confirmed_people
        if session.latest_meta is None:
            session.latest_meta = {}
        risk = session.risk_analyzer.update(
            ServerSpeechResult(status=str(session.latest_meta.get("speech_status", "idle"))),
            tracked_people=confirmed_people,
            face_count=len(faces),
            action_result=action_result,
        )
        risk_presentation = build_user_risk_presentation(risk, action_result)
        cctv = remember_client_cctv_meta(session, client_id, latest_risk_score=risk.score)
        session.latest_meta.update(
            {
                "active_client_count": current_active_clients,
                "cctv_code": cctv["code"],
                "cctv_name": cctv["name"],
                "cctv_location": cctv["location"],
                "cctv_status": cctv["status"],
                "people_count": len(confirmed_people),
                "uncertain_count": sum(
                    1 for person in evaluated_people if person.get("person_state") == "uncertain"
                ),
                "candidate_count": len(evaluated_people),
                "face_count": len(faces),
                "face_detection_enabled": bool(face_detector is not None),
                "person_detect_reused": not should_detect_person,
                "person_detect_interval": detect_interval,
                "latency_ms": round(latency_ms, 1),
                "action_status": action_result.status,
                "action_label": action_result.label,
                "action_label_label": localize_action_label(action_result.label),
                "action_confidence": round(float(action_result.confidence or 0.0), 4),
                "action_abnormal_score": round(float(action_result.abnormal_score or 0.0), 4),
                "action_threshold": round(float(action_result.threshold or 0.0), 4),
                "action_available": bool(action_result.available),
                "action_reason": action_result.reason,
                "action_probabilities": action_result.probabilities or {},
                "risk_score": risk.score,
                "risk_audio_score": risk.audio_score,
                "risk_video_score": risk.video_score,
                "risk_raw_score": risk.raw_score,
                "risk_video_only_score": risk.video_only_score,
                "risk_audio_video_gain": risk.audio_video_gain,
                "risk_audio_confirmed_class": risk.audio_confirmed_class,
                "risk_stage": risk_presentation["stage"],
                "risk_signal_source": risk_presentation["signal_source"],
                "risk_audio_lifted": risk_presentation["audio_lifted"],
                "risk_match_quality": risk.speech_match_quality,
                "risk_level": risk.level,
                "risk_categories": risk_presentation["categories"],
                "risk_reasons": risk_presentation["reasons"],
                "risk_context_flags": list(risk.context_flags),
                "risk_raw_categories": list(risk.categories),
                "risk_raw_reasons": list(risk.reasons),
                "risk_raw_context_flags": list(risk.context_flags),
            }
        )
        upload_hint = build_adaptive_upload_hint(
            latency_ms,
            risk.score,
            action_result.label,
            active_client_count=current_active_clients,
        )
        session.latest_meta["upload_hint"] = upload_hint
        send_backend_event_if_needed(session, risk, action_result)
        if should_log_detection_event(
            session,
            risk.score,
            str(session.latest_meta.get("transcript", "")),
            action_result.label,
        ):
            write_detection_event_log(
                {
                    "client_id": client_id,
                    "timestamp_monotonic": round(monotonic(), 3),
                    "risk_score": risk.score,
                    "risk_audio_score": risk.audio_score,
                    "risk_video_score": risk.video_score,
                    "risk_raw_score": risk.raw_score,
                    "risk_video_only_score": risk.video_only_score,
                    "risk_audio_video_gain": risk.audio_video_gain,
                    "risk_audio_confirmed_class": risk.audio_confirmed_class,
                    "risk_stage": risk_presentation["stage"],
                    "risk_signal_source": risk_presentation["signal_source"],
                    "risk_audio_lifted": risk_presentation["audio_lifted"],
                    "risk_level": risk.level,
                    "risk_categories": risk_presentation["categories"],
                    "risk_reasons": risk_presentation["reasons"],
                    "risk_context_flags": list(risk.context_flags),
                    "risk_raw_categories": list(risk.categories),
                    "risk_raw_reasons": list(risk.reasons),
                    "matched_keywords": list(risk.matched_keywords),
                    "transcript": str(session.latest_meta.get("transcript", "")),
                    "action_label": action_result.label,
                    "action_confidence": round(float(action_result.confidence or 0.0), 4),
                    "action_abnormal_score": round(float(action_result.abnormal_score or 0.0), 4),
                    "people_count": len(confirmed_people),
                    "face_count": len(faces),
                    "latency_ms": round(latency_ms, 1),
                    "upload_hint": upload_hint,
                }
            )
        if args.show_windows:
            cv2.imshow(f"detectWarning server - {client_id}", annotated)
            cv2.waitKey(1)
        return {
            "client_id": client_id,
            "tracked_people": [] if lite else evaluated_people,
            "people_count": len(confirmed_people),
            "candidate_count": len(evaluated_people),
            "faces": [] if lite else [list(map(int, face)) for face in faces],
            "face_count": len(faces),
            "face_detection_enabled": bool(face_detector is not None),
            "person_detect_reused": not should_detect_person,
            "latency_ms": round(latency_ms, 1),
            "action": {
                "available": bool(action_result.available),
                "status": action_result.status,
                "label": action_result.label,
                "label_text": localize_action_label(action_result.label),
                "confidence": round(float(action_result.confidence or 0.0), 4),
                "abnormal_score": round(float(action_result.abnormal_score or 0.0), 4),
                "threshold": round(float(action_result.threshold or 0.0), 4),
                "reason": action_result.reason,
                "probabilities": action_result.probabilities or {},
            },
            "upload_hint": upload_hint,
        }

    def audio_worker() -> None:
        while True:
            job = audio_job_queue.get()
            session = get_session(job.client_id)
            if session.latest_meta is None:
                session.latest_meta = {}
            try:
                transcript, audio_level = speech_recognizer.transcribe_wav_bytes(job.wav_bytes)
                speech_status = "recognized" if transcript else str(speech_recognizer.last_provider or "listening")
                tracked_people = session.latest_people or []
                face_count = int((session.latest_meta or {}).get("face_count", 0))
                risk = session.risk_analyzer.update(
                    ServerSpeechResult(
                        status=speech_status,
                        transcript=transcript,
                        audio_level=audio_level,
                    ),
                    tracked_people=tracked_people,
                    face_count=face_count,
                )
                risk_presentation = build_user_risk_presentation(
                    risk,
                    SimpleNamespace(label=str(session.latest_meta.get("action_label", "normal"))),
                )
                cctv = remember_client_cctv_meta(session, job.client_id, latest_risk_score=risk.score)
                session.latest_meta.update(
                    {
                        "cctv_code": cctv["code"],
                        "cctv_name": cctv["name"],
                        "cctv_location": cctv["location"],
                        "cctv_status": cctv["status"],
                        "speech_status": speech_status,
                        "speech_provider": speech_recognizer.last_provider,
                        "transcript": transcript,
                        "audio_level": round(audio_level, 4),
                        "risk_score": risk.score,
                        "risk_audio_score": risk.audio_score,
                        "risk_video_score": risk.video_score,
                        "risk_raw_score": risk.raw_score,
                        "risk_video_only_score": risk.video_only_score,
                        "risk_audio_video_gain": risk.audio_video_gain,
                        "risk_audio_confirmed_class": risk.audio_confirmed_class,
                        "risk_stage": risk_presentation["stage"],
                        "risk_signal_source": risk_presentation["signal_source"],
                        "risk_audio_lifted": risk_presentation["audio_lifted"],
                        "risk_match_quality": risk.speech_match_quality,
                        "risk_level": risk.level,
                        "risk_categories": risk_presentation["categories"],
                        "risk_reasons": risk_presentation["reasons"],
                        "risk_context_flags": list(risk.context_flags),
                        "risk_raw_categories": list(risk.categories),
                        "risk_raw_reasons": list(risk.reasons),
                        "risk_raw_context_flags": list(risk.context_flags),
                    }
                )
                send_backend_event_if_needed(session, risk, None)
                if should_log_detection_event(session, risk.score, transcript, str(session.latest_meta.get("action_label", "normal"))):
                    write_detection_event_log(
                        {
                            "client_id": job.client_id,
                            "timestamp_monotonic": round(monotonic(), 3),
                            "source": "audio",
                            "risk_score": risk.score,
                            "risk_audio_score": risk.audio_score,
                            "risk_video_score": risk.video_score,
                            "risk_raw_score": risk.raw_score,
                            "risk_video_only_score": risk.video_only_score,
                            "risk_audio_video_gain": risk.audio_video_gain,
                            "risk_audio_confirmed_class": risk.audio_confirmed_class,
                            "risk_stage": risk_presentation["stage"],
                            "risk_signal_source": risk_presentation["signal_source"],
                            "risk_audio_lifted": risk_presentation["audio_lifted"],
                            "risk_level": risk.level,
                            "risk_categories": risk_presentation["categories"],
                            "risk_reasons": risk_presentation["reasons"],
                            "risk_context_flags": list(risk.context_flags),
                            "risk_raw_categories": list(risk.categories),
                            "risk_raw_reasons": list(risk.reasons),
                            "matched_keywords": list(risk.matched_keywords),
                            "match_quality": risk.speech_match_quality,
                            "transcript": transcript,
                            "audio_level": round(audio_level, 4),
                            "people_count": len(tracked_people),
                            "face_count": face_count,
                        }
                    )
                session.latest_meta.pop("speech_error", None)
            except Exception as exc:
                session.latest_meta.update(
                    {
                        "speech_status": "error",
                        "speech_error": str(exc),
                    }
                )
            finally:
                audio_job_queue.task_done()

    threading.Thread(target=audio_worker, name="audio-worker", daemon=True).start()

    return app


def localize_speech_status(status: str) -> str:
    labels = {
        "idle": "idle",
        "listening": "listening",
        "recognized": "recognized",
        "clova": "recognized",
        "clova_empty": "CLOVA empty",
        "clova_error": "CLOVA error",
        "clova_missing_credentials": "CLOVA credentials missing",
        "empty_audio": "empty audio",
        "audio_too_short": "audio too short",
        "audio_too_quiet": "audio too quiet",
        "none": "STT unavailable",
        "error": "error",
    }
    return labels.get(status, status)


def localize_risk_level(level: str) -> str:
    labels = {
        "LOW": "low",
        "ELEVATED": "caution",
        "MEDIUM": "warning",
        "HIGH": "danger",
        "CRITICAL": "critical",
    }
    return labels.get(level, level)


def main() -> None:
    loaded_env_files = load_local_env_files()
    args = parse_args()
    print(f"[detectWarning] YOLO device: {args.yolo_device}")
    print(f"[detectWarning] STT provider: {args.stt_provider}")
    if loaded_env_files:
        print(f"[detectWarning] Loaded env files: {', '.join(loaded_env_files)}")
    print(f"[detectWarning] CLOVA credentials configured: {bool(args.clova_client_id and args.clova_client_secret)}")
    if args.stt_provider != "clova":
        print(f"[detectWarning] Whisper device: {args.stt_device}")
        print(f"[detectWarning] Whisper compute type: {args.stt_compute_type}")
        print(f"[detectWarning] Whisper beam/best_of: {args.stt_beam_size}/{args.stt_best_of}")
    try:
        app = create_app(args)
    except Exception as exc:
        raise RuntimeError(
            "?쒕쾭 ?쒖옉 以?異붾줎 ?μ튂 珥덇린?붿뿉 ?ㅽ뙣?덉뒿?덈떎.\n"
            f"{exc}\n\n"
            "?닿껐 諛⑸쾿:\n"
            "1. Windows ?곗뒪?ы깙?먯꽌 CUDA 吏??PyTorch瑜??ㅼ튂?⑸땲??\n"
            "2. ?먮뒗 ?꾩떆濡?--yolo-device cpu --stt-device cpu 濡??ㅽ뻾?⑸땲??"
        ) from exc
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()









