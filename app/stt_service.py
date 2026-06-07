from __future__ import annotations

import hashlib
import io
import math
import threading
import wave
from collections import deque

import numpy as np
import requests


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


def clova_audio_candidates(audio: np.ndarray) -> list[tuple[str, np.ndarray]]:
    source = np.asarray(audio, dtype=np.float32)
    if source.size == 0:
        return [("raw", source)]
    return [
        ("raw", np.clip(source, -1.0, 1.0).astype(np.float32, copy=False)),
        ("normalized", normalize_audio_for_stt(source, target_rms=0.075, max_gain=5.0)),
        ("boosted", normalize_audio_for_stt(source, target_rms=0.11, max_gain=4.0)),
    ]


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


def wav_duration_seconds(wav_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            frames = int(wav_file.getnframes() or 0)
            sample_rate = int(wav_file.getframerate() or 16000)
        return frames / float(sample_rate or 16000)
    except Exception:
        return 0.0


def stt_audio_gate_reason(audio: np.ndarray, sample_rate: int) -> tuple[str, float]:
    if len(audio) == 0:
        return "empty_audio", 0.0
    audio_level = float(np.sqrt(np.mean(np.square(audio))))
    duration_seconds = len(audio) / float(sample_rate or 16000)
    if duration_seconds < 0.55:
        return "audio_too_short", audio_level
    abs_audio = np.abs(audio)
    peak_level = float(np.max(abs_audio)) if abs_audio.size else 0.0
    if audio_level < 0.006:
        return "audio_too_quiet", audio_level
    if peak_level < 0.035:
        return "audio_peak_too_low", audio_level
    active_ratio = float(np.mean(abs_audio >= 0.012)) if abs_audio.size else 0.0
    if active_ratio < 0.018:
        return "audio_activity_too_low", audio_level
    return "", audio_level


def compute_audio_event_features(audio: np.ndarray, sample_rate: int) -> dict:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return {
            "audio_level": 0.0,
            "audio_peak": 0.0,
            "audio_active_ratio": 0.0,
            "audio_transient_count": 0,
            "audio_crest_factor": 0.0,
            "audio_clipped_ratio": 0.0,
        }
    abs_audio = np.abs(audio)
    audio_level = float(np.sqrt(np.mean(np.square(audio))))
    audio_peak = float(np.max(abs_audio)) if abs_audio.size else 0.0
    active_ratio = float(np.mean(abs_audio >= 0.012)) if abs_audio.size else 0.0
    clipped_ratio = float(np.mean(abs_audio >= 0.96)) if abs_audio.size else 0.0
    crest_factor = audio_peak / max(audio_level, 1e-6)
    window_size = max(int((sample_rate or 16000) * 0.08), 1)
    transient_count = 0
    previous_rms = 0.0
    for start in range(0, len(audio), window_size):
        window = audio[start : start + window_size]
        if window.size < window_size // 2:
            continue
        window_abs = np.abs(window)
        window_peak = float(np.max(window_abs)) if window_abs.size else 0.0
        window_rms = float(np.sqrt(np.mean(np.square(window)))) if window.size else 0.0
        onset_ratio = window_rms / max(previous_rms, 0.006)
        if window_peak >= 0.22 and onset_ratio >= 1.8 and window_rms >= 0.035:
            transient_count += 1
        previous_rms = max(previous_rms * 0.75, window_rms)
    return {
        "audio_level": audio_level,
        "audio_peak": audio_peak,
        "audio_active_ratio": active_ratio,
        "audio_transient_count": int(transient_count),
        "audio_crest_factor": float(crest_factor),
        "audio_clipped_ratio": clipped_ratio,
    }


def should_accept_audio_upload(
    meta: dict,
    duration_seconds: float,
    now: float,
    *,
    min_interval_seconds: float = 3.0,
    max_audio_seconds_per_minute: float = 18.0,
) -> bool:
    duration = max(0.0, float(duration_seconds or 0.0))
    if duration <= 0.0:
        return False
    last_upload_at = float(meta.get("_last_audio_upload_at", 0.0) or 0.0)
    if last_upload_at > 0.0 and now - last_upload_at < min_interval_seconds:
        return False
    raw_window = meta.get("_audio_upload_window")
    window = [
        (float(item[0]), float(item[1]))
        for item in raw_window
        if isinstance(item, (list, tuple)) and len(item) == 2
    ] if isinstance(raw_window, list) else []
    window = [(ts, seconds) for ts, seconds in window if now - ts < 60.0]
    if sum(seconds for _ts, seconds in window) + duration > max_audio_seconds_per_minute:
        meta["_audio_upload_window"] = window
        return False
    window.append((now, duration))
    meta["_audio_upload_window"] = window
    meta["_last_audio_upload_at"] = now
    return True


def reset_audio_upload_budget(meta: dict) -> None:
    meta.pop("_audio_upload_window", None)
    meta.pop("_last_audio_upload_at", None)


def clova_estimated_billed_seconds(duration_seconds: float) -> int:
    duration = max(0.0, float(duration_seconds or 0.0))
    if duration <= 0.0:
        return 0
    return int(min(60, math.ceil(duration / 15.0) * 15))


def normalize_language(language: str) -> str:
    normalized = str(language or "").strip()
    if "-" in normalized:
        normalized = normalized.split("-", 1)[0]
    return normalized.lower() or "ko"


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
        self.total_requests = 0
        self.total_audio_seconds = 0.0
        self.total_estimated_billed_seconds = 0
        self.last_audio_seconds = 0.0
        self.last_estimated_billed_seconds = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if not self.enabled:
            raise RuntimeError("CLOVA credentials are not configured.")
        duration_seconds = len(audio) / float(sample_rate or 16000)
        payload = encode_wav_bytes(audio, sample_rate)
        if len(payload) > 3 * 1024 * 1024:
            raise RuntimeError("CLOVA CSR request is larger than 3MB.")
        headers = {
            "Content-Type": "application/octet-stream",
            "x-ncp-apigw-api-key-id": self.client_id,
            "x-ncp-apigw-api-key": self.client_secret,
        }
        billed_seconds = clova_estimated_billed_seconds(duration_seconds)
        self.total_requests += 1
        self.total_audio_seconds += duration_seconds
        self.total_estimated_billed_seconds += billed_seconds
        self.last_audio_seconds = duration_seconds
        self.last_estimated_billed_seconds = billed_seconds
        try:
            response = requests.post(
                f"https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang={self.language}",
                data=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"CLOVA CSR failed: HTTP {response.status_code} {response.text[:160]}")
            payload_json = response.json()
            self.last_error = ""
            return str(payload_json.get("text", "")).strip()
        except requests.Timeout as exc:
            self.last_error = f"timeout: {exc}"
            raise RuntimeError(f"CLOVA CSR timeout: {exc}") from exc
        except requests.RequestException as exc:
            self.last_error = f"network: {exc}"
            raise RuntimeError(f"CLOVA CSR network error: {exc}") from exc


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
        self.last_clova_attempts: list[str] = []
        self.initial_prompt = (
            "한국어 CCTV 위험상황 감지 음성입니다. "
            "주요 표현: 도와주세요, 살려주세요, 경찰 불러주세요, 신고해주세요, "
            "하지 마세요, 그만해, 멈춰, 때리지 마, 밀지 마, 끌고 가지 마, "
            "소리 지르지 마, 위협하지 마, 칼, 죽여버릴거야, 위험해요."
        )
        self.model = None
        if self.use_whisper:
            try:
                from faster_whisper import WhisperModel
            except Exception as exc:
                raise RuntimeError(
                    "faster-whisper를 불러오지 못했습니다. `pip install -r requirements.txt`를 확인해 주세요."
                ) from exc
            self.model = WhisperModel(
                model_size_or_path=model_size,
                device=device,
                compute_type=compute_type,
            )
        self._lock = threading.Lock()
        self._transcript_cache: dict[str, tuple[str, str, list[str]]] = {}
        self._transcript_cache_order: deque[str] = deque(maxlen=64)

    def _cache_key(self, wav_bytes: bytes) -> str:
        return hashlib.sha1(
            b"|".join(
                [
                    self.provider.encode("utf-8", errors="ignore"),
                    self.language.encode("utf-8", errors="ignore"),
                    wav_bytes,
                ]
            )
        ).hexdigest()

    def _remember_transcript_cache(self, key: str, transcript: str, provider: str, attempts: list[str]) -> None:
        if key not in self._transcript_cache and len(self._transcript_cache_order) >= self._transcript_cache_order.maxlen:
            oldest = self._transcript_cache_order.popleft()
            self._transcript_cache.pop(oldest, None)
        if key not in self._transcript_cache:
            self._transcript_cache_order.append(key)
        self._transcript_cache[key] = (transcript, provider, list(attempts))

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> tuple[str, float]:
        audio, sample_rate = decode_wav_bytes(wav_bytes)
        gate_reason, audio_level = stt_audio_gate_reason(audio, sample_rate)
        if gate_reason:
            self.last_provider = gate_reason
            return "", audio_level
        cache_key = self._cache_key(wav_bytes)
        cached = self._transcript_cache.get(cache_key)
        if cached is not None:
            transcript, provider, attempts = cached
            self.last_provider = provider
            self.last_clova_attempts = list(attempts)
            return transcript, audio_level
        if self.use_clova and self.clova.enabled:
            self.last_clova_attempts = []
            try:
                for profile, candidate_audio in clova_audio_candidates(audio):
                    self.last_clova_attempts.append(profile)
                    transcript = self.clova.transcribe(candidate_audio, sample_rate)
                    if transcript:
                        self.last_provider = "clova" if profile == "raw" else f"clova_{profile}"
                        self._remember_transcript_cache(
                            cache_key,
                            transcript,
                            self.last_provider,
                            self.last_clova_attempts,
                        )
                        return transcript, audio_level
                self.last_provider = "clova_empty"
                self._remember_transcript_cache(cache_key, "", self.last_provider, self.last_clova_attempts)
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
        audio = normalize_audio_for_stt(audio)
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
        self._remember_transcript_cache(cache_key, transcript, self.last_provider, [])
        return transcript, audio_level
