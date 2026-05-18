from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.queues import Queue as MpQueue
from queue import Empty
from time import monotonic


@dataclass
class SpeechResult:
    status: str
    transcript: str = ""
    error: str | None = None
    audio_level: float = 0.0


def _speech_worker(
    result_queue: MpQueue,
    stop_event,
    language: str,
    sample_rate: int,
    phrase_time_limit: float,
    block_duration: float,
    model_size: str,
    compute_type: str,
    input_device: int | None,
    beam_size: int,
    best_of: int,
    no_speech_threshold: float,
) -> None:
    try:
        import numpy as np
        import sounddevice as sounddevice
        from faster_whisper import WhisperModel
    except Exception as exc:
        result_queue.put(
            SpeechResult(
                status="unavailable",
                error=f"Install Whisper packages first: {exc}",
            )
        )
        return

    def normalize_audio_for_stt(audio, target_rms: float = 0.075, max_gain: float = 8.0):
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

    try:
        whisper_model = WhisperModel(
            model_size_or_path=model_size,
            device="auto",
            compute_type=compute_type,
        )
    except Exception as exc:
        result_queue.put(
            SpeechResult(
                status="unavailable",
                error=f"Whisper model load failed: {exc}",
            )
        )
        return

    audio_queue = deque()
    bytes_per_phrase = int(sample_rate * phrase_time_limit * 2)
    min_phrase_bytes = int(sample_rate * 0.55 * 2)
    speech_level = 0.008
    silence_seconds = 0.35
    result_queue.put(SpeechResult(status="loading"))
    last_level_report = 0.0

    def callback(indata, frames, time_info, status) -> None:
        nonlocal last_level_report
        del frames, time_info
        if status:
            result_queue.put(SpeechResult(status="warning", error=str(status)))
        level = float(np.abs(np.frombuffer(indata, dtype=np.int16)).mean() / 32768.0)
        now = monotonic()
        if now - last_level_report >= 0.25:
            result_queue.put(SpeechResult(status="listening", audio_level=level))
            last_level_report = now
        audio_queue.append((bytes(indata), level))

    try:
        if input_device is not None:
            device_info = sounddevice.query_devices(input_device, "input")
        else:
            default_input = sounddevice.default.device[0]
            if default_input in (-1, None):
                result_queue.put(
                    SpeechResult(
                        status="error",
                        error="No default microphone selected. Run with --list-audio-devices and choose --stt-device.",
                    )
                )
                return
            input_device = int(default_input)
            device_info = sounddevice.query_devices(input_device, "input")

        result_queue.put(
            SpeechResult(
                status="loading",
                error=f"Using mic: {device_info['name']}",
            )
        )

        with sounddevice.RawInputStream(
            samplerate=sample_rate,
            blocksize=int(sample_rate * block_duration),
            dtype="int16",
            channels=1,
            device=input_device,
            callback=callback,
        ):
            result_queue.put(SpeechResult(status="listening"))
            buffer = bytearray()
            speech_started_at = None
            last_voice_at = None

            while not stop_event.is_set():
                now = monotonic()
                while audio_queue:
                    chunk, chunk_level = audio_queue.popleft()
                    buffer.extend(chunk)
                    if chunk_level >= speech_level:
                        if speech_started_at is None:
                            speech_started_at = now
                        last_voice_at = now

                reached_max = len(buffer) >= bytes_per_phrase
                speech_finished = (
                    speech_started_at is not None
                    and last_voice_at is not None
                    and len(buffer) >= min_phrase_bytes
                    and (now - last_voice_at) >= silence_seconds
                )
                ready = reached_max or speech_finished
                if not ready:
                    if len(buffer) >= bytes_per_phrase and speech_started_at is None:
                        buffer.clear()
                    stop_event.wait(0.1)
                    continue

                audio = np.frombuffer(bytes(buffer), dtype=np.int16).astype(np.float32) / 32768.0
                buffer.clear()
                speech_started_at = None
                last_voice_at = None
                level = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0

                if level < 0.003:
                    result_queue.put(
                        SpeechResult(
                            status="listening",
                            transcript="",
                            audio_level=level,
                        )
                    )
                    continue

                audio = normalize_audio_for_stt(audio)
                try:
                    result_queue.put(SpeechResult(status="processing", audio_level=level))
                    prompt = (
                        "한국어 위험상황 감지 음성입니다. 살려주세요, 도와주세요, 경찰 불러주세요, "
                        "하지 마세요, 그만해, 멈춰, 손대지 마, 놓아줘, 때리지 마, 끌고 가지 마, "
                        "납치, 칼, 죽여버릴거야, 위험해요 같은 표현을 정확히 받아씁니다."
                    )
                    try:
                        segments, _info = whisper_model.transcribe(
                            audio,
                            language=language,
                            vad_filter=True,
                            vad_parameters={
                                "threshold": 0.35,
                                "min_speech_duration_ms": 160,
                                "min_silence_duration_ms": 350,
                                "speech_pad_ms": 220,
                            },
                            beam_size=beam_size,
                            best_of=best_of,
                            no_speech_threshold=no_speech_threshold,
                            condition_on_previous_text=False,
                            initial_prompt=prompt,
                            temperature=0.0,
                        )
                    except Exception:
                        segments, _info = whisper_model.transcribe(
                            audio,
                            language=language,
                            vad_filter=False,
                            beam_size=beam_size,
                            best_of=best_of,
                            no_speech_threshold=no_speech_threshold,
                            condition_on_previous_text=False,
                            initial_prompt=prompt,
                            temperature=0.0,
                        )
                    transcript = " ".join(
                        segment.text.strip() for segment in segments if segment.text.strip()
                    ).strip()
                except Exception as exc:
                    result_queue.put(
                        SpeechResult(
                            status="error",
                            error=f"Whisper transcription failed: {exc}",
                        )
                    )
                    continue

                if transcript:
                    result_queue.put(
                        SpeechResult(
                            status="recognized",
                            transcript=transcript,
                            audio_level=level,
                        )
                    )
                else:
                    result_queue.put(
                        SpeechResult(
                            status="listening",
                            transcript="",
                            audio_level=level,
                        )
                    )
    except Exception as exc:
        result_queue.put(
            SpeechResult(
                status="error",
                error=f"Microphone start failed: {exc}",
            )
        )


class SpeechToTextListener:
    def __init__(
        self,
        language: str = "ko-KR",
        sample_rate: int = 16000,
        phrase_time_limit: float = 1.8,
        block_duration: float = 0.5,
        model_size: str = "base",
        compute_type: str = "int8",
        input_device: int | None = None,
        beam_size: int = 1,
        best_of: int = 1,
        no_speech_threshold: float = 0.6,
    ) -> None:
        self.language = self._normalize_language(language)
        self.sample_rate = sample_rate
        self.phrase_time_limit = phrase_time_limit
        self.block_duration = block_duration
        self.model_size = model_size
        self.compute_type = compute_type
        self.input_device = input_device
        self.beam_size = beam_size
        self.best_of = best_of
        self.no_speech_threshold = no_speech_threshold
        self._ctx = get_context("spawn")
        self._result_queue = self._ctx.Queue()
        self._stop_event = self._ctx.Event()
        self._process = None
        self._result = SpeechResult(status="idle")
        self._result_updated_at = monotonic()
        self._status_priority = {
            "error": 5,
            "unavailable": 5,
            "warning": 4,
            "recognized": 4,
            "processing": 3,
            "loading": 2,
            "starting": 2,
            "listening": 1,
            "idle": 0,
        }

    def start(self) -> None:
        if self._process is not None:
            return

        self._stop_event.clear()
        self._process = self._ctx.Process(
            target=_speech_worker,
            name="speech-to-text",
            args=(
                self._result_queue,
                self._stop_event,
                self.language,
                self.sample_rate,
                self.phrase_time_limit,
                self.block_duration,
                self.model_size,
                self.compute_type,
                self.input_device,
                self.beam_size,
                self.best_of,
                self.no_speech_threshold,
            ),
            daemon=True,
        )
        self._process.start()
        self._result = SpeechResult(status="starting")

    def stop(self) -> None:
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._process = None

    def get_result(self) -> SpeechResult:
        best_result = self._result
        while True:
            try:
                next_result = self._result_queue.get_nowait()
            except Empty:
                break
            if self._should_replace(best_result, next_result):
                best_result = next_result
                self._result_updated_at = monotonic()
            elif next_result.audio_level > 0:
                best_result.audio_level = next_result.audio_level

        self._result = best_result
        self._expire_stale_status()

        if self._process is not None and not self._process.is_alive():
            if self._result.status in {"starting", "loading", "listening"}:
                self._result = SpeechResult(
                    status="error",
                    error="Speech worker stopped unexpectedly.",
                )

        return SpeechResult(
            status=self._result.status,
            transcript=self._result.transcript,
            error=self._result.error,
            audio_level=self._result.audio_level,
        )

    @staticmethod
    def _normalize_language(language: str) -> str:
        return language.split("-", 1)[0].lower()

    def _should_replace(self, current: SpeechResult, new: SpeechResult) -> bool:
        current_priority = self._status_priority.get(current.status, 0)
        new_priority = self._status_priority.get(new.status, 0)
        if new_priority != current_priority:
            if new.status == "listening" and current.status in {"recognized", "processing"}:
                return monotonic() - self._result_updated_at >= 1.0
            return new_priority > current_priority
        if new.transcript:
            return True
        if new.error:
            return True
        return new.audio_level >= current.audio_level

    def _expire_stale_status(self) -> None:
        age = monotonic() - self._result_updated_at
        if self._result.status == "recognized" and age >= 1.0:
            self._result = SpeechResult(status="listening", audio_level=self._result.audio_level)
            self._result_updated_at = monotonic()
        elif self._result.status == "processing" and age >= max(1.0, self.phrase_time_limit):
            self._result = SpeechResult(status="listening", audio_level=self._result.audio_level)
            self._result_updated_at = monotonic()


def list_input_devices() -> list[tuple[int, str, int, float]]:
    try:
        import sounddevice as sounddevice
    except Exception:
        return []

    devices = []
    for index, device in enumerate(sounddevice.query_devices()):
        if device["max_input_channels"] > 0:
            devices.append(
                (
                    index,
                    str(device["name"]),
                    int(device["max_input_channels"]),
                    float(device["default_samplerate"]),
                )
            )
    return devices
