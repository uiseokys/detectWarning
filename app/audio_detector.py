from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import monotonic

import numpy as np


@dataclass
class SpeechResult:
    status: str
    transcript: str = ""
    error: str | None = None


class SpeechToTextListener:
    def __init__(
        self,
        language: str = "ko-KR",
        sample_rate: int = 16000,
        phrase_time_limit: float = 3.0,
        block_duration: float = 0.5,
        model_size: str = "base",
        compute_type: str = "int8",
    ) -> None:
        self.language = self._normalize_language(language)
        self.sample_rate = sample_rate
        self.phrase_time_limit = phrase_time_limit
        self.block_duration = block_duration
        self.model_size = model_size
        self.compute_type = compute_type
        self._audio_queue: Queue[bytes] = Queue()
        self._stop_event = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._result = SpeechResult(status="idle")
        self._sounddevice = None
        self._whisper_model = None
        self._import_error: str | None = None
        self._bytes_per_phrase = int(self.sample_rate * self.phrase_time_limit * 2)

        try:
            import sounddevice as sounddevice
            from faster_whisper import WhisperModel
        except Exception as exc:
            self._import_error = str(exc)
        else:
            self._sounddevice = sounddevice
            try:
                self._whisper_model = WhisperModel(
                    model_size_or_path=self.model_size,
                    device="auto",
                    compute_type=self.compute_type,
                )
            except Exception as exc:
                self._import_error = f"Whisper model load failed: {exc}"

    def start(self) -> None:
        if self._thread is not None:
            return

        if self._import_error:
            self._set_result(
                SpeechResult(
                    status="unavailable",
                    error=f"Install Whisper packages first: {self._import_error}",
                )
            )
            return

        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="speech-to-text", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_result(self) -> SpeechResult:
        with self._lock:
            return SpeechResult(
                status=self._result.status,
                transcript=self._result.transcript,
                error=self._result.error,
            )

    def _run(self) -> None:
        assert self._sounddevice is not None

        def callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            if status:
                self._set_result(SpeechResult(status="warning", error=str(status)))
            self._audio_queue.put(bytes(indata))

        try:
            with self._sounddevice.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * self.block_duration),
                dtype="int16",
                channels=1,
                callback=callback,
            ):
                self._set_result(SpeechResult(status="listening"))
                buffer = bytearray()
                last_transcribe = monotonic()

                while not self._stop_event.is_set():
                    try:
                        chunk = self._audio_queue.get(timeout=0.2)
                    except Empty:
                        chunk = b""

                    if chunk:
                        buffer.extend(chunk)

                    elapsed = monotonic() - last_transcribe
                    ready = len(buffer) >= self._bytes_per_phrase or (
                        buffer and elapsed >= self.phrase_time_limit
                    )
                    if not ready:
                        continue

                    self._transcribe(bytes(buffer))
                    buffer.clear()
                    last_transcribe = monotonic()
        except Exception as exc:
            self._set_result(
                SpeechResult(
                    status="error",
                    error=f"Microphone start failed: {exc}",
                )
            )

    def _transcribe(self, pcm_data: bytes) -> None:
        assert self._whisper_model is not None

        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        if not np.any(np.abs(audio) > 0.01):
            self._set_result(SpeechResult(status="listening", transcript=""))
            return

        try:
            segments, _info = self._whisper_model.transcribe(
                audio,
                language=self.language,
                vad_filter=True,
                beam_size=5,
                condition_on_previous_text=False,
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            self._set_result(
                SpeechResult(
                    status="error",
                    error=f"Whisper transcription failed: {exc}",
                )
            )
            return

        if transcript:
            self._set_result(SpeechResult(status="recognized", transcript=transcript))
        else:
            self._set_result(SpeechResult(status="listening", transcript=""))

    def _set_result(self, result: SpeechResult) -> None:
        with self._lock:
            self._result = result

    @staticmethod
    def _normalize_language(language: str) -> str:
        return language.split("-", 1)[0].lower()
