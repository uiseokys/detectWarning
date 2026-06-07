from __future__ import annotations

import platform
import queue
import shutil
import socket
import subprocess
import threading

try:
    import psutil
except Exception:
    psutil = None


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
