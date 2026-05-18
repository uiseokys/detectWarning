from __future__ import annotations

import shutil
import subprocess
import threading
import time

from dashboard_runtime import decode_process_output

GPU_STATUS_CACHE: dict[str, object] = {"timestamp": 0.0, "value": None}
GPU_STATUS_CACHE_LOCK = threading.Lock()


def query_gpu_status(*, cache_ttl_seconds: float = 5.0) -> dict:
    now = time.time()
    with GPU_STATUS_CACHE_LOCK:
        cached_timestamp = float(GPU_STATUS_CACHE.get("timestamp") or 0.0)
        cached_value = GPU_STATUS_CACHE.get("value")
        if cached_value is not None and (now - cached_timestamp) <= cache_ttl_seconds:
            return cached_value

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        result = {
            "available": False,
            "summary": "-",
            "detail": "nvidia-smi를 찾지 못했습니다.",
            "device_name": None,
            "utilization_gpu": None,
            "utilization_memory": None,
            "memory_used_mb": None,
            "memory_total_mb": None,
            "memory_percent": None,
            "temperature_c": None,
            "devices": [],
        }
    else:
        command = [
            nvidia_smi,
            "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                timeout=1.5,
                check=True,
            )
            stdout = decode_process_output(completed.stdout)
            devices = []
            for raw_line in stdout.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 7:
                    continue
                gpu_util = _parse_int_or_none(parts[2])
                memory_util = _parse_int_or_none(parts[3])
                memory_used = _parse_float_or_none(parts[4])
                memory_total = _parse_float_or_none(parts[5])
                temperature = _parse_int_or_none(parts[6])
                memory_percent = None
                if memory_used is not None and memory_total not in (None, 0):
                    memory_percent = round((memory_used / memory_total) * 100.0, 1)
                devices.append(
                    {
                        "index": _parse_int_or_none(parts[0]),
                        "name": parts[1],
                        "utilization_gpu": gpu_util,
                        "utilization_memory": memory_util,
                        "memory_used_mb": memory_used,
                        "memory_total_mb": memory_total,
                        "memory_percent": memory_percent,
                        "temperature_c": temperature,
                    }
                )

            primary = None
            if devices:
                primary = max(
                    devices,
                    key=lambda item: (
                        item.get("utilization_gpu") or 0,
                        item.get("memory_percent") or 0.0,
                    ),
                )

            result = {
                "available": bool(primary),
                "summary": _build_gpu_summary(primary),
                "detail": _build_gpu_detail(primary, device_count=len(devices)),
                "device_index": primary.get("index") if primary else None,
                "device_name": primary.get("name") if primary else None,
                "utilization_gpu": primary.get("utilization_gpu") if primary else None,
                "utilization_memory": primary.get("utilization_memory") if primary else None,
                "memory_used_mb": primary.get("memory_used_mb") if primary else None,
                "memory_total_mb": primary.get("memory_total_mb") if primary else None,
                "memory_percent": primary.get("memory_percent") if primary else None,
                "temperature_c": primary.get("temperature_c") if primary else None,
                "devices": devices,
            }
        except (subprocess.SubprocessError, OSError):
            result = {
                "available": False,
                "summary": "-",
                "detail": "GPU 상태를 읽지 못했습니다.",
                "device_index": None,
                "device_name": None,
                "utilization_gpu": None,
                "utilization_memory": None,
                "memory_used_mb": None,
                "memory_total_mb": None,
                "memory_percent": None,
                "temperature_c": None,
                "devices": [],
            }

    with GPU_STATUS_CACHE_LOCK:
        GPU_STATUS_CACHE["timestamp"] = now
        GPU_STATUS_CACHE["value"] = result
    return result


def _parse_int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"n/a", "[not supported]"}:
        return None
    try:
        return int(float(normalized))
    except ValueError:
        return None


def _parse_float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"n/a", "[not supported]"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _format_gib_from_mb(value_mb: float | None) -> str:
    if value_mb is None:
        return "-"
    return f"{value_mb / 1024.0:.1f} GB"


def _build_gpu_summary(primary: dict | None) -> str:
    if not primary:
        return "-"
    gpu_util = primary.get("utilization_gpu")
    if gpu_util is None:
        return "-"
    return f"{gpu_util}%"


def _build_gpu_detail(primary: dict | None, *, device_count: int) -> str:
    if not primary:
        return "GPU 상태를 읽지 못했습니다."
    memory_used = _format_gib_from_mb(primary.get("memory_used_mb"))
    memory_total = _format_gib_from_mb(primary.get("memory_total_mb"))
    memory_percent = primary.get("memory_percent")
    temp = primary.get("temperature_c")
    util_mem = primary.get("utilization_memory")
    index = primary.get("index")
    prefix = f"GPU {index}" if index is not None else "GPU"
    suffix_parts = [f"VRAM {memory_used} / {memory_total}"]
    if memory_percent is not None:
        suffix_parts.append(f"{memory_percent:.1f}%")
    if util_mem is not None:
        suffix_parts.append(f"mem {util_mem}%")
    if temp is not None:
        suffix_parts.append(f"{temp}°C")
    if device_count > 1:
        suffix_parts.append(f"{device_count} GPUs")
    return f"{prefix} | " + " | ".join(suffix_parts)
