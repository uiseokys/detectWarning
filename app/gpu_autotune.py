from __future__ import annotations

from copy import deepcopy
import shutil
import subprocess
from typing import Callable


DEFAULT_GPU_AUTO_TUNE = {
    "enabled": False,
    "reserve_memory_mb": 1536,
    "preprocess": {
        "enabled": True,
        "max_detector_batch_size": 64,
        "max_video_prefetch_workers": 4,
        "max_person_imgsz": 960,
        "max_frames_to_scan": 240,
    },
    "training": {
        "enabled": True,
        "max_batch_size": 96,
        "max_eval_batch_size": 192,
    },
}


def apply_gpu_auto_tune(
    config: dict,
    *,
    stage: str = "all",
    logger: Callable[[str], None] | None = print,
) -> dict:
    tuned_config = deepcopy(config)
    auto_config = resolve_gpu_auto_tune_config(tuned_config)
    if not auto_config.get("enabled"):
        return tuned_config

    status = query_gpu_status_for_config(tuned_config)
    plan = build_gpu_auto_tune_plan(tuned_config, status=status, stage=stage, auto_config=auto_config)
    for message in plan.get("messages", []):
        if logger is not None:
            logger(message)
    if not plan.get("enabled"):
        return tuned_config

    for section, updates in plan.get("updates", {}).items():
        if not isinstance(updates, dict) or not updates:
            continue
        section_payload = tuned_config.setdefault(section, {})
        if not isinstance(section_payload, dict):
            continue
        section_payload.update(updates)
    return tuned_config


def resolve_gpu_auto_tune_config(config: dict) -> dict:
    merged = deepcopy(DEFAULT_GPU_AUTO_TUNE)
    raw = config.get("gpu_auto_tune")
    if raw is None:
        raw = config.get("auto_tune_gpu")
    if isinstance(raw, dict):
        merged = deep_merge(merged, raw)
    return merged


def deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def query_gpu_status_for_config(config: dict) -> dict:
    preprocess_device = str((config.get("preprocess") or {}).get("device") or "")
    training_device = str((config.get("training") or {}).get("device") or "")
    device_index = first_available_int(
        parse_cuda_device_index(preprocess_device),
        parse_cuda_device_index(training_device),
    )
    return query_nvidia_smi(device_index=device_index)


def parse_cuda_device_index(device: str) -> int | None:
    normalized = str(device or "").strip().lower()
    if not normalized.startswith("cuda"):
        return None
    if ":" not in normalized:
        return 0
    suffix = normalized.split(":", 1)[1].strip()
    if not suffix:
        return 0
    try:
        return max(int(suffix), 0)
    except ValueError:
        return 0


def query_nvidia_smi(*, device_index: int | None = None, runner=None) -> dict:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"available": False, "reason": "nvidia-smi not found"}

    command = [
        nvidia_smi,
        "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    if device_index is not None:
        command.insert(1, f"--id={int(device_index)}")

    run = runner or subprocess.run
    try:
        result = run(command, capture_output=True, text=True, timeout=8, check=True)
    except Exception as exc:
        return {"available": False, "reason": str(exc)}

    line = next((item.strip() for item in str(result.stdout or "").splitlines() if item.strip()), "")
    if not line:
        return {"available": False, "reason": "empty nvidia-smi output"}
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 5:
        return {"available": False, "reason": f"unexpected nvidia-smi output: {line}"}
    try:
        return {
            "available": True,
            "index": int(parts[0]),
            "memory_total_mb": int(float(parts[1])),
            "memory_used_mb": int(float(parts[2])),
            "memory_free_mb": int(float(parts[3])),
            "utilization_gpu_percent": int(float(parts[4])),
        }
    except ValueError as exc:
        return {"available": False, "reason": f"invalid nvidia-smi output: {exc}"}


def build_gpu_auto_tune_plan(
    config: dict,
    *,
    status: dict,
    stage: str = "all",
    auto_config: dict | None = None,
) -> dict:
    auto_config = auto_config or resolve_gpu_auto_tune_config(config)
    if not auto_config.get("enabled"):
        return {"enabled": False, "updates": {}, "messages": []}
    if not status.get("available"):
        return {
            "enabled": False,
            "updates": {},
            "messages": [f"[gpu-autotune] disabled: {status.get('reason') or 'GPU status unavailable'}"],
        }

    total_mb = max(int(status.get("memory_total_mb") or 0), 1)
    used_mb = max(int(status.get("memory_used_mb") or 0), 0)
    free_mb = max(int(status.get("memory_free_mb") or (total_mb - used_mb)), 0)
    reserve_mb = max(int(auto_config.get("reserve_memory_mb") or 0), 0)
    free_ratio = max(0.0, min(free_mb / total_mb, 1.0))
    can_grow = free_mb > reserve_mb
    stage_name = str(stage or "all").strip().lower()
    updates: dict[str, dict] = {}
    messages = [
        (
            "[gpu-autotune] "
            f"gpu{status.get('index', 0)} total={total_mb}MB used={used_mb}MB "
            f"free={free_mb}MB util={status.get('utilization_gpu_percent', '?')}%"
        )
    ]

    if can_grow and stage_name in {"all", "prepare"} and (auto_config.get("preprocess") or {}).get("enabled", True):
        preprocess_updates = build_preprocess_updates(
            config.get("preprocess") or {},
            auto_config=auto_config.get("preprocess") or {},
            free_ratio=free_ratio,
            total_mb=total_mb,
        )
        if preprocess_updates:
            updates["preprocess"] = preprocess_updates

    if can_grow and stage_name in {"all", "train"} and (auto_config.get("training") or {}).get("enabled", True):
        training_updates = build_training_updates(
            config.get("training") or {},
            auto_config=auto_config.get("training") or {},
            free_ratio=free_ratio,
            total_mb=total_mb,
        )
        if training_updates:
            updates["training"] = training_updates

    if not can_grow:
        messages.append(f"[gpu-autotune] kept current settings: free VRAM <= reserve ({reserve_mb}MB)")
    else:
        messages.extend(format_update_messages(updates))
        if not updates:
            messages.append("[gpu-autotune] kept current settings: already at safe GPU auto-tune limits")
    return {"enabled": True, "updates": updates, "messages": messages, "status": status}


def build_preprocess_updates(
    preprocess_config: dict,
    *,
    auto_config: dict,
    free_ratio: float,
    total_mb: int,
) -> dict:
    updates: dict[str, int] = {}
    current_detector_batch = positive_int(preprocess_config.get("detector_batch_size"), 8)
    detector_cap = positive_int(auto_config.get("max_detector_batch_size"), 64)
    detector_target = memory_tier_value(total_mb, [(24576, 64), (16384, 48), (12288, 36), (8192, 24)], current_detector_batch)
    detector_target = grow_with_headroom(current_detector_batch, detector_target, detector_cap, free_ratio)
    if detector_target > current_detector_batch:
        updates["detector_batch_size"] = detector_target

    current_prefetch_workers = nonnegative_int(preprocess_config.get("video_prefetch_workers"), 2)
    prefetch_cap = nonnegative_int(auto_config.get("max_video_prefetch_workers"), 4)
    prefetch_target = current_prefetch_workers
    if current_prefetch_workers > 0 and prefetch_cap > current_prefetch_workers:
        prefetch_tier = memory_tier_value(total_mb, [(24576, 6), (16384, 6), (8192, 4)], current_prefetch_workers)
        if free_ratio >= 0.25:
            prefetch_target = min(max(prefetch_target, prefetch_tier), prefetch_cap)
    if prefetch_target > current_prefetch_workers:
        updates["video_prefetch_workers"] = prefetch_target

    current_imgsz = positive_int(preprocess_config.get("person_imgsz"), 640)
    imgsz_cap = positive_int(auto_config.get("max_person_imgsz"), 960)
    imgsz_target = current_imgsz
    if free_ratio >= 0.58 and total_mb >= 12288:
        imgsz_target = max(imgsz_target, min(imgsz_cap, 960))
    elif free_ratio >= 0.38 and total_mb >= 8192:
        imgsz_target = max(imgsz_target, min(imgsz_cap, 768))
    imgsz_target = round_to_multiple(imgsz_target, 32)
    if imgsz_target > current_imgsz:
        updates["person_imgsz"] = imgsz_target

    current_frames = positive_int(preprocess_config.get("max_frames_to_scan"), 160)
    frames_cap = positive_int(auto_config.get("max_frames_to_scan"), 240)
    frames_target = current_frames
    if free_ratio >= 0.58 and total_mb >= 12288:
        frames_target = max(frames_target, min(frames_cap, 240))
    elif free_ratio >= 0.38 and total_mb >= 8192:
        frames_target = max(frames_target, min(frames_cap, 200))
    if frames_target > current_frames:
        updates["max_frames_to_scan"] = frames_target
    return updates


def build_training_updates(
    training_config: dict,
    *,
    auto_config: dict,
    free_ratio: float,
    total_mb: int,
) -> dict:
    updates: dict[str, int] = {}
    current_batch = positive_int(training_config.get("batch_size"), 16)
    batch_cap = positive_int(auto_config.get("max_batch_size"), 96)
    batch_target = memory_tier_value(total_mb, [(24576, 96), (16384, 64), (12288, 48), (8192, 32)], current_batch)
    batch_target = grow_with_headroom(current_batch, batch_target, batch_cap, free_ratio)
    batch_target = round_to_multiple(batch_target, 8)
    if batch_target > current_batch:
        updates["batch_size"] = batch_target

    current_eval = int(training_config.get("eval_batch_size") or 0)
    effective_eval = current_eval if current_eval > 0 else max(current_batch * 2, current_batch)
    eval_cap = positive_int(auto_config.get("max_eval_batch_size"), 192)
    eval_target = max(effective_eval, min(eval_cap, batch_target * 2))
    eval_target = round_to_multiple(eval_target, 8)
    if eval_target > current_eval:
        updates["eval_batch_size"] = eval_target
    return updates


def memory_tier_value(total_mb: int, tiers: list[tuple[int, int]], fallback: int) -> int:
    for minimum_mb, value in tiers:
        if total_mb >= minimum_mb:
            return max(int(value), int(fallback))
    return int(fallback)


def grow_with_headroom(current: int, tier_target: int, cap: int, free_ratio: float) -> int:
    target = max(int(current), int(tier_target))
    if free_ratio >= 0.60:
        target = max(target, int(current * 2))
    elif free_ratio >= 0.40:
        target = max(target, int(current * 1.5))
    return min(max(target, int(current)), int(cap))


def round_to_multiple(value: int, multiple: int) -> int:
    value = int(value)
    multiple = max(int(multiple), 1)
    return max(multiple, (value // multiple) * multiple)


def positive_int(value, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(fallback)
    return max(parsed, 1)


def nonnegative_int(value, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(fallback)
    return max(parsed, 0)


def first_available_int(*values: int | None) -> int | None:
    for value in values:
        if value is not None:
            return value
    return None


def format_update_messages(updates: dict[str, dict]) -> list[str]:
    messages: list[str] = []
    for section, section_updates in updates.items():
        for key, value in section_updates.items():
            messages.append(f"[gpu-autotune] set {section}.{key}={value}")
    return messages
