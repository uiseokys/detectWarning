from __future__ import annotations


DEFAULT_AUTO_NORMAL_RATIO_CONFIG = {
    "enabled": True,
    "min_ratio": 0.20,
    "max_ratio": 0.35,
    "step": 0.05,
    "danger_to_normal_threshold": 0.20,
    "normal_to_danger_threshold": 0.25,
    "predicted_normal_bias_threshold": 0.15,
}


def apply_auto_normal_ratio_to_config(
    config: dict,
    *,
    metric_payloads: list[dict] | tuple[dict, ...] | None = None,
    labels: list[str] | None = None,
) -> dict:
    guideline_config = config.setdefault("guideline_sampling", {})
    if not isinstance(guideline_config, dict):
        config["guideline_sampling"] = {}
        guideline_config = config["guideline_sampling"]

    adjustment = build_auto_normal_ratio_adjustment(
        config,
        metric_payloads=metric_payloads or [],
        labels=labels,
    )
    guideline_config["_auto_normal_ratio_adjustment"] = adjustment
    if adjustment.get("applied"):
        guideline_config["target_normal_clip_ratio"] = adjustment["target_ratio"]
    return adjustment


def build_auto_normal_ratio_adjustment(
    config: dict,
    *,
    metric_payloads: list[dict] | tuple[dict, ...] | None = None,
    labels: list[str] | None = None,
) -> dict:
    guideline_config = config.get("guideline_sampling") if isinstance(config.get("guideline_sampling"), dict) else {}
    auto_config = guideline_config.get("auto_normal_ratio")
    if isinstance(auto_config, bool):
        auto_config = {"enabled": auto_config}
    if auto_config is None:
        auto_config = DEFAULT_AUTO_NORMAL_RATIO_CONFIG
    if not isinstance(auto_config, dict):
        auto_config = {}
    settings = {**DEFAULT_AUTO_NORMAL_RATIO_CONFIG, **auto_config}
    enabled = bool(settings.get("enabled", False))

    current_ratio = clamp_float(
        safe_float(guideline_config.get("target_normal_clip_ratio"), 0.25),
        0.0,
        0.5,
    )
    base_payload = {
        "enabled": enabled,
        "applied": False,
        "source": "none",
        "reason": "disabled" if not enabled else "no_metrics",
        "previous_ratio": round(current_ratio, 4),
        "target_ratio": round(current_ratio, 4),
    }
    if not enabled:
        return base_payload

    payload = select_metric_payload(metric_payloads or [])
    if not payload:
        return base_payload

    final_validation = payload.get("final_validation") if isinstance(payload.get("final_validation"), dict) else {}
    matrix = final_validation.get("confusion_matrix")
    if not isinstance(matrix, list) or not matrix:
        return {**base_payload, "source": "metrics", "reason": "missing_confusion_matrix"}

    resolved_labels = normalize_labels(labels or payload.get("labels") or [])
    if not resolved_labels:
        return {**base_payload, "source": "metrics", "reason": "missing_labels"}

    normal_label = str(guideline_config.get("normal_label") or "normal").strip().lower()
    normal_index = next(
        (index for index, label in enumerate(resolved_labels) if str(label).strip().lower() == normal_label),
        None,
    )
    if normal_index is None:
        return {**base_payload, "source": "metrics", "reason": "normal_label_not_found"}

    size = min(len(resolved_labels), len(matrix))
    total = 0
    danger_support = 0
    danger_as_normal = 0
    normal_support = 0
    normal_as_danger = 0
    predicted_normal = 0
    for row_index in range(size):
        row = matrix[row_index] if isinstance(matrix[row_index], list) else []
        row_values = [safe_int(row[column_index], 0) if column_index < len(row) else 0 for column_index in range(size)]
        row_total = sum(row_values)
        total += row_total
        if normal_index < len(row_values):
            predicted_normal += row_values[normal_index]
        if row_index == normal_index:
            normal_support = row_total
            normal_as_danger = max(row_total - row_values[normal_index], 0)
        else:
            danger_support += row_total
            if normal_index < len(row_values):
                danger_as_normal += row_values[normal_index]

    if total <= 0:
        return {**base_payload, "source": "metrics", "reason": "empty_confusion_matrix"}

    danger_to_normal_rate = danger_as_normal / max(danger_support, 1)
    normal_to_danger_rate = normal_as_danger / max(normal_support, 1)
    predicted_normal_ratio = predicted_normal / max(total, 1)
    true_normal_ratio = normal_support / max(total, 1)
    predicted_normal_bias = predicted_normal_ratio - true_normal_ratio

    min_ratio = clamp_float(safe_float(settings.get("min_ratio"), 0.20), 0.0, 0.5)
    max_ratio = clamp_float(safe_float(settings.get("max_ratio"), 0.35), min_ratio, 0.5)
    step = clamp_float(safe_float(settings.get("step"), 0.05), 0.0, 0.25)
    target_ratio = current_ratio
    reason = "kept"
    if (
        danger_to_normal_rate >= safe_float(settings.get("danger_to_normal_threshold"), 0.20)
        or predicted_normal_bias >= safe_float(settings.get("predicted_normal_bias_threshold"), 0.15)
    ):
        target_ratio = current_ratio - step
        reason = "decrease_normal_ratio"
    elif normal_to_danger_rate >= safe_float(settings.get("normal_to_danger_threshold"), 0.25):
        target_ratio = current_ratio + step
        reason = "increase_normal_ratio"

    target_ratio = clamp_float(target_ratio, min_ratio, max_ratio)
    applied = abs(target_ratio - current_ratio) > 1e-9
    return {
        "enabled": True,
        "applied": applied,
        "source": "metrics",
        "reason": reason if applied else "kept",
        "previous_ratio": round(current_ratio, 4),
        "target_ratio": round(target_ratio, 4),
        "min_ratio": round(min_ratio, 4),
        "max_ratio": round(max_ratio, 4),
        "step": round(step, 4),
        "danger_to_normal_rate": round(danger_to_normal_rate, 4),
        "normal_to_danger_rate": round(normal_to_danger_rate, 4),
        "predicted_normal_ratio": round(predicted_normal_ratio, 4),
        "true_normal_ratio": round(true_normal_ratio, 4),
        "danger_as_normal": danger_as_normal,
        "danger_support": danger_support,
        "normal_as_danger": normal_as_danger,
        "normal_support": normal_support,
    }


def select_metric_payload(metric_payloads: list[dict] | tuple[dict, ...]) -> dict:
    for payload in metric_payloads:
        if isinstance(payload, dict) and isinstance(payload.get("final_validation"), dict):
            return payload
    return {}


def normalize_labels(values) -> list[str]:
    return [str(value).strip() for value in (values or []) if str(value).strip()]


def safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def clamp_float(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))
