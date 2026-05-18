from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from reporting import read_json


def build_guideline_quality_summary(paths: dict, config: dict | None = None) -> dict:
    guideline_config = config.get("guideline_sampling", {}) if isinstance(config, dict) else {}
    target_normal_ratio = parse_float(guideline_config.get("target_normal_clip_ratio"), default=0.25)
    target_normal_ratio = min(max(float(target_normal_ratio or 0.0), 0.0), 0.5)
    summary = {
        "total_clips": 0,
        "source_videos": 0,
        "filekey_count": 0,
        "by_split": {},
        "by_label": {},
        "by_role": {},
        "by_filekey": [],
        "rgb_feature_models": {},
        "rgb_feature_ready": 0,
        "rgb_feature_missing": 0,
        "rgb_ready_ratio": 0.0,
        "xml_matched": 0,
        "xml_missing": 0,
        "xml_match_ratio": 0.0,
        "action_tagged": 0,
        "rgb_only_fallback": 0,
        "person_retry_attempted": 0,
        "person_retry_success": 0,
        "low_weight_clips": 0,
        "low_pose_clips": 0,
        "avg_valid_frames": 0.0,
        "avg_pose_confidence": 0.0,
        "normal_ratio": 0.0,
        "target_normal_ratio": round(target_normal_ratio, 4),
        "normal_target_met": False,
        "imbalance_ratio": 0.0,
        "prepare_source": "",
        "cumulative_filekey_count": 0,
        "health": "pending",
        "notes": [],
    }
    by_split: Counter[str] = Counter()
    by_label: Counter[str] = Counter()
    by_role: Counter[str] = Counter()
    by_filekey: dict[str, dict] = {}
    rgb_feature_models: Counter[str] = Counter()
    valid_frame_total = 0.0
    confidence_total = 0.0
    confidence_count = 0
    source_videos: set[str] = set()
    rgb_path_exists_cache: dict[str, bool] = {}

    for split_name, path_key in (
        ("train", "guideline_prepared_train"),
        ("val", "guideline_prepared_val"),
        ("test", "guideline_prepared_test"),
    ):
        manifest_path = paths.get(path_key)
        if not isinstance(manifest_path, Path) or not manifest_path.exists():
            continue
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    summary["total_clips"] += 1
                    by_split[split_name] += 1
                    target_label = str(row.get("target_label") or "unknown")
                    clip_role = str(row.get("clip_role") or "unknown")
                    by_label[target_label] += 1
                    by_role[clip_role] += 1
                    filekey_item = build_filekey_quality_item(by_filekey, row, target_label, split_name)

                    source_key = str(row.get("video_path") or row.get("source_video_path") or "").strip()
                    if not source_key:
                        item_id = str(row.get("item_id") or "").strip()
                        source_key = item_id.split("__", 1)[0] if item_id else ""
                    if source_key:
                        source_videos.add(source_key)

                    rgb_path = str(row.get("rgb_feature_path") or "").strip()
                    if rgb_path and cached_path_exists(rgb_path, rgb_path_exists_cache):
                        summary["rgb_feature_ready"] += 1
                        filekey_item["rgb_ready"] += 1
                    else:
                        summary["rgb_feature_missing"] += 1
                        filekey_item["rgb_missing"] += 1
                    rgb_model = str(row.get("rgb_feature_model") or "").strip()
                    if rgb_model:
                        rgb_feature_models[rgb_model] += 1
                    xml_path = str(row.get("xml_path") or "").strip()
                    if xml_path and cached_path_exists(xml_path, rgb_path_exists_cache):
                        summary["xml_matched"] += 1
                        filekey_item["xml_matched"] += 1
                    else:
                        summary["xml_missing"] += 1
                        filekey_item["xml_missing"] += 1
                    if row.get("aux_actions"):
                        summary["action_tagged"] += 1
                    if row.get("rgb_only_fallback"):
                        summary["rgb_only_fallback"] += 1
                        filekey_item["rgb_fallback"] += 1
                    person_retry = row.get("person_retry") if isinstance(row.get("person_retry"), dict) else {}
                    if person_retry.get("attempted"):
                        summary["person_retry_attempted"] += 1
                        filekey_item["person_retry_attempted"] += 1
                    if person_retry.get("success"):
                        summary["person_retry_success"] += 1
                        filekey_item["person_retry_success"] += 1
                    if parse_float(row.get("sample_weight"), default=1.0) < 0.5:
                        summary["low_weight_clips"] += 1
                        filekey_item["low_weight"] += 1
                    valid_frames = parse_float(row.get("valid_frames"), default=0.0)
                    valid_frame_total += valid_frames
                    if valid_frames < 10:
                        summary["low_pose_clips"] += 1
                        filekey_item["low_pose"] += 1
                    confidence = parse_float(row.get("avg_pose_confidence"), default=None)
                    if confidence is not None:
                        confidence_total += confidence
                        confidence_count += 1
        except OSError:
            continue

    total = int(summary["total_clips"])
    summary["source_videos"] = len(source_videos)
    summary["by_split"] = dict(sorted(by_split.items()))
    summary["by_label"] = dict(sorted(by_label.items()))
    summary["by_role"] = dict(sorted(by_role.items()))
    summary["rgb_feature_models"] = dict(sorted(rgb_feature_models.items()))
    summary["filekey_count"] = len(by_filekey)
    summary["by_filekey"] = compact_filekey_quality(by_filekey)
    summary["avg_valid_frames"] = round(valid_frame_total / max(total, 1), 2) if total else 0.0
    summary["avg_pose_confidence"] = round(confidence_total / max(confidence_count, 1), 4) if confidence_count else 0.0
    summary["rgb_ready_ratio"] = round(int(summary["rgb_feature_ready"]) / max(total, 1), 4) if total else 0.0
    summary["xml_match_ratio"] = round(int(summary["xml_matched"]) / max(total, 1), 4) if total else 0.0
    summary["normal_ratio"] = round(int(by_label.get("normal", 0)) / max(total, 1), 4) if total else 0.0
    summary["normal_target_met"] = bool(total > 0 and float(summary["normal_ratio"]) >= max(target_normal_ratio * 0.8, 0.05))
    nonzero_counts = [int(count) for count in by_label.values() if int(count) > 0]
    if nonzero_counts:
        summary["imbalance_ratio"] = round(max(nonzero_counts) / max(min(nonzero_counts), 1), 2)

    prepare_summary = read_json(paths["manifests_dir"] / "guideline_prepare_summary.json") or {}
    summary["prepare_summary"] = prepare_summary
    summary["prepare_source"] = str(prepare_summary.get("source") or "").strip()
    summary["cumulative_filekey_count"] = len(
        collect_manifest_filekeys(paths, ("prepared_train", "prepared_val", "prepared_test"))
    )

    health, notes = guideline_quality_health(summary, total=total, target_normal_ratio=target_normal_ratio)
    summary["health"] = health
    summary["notes"] = notes
    return summary


def cached_path_exists(path_text: str, cache: dict[str, bool]) -> bool:
    if path_text not in cache:
        cache[path_text] = Path(path_text).exists()
    return cache[path_text]


def build_filekey_quality_item(by_filekey: dict[str, dict], row: dict, target_label: str, split_name: str) -> dict:
    filekey = normalize_guideline_filekey(row)
    item = by_filekey.setdefault(
        filekey,
        {
            "filekey": filekey,
            "total": 0,
            "normal": 0,
            "danger": 0,
            "rgb_ready": 0,
            "rgb_missing": 0,
            "xml_matched": 0,
            "xml_missing": 0,
            "low_pose": 0,
            "low_weight": 0,
            "rgb_fallback": 0,
            "person_retry_attempted": 0,
            "person_retry_success": 0,
            "labels": Counter(),
            "splits": Counter(),
            "zip_groups": Counter(),
        },
    )
    item["total"] += 1
    item["labels"][target_label] += 1
    item["splits"][split_name] += 1
    if target_label == "normal":
        item["normal"] += 1
    else:
        item["danger"] += 1
    zip_group = str(row.get("source_zip_group") or row.get("zip_group") or "").strip()
    if zip_group:
        item["zip_groups"][zip_group] += 1
    return item


def compact_filekey_quality(by_filekey: dict[str, dict]) -> list[dict]:
    return [
        {
            "filekey": item["filekey"],
            "total": item["total"],
            "normal": item["normal"],
            "danger": item["danger"],
            "normal_ratio": round(item["normal"] / max(item["total"], 1), 4),
            "rgb_ready": item["rgb_ready"],
            "rgb_missing": item["rgb_missing"],
            "xml_matched": item["xml_matched"],
            "xml_missing": item["xml_missing"],
            "xml_match_ratio": round(item["xml_matched"] / max(item["total"], 1), 4),
            "low_pose": item["low_pose"],
            "low_weight": item["low_weight"],
            "rgb_fallback": item["rgb_fallback"],
            "person_retry_attempted": item["person_retry_attempted"],
            "person_retry_success": item["person_retry_success"],
            "labels": dict(item["labels"].most_common(6)),
            "splits": dict(item["splits"]),
            "zip_groups": dict(item["zip_groups"]),
        }
        for item in sorted(by_filekey.values(), key=lambda value: int(value["total"]), reverse=True)[:12]
    ]


def guideline_quality_health(summary: dict, *, total: int, target_normal_ratio: float) -> tuple[str, list[str]]:
    notes = []
    health = "good"
    if total <= 0:
        health = "pending"
        notes.append("guideline clip manifest가 아직 없습니다.")
    if total > 0 and float(summary["rgb_ready_ratio"]) < 0.95:
        health = "warning"
        notes.append("RGB/I3D feature가 없는 clip이 있습니다.")
    if total > 0 and float(summary["xml_match_ratio"]) < 0.9:
        health = "warning"
        notes.append("XML event match ratio is below 90%; danger clips without XML are kept with lower sample weight.")
    low_xml_filekeys = [
        item
        for item in summary.get("by_filekey", [])
        if isinstance(item, dict)
        and int(item.get("danger") or 0) > 0
        and float(item.get("xml_match_ratio") or 0.0) < 0.8
    ]
    if low_xml_filekeys:
        health = "warning"
        filekeys = ", ".join(str(item.get("filekey") or "-") for item in low_xml_filekeys[:4])
        notes.append(f"Low XML match filekeys: {filekeys}")
    if total > 0 and int(summary["low_pose_clips"]) > 0:
        health = "warning"
        notes.append("pose 유효 프레임이 낮은 clip이 있습니다.")
    if total > 0 and int(summary["rgb_only_fallback"]) > 0:
        notes.append("pose 실패 clip은 RGB/I3D fallback 샘플로 유지됩니다.")
    if float(summary["imbalance_ratio"]) >= 3.0:
        health = "warning"
        notes.append("클래스 분포 차이가 큽니다.")
    if total > 0 and float(summary["normal_ratio"]) < max(target_normal_ratio * 0.8, 0.05):
        health = "warning"
        notes.append("normal clip 비율이 목표보다 낮습니다.")
    if (
        total > 0
        and str(summary.get("prepare_source") or "").strip().lower() == "current"
        and int(summary.get("cumulative_filekey_count") or 0) > int(summary.get("filekey_count") or 0)
    ):
        health = "warning"
        notes.append("guideline manifest가 current 기준으로 만들어져 누적 filekey 일부가 빠졌습니다. cumulative 기준으로 guideline 재시작이 필요합니다.")
    if total > 0 and not summary["rgb_feature_models"]:
        notes.append("RGB/I3D model 메타데이터가 없어 feature 재시작이 필요할 수 있습니다.")
    return health, notes


def collect_manifest_filekeys(paths: dict, path_keys: tuple[str, ...]) -> set[str]:
    filekeys: set[str] = set()
    for path_key in path_keys:
        manifest_path = paths.get(path_key)
        if not isinstance(manifest_path, Path) or not manifest_path.exists():
            continue
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    filekeys.add(normalize_guideline_filekey(row))
        except OSError:
            continue
    filekeys.discard("unknown_filekey")
    return filekeys


def normalize_guideline_filekey(row: dict) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for container in (row, metadata):
        for key in (
            "source_filekey",
            "filekey",
            "file_key",
            "fileKey",
            "aihub_filekey",
            "job_filekey",
            "dataset_filekey",
            "rgb_feature_filekey",
        ):
            value = container.get(key) if isinstance(container, dict) else None
            if isinstance(value, list):
                value = ",".join(str(item).strip() for item in value if str(item).strip())
            text = str(value or "").strip()
            if text:
                return text
    for key in ("video_path", "source_video_path", "rgb_feature_path", "pose_path"):
        value = str(row.get(key) or metadata.get(key) or "").strip()
        for part in Path(value).parts:
            if part.lower().startswith("job_"):
                parts = part.split("_")
                return parts[1] if len(parts) > 1 and parts[1].isdigit() else part
    return "unknown_filekey"


def parse_float(value, *, default: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
