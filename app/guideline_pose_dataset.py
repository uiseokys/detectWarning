from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from action_training_pipeline import (
    decide_prepare_sample_usage,
    get_target_labels,
    load_config,
    read_jsonl_entries,
    resolve_paths,
    slugify,
    write_pipeline_status,
    write_jsonl_entries,
)
from dashboard_normal_ratio import apply_auto_normal_ratio_to_config
from detector import FaceDetector, PersonDetector
from pipeline_prepare import build_empty_pose_sequence, extract_pose_sequence_from_payload, load_video_sequence_payload
from reporting import read_json, write_json_atomic


@dataclass(frozen=True)
class ClipSpec:
    item_id: str
    target_label: str
    source_label: str
    start_seconds: float
    end_seconds: float
    clip_role: str
    event_label: str
    aux_actions: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AIHub guideline 기반으로 XML event/context/normal 구간을 clip 단위 pose manifest로 다시 만듭니다."
        )
    )
    parser.add_argument("--config", default="configs/action_training.aihub_shell.example.json")
    parser.add_argument("--source", choices=("cumulative", "active", "current"), default="cumulative")
    parser.add_argument("--splits", default="train,val,test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    paths = resolve_paths(config, config_path.parent)
    result = build_guideline_pose_dataset(
        config=config,
        paths=paths,
        source=args.source,
        splits=[part.strip() for part in args.splits.split(",") if part.strip()],
    )
    print(f"[guideline] summary: {result['summary_path']}")


def build_guideline_pose_dataset(
    *,
    config: dict,
    paths: dict,
    source: str = "cumulative",
    splits: list[str] | None = None,
) -> dict:
    preprocess_config = config.get("preprocess", {}) or {}
    target_labels = get_target_labels(config)
    apply_auto_normal_ratio_to_config(
        config,
        metric_payloads=[
            read_json(paths["artifacts_dir"] / "metrics.json"),
            read_json(paths["training_progress"]),
        ],
        labels=target_labels,
    )
    guideline_config = resolve_guideline_config(config)
    label_to_idx = {label: index for index, label in enumerate(target_labels)}
    splits = splits or ["train", "val", "test"]
    source_manifests = resolve_source_manifests(paths, source=source)
    source_rows_by_split = {
        split_name: load_guideline_source_rows(
            paths,
            source=source,
            split_name=split_name,
            prepared_manifest=source_manifests[split_name],
            guideline_config=guideline_config,
        )
        for split_name in splits
    }
    output_manifests = resolve_output_manifests(paths)
    output_pose_dir = paths["workspace_dir"] / "prepared_pose_guideline"
    output_pose_dir.mkdir(parents=True, exist_ok=True)

    xml_index = build_xml_index(paths, guideline_config)
    device = str(preprocess_config.get("device", "cuda:0"))
    pose_mode = str(guideline_config.get("pose_mode") or "reuse_existing").strip().lower()
    should_extract_pose = pose_mode == "extract"
    person_detector = None
    face_detector = None
    if should_extract_pose:
        person_detector = PersonDetector(
            score_threshold=float(preprocess_config.get("person_score_threshold", 0.25)),
            resize_width=int(preprocess_config.get("person_imgsz", 640)),
            device=device,
        )
        face_detector = FaceDetector()
    sequence_length = max(int(preprocess_config.get("sequence_length", 64)), 1)
    max_frames_to_scan = max(int(preprocess_config.get("max_frames_to_scan", 220)), sequence_length)
    detector_batch_size = max(int(preprocess_config.get("detector_batch_size", 8)), 1)
    compress_prepared_pose = bool(preprocess_config.get("compress_prepared_pose", False))

    summary = {
        "source": source,
        "labels": target_labels,
        "xml_files_indexed": len(xml_index),
        "splits": {},
        "skipped": Counter(),
    }
    total_videos = sum(len(source_rows_by_split[split_name]) for split_name in splits)
    processed_videos = 0
    total_clips_kept = 0
    last_progress_at = 0.0

    def report_progress(
        *,
        split_name: str,
        row_index: int,
        split_total: int,
        clips_seen: int,
        force: bool = False,
        detail: str = "",
    ) -> None:
        nonlocal last_progress_at
        now = time.monotonic()
        if not force and now - last_progress_at < 10.0:
            return
        last_progress_at = now
        progress = processed_videos / max(total_videos, 1)
        message = (
            f"guideline_clips running: split={split_name} video={row_index}/{split_total}, "
            f"total_videos={processed_videos}/{total_videos}, clips={total_clips_kept}"
        )
        if detail:
            message += f", {detail}"
        print(f"[guideline] {message}", flush=True)
        write_pipeline_status(
            paths,
            stage="guideline",
            state="running",
            message=message,
            stage_progress=progress,
            current_split=split_name,
            current_split_video=row_index,
            current_split_total=split_total,
            processed_videos=processed_videos,
            total_videos=total_videos,
            guideline_clips_kept=total_clips_kept,
        )

    for split_name in splits:
        rows = source_rows_by_split[split_name]
        prepared_rows: list[dict] = []
        split_counts = Counter()
        split_pose_missing_rows = sum(1 for row in rows if row.get("_guideline_added_from_split_missing_prepared"))
        for row_index, row in enumerate(rows, start=1):
            video_path = Path(str(row.get("video_path") or ""))
            source_filekey = resolve_row_filekey(row)
            source_video_id = resolve_row_source_video_id(row, video_path)
            source_zip_group = resolve_row_zip_group(row, video_path)
            report_progress(
                split_name=split_name,
                row_index=row_index,
                split_total=len(rows),
                clips_seen=len(prepared_rows),
                force=row_index == 1,
                detail=f"video={video_path.name}",
            )
            if not video_path.exists():
                summary["skipped"]["video_missing"] += 1
                processed_videos += 1
                continue
            duration_seconds = probe_video_duration(video_path)
            if duration_seconds <= 0:
                summary["skipped"]["duration_missing"] += 1
                processed_videos += 1
                continue
            xml_path = find_xml_for_video(video_path, row, xml_index)
            clips = build_clip_specs(
                row=row,
                video_path=video_path,
                duration_seconds=duration_seconds,
                xml_path=xml_path,
                config=config,
                guideline_config=guideline_config,
            )
            for clip in clips:
                if clip.target_label not in label_to_idx:
                    summary["skipped"]["label_not_enabled"] += 1
                    continue
                pose_fallback_mode = ""
                reused_pose_path = ""
                if should_extract_pose:
                    try:
                        payload = load_video_sequence_payload(
                            video_path=video_path,
                            sequence_length=sequence_length,
                            max_frames_to_scan=max_frames_to_scan,
                            clip_start_seconds=clip.start_seconds,
                            clip_end_seconds=clip.end_seconds,
                        )
                        sequence = extract_pose_sequence_from_payload(
                            payload=payload,
                            person_detector=person_detector,
                            face_detector=face_detector,
                            detector_batch_size=detector_batch_size,
                            allow_rejected_pose_fallback=bool(
                                preprocess_config.get("allow_rejected_pose_fallback", True)
                            ),
                            fallback_min_keypoints=int(preprocess_config.get("fallback_min_keypoints", 5)),
                            fallback_min_detection_confidence=float(
                                preprocess_config.get("fallback_min_detection_confidence", 0.2)
                            ),
                            fallback_min_person_score=int(preprocess_config.get("fallback_min_person_score", 30)),
                        )
                    except Exception as exc:
                        if not guideline_config["keep_pose_failed_clips"]:
                            summary["skipped"][f"pose_error:{type(exc).__name__}"] += 1
                            continue
                        pose_fallback_mode = f"pose_error:{type(exc).__name__}"
                        sequence = build_empty_guideline_sequence(sequence_length, skip_reason=pose_fallback_mode)
                else:
                    sequence, reused_pose_path, pose_fallback_mode = load_reusable_pose_sequence(
                        row,
                        sequence_length=sequence_length,
                    )
                    if pose_fallback_mode and not guideline_config["keep_pose_failed_clips"]:
                        summary["skipped"][pose_fallback_mode] += 1
                        continue

                keep_sample, skip_reason, recovery_actions = decide_prepare_sample_usage(
                    sequence,
                    sequence_length=sequence_length,
                    min_frames_with_person=int(
                        guideline_config.get(
                            "min_pose_valid_frames",
                            preprocess_config.get("min_frames_with_person", 10),
                        )
                    ),
                    fallback_min_frames_with_person=int(
                        guideline_config.get(
                            "fallback_min_pose_valid_frames",
                            preprocess_config.get("fallback_min_frames_with_person", 6),
                        )
                    ),
                    min_confirmed_frames_with_person=int(
                        guideline_config.get(
                            "min_confirmed_pose_frames",
                            preprocess_config.get("min_confirmed_frames_with_person", 3),
                        )
                    ),
                    min_total_keypoints=int(
                        guideline_config.get(
                            "min_pose_total_keypoints",
                            preprocess_config.get("min_total_keypoints", 40),
                        )
                    ),
                    max_missing_frames_ratio=float(
                        guideline_config.get(
                            "max_pose_missing_frames_ratio",
                            preprocess_config.get("max_missing_frames_ratio", 0.75),
                        )
                    ),
                    max_fallback_frames_ratio=float(
                        guideline_config.get(
                            "max_pose_fallback_frames_ratio",
                            preprocess_config.get("max_fallback_frames_ratio", 0.45),
                        )
                    ),
                    allow_partial_pose=bool(preprocess_config.get("allow_partial_pose", True)),
                    allow_padding=bool(preprocess_config.get("allow_padding", True)),
                )
                if not keep_sample:
                    if guideline_config["keep_pose_failed_clips"]:
                        recovery_actions = sorted(
                            set([*recovery_actions, "pose_quality_fallback", str(skip_reason or "pose_low_quality")])
                        )
                        pose_fallback_mode = pose_fallback_mode or str(skip_reason or "pose_low_quality")
                    else:
                        summary["skipped"][skip_reason] += 1
                        continue

                if reused_pose_path:
                    pose_path = Path(reused_pose_path)
                    recovery_actions = sorted(set([*recovery_actions, "reuse_existing_pose"]))
                else:
                    pose_path = build_pose_path(
                        output_pose_dir,
                        split_name=split_name,
                        target_label=clip.target_label,
                        clip=clip,
                        video_path=video_path,
                    )
                    pose_path.parent.mkdir(parents=True, exist_ok=True)
                    save_npz = np.savez_compressed if compress_prepared_pose else np.savez
                    save_npz(
                        pose_path,
                        pose=sequence["pose"],
                        mask=sequence["mask"],
                        label_idx=np.int64(label_to_idx[clip.target_label]),
                    )
                clip_row = {
                    **row,
                    "item_id": clip.item_id,
                    "filekey": source_filekey,
                    "source_filekey": source_filekey,
                    "source_video_id": source_video_id,
                    "source_zip_group": source_zip_group,
                    "source_label": clip.source_label,
                    "target_label": clip.target_label,
                    "label_idx": label_to_idx[clip.target_label],
                    "pose_path": str(pose_path.resolve()),
                    "clip_start_seconds": round(clip.start_seconds, 3),
                    "clip_end_seconds": round(clip.end_seconds, 3),
                    "clip_role": clip.clip_role,
                    "event_label": clip.event_label,
                    "aux_actions": list(clip.aux_actions),
                    "xml_path": str(xml_path) if xml_path else "",
                    "valid_frames": sequence["valid_frames"],
                    "confirmed_frames": sequence["confirmed_frames"],
                    "fallback_frames": sequence.get("fallback_frames", 0),
                    "total_valid_keypoints": sequence.get("total_valid_keypoints", 0),
                    "avg_pose_confidence": round(float(sequence.get("avg_pose_confidence", 0.0) or 0.0), 6),
                    "chosen_track_id": sequence["chosen_track_id"],
                    "recovery_actions": recovery_actions,
                    "pose_quality": build_pose_quality_payload(sequence, sequence_length=sequence_length),
                    "pose_fallback_mode": pose_fallback_mode,
                    "rgb_only_fallback": bool(pose_fallback_mode),
                    "sample_weight": resolve_sample_weight(
                        clip,
                        guideline_config,
                        sequence=sequence,
                        sequence_length=sequence_length,
                        pose_fallback_mode=pose_fallback_mode,
                    ),
                    "frame_stats": sequence.get("frame_stats", {}),
                }
                maybe_attach_rgb_feature_path(clip_row, guideline_config)
                maybe_raise_rgb_ready_fallback_sample_weight(clip_row, guideline_config)
                maybe_apply_xml_missing_sample_weight(clip_row, guideline_config)
                prepared_rows.append(clip_row)
                total_clips_kept += 1
                split_counts[clip.target_label] += 1
                if total_clips_kept % 10 == 0:
                    report_progress(
                        split_name=split_name,
                        row_index=row_index,
                        split_total=len(rows),
                        clips_seen=len(prepared_rows),
                        detail=f"video={video_path.name}",
                    )

            processed_videos += 1
            if row_index == 1 or row_index % 25 == 0 or row_index == len(rows):
                report_progress(
                    split_name=split_name,
                    row_index=row_index,
                    split_total=len(rows),
                    clips_seen=len(prepared_rows),
                    force=True,
                    detail=f"split_clips={len(prepared_rows)}",
                )

        prepared_rows = merge_existing_guideline_output_rows(output_manifests[split_name], prepared_rows)
        write_jsonl_entries(output_manifests[split_name], prepared_rows)
        summary["splits"][split_name] = {
            "videos": len(rows),
            "pose_skipped_rows_added": split_pose_missing_rows,
            "clips": len(prepared_rows),
            "by_label": dict(sorted(split_counts.items())),
            "manifest": str(output_manifests[split_name]),
        }
        print(f"[guideline] {split_name}: {len(prepared_rows)} clips -> {output_manifests[split_name]}", flush=True)

    summary["skipped"] = dict(summary["skipped"])
    summary_path = paths["manifests_dir"] / "guideline_prepare_summary.json"
    write_json_atomic(summary_path, summary)
    print(f"[guideline] summary: {summary_path}", flush=True)
    return {"summary_path": str(summary_path), "manifests": {k: str(v) for k, v in output_manifests.items()}}


def resolve_guideline_config(config: dict) -> dict:
    raw = config.get("guideline_sampling") if isinstance(config.get("guideline_sampling"), dict) else {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "normal_label": str(raw.get("normal_label") or "normal").strip(),
        "include_normal": bool(raw.get("include_normal", True)),
        "context_before_seconds": float(raw.get("context_before_seconds", 12.0)),
        "context_after_seconds": float(raw.get("context_after_seconds", 12.0)),
        "clip_seconds": float(raw.get("clip_seconds", 6.0)),
        "clip_stride_seconds": float(raw.get("clip_stride_seconds", 6.0)),
        "normal_clip_seconds": float(raw.get("normal_clip_seconds", raw.get("clip_seconds", 6.0))),
        "normal_stride_seconds": float(raw.get("normal_stride_seconds", 30.0)),
        "max_event_context_clips_per_video": int(raw.get("max_event_context_clips_per_video", 16)),
        "max_normal_clips_per_video": int(raw.get("max_normal_clips_per_video", 8)),
        "max_total_clips_per_video": int(raw.get("max_total_clips_per_video", 24)),
        "target_normal_clip_ratio": float(raw.get("target_normal_clip_ratio", 0.25)),
        "pose_mode": str(raw.get("pose_mode") or "reuse_existing").strip().lower(),
        "fallback_event_start_ratio": float(raw.get("fallback_event_start_ratio", 0.25)),
        "fallback_event_end_ratio": float(raw.get("fallback_event_end_ratio", 0.75)),
        "event_sample_weight": float(raw.get("event_sample_weight", 1.0)),
        "normal_sample_weight": float(raw.get("normal_sample_weight", 1.0)),
        "hard_negative_sample_weight": float(raw.get("hard_negative_sample_weight", 1.5)),
        "keep_pose_failed_clips": bool(raw.get("keep_pose_failed_clips", True)),
        "include_pose_skipped_videos": bool(raw.get("include_pose_skipped_videos", True)),
        "min_pose_valid_frames": int(raw.get("min_pose_valid_frames", 1)),
        "fallback_min_pose_valid_frames": int(raw.get("fallback_min_pose_valid_frames", 1)),
        "min_confirmed_pose_frames": int(raw.get("min_confirmed_pose_frames", 0)),
        "min_pose_total_keypoints": int(raw.get("min_pose_total_keypoints", 1)),
        "max_pose_missing_frames_ratio": float(raw.get("max_pose_missing_frames_ratio", 1.0)),
        "max_pose_fallback_frames_ratio": float(raw.get("max_pose_fallback_frames_ratio", 1.0)),
        "low_pose_valid_frame_ratio": float(raw.get("low_pose_valid_frame_ratio", 0.25)),
        "low_pose_confidence": float(raw.get("low_pose_confidence", 0.12)),
        "low_pose_sample_weight_multiplier": float(raw.get("low_pose_sample_weight_multiplier", 0.35)),
        "rgb_only_sample_weight_multiplier": float(raw.get("rgb_only_sample_weight_multiplier", 0.15)),
        "rgb_ready_fallback_sample_weight_multiplier": float(
            raw.get("rgb_ready_fallback_sample_weight_multiplier", 0.45)
        ),
        "xml_missing_sample_weight_multiplier": float(raw.get("xml_missing_sample_weight_multiplier", 0.35)),
        "xml_roots": [str(item) for item in raw.get("xml_roots", []) if str(item).strip()],
        "rgb_feature_dir": str(raw.get("rgb_feature_dir") or "").strip(),
    }


def resolve_source_manifests(paths: dict, *, source: str) -> dict[str, Path]:
    prefix = {
        "cumulative": "prepared",
        "active": "active_prepared",
        "current": "current_prepared",
    }[source]
    return {
        "train": paths[f"{prefix}_train"],
        "val": paths[f"{prefix}_val"],
        "test": paths[f"{prefix}_test"],
    }


def resolve_split_source_manifests(paths: dict, *, source: str) -> dict[str, Path]:
    if source == "cumulative":
        prefix = "split"
    elif source == "current":
        prefix = "current_split"
    else:
        return {}
    return {
        "train": paths[f"{prefix}_train"],
        "val": paths[f"{prefix}_val"],
        "test": paths[f"{prefix}_test"],
    }


def load_guideline_source_rows(
    paths: dict,
    *,
    source: str,
    split_name: str,
    prepared_manifest: Path,
    guideline_config: dict,
) -> list[dict]:
    prepared_rows = read_jsonl_entries(prepared_manifest)
    if not guideline_config.get("include_pose_skipped_videos", True):
        return prepared_rows

    split_manifest = resolve_split_source_manifests(paths, source=source).get(split_name)
    if not split_manifest:
        return prepared_rows

    split_rows = read_jsonl_entries(split_manifest)
    if not split_rows:
        return prepared_rows

    existing = {guideline_row_identity(row) for row in prepared_rows}
    merged = list(prepared_rows)
    for row in split_rows:
        identity = guideline_row_identity(row)
        if identity in existing:
            continue
        fallback_row = dict(row)
        fallback_row["_guideline_added_from_split_missing_prepared"] = True
        fallback_row["pose_fallback_mode"] = str(fallback_row.get("pose_fallback_mode") or "pose_reuse_missing")
        fallback_row["rgb_only_fallback"] = True
        metadata = fallback_row.get("metadata") if isinstance(fallback_row.get("metadata"), dict) else {}
        fallback_row["metadata"] = {
            **metadata,
            "guideline_source": "split_missing_prepared",
        }
        merged.append(fallback_row)
        existing.add(identity)
    return merged


def guideline_row_identity(row: dict) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for value in (
        row.get("video_path"),
        metadata.get("source_path"),
        row.get("item_id"),
        row.get("source_video_id"),
    ):
        text = str(value or "").strip()
        if text:
            return normalize_match_key(text)
    return hashlib.sha1(json.dumps(row, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def merge_existing_guideline_output_rows(manifest_path: Path, new_rows: list[dict]) -> list[dict]:
    if not manifest_path.exists():
        return new_rows
    touched_filekeys = {
        str(row.get("source_filekey") or row.get("filekey") or "").strip()
        for row in new_rows
        if str(row.get("source_filekey") or row.get("filekey") or "").strip()
    }
    seen = {guideline_clip_identity(row) for row in new_rows}
    merged = list(new_rows)
    for row in read_jsonl_entries(manifest_path):
        filekey = str(row.get("source_filekey") or row.get("filekey") or "").strip()
        if filekey and filekey in touched_filekeys:
            continue
        identity = guideline_clip_identity(row)
        if not identity or identity in seen:
            continue
        merged.append(row)
        seen.add(identity)
    return merged


def guideline_clip_identity(row: dict) -> str:
    item_id = str(row.get("item_id") or "").strip()
    if item_id:
        return f"item::{normalize_match_key(item_id)}"
    pose_path = str(row.get("pose_path") or "").strip()
    if pose_path:
        return f"pose::{normalize_match_key(pose_path)}"
    payload = {
        "filekey": str(row.get("source_filekey") or row.get("filekey") or "").strip(),
        "video": str(row.get("video_path") or row.get("source_video_path") or "").strip(),
        "start": row.get("clip_start_seconds"),
        "end": row.get("clip_end_seconds"),
        "label": row.get("target_label"),
        "role": row.get("clip_role"),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def resolve_output_manifests(paths: dict) -> dict[str, Path]:
    return {
        "train": paths["manifests_dir"] / "guideline_prepared_train.jsonl",
        "val": paths["manifests_dir"] / "guideline_prepared_val.jsonl",
        "test": paths["manifests_dir"] / "guideline_prepared_test.jsonl",
    }


def build_xml_index(paths: dict, guideline_config: dict) -> dict[str, Path]:
    roots = []
    xml_cache_dir = paths.get("xml_cache_dir")
    if isinstance(xml_cache_dir, Path):
        roots.append(xml_cache_dir)
    roots.extend([paths["extracted_dir"], paths["import_dir"], paths["raw_dir"]])
    for value in guideline_config["xml_roots"]:
        path = Path(value).expanduser()
        roots.append(path if path.is_absolute() else paths["workspace_dir"] / path)
    index: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for xml_path in root.rglob("*.xml"):
            keys = {normalize_match_key(xml_path.stem)}
            keys.add(normalize_match_key(xml_path.name))
            for key in keys:
                index.setdefault(key, xml_path)
    return index


def find_xml_for_video(video_path: Path, row: dict, xml_index: dict[str, Path]) -> Path | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for container in (row, metadata):
        if not isinstance(container, dict):
            continue
        for direct_key in ("xml_path", "source_xml_path", "annotation_xml_path"):
            direct_path = resolve_existing_xml_path(container.get(direct_key))
            if direct_path is not None:
                return direct_path
    candidates = [
        video_path.stem,
        video_path.name,
        str(row.get("item_id") or ""),
        Path(str(metadata.get("source_path") or "")).stem,
    ]
    for candidate in candidates:
        key = normalize_match_key(candidate)
        if key in xml_index:
            return xml_index[key]
    video_key = normalize_match_key(video_path.stem)
    for key, xml_path in xml_index.items():
        if video_key and (video_key in key or key in video_key):
            return xml_path
    return None


def resolve_existing_xml_path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text or text in {".", ".."}:
        return None
    path = Path(text).expanduser()
    if path.is_file() and path.suffix.lower() == ".xml":
        return path
    return None


def normalize_match_key(value: str) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def resolve_row_filekey(row: dict) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for container in (row, metadata):
        for key in (
            "filekey",
            "file_key",
            "fileKey",
            "aihub_filekey",
            "source_filekey",
            "job_filekey",
            "dataset_filekey",
        ):
            value = container.get(key) if isinstance(container, dict) else None
            if isinstance(value, list):
                value = ",".join(str(item).strip() for item in value if str(item).strip())
            text = str(value or "").strip()
            if text:
                return text
    for key in ("source_path", "archive_path", "zip_filename", "video_path"):
        value = str(metadata.get(key) or row.get(key) or "").strip()
        match = next((part for part in Path(value).parts if part.lower().startswith("job_")), "")
        if match:
            parts = match.split("_")
            return parts[1] if len(parts) > 1 and parts[1].isdigit() else match
    return "unknown_filekey"


def resolve_row_source_video_id(row: dict, video_path: Path) -> str:
    raw_item_id = str(row.get("item_id") or "").strip()
    if raw_item_id:
        return raw_item_id.split("__", 1)[0]
    return video_path.stem


def resolve_row_zip_group(row: dict, video_path: Path) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    text = " ".join(
        str(value or "")
        for value in (
            row.get("zip_group"),
            metadata.get("zip_group"),
            metadata.get("zip_filename"),
            metadata.get("source_path"),
            row.get("video_path"),
            video_path,
        )
    ).lower()
    for group in ("outsidedoor", "insidedoor", "inside_croki"):
        if group in text:
            return group
    return ""


def probe_video_duration(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0.0
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    if frames <= 0 or fps <= 0:
        return 0.0
    return frames / fps


def build_clip_specs(
    *,
    row: dict,
    video_path: Path,
    duration_seconds: float,
    xml_path: Path | None,
    config: dict,
    guideline_config: dict,
) -> list[ClipSpec]:
    events = parse_xml_events(xml_path) if xml_path else []
    if not events:
        events = [fallback_event(row, duration_seconds, guideline_config)]
    label_mapping = config.get("dataset", {}).get("label_mapping", {}) or {}
    event_clips: list[ClipSpec] = []
    normal_clips: list[ClipSpec] = []
    event_ranges: list[tuple[float, float]] = []
    for event in events:
        target_label = map_event_label(event["label"], row, label_mapping)
        if not target_label:
            continue
        start = max(float(event["start"]) - guideline_config["context_before_seconds"], 0.0)
        end = min(float(event["end"]) + guideline_config["context_after_seconds"], duration_seconds)
        if end <= start:
            continue
        event_ranges.append((start, end))
        event_clips.extend(
            make_sliding_clips(
                row=row,
                video_path=video_path,
                target_label=target_label,
                source_label=str(event["label"] or row.get("source_label") or target_label),
                event_label=str(event["label"] or target_label),
                clip_role="event_context",
                start=start,
                end=end,
                clip_seconds=guideline_config["clip_seconds"],
                stride_seconds=guideline_config["clip_stride_seconds"],
                aux_actions=tuple(event.get("actions") or ()),
            )
        )
    if guideline_config["include_normal"]:
        normal_label = guideline_config["normal_label"]
        for start, end in normal_intervals(duration_seconds, event_ranges):
            normal_clips.extend(
                make_sliding_clips(
                    row=row,
                    video_path=video_path,
                    target_label=normal_label,
                    source_label=normal_label,
                    event_label=normal_label,
                    clip_role="normal_context",
                    start=start,
                    end=end,
                    clip_seconds=guideline_config["normal_clip_seconds"],
                    stride_seconds=guideline_config["normal_stride_seconds"],
                    aux_actions=(),
                )
            )
    event_clips = limit_clips_evenly(
        event_clips,
        int(guideline_config.get("max_event_context_clips_per_video", 0) or 0),
    )
    normal_clips = select_normal_clips_for_ratio(event_clips, normal_clips, guideline_config)
    max_total = int(guideline_config.get("max_total_clips_per_video", 0) or 0)
    if max_total > 0 and len(event_clips) + len(normal_clips) > max_total:
        event_clips = limit_clips_evenly(event_clips, max(max_total - len(normal_clips), 0))
    return [*event_clips, *normal_clips]


def select_normal_clips_for_ratio(
    event_clips: list[ClipSpec],
    normal_clips: list[ClipSpec],
    guideline_config: dict,
) -> list[ClipSpec]:
    if not normal_clips:
        return []
    max_normal = int(guideline_config.get("max_normal_clips_per_video", 0) or 0)
    target_ratio = min(max(float(guideline_config.get("target_normal_clip_ratio", 0.25) or 0.0), 0.0), 0.5)
    if target_ratio <= 0.0:
        target_count = len(normal_clips)
    elif not event_clips:
        target_count = max_normal if max_normal > 0 else len(normal_clips)
    else:
        target_count = math.ceil(len(event_clips) * target_ratio / max(1.0 - target_ratio, 0.01))
    if max_normal > 0:
        target_count = min(target_count, max_normal)
    target_count = max(target_count, 1 if event_clips else 0)
    return limit_clips_evenly(normal_clips, target_count)


def limit_clips_evenly(clips: list[ClipSpec], max_count: int) -> list[ClipSpec]:
    max_count = int(max_count or 0)
    if max_count <= 0 or len(clips) <= max_count:
        return clips
    if max_count == 1:
        return [clips[len(clips) // 2]]
    indices = np.linspace(0, len(clips) - 1, num=max_count)
    selected = sorted({int(round(index)) for index in indices})
    while len(selected) < max_count:
        for index in range(len(clips)):
            if index not in selected:
                selected.append(index)
                if len(selected) >= max_count:
                    break
    return [clips[index] for index in sorted(selected[:max_count])]


def fallback_event(row: dict, duration_seconds: float, guideline_config: dict) -> dict:
    start_ratio = min(max(guideline_config["fallback_event_start_ratio"], 0.0), 0.95)
    end_ratio = min(max(guideline_config["fallback_event_end_ratio"], start_ratio + 0.01), 1.0)
    return {
        "label": str(row.get("source_label") or row.get("target_label") or ""),
        "start": duration_seconds * start_ratio,
        "end": duration_seconds * end_ratio,
        "actions": (),
    }


def parse_xml_events(xml_path: Path | None) -> list[dict]:
    if not xml_path or not xml_path.is_file() or xml_path.suffix.lower() != ".xml":
        return []
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError, PermissionError):
        return []
    events: list[dict] = []
    for element in root.iter():
        attrs = {str(k).lower(): str(v) for k, v in element.attrib.items()}
        start = parse_time_attr(attrs, ("starttime", "start_time", "start", "startsec", "startsecond"))
        duration = parse_time_attr(attrs, ("duration", "dur", "durationtime"))
        end = parse_time_attr(attrs, ("endtime", "end_time", "end", "endsec", "endsecond"))
        if start is None:
            continue
        if end is None and duration is not None:
            end = start + duration
        if end is None or end <= start:
            continue
        label = first_attr(attrs, ("eventname", "event_name", "event", "label", "name", "actionname", "action"))
        actions = tuple(sorted(collect_action_names(element)))
        events.append({"label": label, "start": start, "end": end, "actions": actions})
    return merge_duplicate_events(events)


def parse_time_attr(attrs: dict[str, str], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = attrs.get(name)
        if value is None:
            continue
        parsed = parse_time_value(value)
        if parsed is not None:
            return parsed
    return None


def parse_time_value(value: str) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        parts = [float(part) for part in text.split(":")]
        total = 0.0
        for part in parts:
            total = total * 60.0 + part
        return total
    try:
        return float(text)
    except ValueError:
        return None


def first_attr(attrs: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = attrs.get(name)
        if value:
            return value.strip()
    return ""


def collect_action_names(element: ET.Element) -> set[str]:
    actions: set[str] = set()
    for child in element.iter():
        attrs = {str(k).lower(): str(v) for k, v in child.attrib.items()}
        name = first_attr(attrs, ("actionname", "action_name", "action", "label", "name"))
        if name:
            actions.add(name)
    return actions


def merge_duplicate_events(events: list[dict]) -> list[dict]:
    unique: dict[tuple[str, int, int], dict] = {}
    for event in events:
        key = (str(event["label"]), int(float(event["start"]) * 1000), int(float(event["end"]) * 1000))
        if key not in unique:
            unique[key] = event
    return list(unique.values())


def map_event_label(label: str, row: dict, label_mapping: dict) -> str:
    raw = str(label or "").strip()
    if raw in label_mapping:
        return str(label_mapping[raw])
    source = str(row.get("source_label") or "").strip()
    if source in label_mapping:
        return str(label_mapping[source])
    return str(row.get("target_label") or raw).strip()


def make_sliding_clips(
    *,
    row: dict,
    video_path: Path,
    target_label: str,
    source_label: str,
    event_label: str,
    clip_role: str,
    start: float,
    end: float,
    clip_seconds: float,
    stride_seconds: float,
    aux_actions: tuple[str, ...],
) -> list[ClipSpec]:
    clip_seconds = max(float(clip_seconds), 0.5)
    stride_seconds = max(float(stride_seconds), 0.5)
    if end - start <= clip_seconds:
        starts = [start]
    else:
        count = int(math.floor((end - start - clip_seconds) / stride_seconds)) + 1
        starts = [start + index * stride_seconds for index in range(max(count, 1))]
        tail_start = max(end - clip_seconds, start)
        if starts and abs(starts[-1] - tail_start) > 0.25:
            starts.append(tail_start)
    clips = []
    for clip_start in starts:
        clip_end = min(clip_start + clip_seconds, end)
        if clip_end <= clip_start:
            continue
        clip_id = hashlib.sha1(
            f"{row.get('item_id')}|{video_path}|{target_label}|{clip_role}|{clip_start:.3f}|{clip_end:.3f}".encode(
                "utf-8",
                errors="ignore",
            )
        ).hexdigest()[:16]
        clips.append(
            ClipSpec(
                item_id=f"{row.get('item_id') or video_path.stem}__{clip_role}_{clip_id}",
                target_label=target_label,
                source_label=source_label,
                start_seconds=clip_start,
                end_seconds=clip_end,
                clip_role=clip_role,
                event_label=event_label,
                aux_actions=aux_actions,
            )
        )
    return clips


def normal_intervals(duration_seconds: float, event_ranges: list[tuple[float, float]]) -> Iterable[tuple[float, float]]:
    merged = []
    for start, end in sorted(event_ranges):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            yield cursor, start
        cursor = max(cursor, end)
    if cursor < duration_seconds:
        yield cursor, duration_seconds


def build_pose_path(
    output_pose_dir: Path,
    *,
    split_name: str,
    target_label: str,
    clip: ClipSpec,
    video_path: Path,
) -> Path:
    name = f"{slugify(video_path.stem)[:42]}_{slugify(clip.clip_role)}_{clip.item_id[-16:]}.npz"
    return output_pose_dir / split_name / slugify(target_label) / name


def build_empty_guideline_sequence(sequence_length: int, *, skip_reason: str) -> dict:
    return build_empty_pose_sequence(
        sequence_length,
        skip_reason=skip_reason,
        frame_stats={
            "sampled_frames": 0,
            "loaded_frames": 0,
            "detection_frames": 0,
            "candidate_frames": 0,
            "accepted_frames": 0,
            "fallback_frames": 0,
            "rejected_candidates": 0,
            "rejection_reasons": {},
        },
    )


def load_reusable_pose_sequence(row: dict, *, sequence_length: int) -> tuple[dict, str, str]:
    pose_path = Path(str(row.get("pose_path") or ""))
    if not pose_path.exists():
        return build_empty_guideline_sequence(sequence_length, skip_reason="pose_reuse_missing"), "", "pose_reuse_missing"
    try:
        with np.load(pose_path, allow_pickle=False) as loaded:
            pose = np.asarray(loaded["pose"], dtype=np.float32)
            mask = np.asarray(loaded["mask"], dtype=np.float32)
    except Exception as exc:
        reason = f"pose_reuse_error:{type(exc).__name__}"
        return build_empty_guideline_sequence(sequence_length, skip_reason=reason), "", reason

    valid_frames = int(np.asarray(mask > 0).sum())
    confidence = pose[..., 2] if pose.ndim == 3 and pose.shape[-1] >= 3 else np.zeros((pose.shape[0], 1), dtype=np.float32)
    positive_confidence = confidence[confidence > 0]
    avg_confidence = float(positive_confidence.mean()) if positive_confidence.size else 0.0
    valid_keypoints = int((confidence > 0).sum()) if confidence.size else 0
    recovery_actions = sorted(set([*normalize_string_list(row.get("recovery_actions")), "reuse_existing_pose"]))
    return (
        {
            "pose": pose,
            "mask": mask,
            "valid_frames": valid_frames,
            "confirmed_frames": int(row.get("confirmed_frames", valid_frames) or valid_frames),
            "fallback_frames": int(row.get("fallback_frames", 0) or 0),
            "total_valid_keypoints": int(row.get("total_valid_keypoints", valid_keypoints) or valid_keypoints),
            "avg_pose_confidence": float(row.get("avg_pose_confidence", avg_confidence) or avg_confidence),
            "chosen_track_id": int(row.get("chosen_track_id", -1) or -1),
            "skip_reason": None if valid_frames > 0 else "pose_reuse_empty",
            "recovery_actions": recovery_actions,
            "frame_stats": row.get("frame_stats") if isinstance(row.get("frame_stats"), dict) else {},
        },
        str(pose_path.resolve()),
        "" if valid_frames > 0 else "pose_reuse_empty",
    )


def normalize_string_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def build_pose_quality_payload(sequence: dict, *, sequence_length: int) -> dict:
    valid_frames = int(sequence.get("valid_frames", 0) or 0)
    confirmed_frames = int(sequence.get("confirmed_frames", 0) or 0)
    fallback_frames = int(sequence.get("fallback_frames", 0) or 0)
    total_keypoints = int(sequence.get("total_valid_keypoints", 0) or 0)
    avg_confidence = float(sequence.get("avg_pose_confidence", 0.0) or 0.0)
    valid_ratio = valid_frames / max(int(sequence_length), 1)
    return {
        "valid_frames": valid_frames,
        "valid_frame_ratio": round(valid_ratio, 6),
        "confirmed_frames": confirmed_frames,
        "fallback_frames": fallback_frames,
        "total_valid_keypoints": total_keypoints,
        "avg_pose_confidence": round(avg_confidence, 6),
        "skip_reason": sequence.get("skip_reason") or "",
    }


def pose_quality_multiplier(
    sequence: dict,
    guideline_config: dict,
    *,
    sequence_length: int,
    pose_fallback_mode: str = "",
) -> float:
    valid_frames = int(sequence.get("valid_frames", 0) or 0)
    avg_confidence = float(sequence.get("avg_pose_confidence", 0.0) or 0.0)
    valid_ratio = valid_frames / max(int(sequence_length), 1)
    if pose_fallback_mode or valid_frames <= 0:
        return max(float(guideline_config["rgb_only_sample_weight_multiplier"]), 0.0)
    is_low_pose = (
        valid_ratio < float(guideline_config["low_pose_valid_frame_ratio"])
        or avg_confidence < float(guideline_config["low_pose_confidence"])
    )
    if is_low_pose:
        return max(float(guideline_config["low_pose_sample_weight_multiplier"]), 0.0)
    return 1.0


def resolve_sample_weight(
    clip: ClipSpec,
    guideline_config: dict,
    *,
    sequence: dict | None = None,
    sequence_length: int = 1,
    pose_fallback_mode: str = "",
) -> float:
    if clip.clip_role == "normal_context":
        base_weight = guideline_config["normal_sample_weight"]
    elif clip.clip_role == "hard_negative":
        base_weight = guideline_config["hard_negative_sample_weight"]
    else:
        base_weight = guideline_config["event_sample_weight"]
    if sequence is None:
        return float(base_weight)
    return round(
        float(base_weight)
        * pose_quality_multiplier(
            sequence,
            guideline_config,
            sequence_length=sequence_length,
            pose_fallback_mode=pose_fallback_mode,
        ),
        6,
    )


def maybe_raise_rgb_ready_fallback_sample_weight(row: dict, guideline_config: dict) -> None:
    if not row.get("rgb_only_fallback") and not str(row.get("pose_fallback_mode") or "").strip():
        return
    feature_path = str(row.get("rgb_feature_path") or "").strip()
    if not feature_path or not Path(feature_path).exists():
        return
    multiplier = max(float(guideline_config.get("rgb_ready_fallback_sample_weight_multiplier", 0.45)), 0.0)
    base_weight = resolve_clip_base_sample_weight(str(row.get("clip_role") or ""), guideline_config)
    boosted_weight = round(base_weight * multiplier, 6)
    row["sample_weight"] = max(float(row.get("sample_weight", 0.0) or 0.0), boosted_weight)
    row["sample_weight_reason"] = append_sample_weight_reason(row.get("sample_weight_reason"), "rgb_ready_fallback")


def maybe_apply_xml_missing_sample_weight(row: dict, guideline_config: dict) -> None:
    if str(row.get("xml_path") or "").strip():
        return
    if str(row.get("target_label") or "").strip().lower() == str(guideline_config.get("normal_label") or "normal"):
        return
    if str(row.get("clip_role") or "").strip() == "normal_context":
        return
    multiplier = max(float(guideline_config.get("xml_missing_sample_weight_multiplier", 0.35)), 0.0)
    base_weight = resolve_clip_base_sample_weight(str(row.get("clip_role") or ""), guideline_config)
    capped_weight = round(base_weight * multiplier, 6)
    current_weight = float(row.get("sample_weight", base_weight) or base_weight)
    row["sample_weight"] = min(current_weight, capped_weight)
    row["sample_weight_reason"] = append_sample_weight_reason(row.get("sample_weight_reason"), "xml_missing")


def append_sample_weight_reason(existing: object, reason: str) -> str:
    reasons = [part for part in str(existing or "").split("+") if part]
    if reason not in reasons:
        reasons.append(reason)
    return "+".join(reasons)


def resolve_clip_base_sample_weight(clip_role: str, guideline_config: dict) -> float:
    if clip_role == "normal_context":
        return float(guideline_config["normal_sample_weight"])
    if clip_role == "hard_negative":
        return float(guideline_config["hard_negative_sample_weight"])
    return float(guideline_config["event_sample_weight"])


def maybe_attach_rgb_feature_path(row: dict, guideline_config: dict) -> None:
    feature_dir = guideline_config.get("rgb_feature_dir")
    if not feature_dir:
        return
    base = Path(feature_dir).expanduser()
    candidates = [
        base / f"{slugify(str(row.get('item_id') or ''))}.npz",
        base / f"{Path(str(row.get('video_path') or '')).stem}.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            row["rgb_feature_path"] = str(candidate.resolve())
            return


if __name__ == "__main__":
    main()
