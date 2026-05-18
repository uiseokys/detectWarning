from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from action_training_pipeline import read_jsonl_entries, resolve_paths, write_jsonl_entries
from reporting import write_json_atomic
from training_config import load_action_training_config


MAX_RGB_FEATURE_PATH_LENGTH = 240


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="guideline clip manifest의 각 clip에서 RGB 3D-CNN feature를 추출해 manifest에 rgb_feature_path를 붙입니다."
    )
    parser.add_argument("--config", default="configs/action_training.aihub_shell.example.json")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--model", default="i3d_r50", choices=("i3d_r50", "r3d_18", "mc3_18", "r2plus1d_18"))
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=112)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--reuse-existing-only",
        action="store_true",
        help="Do not load a model or extract videos; only attach RGB feature files that already exist.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="If the requested RGB/I3D model cannot load, fall back to a torchvision model and record the actual model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_action_training_config(config_path)
    paths = resolve_paths(config, config_path.parent)
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    extractor = None
    actual_model_name = args.model
    frames = max(int(args.frames), 4)
    image_size = max(int(args.size), 64)
    device_text = args.device
    if not args.reuse_existing_only:
        extractor = RgbFeatureExtractor(
            model_name=args.model,
            device=args.device,
            frames=args.frames,
            image_size=args.size,
            allow_fallback=bool(args.allow_fallback),
        )
        actual_model_name = extractor.actual_model_name
        frames = extractor.frames
        image_size = extractor.image_size
        device_text = str(extractor.device)
    output_dir = paths["workspace_dir"] / "rgb_clip_features" / actual_model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output_dir / "feature_run_manifest.json",
        {
            "requested_model": args.model,
            "actual_model": actual_model_name,
            "frames": frames,
            "image_size": image_size,
            "device": device_text,
            "allow_fallback": bool(args.allow_fallback),
            "reuse_existing_only": bool(args.reuse_existing_only),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    manifests = {
        "train": paths["guideline_prepared_train"],
        "val": paths["guideline_prepared_val"],
        "test": paths["guideline_prepared_test"],
    }
    source_manifests = {
        "train": [paths.get("prepared_train"), paths.get("split_train"), paths.get("current_prepared_train"), paths.get("current_split_train")],
        "val": [paths.get("prepared_val"), paths.get("split_val"), paths.get("current_prepared_val"), paths.get("current_split_val")],
        "test": [paths.get("prepared_test"), paths.get("split_test"), paths.get("current_prepared_test"), paths.get("current_split_test")],
    }
    for split_name in splits:
        rows = read_jsonl_entries(manifests[split_name])
        rows = recover_existing_rgb_feature_rows(
            rows,
            split_name=split_name,
            output_dir=output_dir,
            model_name=actual_model_name,
            requested_model=args.model,
            source_manifests=source_manifests.get(split_name) or [],
            config=config,
        )
        updated = []
        for index, row in enumerate(rows, start=1):
            filekey = safe_path_part(resolve_row_filekey(row))
            existing_feature_path = str(row.get("rgb_feature_path") or "").strip()
            feature_path = (
                Path(existing_feature_path)
                if existing_feature_path
                else build_rgb_feature_path(output_dir, split_name=split_name, filekey=filekey, row=row)
            )
            if should_use_short_rgb_feature_path(feature_path):
                feature_path = build_short_rgb_feature_path(output_dir, split_name=split_name, filekey=filekey, row=row)
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            if not feature_path.exists():
                video_path = Path(str(row.get("video_path") or ""))
                if args.reuse_existing_only or row.get("rgb_catalog_only") or not video_path.exists():
                    print(f"[rgb] skipped missing source video: split={split_name} item={row.get('item_id')}")
                    continue
                if extractor is None:
                    raise RuntimeError("RGB extractor was not initialized.")
                feature = extractor.extract_clip(
                    video_path,
                    start_seconds=float(row.get("clip_start_seconds", 0.0) or 0.0),
                    end_seconds=float(row.get("clip_end_seconds", 0.0) or 0.0),
                )
                feature_path = save_rgb_feature_with_fallback(
                    feature_path,
                    feature,
                    output_dir=output_dir,
                    split_name=split_name,
                    filekey=filekey,
                    row=row,
                )
            write_rgb_feature_sidecar(feature_path, row)
            updated.append(
                apply_rgb_ready_fallback_sample_weight(
                    {
                    **row,
                    "rgb_feature_path": str(feature_path.resolve()),
                    "rgb_feature_model": actual_model_name,
                    "rgb_feature_requested_model": args.model,
                    "rgb_feature_filekey": filekey,
                    "rgb_feature_frames": frames,
                    "rgb_feature_size": image_size,
                    "rgb_feature_fallback_used": actual_model_name != args.model,
                    },
                    config,
                )
            )
            if index == 1 or index % 50 == 0 or index == len(rows):
                print(f"[rgb] {split_name} {index}/{len(rows)}")
        write_jsonl_entries(manifests[split_name], updated)
        print(f"[rgb] updated {split_name}: {manifests[split_name]}")


def recover_existing_rgb_feature_rows(
    rows: list[dict],
    *,
    split_name: str,
    output_dir: Path,
    model_name: str,
    requested_model: str,
    source_manifests: list[Path | None],
    config: dict,
) -> list[dict]:
    feature_root = output_dir / split_name
    if not feature_root.exists():
        return rows

    pose_index = build_source_row_index(source_manifests)
    existing_keys = {build_rgb_row_key(row) for row in rows}
    existing_identities = {build_row_identity_key(row) for row in rows}
    recovered: list[dict] = []
    for feature_path in sorted(feature_root.glob("*/*.npz")):
        row = build_recovered_rgb_row(
            feature_path,
            split_name=split_name,
            model_name=model_name,
            requested_model=requested_model,
            pose_index=pose_index,
            config=config,
        )
        if row is None:
            continue
        key = build_rgb_row_key(row)
        identity_key = build_row_identity_key(row)
        if key in existing_keys or identity_key in existing_identities:
            continue
        recovered.append(row)
        existing_keys.add(key)
        existing_identities.add(identity_key)

    if recovered:
        print(f"[rgb] recovered existing feature rows {split_name}: +{len(recovered)} from {feature_root}")
    return [*rows, *recovered]


def build_source_row_index(source_manifests: list[Path | None]) -> dict[tuple[str, str], dict]:
    index: dict[tuple[str, str], dict] = {}
    for manifest_path in source_manifests:
        if not isinstance(manifest_path, Path) or not manifest_path.exists():
            continue
        for row in read_jsonl_entries(manifest_path):
            filekeys = resolve_row_filekeys(row)
            video_keys = resolve_video_identity_keys(row)
            if not filekeys or not video_keys:
                continue
            for filekey in filekeys:
                for video_key in video_keys:
                    key = (safe_path_part(filekey), video_key)
                    existing = index.get(key)
                    if existing is None or (not row_xml_path(existing) and row_xml_path(row)):
                        index[key] = row
    return index


def build_recovered_rgb_row(
    feature_path: Path,
    *,
    split_name: str,
    model_name: str,
    requested_model: str,
    pose_index: dict[tuple[str, str], dict],
    config: dict,
) -> dict | None:
    filekey = safe_path_part(feature_path.parent.name)
    item_id = feature_path.stem
    sidecar_row = read_rgb_feature_sidecar(feature_path)
    prefix = f"{filekey}__"
    if item_id.startswith(prefix):
        item_id = item_id[len(prefix) :]
    clip_role, base_item_id = split_feature_item_id(item_id)
    target_label = infer_rgb_feature_label(base_item_id, clip_role=clip_role, config=config)
    if not target_label:
        return None
    label_to_idx = {
        str(label): index
        for index, label in enumerate((config.get("dataset") or {}).get("target_labels") or [])
    }

    pose_row = None
    for video_key in resolve_video_identity_keys({"item_id": base_item_id}):
        pose_row = pose_index.get((filekey, video_key))
        if pose_row is not None:
            break

    if sidecar_row and resolve_row_filekey(sidecar_row) == filekey:
        pose_row = {**sidecar_row, **(pose_row or {})}
    if pose_row is None:
        return None

    recovered = dict(pose_row)
    xml_path = row_xml_path(recovered)
    recovered.update(
        {
            "item_id": recovered.get("item_id") or item_id,
            "source_item_id": recovered.get("source_item_id") or base_item_id,
            "target_label": target_label,
            "label_idx": label_to_idx.get(target_label, recovered.get("label_idx")),
            "source_label": recovered.get("source_label") or target_label,
            "filekey": filekey,
            "source_filekey": filekey,
            "split": split_name,
            "clip_role": clip_role or recovered.get("clip_role") or "event_context",
            "rgb_feature_path": str(feature_path.resolve()),
            "rgb_feature_model": model_name,
            "rgb_feature_requested_model": requested_model,
            "rgb_feature_filekey": filekey,
            "rgb_feature_recovered": True,
        }
    )
    if xml_path and not str(recovered.get("xml_path") or "").strip():
        recovered["xml_path"] = xml_path
    return recovered


def read_rgb_feature_sidecar(feature_path: Path) -> dict:
    sidecar_path = feature_path.with_suffix(".json")
    if not sidecar_path.exists():
        return {}
    try:
        with sidecar_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    row = payload.get("row") if isinstance(payload, dict) else {}
    return row if isinstance(row, dict) else {}


def write_rgb_feature_sidecar(feature_path: Path, row: dict) -> None:
    sidecar_path = feature_path.with_suffix(".json")
    payload = {
        "schema": "rgb_feature_sidecar.v1",
        "row": {
            key: value
            for key, value in row.items()
            if key not in {"feature", "process", "runtime_config"}
        },
    }
    try:
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = sidecar_path.with_name(f".{sidecar_path.name}.{hashlib.sha1(str(sidecar_path).encode('utf-8')).hexdigest()[:8]}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        temp_path.replace(sidecar_path)
    except OSError:
        pass


def row_xml_path(row: dict) -> str:
    if not isinstance(row, dict):
        return ""
    direct = valid_xml_path_text(row.get("xml_path") or row.get("source_xml_path") or row.get("annotation_xml_path"))
    if direct:
        return direct
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return valid_xml_path_text(
        metadata.get("xml_path")
        or metadata.get("source_xml_path")
        or metadata.get("annotation_xml_path")
        or ""
    )


def valid_xml_path_text(value: object) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."}:
        return ""
    path = Path(text).expanduser()
    if path.is_file() and path.suffix.lower() == ".xml":
        return str(path)
    return ""


def split_feature_item_id(item_id: str) -> tuple[str, str]:
    for marker, role in (
        ("__normal_context_", "normal_context"),
        ("__event_context_", "event_context"),
        ("__hard_negative_", "hard_negative"),
    ):
        if marker in item_id:
            return role, item_id.split(marker, 1)[0]
    return "", item_id


def infer_rgb_feature_label(base_item_id: str, *, clip_role: str, config: dict) -> str:
    guideline_config = config.get("guideline_sampling") if isinstance(config.get("guideline_sampling"), dict) else {}
    normal_label = str(guideline_config.get("normal_label") or "normal")
    if clip_role == "normal_context":
        return normal_label

    dataset_config = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    target_labels = {str(label) for label in dataset_config.get("target_labels") or []}
    label_mapping = dataset_config.get("label_mapping") if isinstance(dataset_config.get("label_mapping"), dict) else {}
    lowered = str(base_item_id or "").lower()
    for source_label, target_label in sorted(label_mapping.items(), key=lambda item: len(str(item[0])), reverse=True):
        source_text = str(source_label or "").strip().lower()
        target_text = str(target_label or "").strip()
        if source_text and source_text in lowered and (not target_labels or target_text in target_labels):
            return target_text
    return ""


def build_rgb_row_key(row: dict) -> str:
    feature_path = str(row.get("rgb_feature_path") or "").strip()
    if feature_path:
        return f"rgb::{feature_path.lower()}"
    return f"item::{build_row_identity_key(row)}"


def build_row_identity_key(row: dict) -> str:
    return f"{safe_path_part(resolve_row_filekey(row))}::{safe_path_part(str(row.get('item_id') or row.get('source_item_id') or ''))}"


def resolve_row_filekeys(row: dict) -> list[str]:
    values: list[str] = []
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for container in (row, metadata):
        if not isinstance(container, dict):
            continue
        for key in (
            "filekey",
            "file_key",
            "fileKey",
            "aihub_filekey",
            "source_filekey",
            "job_filekey",
            "dataset_filekey",
        ):
            value = container.get(key)
            if isinstance(value, list):
                values.extend(str(item).strip() for item in value if str(item).strip())
            elif str(value or "").strip():
                values.append(str(value).strip())
    if not values:
        values.append(resolve_row_filekey(row))
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        safe = safe_path_part(value)
        if safe and safe not in seen and safe != "unknown_filekey":
            unique.append(safe)
            seen.add(safe)
    return unique


def resolve_video_identity_keys(row: dict) -> list[str]:
    candidates: list[str] = []
    for key in ("video_path", "source_video_path"):
        value = str(row.get(key) or "").strip()
        if value:
            candidates.append(Path(value).name.lower())
    item_id = str(row.get("item_id") or row.get("source_item_id") or "").strip()
    if item_id:
        candidates.append(item_id.lower())
        parts = [part for part in item_id.replace("\\", "/").split("/") if part]
        candidates.extend(part.lower() for part in parts if part.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".wmv")))
        for marker in (".mp4_", ".avi_", ".mov_", ".mkv_", ".wmv_"):
            if marker in item_id.lower():
                candidates.append(item_id.split(marker, 1)[1].lower())
    seen: set[str] = set()
    unique: list[str] = []
    for value in candidates:
        cleaned = value.strip().replace("\\", "/")
        if cleaned and cleaned not in seen:
            unique.append(cleaned)
            seen.add(cleaned)
    return unique


def apply_rgb_ready_fallback_sample_weight(row: dict, config: dict) -> dict:
    if not row.get("rgb_only_fallback") and not str(row.get("pose_fallback_mode") or "").strip():
        return row
    guideline_config = config.get("guideline_sampling") if isinstance(config.get("guideline_sampling"), dict) else {}
    multiplier = max(float(guideline_config.get("rgb_ready_fallback_sample_weight_multiplier", 0.45) or 0.45), 0.0)
    base_weight = resolve_clip_base_sample_weight(str(row.get("clip_role") or ""), guideline_config)
    boosted_weight = round(base_weight * multiplier, 6)
    row["sample_weight"] = max(float(row.get("sample_weight", 0.0) or 0.0), boosted_weight)
    row["sample_weight_reason"] = append_sample_weight_reason(row.get("sample_weight_reason"), "rgb_ready_fallback")
    if should_downweight_missing_xml(row, guideline_config):
        xml_multiplier = max(float(guideline_config.get("xml_missing_sample_weight_multiplier", 0.35) or 0.35), 0.0)
        row["sample_weight"] = min(float(row.get("sample_weight", base_weight) or base_weight), round(base_weight * xml_multiplier, 6))
        row["sample_weight_reason"] = append_sample_weight_reason(row.get("sample_weight_reason"), "xml_missing")
    return row


def should_downweight_missing_xml(row: dict, guideline_config: dict) -> bool:
    if str(row.get("xml_path") or "").strip():
        return False
    normal_label = str(guideline_config.get("normal_label") or "normal").strip().lower()
    if str(row.get("target_label") or "").strip().lower() == normal_label:
        return False
    if str(row.get("clip_role") or "").strip() == "normal_context":
        return False
    return True


def append_sample_weight_reason(existing: object, reason: str) -> str:
    reasons = [part for part in str(existing or "").split("+") if part]
    if reason not in reasons:
        reasons.append(reason)
    return "+".join(reasons)


def resolve_clip_base_sample_weight(clip_role: str, guideline_config: dict) -> float:
    if clip_role == "normal_context":
        return float(guideline_config.get("normal_sample_weight", 0.8) or 0.8)
    if clip_role == "hard_negative":
        return float(guideline_config.get("hard_negative_sample_weight", 1.5) or 1.5)
    return float(guideline_config.get("event_sample_weight", 1.0) or 1.0)


class RgbFeatureExtractor:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        frames: int,
        image_size: int,
        allow_fallback: bool = False,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.frames = max(int(frames), 4)
        self.image_size = max(int(image_size), 64)
        self.model, self.actual_model_name = build_video_model(model_name, allow_fallback=allow_fallback)
        self.model = self.model.to(self.device).eval()
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645], device=self.device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.22803, 0.22145, 0.216989], device=self.device).view(1, 3, 1, 1, 1)

    @torch.inference_mode()
    def extract_clip(self, video_path: Path, *, start_seconds: float, end_seconds: float) -> np.ndarray:
        frames = read_rgb_clip_frames(
            video_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            frame_count=self.frames,
            image_size=self.image_size,
        )
        tensor = torch.from_numpy(frames).to(self.device)
        tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        output = self.model(tensor)
        return output.detach().float().cpu().numpy().ravel()

    @torch.inference_mode()
    def extract_frames(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        if not frames_bgr:
            frames_bgr = [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)]
        indices = np.linspace(0, len(frames_bgr) - 1, num=self.frames, dtype=int).tolist()
        frames = []
        for index in indices:
            frame = frames_bgr[index]
            if frame is None or getattr(frame, "size", 0) == 0:
                frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(resize_center_crop(frame_rgb, self.image_size))
        tensor = torch.from_numpy(np.stack(frames).astype(np.uint8)).to(self.device)
        tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        output = self.model(tensor)
        return output.detach().float().cpu().numpy().ravel()


def build_video_model(model_name: str, *, allow_fallback: bool = False) -> tuple[torch.nn.Module, str]:
    if model_name == "i3d_r50":
        try:
            torch.hub.set_dir(str(Path(".torch_hub").resolve()))
            model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=True)
            if hasattr(model, "blocks") and len(model.blocks) > 0:
                model.blocks[-1] = torch.nn.Identity()
            return model, "i3d_r50"
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(
                    "i3d_r50 pretrained load failed. Install pytorchvideo/fvcore/iopath or rerun with --allow-fallback."
                ) from exc
            print(f"[rgb] i3d_r50 pretrained load failed, falling back to r3d_18: {exc}")
            model_name = "r3d_18"
    try:
        from torchvision.models.video import mc3_18, r2plus1d_18, r3d_18
        from torchvision.models.video import MC3_18_Weights, R2Plus1D_18_Weights, R3D_18_Weights
    except Exception as exc:
        raise RuntimeError("RGB feature 추출에는 torchvision video model이 필요합니다.") from exc

    builders = {
        "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT),
        "mc3_18": (mc3_18, MC3_18_Weights.DEFAULT),
        "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights.DEFAULT),
    }
    builder, weights = builders[model_name]
    try:
        model = builder(weights=weights)
    except Exception:
        model = builder(weights=None)
    if hasattr(model, "fc"):
        model.fc = torch.nn.Identity()
    return model, model_name


def read_rgb_clip_frames(
    video_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    frame_count: int,
    image_size: int,
) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"video open failed: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or total_frames <= 0:
        capture.release()
        raise RuntimeError(f"video metadata missing: {video_path}")
    start_frame = max(int(max(start_seconds, 0.0) * fps), 0)
    end_frame = int(end_seconds * fps) if end_seconds > 0 else total_frames
    end_frame = min(max(end_frame, start_frame + 1), total_frames)
    indices = np.linspace(start_frame, end_frame - 1, num=frame_count, dtype=int).tolist()
    frames = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            if frames:
                frames.append(frames[-1].copy())
                continue
            frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_center_crop(frame, image_size)
        frames.append(frame)
    capture.release()
    return np.stack(frames).astype(np.uint8)


def resize_center_crop(frame: np.ndarray, image_size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = image_size / max(min(height, width), 1)
    resized = cv2.resize(frame, (max(int(width * scale), image_size), max(int(height * scale), image_size)))
    new_height, new_width = resized.shape[:2]
    y0 = max((new_height - image_size) // 2, 0)
    x0 = max((new_width - image_size) // 2, 0)
    return resized[y0 : y0 + image_size, x0 : x0 + image_size]


def build_rgb_feature_path(output_dir: Path, *, split_name: str, filekey: str, row: dict) -> Path:
    feature_path = output_dir / split_name / filekey / f"{safe_id(row)}.npz"
    if should_use_short_rgb_feature_path(feature_path):
        return build_short_rgb_feature_path(output_dir, split_name=split_name, filekey=filekey, row=row)
    return feature_path


def build_short_rgb_feature_path(output_dir: Path, *, split_name: str, filekey: str, row: dict) -> Path:
    item_id = str(row.get("item_id") or row.get("source_item_id") or row.get("pose_path") or "")
    digest_source = json.dumps(
        {
            "filekey": filekey,
            "item_id": item_id,
            "clip_start_seconds": row.get("clip_start_seconds"),
            "clip_end_seconds": row.get("clip_end_seconds"),
            "target_label": row.get("target_label"),
            "split": split_name,
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:20]
    role = safe_path_part(str(row.get("clip_role") or "clip"))[:24]
    label = safe_path_part(str(row.get("target_label") or "unknown"))[:24]
    return output_dir / split_name / filekey / f"{filekey}__{label}__{role}__{digest}.npz"


def should_use_short_rgb_feature_path(feature_path: Path) -> bool:
    text = str(feature_path)
    return len(text) >= MAX_RGB_FEATURE_PATH_LENGTH or len(feature_path.name) >= 140


def save_rgb_feature_with_fallback(
    feature_path: Path,
    feature: np.ndarray,
    *,
    output_dir: Path,
    split_name: str,
    filekey: str,
    row: dict,
) -> Path:
    payload = feature.astype(np.float32, copy=False)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(feature_path, feature=payload)
        return feature_path
    except (FileNotFoundError, OSError):
        fallback_path = build_short_rgb_feature_path(output_dir, split_name=split_name, filekey=filekey, row=row)
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fallback_path, feature=payload)
        return fallback_path


def safe_id(row: dict) -> str:
    filekey = resolve_row_filekey(row)
    value = f"{filekey}__{row.get('item_id') or row.get('pose_path') or ''}"
    return safe_path_part(value)[-120:]


def safe_path_part(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "").strip())
    return safe.strip("._-") or "unknown"


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
    for key in ("video_path", "source_video_path", "rgb_feature_path", "pose_path"):
        value = str(row.get(key) or metadata.get(key) or "").strip()
        for part in Path(value).parts:
            if part.lower().startswith("job_"):
                parts = part.split("_")
                return parts[1] if len(parts) > 1 and parts[1].isdigit() else part
    return "unknown_filekey"


if __name__ == "__main__":
    main()
