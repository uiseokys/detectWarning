from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from action_training_pipeline import (
    build_stage_timing_entry,
    current_timestamp_iso,
    download_dataset,
    load_config,
    resolve_paths,
    split_dataset,
    validate_source_config,
    write_pipeline_status,
)
from reporting import write_json_atomic


LABELS = ["normal", "violence"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class Metrics:
    loss: float
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    confusion_matrix: list[list[int]]
    per_class: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fight/noFight 2D CNN + BiLSTM attention model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reuse-split", action="store_true", help="Reuse current_split manifests when present.")
    return parser.parse_args()


def resolve_binary_labels(config: dict) -> list[str]:
    cfg = config.get("fight_bilstm") if isinstance(config.get("fight_bilstm"), dict) else {}
    raw_labels = cfg.get("labels")
    if isinstance(raw_labels, list):
        labels = [str(label).strip() for label in raw_labels if str(label).strip()]
        if len(labels) == 2 and "normal" in labels:
            return labels
    positive_label = str(cfg.get("positive_label") or "violence").strip() or "violence"
    return ["normal", positive_label]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not isinstance(path, Path) or not path.exists() or not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("target_label") or "") in LABELS and Path(str(row.get("video_path") or "")).exists():
                rows.append(row)
    return rows


def materialize_pose_binary_manifest(rows: list[dict], output_path: Path) -> Path:
    label_to_idx = {label: index for index, label in enumerate(LABELS)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            label = str(row.get("target_label") or "").strip()
            pose_path = str(row.get("pose_path") or "").strip()
            if label not in label_to_idx or not pose_path or not Path(pose_path).exists():
                continue
            next_row = dict(row)
            next_row["label"] = label
            next_row["target_label"] = label
            next_row["label_idx"] = label_to_idx[label]
            handle.write(json.dumps(next_row, ensure_ascii=False) + "\n")
    return output_path


def sample_video_frames(video_path: Path, frame_count: int, image_size: int) -> torch.Tensor:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"video open failed: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        capture.release()
        raise RuntimeError(f"video has no frames: {video_path}")
    indices = np.linspace(0, max(total_frames - 1, 0), num=frame_count, dtype=int).tolist()
    frames: list[torch.Tensor] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok or frame is None:
            frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_center_crop(frame, image_size)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frames.append((tensor - IMAGENET_MEAN) / IMAGENET_STD)
    capture.release()
    return torch.stack(frames, dim=0)


def resize_center_crop(frame: np.ndarray, image_size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = image_size / float(min(height, width))
    resized = cv2.resize(frame, (max(int(round(width * scale)), image_size), max(int(round(height * scale)), image_size)))
    y = max((resized.shape[0] - image_size) // 2, 0)
    x = max((resized.shape[1] - image_size) // 2, 0)
    return resized[y : y + image_size, x : x + image_size].copy()


class FightFrameSequenceDataset(Dataset):
    def __init__(self, rows: list[dict], *, frame_count: int, image_size: int, augment: bool = False) -> None:
        self.rows = rows
        self.frame_count = frame_count
        self.image_size = image_size
        self.augment = bool(augment)
        self.label_to_idx = {label: index for index, label in enumerate(LABELS)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        frames = sample_video_frames(Path(str(row["video_path"])), self.frame_count, self.image_size)
        if self.augment:
            frames = augment_frame_sequence(frames)
        label = self.label_to_idx[str(row["target_label"])]
        return frames, torch.tensor(label, dtype=torch.long)


def augment_frame_sequence(frames: torch.Tensor) -> torch.Tensor:
    output = frames
    if random.random() < 0.5:
        output = torch.flip(output, dims=[3])
    if random.random() < 0.35:
        brightness = random.uniform(0.88, 1.12)
        contrast = random.uniform(0.90, 1.10)
        mean = output.mean(dim=(2, 3), keepdim=True)
        output = (output - mean) * contrast + mean
        output = output * brightness
    return output


class CnnBiLstmAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        freeze_cnn: bool,
        pretrained_cnn: bool,
        unfreeze_cnn_tail: bool = False,
    ) -> None:
        super().__init__()
        self.cnn, feature_dim, pretrained_loaded = build_frame_encoder(pretrained=pretrained_cnn)
        self.pretrained_loaded = pretrained_loaded
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
            if unfreeze_cnn_tail:
                unfreeze_frame_encoder_tail(self.cnn)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=max(num_layers, 1),
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, len(LABELS))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = frames.shape
        features = self.cnn(frames.reshape(batch * steps, channels, height, width))
        features = features.reshape(batch, steps, -1)
        sequence, _hidden = self.lstm(features)
        weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return self.classifier(self.dropout(pooled))


def unfreeze_frame_encoder_tail(model: nn.Module) -> None:
    for attr in ("layer4", "features"):
        module = getattr(model, attr, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True
            return
    trainable_params = list(model.parameters())[-4:]
    for param in trainable_params:
        param.requires_grad = True


def build_optimizer_parameter_groups(
    model: CnnBiLstmAttention,
    *,
    base_lr: float,
    cnn_lr_multiplier: float,
) -> list[dict]:
    cnn_param_ids = {id(param) for param in model.cnn.parameters() if param.requires_grad}
    cnn_params = [param for param in model.cnn.parameters() if param.requires_grad]
    head_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in cnn_param_ids
    ]
    groups: list[dict] = []
    if head_params:
        groups.append({"params": head_params, "lr": float(base_lr)})
    if cnn_params:
        groups.append({"params": cnn_params, "lr": float(base_lr) * float(cnn_lr_multiplier)})
    return groups


def build_frame_encoder(*, pretrained: bool) -> tuple[nn.Module, int, bool]:
    try:
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        try:
            model = resnet18(weights=weights)
            pretrained_loaded = bool(weights is not None)
        except Exception:
            model = resnet18(weights=None)
            pretrained_loaded = False
        feature_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        return model, feature_dim, pretrained_loaded
    except Exception:
        return SmallFrameCnn(), 256, False


class SmallFrameCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.net(frames)


def class_weights(rows: list[dict], device: torch.device) -> torch.Tensor:
    counts = Counter(str(row.get("target_label")) for row in rows)
    total = sum(counts.values()) or 1
    weights = [total / max(len(LABELS) * counts.get(label, 1), 1) for label in LABELS]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_sampler(rows: list[dict]) -> WeightedRandomSampler:
    counts = Counter(str(row.get("target_label")) for row in rows)
    weights = [1.0 / max(counts.get(str(row.get("target_label")), 1), 1) for row in rows]
    return WeightedRandomSampler(weights, num_samples=len(rows), replacement=True)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Metrics:
    y_true, y_pred, _probabilities, mean_loss = predict_loader(model, loader, device, criterion)
    return compute_metrics(y_true, y_pred, mean_loss)


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[list[int], list[int], np.ndarray, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    probability_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for frames, labels in loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(frames)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            batch_size = int(labels.numel())
            total_loss += float(loss.detach().cpu()) * batch_size
            total_count += batch_size
            y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
            y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist())
            probability_rows.append(probabilities.detach().cpu().numpy().astype(np.float64))
    probability_matrix = np.concatenate(probability_rows, axis=0) if probability_rows else np.zeros((0, len(LABELS)))
    return y_true, y_pred, probability_matrix, total_loss / max(total_count, 1)


def compute_metrics(y_true: list[int], y_pred: list[int], loss: float) -> Metrics:
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for target, pred in zip(y_true, y_pred):
        if 0 <= target < len(LABELS) and 0 <= pred < len(LABELS):
            confusion[target, pred] += 1
    per_class: list[dict] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for index, label in enumerate(LABELS):
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
        per_class.append(
            {
                "class_index": index,
                "label": label,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": support,
            }
        )
    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else 0.0
    return Metrics(
        loss=float(loss),
        accuracy=accuracy,
        macro_f1=float(sum(f1s) / len(f1s)),
        balanced_accuracy=float(sum(recalls) / len(recalls)),
        confusion_matrix=confusion.astype(int).tolist(),
        per_class=per_class,
    )


def select_binary_threshold(
    y_true: list[int],
    violence_probabilities: np.ndarray,
    *,
    loss: float = 0.0,
    min_normal_recall: float = 0.65,
) -> tuple[float, Metrics]:
    probabilities = np.nan_to_num(np.asarray(violence_probabilities, dtype=np.float64).ravel(), nan=0.0)
    if probabilities.size != len(y_true):
        raise ValueError("violence_probabilities length must match y_true")
    thresholds = np.round(np.arange(0.30, 0.901, 0.01), 2)
    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, (probabilities >= best_threshold).astype(int).tolist(), loss)
    best_score = threshold_selection_score(best_metrics, min_normal_recall=min_normal_recall)
    for threshold in thresholds:
        y_pred = (probabilities >= float(threshold)).astype(int).tolist()
        metrics = compute_metrics(y_true, y_pred, loss)
        score = threshold_selection_score(metrics, min_normal_recall=min_normal_recall)
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12
            and metrics.per_class[0]["recall"] > best_metrics.per_class[0]["recall"]
        ):
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def threshold_selection_score(metrics: Metrics, *, min_normal_recall: float) -> float:
    normal_recall = float(metrics.per_class[0].get("recall") or 0.0)
    violence_recall = float(metrics.per_class[1].get("recall") or 0.0)
    penalty = max(0.0, min_normal_recall - normal_recall) * 0.35
    collapse_penalty = max(0.0, 0.55 - violence_recall) * 0.15
    return float(metrics.macro_f1) + normal_recall * 0.04 - penalty - collapse_penalty


def fuse_rgb_pose_probabilities(
    rows: list[dict],
    rgb_violence_probabilities: np.ndarray,
    pose_violence_by_item_id: dict[str, float],
    *,
    pose_weight: float,
) -> np.ndarray:
    rgb_values = np.nan_to_num(np.asarray(rgb_violence_probabilities, dtype=np.float64).ravel(), nan=0.0)
    weight = min(max(float(pose_weight), 0.0), 1.0)
    fused = rgb_values.copy()
    for index, row in enumerate(rows[: rgb_values.size]):
        item_id = str(row.get("item_id") or "").strip()
        if not item_id or item_id not in pose_violence_by_item_id:
            continue
        pose_value = float(np.clip(pose_violence_by_item_id[item_id], 0.0, 1.0))
        fused[index] = (1.0 - weight) * float(rgb_values[index]) + weight * pose_value
    return np.clip(fused, 0.0, 1.0)


def config_float(config: dict, key: str, default: float) -> float:
    value = config.get(key)
    if value is None or value == "":
        return float(default)
    return float(value)


def metrics_to_dict(metrics: Metrics) -> dict:
    return {
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "balanced_accuracy": metrics.balanced_accuracy,
        "confusion_matrix": metrics.confusion_matrix,
        "per_class": metrics.per_class,
    }


def train_pose_binary_fusion(
    *,
    config: dict,
    paths: dict,
    rows_by_split: dict[str, list[dict]],
    output_dir: Path,
) -> dict:
    cfg = config.get("fight_bilstm") if isinstance(config.get("fight_bilstm"), dict) else {}
    pose_dir = output_dir / "pose_binary"
    manifest_dir = pose_dir / "manifests"
    manifests = {
        split: materialize_pose_binary_manifest(rows, manifest_dir / f"{split}.jsonl")
        for split, rows in rows_by_split.items()
        if split in {"train", "val", "test"}
    }
    counts = {split: count_jsonl(path) for split, path in manifests.items()}
    payload: dict = {
        "enabled": True,
        "labels": LABELS,
        "train_samples": counts.get("train", 0),
        "val_samples": counts.get("val", 0),
        "test_samples": counts.get("test", 0),
        "pose_weight": config_float(cfg, "pose_fusion_weight", 0.25),
    }
    if counts.get("train", 0) <= 0 or counts.get("val", 0) <= 0:
        payload.update({"available": False, "reason": "not_enough_pose_binary_samples"})
        write_json_atomic(pose_dir / "metrics.json", payload)
        return payload
    from action_model import train_action_classifier

    try:
        artifacts = train_action_classifier(
            train_manifest=manifests["train"],
            val_manifest=manifests["val"],
            output_dir=pose_dir,
            labels=LABELS,
            epochs=int(cfg.get("pose_epochs") or 18),
            batch_size=int(cfg.get("pose_batch_size") or 48),
            eval_batch_size=int(cfg.get("pose_eval_batch_size") or 96),
            learning_rate=float(cfg.get("pose_learning_rate") or 3e-4),
            weight_decay=float(cfg.get("pose_weight_decay") or 0.003),
            hidden_dim=int(cfg.get("pose_hidden_dim") or 128),
            num_layers=int(cfg.get("pose_num_layers") or 2),
            dropout=float(cfg.get("pose_dropout") or 0.35),
            temporal_pooling=str(cfg.get("pose_temporal_pooling") or "mean_max"),
            label_smoothing=float(cfg.get("pose_label_smoothing") or 0.02),
            loss_name=str(cfg.get("pose_loss") or "focal"),
            focal_gamma=float(cfg.get("pose_focal_gamma") or 1.0),
            class_weight="balanced",
            balanced_sampler=True,
            grad_clip_norm=1.0,
            seed=int(cfg.get("seed") or config.get("split", {}).get("seed") or 42),
            deterministic=False,
            num_workers=0,
            device=str(cfg.get("device") or config.get("training", {}).get("device") or "cuda"),
            amp=True,
            amp_dtype="auto",
            compile_model=False,
            dataset_cache_size=512,
            prefetch_factor=2,
            persistent_workers=False,
            pin_memory="auto",
            early_stopping_patience=int(cfg.get("pose_early_stopping_patience") or 5),
            early_stopping_min_delta=0.001,
            selection_metric="macro_f1",
            overfit_guard_enabled=True,
            overfit_guard_min_epoch=6,
            overfit_guard_loss_gap=1.5,
            overfit_guard_patience=2,
            progress_path=pose_dir / "training_progress.json",
            resume_from=None,
            max_duplicate_pose_label_samples=0,
        )
        metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        payload.update({"available": False, "reason": str(exc)})
        write_json_atomic(pose_dir / "metrics.json", payload)
        return payload
    holdout_metrics = {}
    if counts.get("test", 0) > 0:
        try:
            holdout_metrics = evaluate_pose_binary_checkpoint(
                model_path=artifacts.best_model_path,
                manifest_path=manifests.get("test"),
                batch_size=int(cfg.get("pose_eval_batch_size") or 96),
                device=str(cfg.get("device") or config.get("training", {}).get("device") or "cuda"),
            ).get("metrics", {})
        except Exception as exc:
            payload["holdout_warning"] = str(exc)
    payload.update(
        {
            "available": True,
            "metrics_path": str(artifacts.metrics_path),
            "model_path": str(artifacts.best_model_path),
            "final_validation": metrics.get("final_validation", {}),
            "holdout_test": holdout_metrics,
        }
    )
    write_json_atomic(pose_dir / "fusion_summary.json", payload)
    return payload


def count_jsonl(path: Path | None) -> int:
    if not isinstance(path, Path) or not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def evaluate_pose_binary_checkpoint(*, model_path: Path, manifest_path: Path | None, batch_size: int, device: str) -> dict:
    if not isinstance(manifest_path, Path) or not manifest_path.exists():
        return {"probabilities": {}, "metrics": {}}
    from action_model import PoseSequenceDataset, _build_temporal_pose_classifier

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    labels = [str(label) for label in checkpoint.get("labels") or LABELS]
    use_device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    dataset = PoseSequenceDataset(manifest_path, cache_size=0, max_duplicate_pose_label_samples=0)
    loader = DataLoader(dataset, batch_size=max(int(batch_size), 1), shuffle=False, num_workers=0)
    first_pose, _first_mask, _first_label = dataset[0]
    model = _build_temporal_pose_classifier(
        temporal_pooling=str(checkpoint.get("temporal_pooling") or "mean"),
        num_joints=int(checkpoint.get("num_joints") or first_pose.shape[1]),
        input_dim=int(checkpoint.get("input_dim") or first_pose.shape[2]),
        hidden_dim=int(checkpoint.get("hidden_dim") or 128),
        num_layers=int(checkpoint.get("num_layers") or 2),
        num_classes=len(labels),
        dropout=float(checkpoint.get("dropout") or 0.0),
    ).to(use_device)
    model.load_state_dict(checkpoint.get("model_state_dict") or {})
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities_by_item: dict[str, float] = {}
    violence_index = labels.index("violence") if "violence" in labels else min(1, len(labels) - 1)
    sample_offset = 0
    with torch.inference_mode():
        for pose, mask, label_tensor in loader:
            pose = pose.to(use_device)
            mask = mask.to(use_device)
            probs = torch.softmax(model(pose, mask).float(), dim=1).cpu().numpy()
            pred = np.argmax(probs, axis=1).astype(int).tolist()
            true = label_tensor.numpy().astype(int).tolist()
            y_true.extend(true)
            y_pred.extend(pred)
            for batch_index, probability_row in enumerate(probs):
                sample = dataset.samples[sample_offset + batch_index]
                item_id = str(sample.get("item_id") or "").strip()
                if item_id:
                    probabilities_by_item[item_id] = float(probability_row[violence_index])
            sample_offset += len(true)
    return {
        "probabilities": probabilities_by_item,
        "metrics": metrics_to_dict(compute_metrics(y_true, y_pred, 0.0)),
    }


def train_bilstm(config: dict, paths: dict, manifests: dict[str, Path]) -> dict:
    global LABELS
    cfg = config.get("fight_bilstm") if isinstance(config.get("fight_bilstm"), dict) else {}
    LABELS = resolve_binary_labels(config)
    positive_label = next((label for label in LABELS if label != "normal"), LABELS[-1])
    output_dir = paths["artifacts_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = int(cfg.get("frame_count") or 10)
    image_size = int(cfg.get("image_size") or 224)
    batch_size = int(cfg.get("batch_size") or 12)
    epochs = int(cfg.get("epochs") or 24)
    seed = int(cfg.get("seed") or config.get("split", {}).get("seed") or 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rows_by_split = {split: read_jsonl(path) for split, path in manifests.items()}
    pose_rows_by_split = {
        "train": read_jsonl(paths.get("current_prepared_train", Path())),
        "val": read_jsonl(paths.get("current_prepared_val", Path())),
        "test": read_jsonl(paths.get("current_prepared_test", Path())),
    }
    train_rows = rows_by_split.get("train") or []
    val_rows = rows_by_split.get("val") or []
    test_rows = rows_by_split.get("test") or []
    if not train_rows or not val_rows:
        raise RuntimeError("BiLSTM 학습에 필요한 train/val split manifest가 부족합니다.")
    if len(set(row["target_label"] for row in train_rows)) < 2 or len(set(row["target_label"] for row in val_rows)) < 2:
        raise RuntimeError("BiLSTM 학습에는 train/val 양쪽에 normal, violence가 모두 필요합니다.")

    device = torch.device(str(cfg.get("device") or config.get("training", {}).get("device") or "cuda") if torch.cuda.is_available() else "cpu")
    train_dataset = FightFrameSequenceDataset(
        train_rows,
        frame_count=frame_count,
        image_size=image_size,
        augment=bool(cfg.get("augment_train", True)),
    )
    val_dataset = FightFrameSequenceDataset(val_rows, frame_count=frame_count, image_size=image_size)
    test_dataset = FightFrameSequenceDataset(test_rows, frame_count=frame_count, image_size=image_size) if test_rows else None
    num_workers = int(cfg.get("num_workers") or 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=build_sampler(train_rows),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_dataset, batch_size=max(batch_size, 1), shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    test_loader = (
        DataLoader(test_dataset, batch_size=max(batch_size, 1), shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
        if test_dataset is not None
        else None
    )

    model = CnnBiLstmAttention(
        hidden_dim=int(cfg.get("hidden_dim") or 192),
        num_layers=int(cfg.get("num_layers") or 1),
        dropout=float(cfg.get("dropout") or 0.35),
        freeze_cnn=bool(cfg.get("freeze_cnn", True)),
        pretrained_cnn=bool(cfg.get("pretrained_cnn", True)),
        unfreeze_cnn_tail=bool(cfg.get("unfreeze_cnn_tail", True)),
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_rows, device),
        label_smoothing=config_float(cfg, "label_smoothing", 0.0),
    )
    base_lr = float(cfg.get("learning_rate") or 3e-4)
    optimizer = torch.optim.AdamW(
        build_optimizer_parameter_groups(
            model,
            base_lr=base_lr,
            cnn_lr_multiplier=config_float(cfg, "cnn_lr_multiplier", 1.0),
        ),
        lr=base_lr,
        weight_decay=float(cfg.get("weight_decay") or 1e-3),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(cfg.get("amp", True)))
    patience = int(cfg.get("early_stopping_patience") or 6)
    selection_metric = str(cfg.get("selection_metric") or "macro_f1")
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    progress_path = output_dir / "training_progress.json"
    started_at = datetime.now(UTC).isoformat()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for frames, labels in train_loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and bool(cfg.get("amp", True))):
                logits = model(frames)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip_norm") or 1.0))
            scaler.step(optimizer)
            scaler.update()
            batch_size_actual = int(labels.numel())
            train_loss += float(loss.detach().cpu()) * batch_size_actual
            train_count += batch_size_actual
        val_metrics = evaluate(model, val_loader, device, criterion)
        row = {
            "epoch": epoch,
            "epochs_total": epochs,
            "train_loss": train_loss / max(train_count, 1),
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
            "val_macro_f1": val_metrics.macro_f1,
            "val_balanced_accuracy": val_metrics.balanced_accuracy,
        }
        history.append(row)
        score = float(row.get(f"val_{selection_metric}") or row.get("val_macro_f1") or 0.0)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "labels": LABELS,
                    "model_type": "resnet18_bilstm_attention",
                    "frame_count": frame_count,
                    "image_size": image_size,
                    "config": cfg,
                    "pretrained_loaded": bool(getattr(model, "pretrained_loaded", False)),
                },
                output_dir / "best_fight_bilstm.pt",
            )
        else:
            stale_epochs += 1
        write_json_atomic(
            progress_path,
            {
                "state": "running",
                "model_type": "resnet18_bilstm_attention",
                "labels": LABELS,
                "epoch": epoch,
                "epochs_total": epochs,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_score,
                "latest": row,
                "history": history,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        print(
            f"[bilstm][epoch] {epoch:03d}/{epochs:03d} "
            f"train_loss={row['train_loss']:.4f} val_acc={val_metrics.accuracy:.4f} "
            f"val_f1={val_metrics.macro_f1:.4f} best={best_score:.4f}",
            flush=True,
        )
        if stale_epochs >= patience:
            print(f"[bilstm] early stopping at epoch {epoch} patience={patience}", flush=True)
            break

    checkpoint = torch.load(output_dir / "best_fight_bilstm.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_true, val_argmax_pred, val_probs, val_loss = predict_loader(model, val_loader, device, criterion)
    final_validation_argmax = compute_metrics(val_true, val_argmax_pred, val_loss)
    min_normal_recall = float(cfg.get("threshold_min_normal_recall") or 0.70)
    selected_threshold, final_validation = select_binary_threshold(
        val_true,
        val_probs[:, 1] if val_probs.size else np.zeros((0,), dtype=np.float64),
        loss=val_loss,
        min_normal_recall=min_normal_recall,
    )
    holdout_test = None
    holdout_test_argmax = None
    if test_loader is not None:
        test_true, test_argmax_pred, test_probs, test_loss = predict_loader(model, test_loader, device, criterion)
        holdout_test_argmax = compute_metrics(test_true, test_argmax_pred, test_loss)
        holdout_test = compute_metrics(
            test_true,
            (test_probs[:, 1] >= selected_threshold).astype(int).tolist() if test_probs.size else [],
            test_loss,
        )
    pose_fusion = {}
    fusion_validation = None
    fusion_holdout = None
    if bool(cfg.get("pose_fusion_enabled", True)):
        pose_fusion = train_pose_binary_fusion(
            config=config,
            paths=paths,
            rows_by_split=pose_rows_by_split,
            output_dir=output_dir,
        )
        if pose_fusion.get("available"):
            try:
                pose_model_path = Path(str(pose_fusion.get("model_path")))
                val_pose = evaluate_pose_binary_checkpoint(
                    model_path=pose_model_path,
                    manifest_path=output_dir / "pose_binary" / "manifests" / "val.jsonl",
                    batch_size=int(cfg.get("pose_eval_batch_size") or 96),
                    device=str(cfg.get("device") or config.get("training", {}).get("device") or "cuda"),
                )
                fused_val_probs = fuse_rgb_pose_probabilities(
                    val_rows,
                    val_probs[:, 1] if val_probs.size else np.zeros((0,), dtype=np.float64),
                    val_pose.get("probabilities", {}),
                    pose_weight=float(pose_fusion.get("pose_weight") or 0.25),
                )
                _fusion_threshold, fusion_validation = select_binary_threshold(
                    val_true,
                    fused_val_probs,
                    loss=val_loss,
                    min_normal_recall=min_normal_recall,
                )
                pose_fusion["decision_threshold"] = _fusion_threshold
                if test_loader is not None and test_probs is not None:
                    test_pose = evaluate_pose_binary_checkpoint(
                        model_path=pose_model_path,
                        manifest_path=output_dir / "pose_binary" / "manifests" / "test.jsonl",
                        batch_size=int(cfg.get("pose_eval_batch_size") or 96),
                        device=str(cfg.get("device") or config.get("training", {}).get("device") or "cuda"),
                    )
                    fused_test_probs = fuse_rgb_pose_probabilities(
                        test_rows,
                        test_probs[:, 1] if test_probs.size else np.zeros((0,), dtype=np.float64),
                        test_pose.get("probabilities", {}),
                        pose_weight=float(pose_fusion.get("pose_weight") or 0.25),
                    )
                    fusion_holdout = compute_metrics(
                        test_true,
                        (fused_test_probs >= _fusion_threshold).astype(int).tolist(),
                        test_loss,
                    )
            except Exception as exc:
                pose_fusion["fusion_warning"] = str(exc)
    rgb_holdout_score = float(holdout_test.macro_f1) if holdout_test is not None else float(final_validation.macro_f1)
    fusion_holdout_score = float(fusion_holdout.macro_f1) if fusion_holdout is not None else -1.0
    recommended_result = "fusion" if fusion_holdout_score >= rgb_holdout_score else "rgb_bilstm"
    display_validation = fusion_validation if recommended_result == "fusion" and fusion_validation is not None else final_validation
    display_holdout = fusion_holdout if recommended_result == "fusion" and fusion_holdout is not None else holdout_test
    metrics = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "started_at": started_at,
        "model_type": "resnet18_bilstm_attention",
        "labels": LABELS,
        "positive_label": positive_label,
        "frame_count": frame_count,
        "image_size": image_size,
        "decision_threshold": {
            "violence_probability": selected_threshold,
            "selected_on": "validation",
            "min_normal_recall": min_normal_recall,
            "score": threshold_selection_score(final_validation, min_normal_recall=min_normal_recall),
        },
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_score, 6),
        "history": history,
        "train_distribution": dict(Counter(row["target_label"] for row in train_rows)),
        "val_distribution": dict(Counter(row["target_label"] for row in val_rows)),
        "test_distribution": dict(Counter(row["target_label"] for row in test_rows)),
        "final_validation": metrics_to_dict(display_validation),
        "rgb_final_validation": metrics_to_dict(final_validation),
        "final_validation_argmax": metrics_to_dict(final_validation_argmax),
        "holdout_test": metrics_to_dict(display_holdout) if display_holdout is not None else {},
        "rgb_holdout_test": metrics_to_dict(holdout_test) if holdout_test is not None else {},
        "holdout_test_argmax": metrics_to_dict(holdout_test_argmax) if holdout_test_argmax is not None else {},
        "pose_binary_fusion": pose_fusion,
        "fusion_validation": metrics_to_dict(fusion_validation) if fusion_validation is not None else {},
        "fusion_holdout_test": metrics_to_dict(fusion_holdout) if fusion_holdout is not None else {},
        "recommended_result": recommended_result,
        "recommended_validation": metrics_to_dict(display_validation),
        "recommended_holdout_test": metrics_to_dict(display_holdout) if display_holdout is not None else {},
        "artifacts": {
            "model_path": str(output_dir / "best_fight_bilstm.pt"),
            "progress_path": str(progress_path),
        },
        "pretrained_loaded": bool(checkpoint.get("pretrained_loaded")),
    }
    checkpoint["decision_threshold"] = metrics["decision_threshold"]
    torch.save(checkpoint, output_dir / "best_fight_bilstm.pt")
    write_json_atomic(output_dir / "metrics.json", metrics)
    write_json_atomic(output_dir / "labels.json", LABELS)
    write_json_atomic(progress_path, {**metrics, "state": "completed", "updated_at": datetime.now(UTC).isoformat()})
    return metrics


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    validate_source_config(config, config_path, stage="download")
    paths = resolve_paths(config, config_path.parent)
    pipeline_started_at = current_timestamp_iso()
    stage_timings: dict[str, dict] = {}
    write_pipeline_status(
        paths,
        stage="starting",
        state="running",
        message="fight/noFight BiLSTM 파인튜닝을 시작합니다.",
        config_path=str(config_path),
        pipeline_started_at=pipeline_started_at,
        stage_progress=0.0,
    )
    try:
        split_paths = {
            "train": paths["current_split_train"],
            "val": paths["current_split_val"],
            "test": paths["current_split_test"],
        }
        if not args.reuse_split or not split_paths["train"].exists() or not split_paths["val"].exists():
            stage_started_at = current_timestamp_iso()
            write_pipeline_status(
                paths,
                stage="download",
                state="running",
                message="Surveillance Fight Dataset을 다운로드/동기화하는 중입니다.",
                stage_progress=0.05,
                pipeline_started_at=pipeline_started_at,
                stage_started_at=stage_started_at,
                stage_timings=stage_timings,
            )
            downloaded = download_dataset(config, paths)
            split_paths = split_dataset(downloaded, config, paths)
            stage_timings["download"] = build_stage_timing_entry(stage_started_at, current_timestamp_iso())

        stage_started_at = current_timestamp_iso()
        write_pipeline_status(
            paths,
            stage="train",
            state="running",
            message="2D CNN + BiLSTM attention fight/noFight 모델을 학습하는 중입니다.",
            stage_progress=0.35,
            pipeline_started_at=pipeline_started_at,
            stage_started_at=stage_started_at,
            stage_timings=stage_timings,
        )
        metrics = train_bilstm(config, paths, split_paths)
        stage_timings["train"] = build_stage_timing_entry(stage_started_at, current_timestamp_iso())
        write_pipeline_status(
            paths,
            stage="completed",
            state="completed",
            message="fight/noFight BiLSTM 파인튜닝이 완료되었습니다.",
            stage_progress=1.0,
            pipeline_started_at=pipeline_started_at,
            stage_started_at=stage_started_at,
            stage_timings=stage_timings,
            final_validation=metrics.get("final_validation", {}),
            holdout_test=metrics.get("holdout_test", {}),
        )
    except Exception as exc:
        write_pipeline_status(
            paths,
            stage="error",
            state="error",
            message=str(exc),
            stage_progress=1.0,
            pipeline_started_at=pipeline_started_at,
            stage_timings=stage_timings,
        )
        raise


if __name__ == "__main__":
    main()
