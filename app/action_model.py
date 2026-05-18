from __future__ import annotations

import inspect
import json
import os
import platform
import random
import time
import uuid
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from reporting import analyze_class_balance, write_json_atomic

DATALOADER_SUPPORTS_PIN_MEMORY_DEVICE = "pin_memory_device" in inspect.signature(DataLoader).parameters
POSE_COORD_CLIP_RANGE = (-0.5, 1.5)
POSE_CONFIDENCE_CLIP_RANGE = (0.0, 1.0)
MAX_MISCLASSIFIED_EXAMPLES = 200
MAX_ERROR_EXAMPLES_PER_GROUP = 20
MAX_DUPLICATE_POSE_LABEL_SAMPLES = 3


@dataclass
class TrainingArtifacts:
    best_model_path: Path
    metrics_path: Path
    labels_path: Path
    history: list[dict]


class PoseSequenceDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        *,
        cache_size: int = 0,
        max_duplicate_pose_label_samples: int = MAX_DUPLICATE_POSE_LABEL_SAMPLES,
    ) -> None:
        self.samples = []
        self.skipped_missing_pose_path = 0
        self.skipped_conflicting_pose_label = 0
        self.skipped_duplicate_pose_label = 0
        self.cache_size = max(int(cache_size), 0)
        self._cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()
        loaded_samples: list[dict] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                pose_path = str(sample.get("pose_path") or "").strip()
                if not pose_path or not Path(pose_path).exists():
                    self.skipped_missing_pose_path += 1
                    continue
                loaded_samples.append(sample)
        filtered_samples, self.skipped_conflicting_pose_label = _filter_conflicting_pose_label_samples(loaded_samples)
        self.max_duplicate_pose_label_samples = int(max_duplicate_pose_label_samples)
        self.samples, self.skipped_duplicate_pose_label = _limit_duplicate_pose_label_samples(
            filtered_samples,
            max_per_pose_label=self.max_duplicate_pose_label_samples,
        )
        if not self.samples:
            raise RuntimeError(
                f"Pose 학습용 샘플이 없습니다: {manifest_path} "
                f"(pose_path 없음/파일 없음: {self.skipped_missing_pose_path})"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        cache_key = "::".join(
            [
                str(sample.get("pose_path") or ""),
                str(sample.get("label_idx") or ""),
                str(sample.get("item_id") or index),
            ]
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

        loaded_sample = self._load_sample(sample)
        if self.cache_size > 0:
            self._cache[cache_key] = loaded_sample
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return loaded_sample

    @staticmethod
    def _load_sample(sample: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pose_path = Path(sample["pose_path"])
        with np.load(pose_path, allow_pickle=False) as loaded:
            pose = np.asarray(loaded["pose"], dtype=np.float32)
            mask = np.asarray(loaded["mask"], dtype=np.float32)
        pose, mask = _sanitize_pose_arrays(pose, mask)
        label_idx = int(sample["label_idx"])
        return (
            torch.from_numpy(pose),
            torch.from_numpy(mask),
            torch.tensor(label_idx, dtype=torch.long),
        )


def _filter_conflicting_pose_label_samples(samples: list[dict]) -> tuple[list[dict], int]:
    by_pose: dict[str, list[dict]] = {}
    for sample in samples:
        pose_key = str(sample.get("pose_path") or "").strip()
        by_pose.setdefault(pose_key, []).append(sample)

    kept: list[dict] = []
    skipped = 0
    for group in by_pose.values():
        labels = {str(sample.get("target_label") or sample.get("label_idx") or "") for sample in group}
        if len(labels) <= 1:
            kept.extend(group)
            continue

        non_normal = [
            sample
            for sample in group
            if str(sample.get("target_label") or "").strip().lower() != "normal"
            and str(sample.get("clip_role") or "").strip() != "normal_context"
        ]
        if non_normal:
            label_counts: dict[str, int] = {}
            for sample in non_normal:
                label_key = str(sample.get("target_label") or sample.get("label_idx") or "")
                label_counts[label_key] = label_counts.get(label_key, 0) + 1
            dominant_label = max(label_counts.items(), key=lambda item: item[1])[0]
            selected = [
                sample
                for sample in non_normal
                if str(sample.get("target_label") or sample.get("label_idx") or "") == dominant_label
            ]
        else:
            selected = [group[0]]
        kept.extend(selected)
        skipped += len(group) - len(selected)
    return kept, skipped


def _drop_empty_training_labels(
    train_dataset: PoseSequenceDataset,
    val_dataset: PoseSequenceDataset,
    labels: list[str],
) -> tuple[list[str], list[str]]:
    train_counts = _count_sample_label_indices(train_dataset.samples, len(labels))
    val_counts = _count_sample_label_indices(val_dataset.samples, len(labels))
    keep_indices = [
        index
        for index, _label in enumerate(labels)
        if train_counts.get(index, 0) > 0 and val_counts.get(index, 0) > 0
    ]
    if len(keep_indices) == len(labels) or len(keep_indices) < 2:
        return labels, []

    index_map = {old_index: new_index for new_index, old_index in enumerate(keep_indices)}
    filtered_labels = [labels[index] for index in keep_indices]
    dropped_labels = [label for index, label in enumerate(labels) if index not in index_map]
    train_dataset.samples = _remap_samples_to_kept_labels(train_dataset.samples, labels, index_map)
    val_dataset.samples = _remap_samples_to_kept_labels(val_dataset.samples, labels, index_map)
    return filtered_labels, dropped_labels


def _count_sample_label_indices(samples: list[dict], num_labels: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    for sample in samples:
        try:
            label_idx = int(sample.get("label_idx"))
        except (TypeError, ValueError):
            continue
        if 0 <= label_idx < num_labels:
            counts[label_idx] = counts.get(label_idx, 0) + 1
    return counts


def _remap_samples_to_kept_labels(
    samples: list[dict],
    labels: list[str],
    index_map: dict[int, int],
) -> list[dict]:
    remapped: list[dict] = []
    for sample in samples:
        try:
            old_index = int(sample.get("label_idx"))
        except (TypeError, ValueError):
            continue
        if old_index not in index_map:
            continue
        next_sample = dict(sample)
        next_sample["source_label_idx"] = old_index
        next_sample["source_target_label"] = next_sample.get("target_label") or labels[old_index]
        next_sample["label_idx"] = index_map[old_index]
        next_sample["target_label"] = labels[old_index]
        next_sample["label"] = labels[old_index]
        remapped.append(next_sample)
    return remapped


def _limit_duplicate_pose_label_samples(
    samples: list[dict],
    *,
    max_per_pose_label: int = MAX_DUPLICATE_POSE_LABEL_SAMPLES,
) -> tuple[list[dict], int]:
    if max_per_pose_label <= 0:
        return list(samples), 0

    grouped: dict[tuple[str, int], list[dict]] = {}
    passthrough: list[dict] = []
    for sample in samples:
        pose_key = str(sample.get("pose_path") or "").strip()
        try:
            label_idx = int(sample.get("label_idx"))
        except (TypeError, ValueError):
            passthrough.append(sample)
            continue
        if not pose_key:
            passthrough.append(sample)
            continue
        grouped.setdefault((pose_key, label_idx), []).append(sample)

    kept = list(passthrough)
    skipped = 0
    for group in grouped.values():
        ranked = sorted(group, key=_duplicate_pose_sample_rank)
        selected = ranked[:max_per_pose_label]
        kept.extend(selected)
        skipped += max(0, len(group) - len(selected))
    return kept, skipped


def _duplicate_pose_sample_rank(sample: dict) -> tuple:
    role = str(sample.get("clip_role") or "").strip()
    is_event = 0 if role != "normal_context" else 1
    try:
        sample_weight = -float(sample.get("sample_weight") or 1.0)
    except (TypeError, ValueError):
        sample_weight = -1.0
    try:
        valid_frames = -int(sample.get("valid_frames") or sample.get("frames_with_person") or 0)
    except (TypeError, ValueError):
        valid_frames = 0
    return (
        is_event,
        sample_weight,
        valid_frames,
        str(sample.get("item_id") or sample.get("video") or sample.get("source_video") or ""),
    )


def _sanitize_pose_arrays(pose: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.nan_to_num(
        np.asarray(pose, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    mask = np.nan_to_num(
        np.asarray(mask, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if pose.ndim >= 1 and pose.shape[-1] >= 2:
        pose[..., 0:2] = np.clip(pose[..., 0:2], *POSE_COORD_CLIP_RANGE)
    if pose.ndim >= 1 and pose.shape[-1] >= 3:
        pose[..., 2] = np.clip(pose[..., 2], *POSE_CONFIDENCE_CLIP_RANGE)
    mask = np.clip(mask, 0.0, 1.0)
    return pose.astype(np.float32, copy=False), mask.astype(np.float32, copy=False)


class TemporalPoseClassifier(nn.Module):
    def __init__(
        self,
        num_joints: int,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        feature_dim = num_joints * input_dim
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pose: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_joints, input_dim = pose.shape
        flattened = pose.view(batch_size, time_steps, num_joints * input_dim)
        encoded = self.input_proj(flattened)
        temporal_out, _hidden = self.temporal_encoder(encoded)
        mask = mask.unsqueeze(-1)
        pooled = (temporal_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)


class MeanMaxTemporalPoseClassifier(nn.Module):
    def __init__(
        self,
        num_joints: int,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        feature_dim = num_joints * input_dim
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        pooled_dim = hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pose: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_joints, input_dim = pose.shape
        flattened = pose.view(batch_size, time_steps, num_joints * input_dim)
        encoded = self.input_proj(flattened)
        temporal_out, _hidden = self.temporal_encoder(encoded)

        mask_float = mask.unsqueeze(-1).to(dtype=temporal_out.dtype)
        mean_pooled = (temporal_out * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)

        valid_mask = mask.unsqueeze(-1).to(dtype=torch.bool)
        min_value = torch.finfo(temporal_out.dtype).min
        masked_temporal = temporal_out.masked_fill(~valid_mask, min_value)
        max_pooled = masked_temporal.max(dim=1).values
        has_valid_frame = valid_mask.any(dim=1)
        max_pooled = torch.where(has_valid_frame, max_pooled, torch.zeros_like(max_pooled))

        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)
        return self.classifier(pooled)


def _resolve_temporal_pooling(value: str | None) -> str:
    normalized = str(value or "mean").strip().lower().replace("-", "_")
    if normalized in {"mean_max", "meanmax", "max_mean"}:
        return "mean_max"
    return "mean"


def _build_temporal_pose_classifier(
    *,
    temporal_pooling: str | None,
    num_joints: int,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_classes: int,
    dropout: float,
) -> nn.Module:
    pooling = _resolve_temporal_pooling(temporal_pooling)
    model_class = MeanMaxTemporalPoseClassifier if pooling == "mean_max" else TemporalPoseClassifier
    return model_class(
        num_joints=num_joints,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    )


class FocalLoss(nn.Module):
    def __init__(
        self,
        *,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if weight is None:
            self.register_buffer("weight", None)
        else:
            self.register_buffer("weight", weight.detach().clone())
        self.gamma = max(float(gamma), 0.0)
        self.label_smoothing = max(0.0, min(float(label_smoothing), 0.999))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        base_ce = F.cross_entropy(
            logits,
            target,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-base_ce).clamp(min=0.0, max=1.0)
        return ((1.0 - pt) ** self.gamma) * ce_loss


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


def _can_enable_compile() -> tuple[bool, str]:
    if not hasattr(torch, "compile"):
        return False, "torch.compile 미지원 환경입니다."
    return True, ""


def _resolve_compile_backend(
    *,
    requested_backend: str | None,
    device: str,
) -> list[tuple[str | None, str]]:
    normalized = str(requested_backend or "auto").strip().lower()
    is_windows = platform.system().lower() == "windows"
    use_cuda = _uses_cuda(device)
    has_triton = _has_triton()

    if normalized and normalized not in {"", "auto"}:
        return [(normalized, normalized)]

    if is_windows:
        return [
            ("aot_eager", "aot_eager"),
            ("eager", "eager"),
        ]

    if use_cuda and has_triton:
        return [
            (None, "inductor"),
            ("aot_eager", "aot_eager"),
            ("eager", "eager"),
        ]

    return [
        ("aot_eager", "aot_eager"),
        ("eager", "eager"),
    ]


def _compile_model_safely(
    base_model: nn.Module,
    *,
    compile_model: bool,
    compile_backend: str | None,
    device: str,
) -> tuple[nn.Module, bool, str]:
    if not compile_model:
        return base_model, False, "disabled"

    can_compile, compile_reason = _can_enable_compile()
    if not can_compile:
        print(f"[train] torch.compile을 건너뜁니다: {compile_reason}")
        return base_model, False, "disabled"

    import torch._dynamo

    torch._dynamo.config.suppress_errors = True
    candidates = _resolve_compile_backend(
        requested_backend=compile_backend,
        device=device,
    )

    last_error = ""
    for backend, backend_label in candidates:
        try:
            kwargs = {}
            if backend is not None:
                kwargs["backend"] = backend
            compiled = torch.compile(base_model, **kwargs)
            return compiled, True, backend_label
        except Exception as exc:
            last_error = str(exc)
            print(f"[train] torch.compile backend {backend_label} 실패: {exc}")

    if last_error:
        print(f"[train] torch.compile을 건너뜁니다: {last_error}")
    return base_model, False, "disabled"


def train_action_classifier(
    *,
    train_manifest: Path,
    val_manifest: Path,
    output_dir: Path,
    labels: list[str],
    epochs: int = 20,
    batch_size: int = 16,
    eval_batch_size: int | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    temporal_pooling: str = "mean",
    label_smoothing: float = 0.05,
    loss_name: str = "cross_entropy",
    focal_gamma: float = 2.0,
    class_weight: bool | str = "balanced",
    class_weight_multipliers: dict | None = None,
    balanced_sampler: bool | str = "auto",
    grad_clip_norm: float = 1.0,
    seed: int | None = 42,
    deterministic: bool = False,
    num_workers: int | str = 0,
    device: str = "cuda",
    amp: bool = True,
    amp_dtype: str = "auto",
    compile_model: bool = False,
    compile_backend: str | None = None,
    dataset_cache_size: int = 2048,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    pin_memory: bool | str = "auto",
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.001,
    selection_metric: str = "macro_f1",
    overfit_guard_enabled: bool = True,
    overfit_guard_min_epoch: int = 8,
    overfit_guard_loss_gap: float = 0.45,
    overfit_guard_patience: int = 3,
    imbalance_warn_min_samples: int = 8,
    imbalance_warn_ratio: float = 5.0,
    progress_path: Path | None = None,
    resume_from: Path | None = None,
    max_duplicate_pose_label_samples: int = MAX_DUPLICATE_POSE_LABEL_SAMPLES,
) -> TrainingArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.json"
    metrics_path = output_dir / "metrics.json"
    best_model_path = output_dir / "best_action_model.pt"

    effective_seed = _resolve_training_seed(seed)
    if effective_seed is not None:
        _set_training_seed(effective_seed, deterministic=deterministic)

    use_cuda = _uses_cuda(device)
    use_amp = bool(amp and use_cuda and torch.cuda.is_available())
    resolved_amp_dtype, resolved_amp_dtype_label = _resolve_amp_dtype(
        device=device,
        enabled=use_amp,
        requested_dtype=amp_dtype,
    )
    resolved_batch_size = max(int(batch_size), 1)
    resolved_eval_batch_size = _resolve_eval_batch_size(eval_batch_size, train_batch_size=resolved_batch_size)
    resolved_num_workers = _resolve_num_workers(num_workers, batch_size=resolved_batch_size)
    resolved_prefetch_factor = max(int(prefetch_factor), 1)
    use_persistent_workers = bool(persistent_workers and resolved_num_workers > 0)
    use_pin_memory = _resolve_pin_memory(pin_memory, use_cuda=use_cuda)
    resolved_pin_memory_device = _resolve_pin_memory_device(
        use_pin_memory=use_pin_memory,
        device=device,
    )
    effective_cache_size = _resolve_dataset_cache_size(
        dataset_cache_size,
        num_workers=resolved_num_workers,
    )

    train_dataset = PoseSequenceDataset(
        train_manifest,
        cache_size=effective_cache_size,
        max_duplicate_pose_label_samples=max_duplicate_pose_label_samples,
    )
    val_dataset = PoseSequenceDataset(
        val_manifest,
        cache_size=effective_cache_size,
        max_duplicate_pose_label_samples=max_duplicate_pose_label_samples,
    )
    skipped_missing_pose_total = (
        int(getattr(train_dataset, "skipped_missing_pose_path", 0))
        + int(getattr(val_dataset, "skipped_missing_pose_path", 0))
    )
    skipped_conflicting_pose_total = (
        int(getattr(train_dataset, "skipped_conflicting_pose_label", 0))
        + int(getattr(val_dataset, "skipped_conflicting_pose_label", 0))
    )
    skipped_duplicate_pose_total = (
        int(getattr(train_dataset, "skipped_duplicate_pose_label", 0))
        + int(getattr(val_dataset, "skipped_duplicate_pose_label", 0))
    )
    if skipped_missing_pose_total > 0:
        print(
            "[train] pose dataset filtered RGB/I3D-only fallback rows "
            f"train={getattr(train_dataset, 'skipped_missing_pose_path', 0)} "
            f"val={getattr(val_dataset, 'skipped_missing_pose_path', 0)}"
        )
    if skipped_conflicting_pose_total > 0:
        print(
            "[train] pose dataset filtered conflicting pose-label rows "
            f"train={getattr(train_dataset, 'skipped_conflicting_pose_label', 0)} "
            f"val={getattr(val_dataset, 'skipped_conflicting_pose_label', 0)}"
        )
    if skipped_duplicate_pose_total > 0:
        print(
            "[train] pose dataset capped duplicate pose-label rows "
            f"max_per_pose_label={max_duplicate_pose_label_samples} "
            f"train={getattr(train_dataset, 'skipped_duplicate_pose_label', 0)} "
            f"val={getattr(val_dataset, 'skipped_duplicate_pose_label', 0)}"
        )

    label_mapping_payload = _build_label_mapping_payload(labels)
    labels, dropped_labels = _drop_empty_training_labels(train_dataset, val_dataset, labels)
    if dropped_labels:
        print(
            "[train] dropped labels with no pose-ready train/val samples: "
            + ", ".join(dropped_labels)
        )
        label_mapping_payload = _build_label_mapping_payload(labels)
    _validate_dataset_label_mapping(train_dataset.samples, labels, split_name="train")
    _validate_dataset_label_mapping(val_dataset.samples, labels, split_name="val")
    train_distribution = _summarize_class_distribution(
        train_dataset.samples,
        labels,
        min_samples=imbalance_warn_min_samples,
        ratio_warn=imbalance_warn_ratio,
    )
    val_distribution = _summarize_class_distribution(
        val_dataset.samples,
        labels,
        min_samples=imbalance_warn_min_samples,
        ratio_warn=imbalance_warn_ratio,
    )
    train_sample_count = len(train_dataset)
    val_sample_count = len(val_dataset)

    _configure_training_acceleration(device=device, use_cuda=use_cuda, deterministic=deterministic)
    class_weight_multiplier_tensor, class_weight_multiplier_payload = _resolve_class_weight_multipliers(
        labels,
        class_weight_multipliers,
    )

    train_generator = _build_torch_generator(effective_seed)
    sampler_generator = _build_torch_generator(None if effective_seed is None else effective_seed + 1)
    worker_init_fn = _build_worker_init_fn(effective_seed)
    train_sampler, sampler_mode = _build_balanced_sampler(
        train_dataset.samples,
        num_classes=len(labels),
        requested=balanced_sampler,
        distribution=train_distribution,
        generator=sampler_generator,
        class_weight_multipliers=class_weight_multiplier_tensor,
    )

    def build_training_loaders(loader_num_workers: int) -> tuple[DataLoader, DataLoader, bool]:
        loader_persistent_workers = bool(persistent_workers and loader_num_workers > 0)
        train_data_loader = _build_dataloader(
            dataset=train_dataset,
            batch_size=resolved_batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=loader_num_workers,
            pin_memory=use_pin_memory,
            pin_memory_device=resolved_pin_memory_device,
            prefetch_factor=resolved_prefetch_factor,
            persistent_workers=loader_persistent_workers,
            generator=train_generator,
            worker_init_fn=worker_init_fn,
        )
        val_data_loader = _build_dataloader(
            dataset=val_dataset,
            batch_size=resolved_eval_batch_size,
            shuffle=False,
            sampler=None,
            num_workers=loader_num_workers,
            pin_memory=use_pin_memory,
            pin_memory_device=resolved_pin_memory_device,
            prefetch_factor=resolved_prefetch_factor,
            persistent_workers=loader_persistent_workers,
            generator=None,
            worker_init_fn=worker_init_fn,
        )
        return train_data_loader, val_data_loader, loader_persistent_workers

    train_loader, val_loader, use_persistent_workers = build_training_loaders(resolved_num_workers)

    first_pose, _first_mask, _first_label = train_dataset[0]
    resolved_temporal_pooling = _resolve_temporal_pooling(temporal_pooling)
    base_model = _build_temporal_pose_classifier(
        temporal_pooling=resolved_temporal_pooling,
        num_joints=first_pose.shape[1],
        input_dim=first_pose.shape[2],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=len(labels),
        dropout=dropout,
    ).to(device)
    model = base_model

    resumed_from_checkpoint = False
    resume_mode = "fresh"
    if resume_from is not None and resume_from.exists():
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        checkpoint_labels = list(checkpoint.get("labels", []))
        checkpoint_state = checkpoint.get("model_state_dict", {}) or {}
        if checkpoint_labels == list(labels):
            try:
                base_model.load_state_dict(checkpoint_state, strict=False)
                resumed_from_checkpoint = True
                resume_mode = "full"
            except RuntimeError as exc:
                loaded_count, skipped_keys = _load_compatible_state_dict(base_model, checkpoint_state)
                if loaded_count > 0:
                    resumed_from_checkpoint = True
                    resume_mode = "partial"
                    print(
                        "[train] checkpoint를 부분 warm-start로 읽었습니다. "
                        f"loaded={loaded_count} skipped={len(skipped_keys)} reason={exc}"
                    )
                else:
                    print(f"[train] checkpoint resume을 건너뜁니다: {exc}")
        else:
            loaded_count, skipped_keys = _load_compatible_state_dict(base_model, checkpoint_state)
            if loaded_count > 0:
                resumed_from_checkpoint = True
                resume_mode = "partial"
                print(
                    "[train] 기존 체크포인트를 부분 warm-start로 이어받습니다.\n"
                    f"- checkpoint labels: {checkpoint_labels}\n"
                    f"- current labels: {labels}\n"
                    f"- loaded tensors: {loaded_count}\n"
                    f"- skipped tensors: {len(skipped_keys)}"
                )
            else:
                print(
                    "[train] 기존 체크포인트 라벨 구성이 현재 설정과 달라서 resume을 건너뜁니다.\n"
                    f"- checkpoint labels: {checkpoint_labels}\n"
                    f"- current labels: {labels}"
                )

    model, compiled_model, compiled_backend = _compile_model_safely(
        base_model,
        compile_model=compile_model,
        compile_backend=compile_backend,
        device=device,
    )

    class_weights, class_weight_mode = _resolve_class_weights(
        train_dataset.samples,
        num_classes=len(labels),
        requested=class_weight,
    )
    if class_weight_multiplier_tensor is not None and class_weights is not None:
        class_weight_mode = f"{class_weight_mode}+multipliers"
        class_weights = class_weights * class_weight_multiplier_tensor
    class_weights_for_loss = class_weights.to(device) if class_weights is not None else None
    criterion, resolved_loss_name = _build_loss_function(
        loss_name=loss_name,
        class_weights=class_weights_for_loss,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=max(float(weight_decay), 0.0),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = _create_grad_scaler(enabled=use_amp)

    history: list[dict] = []
    resolved_selection_metric = _resolve_selection_metric(selection_metric)
    best_selection_score = -1.0
    best_val_f1 = -1.0
    best_epoch = 0
    effective_patience = max(int(early_stopping_patience), 0)
    effective_min_delta = max(float(early_stopping_min_delta), 0.0)
    effective_overfit_guard_enabled = bool(overfit_guard_enabled)
    effective_overfit_guard_min_epoch = max(int(overfit_guard_min_epoch), 1)
    effective_overfit_guard_loss_gap = max(float(overfit_guard_loss_gap), 0.0)
    effective_overfit_guard_patience = max(int(overfit_guard_patience), 1)
    epochs_without_improvement = 0
    stopped_early = False
    stop_reason: str | None = None
    effective_grad_clip_norm = max(float(grad_clip_norm), 0.0)
    training_options = {
        **label_mapping_payload,
        "loss_name": resolved_loss_name,
        "class_weight_mode": class_weight_mode,
        "class_weights": _tensor_to_float_list(class_weights),
        "class_weight_multipliers": class_weight_multiplier_payload,
        "balanced_sampler": sampler_mode,
        "label_smoothing": max(0.0, min(float(label_smoothing), 0.999)),
        "focal_gamma": max(float(focal_gamma), 0.0),
        "weight_decay": max(float(weight_decay), 0.0),
        "grad_clip_norm": effective_grad_clip_norm,
        "overfit_guard": {
            "enabled": effective_overfit_guard_enabled,
            "min_epoch": effective_overfit_guard_min_epoch,
            "loss_gap": effective_overfit_guard_loss_gap,
            "patience": effective_overfit_guard_patience,
            "metric": f"val_{resolved_selection_metric}",
        },
        "selection_metric": resolved_selection_metric,
        "seed": effective_seed,
        "deterministic": bool(deterministic),
        "temporal_pooling": resolved_temporal_pooling,
    }

    print(
        "[train] acceleration "
        f"device={device} "
        f"amp={'on' if use_amp else 'off'} "
        f"amp_dtype={resolved_amp_dtype_label} "
        f"compile={'on' if compiled_model else 'off'} "
        f"compile_backend={compiled_backend} "
        f"resume={resume_mode} "
        f"batch(train/val)={resolved_batch_size}/{resolved_eval_batch_size} "
        f"cache={effective_cache_size} "
        f"workers={resolved_num_workers} "
        f"pin_memory={'on' if use_pin_memory else 'off'} "
        f"prefetch={resolved_prefetch_factor if resolved_num_workers > 0 else 0} "
        f"persistent={'on' if use_persistent_workers else 'off'} "
        f"loss={resolved_loss_name} "
        f"pooling={resolved_temporal_pooling} "
        f"class_weight={class_weight_mode} "
        f"class_weight_multipliers={class_weight_multiplier_payload or 'none'} "
        f"sampler={sampler_mode} "
        f"weight_decay={max(float(weight_decay), 0.0):.6g} "
        f"grad_clip={effective_grad_clip_norm:.4g} "
        f"seed={effective_seed if effective_seed is not None else 'none'}"
    )
    print(
        "[train][start] "
        f"epochs={epochs} "
        f"samples(train/val)={train_sample_count}/{val_sample_count} "
        f"labels={len(labels)} "
        f"resume={resume_mode}"
    )
    if train_distribution.get("messages"):
        print(
            "[train] class-balance "
            f"train={train_distribution.get('severity')} | "
            + " / ".join(str(message) for message in train_distribution.get("messages", []))
        )
    if val_distribution.get("messages"):
        print(
            "[train] class-balance "
            f"val={val_distribution.get('severity')} | "
            + " / ".join(str(message) for message in val_distribution.get("messages", []))
        )

    if progress_path is not None:
        _write_progress(
            progress_path,
            {
                "state": "running",
                "device": device,
                "epochs_total": epochs,
                "epochs_completed": 0,
                "best_val_macro_f1": None,
                "best_epoch": None,
                "latest": None,
                "history": history,
                "labels": labels,
                "resumed_from_checkpoint": resumed_from_checkpoint,
                "resume_mode": resume_mode,
                "train_samples": train_sample_count,
                "val_samples": val_sample_count,
                "skipped_missing_pose_path": {
                    "train": int(getattr(train_dataset, "skipped_missing_pose_path", 0)),
                    "val": int(getattr(val_dataset, "skipped_missing_pose_path", 0)),
                },
                "amp_enabled": use_amp,
                "amp_dtype": resolved_amp_dtype_label,
                "compile_enabled": compiled_model,
                "compile_backend": compiled_backend,
                "batch_size": resolved_batch_size,
                "eval_batch_size": resolved_eval_batch_size,
                "dataset_cache_size": effective_cache_size,
                "num_workers": resolved_num_workers,
                "pin_memory": use_pin_memory,
                "pin_memory_device": resolved_pin_memory_device,
                "prefetch_factor": resolved_prefetch_factor if resolved_num_workers > 0 else 0,
                "persistent_workers": use_persistent_workers,
                **training_options,
                "train_distribution": train_distribution,
                "val_distribution": val_distribution,
                "stopped_early": False,
                "stop_reason": None,
                    "early_stopping": {
                        "enabled": effective_patience > 0,
                        "patience": effective_patience,
                        "min_delta": effective_min_delta,
                        "metric": "val_macro_f1",
                        "epochs_without_improvement": 0,
                        "overfit_guard": training_options["overfit_guard"],
                    },
                },
            )

    for epoch in range(1, epochs + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        try:
            train_loss = _run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                train=True,
                use_amp=use_amp,
                amp_dtype=resolved_amp_dtype,
                scaler=scaler,
                grad_clip_norm=effective_grad_clip_norm,
            )
        except PermissionError as exc:
            if resolved_num_workers <= 0 or not _is_dataloader_worker_permission_error(exc):
                raise
            print(
                "[train][warning] dataloader worker startup failed; "
                "retrying with num_workers=0. "
                f"reason={exc}"
            )
            resolved_num_workers = 0
            train_loader, val_loader, use_persistent_workers = build_training_loaders(resolved_num_workers)
            train_loss = _run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                train=True,
                use_amp=use_amp,
                amp_dtype=resolved_amp_dtype,
                scaler=scaler,
                grad_clip_norm=effective_grad_clip_norm,
            )
        val_metrics = _evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=len(labels),
            label_names=labels,
            use_amp=use_amp,
            amp_dtype=resolved_amp_dtype,
        )
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_cross_entropy_loss": round(val_metrics["cross_entropy_loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
            "val_macro_f1_supported": round(val_metrics["macro_f1_supported"], 6),
            "val_balanced_accuracy": round(val_metrics["balanced_accuracy"], 6),
            "val_loss_gap": round(val_metrics["loss"] - train_loss, 6),
            "val_mean_true_confidence": round(val_metrics["mean_true_confidence"], 6),
            "val_mean_pred_confidence": round(val_metrics["mean_pred_confidence"], 6),
            "learning_rate": round(current_lr, 8),
        }
        history.append(epoch_metrics)
        epoch_progress = epoch / max(epochs, 1)
        progress_width = 24
        progress_filled = min(progress_width, max(0, round(epoch_progress * progress_width)))
        progress_bar = "#" * progress_filled + "-" * (progress_width - progress_filled)
        print(
            "[train][epoch] "
            f"{epoch:03d}/{epochs:03d} "
            f"{epoch_progress * 100:6.1f}% "
            f"[{progress_bar}] "
            f"loss train={train_loss:.4f} val={val_metrics['loss']:.4f} ce={val_metrics['cross_entropy_loss']:.4f} "
            f"score acc={val_metrics['accuracy']:.4f} f1={val_metrics['macro_f1']:.4f} "
            f"best_{resolved_selection_metric}="
            f"{max(best_selection_score, _metric_value(val_metrics, resolved_selection_metric)):.4f}"
        )

        current_selection_score = _metric_value(val_metrics, resolved_selection_metric)
        improved = current_selection_score > (best_selection_score + effective_min_delta)
        if best_epoch == 0 or improved:
            best_selection_score = current_selection_score
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_torch_checkpoint_atomic(
                best_model_path,
                {
                    "model_state_dict": base_model.state_dict(),
                    "labels": labels,
                    **label_mapping_payload,
                    "num_joints": int(first_pose.shape[1]),
                    "input_dim": int(first_pose.shape[2]),
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "temporal_pooling": resolved_temporal_pooling,
                    "best_epoch": best_epoch,
                    "best_selection_metric": resolved_selection_metric,
                    "best_selection_score": round(best_selection_score, 6),
                    "best_val_macro_f1": round(best_val_f1, 6),
                    "best_validation": val_metrics,
                    "training_options": training_options,
                },
            )
        else:
            epochs_without_improvement += 1

        if progress_path is not None:
            _write_progress(
                progress_path,
                {
                    "state": "running",
                    "device": device,
                    "epochs_total": epochs,
                    "epochs_completed": epoch,
                    "best_val_macro_f1": round(best_val_f1, 6),
                    "best_epoch": best_epoch,
                    "latest": epoch_metrics,
                    "history": history,
                    "labels": labels,
                    "resumed_from_checkpoint": resumed_from_checkpoint,
                    "resume_mode": resume_mode,
                    "train_samples": train_sample_count,
                    "val_samples": val_sample_count,
                    "skipped_missing_pose_path": {
                        "train": int(getattr(train_dataset, "skipped_missing_pose_path", 0)),
                        "val": int(getattr(val_dataset, "skipped_missing_pose_path", 0)),
                    },
                    "amp_enabled": use_amp,
                    "amp_dtype": resolved_amp_dtype_label,
                    "compile_enabled": compiled_model,
                    "compile_backend": compiled_backend,
                    "batch_size": resolved_batch_size,
                    "eval_batch_size": resolved_eval_batch_size,
                    "dataset_cache_size": effective_cache_size,
                    "num_workers": resolved_num_workers,
                    "pin_memory": use_pin_memory,
                    "pin_memory_device": resolved_pin_memory_device,
                    "prefetch_factor": resolved_prefetch_factor if resolved_num_workers > 0 else 0,
                    "persistent_workers": use_persistent_workers,
                    **training_options,
                    "train_distribution": train_distribution,
                    "val_distribution": val_distribution,
                    "stopped_early": False,
                    "stop_reason": None,
                    "early_stopping": {
                        "enabled": effective_patience > 0,
                        "patience": effective_patience,
                        "min_delta": effective_min_delta,
                        "metric": "val_macro_f1",
                        "epochs_without_improvement": epochs_without_improvement,
                        "overfit_guard": training_options["overfit_guard"],
                    },
                },
            )

        current_loss_gap = float(val_metrics["loss"] - train_loss)
        overfit_guard_triggered = (
            effective_overfit_guard_enabled
            and epoch >= effective_overfit_guard_min_epoch
            and epochs_without_improvement >= effective_overfit_guard_patience
            and current_loss_gap >= effective_overfit_guard_loss_gap
        )
        if overfit_guard_triggered:
            stopped_early = True
            stop_reason = (
                "overfit guard: validation loss gap이 커지고 "
                f"val_macro_f1이 {epochs_without_improvement} epoch 동안 개선되지 않아 조기 종료합니다. "
                f"loss_gap={current_loss_gap:.4f}, threshold={effective_overfit_guard_loss_gap:.4f}"
            )
            print(f"[train] overfit guard triggered: {stop_reason}")
            break

        if effective_patience > 0 and epochs_without_improvement >= effective_patience:
            stopped_early = True
            stop_reason = (
                f"val_macro_f1가 {effective_patience} epoch 동안 "
                f"{effective_min_delta:.4f} 이상 개선되지 않아 조기 종료합니다."
            )
            print(f"[train] early stopping triggered: {stop_reason}")
            break

    write_json_atomic(labels_path, {"labels": labels, **label_mapping_payload})

    if best_model_path.exists():
        best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
        best_state = best_checkpoint.get("model_state_dict", {}) or {}
        if best_state:
            base_model.load_state_dict(best_state, strict=False)

    final_metrics = _evaluate(
        model=base_model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=len(labels),
        label_names=labels,
        use_amp=use_amp,
        amp_dtype=resolved_amp_dtype,
    )
    error_analysis_path = output_dir / "validation_error_analysis.json"
    false_negative_path = output_dir / "false_negative_examples.json"
    confusion_pair_path = output_dir / "confusion_pair_examples.json"
    error_analysis = _build_validation_error_analysis(final_metrics, labels=labels)
    write_json_atomic(error_analysis_path, error_analysis)
    write_json_atomic(false_negative_path, error_analysis.get("false_negative_examples", {}))
    write_json_atomic(confusion_pair_path, error_analysis.get("confusion_pair_examples", []))
    write_json_atomic(
        metrics_path,
        {
            "history": history,
            "final_validation": final_metrics,
            "validation_error_analysis": error_analysis.get("summary", {}),
            "validation_error_analysis_path": str(error_analysis_path),
            "false_negative_examples_path": str(false_negative_path),
            "confusion_pair_examples_path": str(confusion_pair_path),
            "labels": labels,
            "best_epoch": best_epoch,
            "best_val_macro_f1": round(best_val_f1, 6),
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "resume_mode": resume_mode,
            "train_samples": train_sample_count,
            "val_samples": val_sample_count,
            "amp_enabled": use_amp,
            "amp_dtype": resolved_amp_dtype_label,
            "compile_enabled": compiled_model,
            "compile_backend": compiled_backend,
            "batch_size": resolved_batch_size,
            "eval_batch_size": resolved_eval_batch_size,
            "dataset_cache_size": effective_cache_size,
            "num_workers": resolved_num_workers,
            "pin_memory": use_pin_memory,
            "pin_memory_device": resolved_pin_memory_device,
            "prefetch_factor": resolved_prefetch_factor if resolved_num_workers > 0 else 0,
            "persistent_workers": use_persistent_workers,
            **training_options,
            "train_distribution": train_distribution,
            "val_distribution": val_distribution,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "early_stopping": {
                "enabled": effective_patience > 0,
                "patience": effective_patience,
                "min_delta": effective_min_delta,
                "metric": "val_macro_f1",
                "epochs_without_improvement": epochs_without_improvement,
            },
        },
    )
    print(f"[train] validation error analysis: {error_analysis_path}")
    print(f"[train] false negatives: {false_negative_path}")
    print(f"[train] confusion pairs: {confusion_pair_path}")

    if progress_path is not None:
        _write_progress(
            progress_path,
            {
                "state": "completed",
                "device": device,
                "epochs_total": epochs,
                "epochs_completed": len(history),
                "best_val_macro_f1": round(best_val_f1, 6),
                "best_epoch": best_epoch,
                "latest": history[-1] if history else None,
                "history": history,
                "labels": labels,
                "final_validation": final_metrics,
                "validation_error_analysis": error_analysis.get("summary", {}),
                "validation_error_analysis_path": str(error_analysis_path),
                "false_negative_examples_path": str(false_negative_path),
                "confusion_pair_examples_path": str(confusion_pair_path),
                "resumed_from_checkpoint": resumed_from_checkpoint,
                "resume_mode": resume_mode,
                "train_samples": train_sample_count,
                "val_samples": val_sample_count,
                "amp_enabled": use_amp,
                "amp_dtype": resolved_amp_dtype_label,
                "compile_enabled": compiled_model,
                "compile_backend": compiled_backend,
                "batch_size": resolved_batch_size,
                "eval_batch_size": resolved_eval_batch_size,
                "dataset_cache_size": effective_cache_size,
                "num_workers": resolved_num_workers,
                "pin_memory": use_pin_memory,
                "pin_memory_device": resolved_pin_memory_device,
                "prefetch_factor": resolved_prefetch_factor if resolved_num_workers > 0 else 0,
                "persistent_workers": use_persistent_workers,
                **training_options,
                "train_distribution": train_distribution,
                "val_distribution": val_distribution,
                "stopped_early": stopped_early,
                "stop_reason": stop_reason,
                "early_stopping": {
                    "enabled": effective_patience > 0,
                    "patience": effective_patience,
                    "min_delta": effective_min_delta,
                    "metric": "val_macro_f1",
                    "epochs_without_improvement": epochs_without_improvement,
                },
            },
        )

    return TrainingArtifacts(
        best_model_path=best_model_path,
        metrics_path=metrics_path,
        labels_path=labels_path,
        history=history,
    )


def _build_class_weights(samples: list[dict], num_classes: int) -> torch.Tensor:
    counts = np.ones(num_classes, dtype=np.float32)
    for sample in samples:
        label_idx = _safe_label_index(sample.get("label_idx"), num_classes=num_classes)
        if label_idx >= 0:
            counts[label_idx] += 1.0
    weights = counts.sum() / (counts * len(counts))
    return torch.tensor(weights, dtype=torch.float32)


def _summarize_class_distribution(
    samples: list[dict],
    labels: list[str],
    *,
    min_samples: int,
    ratio_warn: float,
) -> dict:
    by_label: dict[str, int] = {label: 0 for label in labels}
    for sample in samples:
        label_idx = _safe_label_index(sample.get("label_idx"), num_classes=len(labels))
        if 0 <= label_idx < len(labels):
            by_label[labels[label_idx]] = int(by_label.get(labels[label_idx], 0) or 0) + 1
    summary = analyze_class_balance(
        labels,
        by_label,
        min_samples=min_samples,
        ratio_warn=ratio_warn,
    )
    summary["total_samples"] = len(samples)
    return summary


def _safe_label_index(value, *, num_classes: int | None = None) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return -1
    if num_classes is not None and not 0 <= index < num_classes:
        return -1
    return index


def _resolve_selection_metric(value: str | None) -> str:
    metric = str(value or "macro_f1").strip().lower()
    aliases = {
        "acc": "accuracy",
        "f1": "macro_f1",
        "macro": "macro_f1",
        "balanced": "balanced_accuracy",
        "balanced_acc": "balanced_accuracy",
    }
    metric = aliases.get(metric, metric)
    allowed = {"accuracy", "macro_f1", "macro_f1_supported", "balanced_accuracy"}
    return metric if metric in allowed else "macro_f1"


def _metric_value(metrics: dict, metric: str) -> float:
    try:
        return float(metrics.get(metric) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _build_label_mapping_payload(labels: list[str]) -> dict:
    class_to_idx = {label: index for index, label in enumerate(labels)}
    idx_to_class = {str(index): label for label, index in class_to_idx.items()}
    return {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }


def _validate_dataset_label_mapping(samples: list[dict], labels: list[str], *, split_name: str) -> None:
    mismatches: list[str] = []
    for index, sample in enumerate(samples):
        label_idx = _safe_label_index(sample.get("label_idx"), num_classes=len(labels))
        target_label = str(sample.get("target_label") or "").strip()
        if label_idx < 0:
            mismatches.append(
                f"{split_name}[{index}] "
                f"label_idx={sample.get('label_idx')} "
                f"target_label={target_label or '-'}"
            )
            continue
        expected_label = labels[label_idx]
        if target_label and target_label != expected_label:
            mismatches.append(
                f"{split_name}[{index}] "
                f"label_idx={label_idx} "
                f"expected={expected_label} "
                f"target_label={target_label}"
            )
    if mismatches:
        examples = "\n".join(f"- {message}" for message in mismatches[:8])
        raise RuntimeError(
            "학습 manifest의 label_idx와 target_label 매핑이 현재 labels 순서와 일치하지 않습니다.\n"
            f"{examples}\n"
            "prepare/train manifest를 다시 materialize 하거나 dataset.target_labels 순서를 확인해 주세요."
        )


def _resolve_training_seed(seed: int | str | None) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, str):
        normalized = seed.strip().lower()
        if normalized in {"", "none", "off", "false"}:
            return None
        return int(normalized)
    return int(seed)


def _set_training_seed(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic and hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def _build_torch_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _build_worker_init_fn(seed: int | None):
    if seed is None:
        return None
    return partial(_seed_worker_with_base, base_seed=int(seed))


def _seed_worker_with_base(worker_id: int, *, base_seed: int) -> None:
    worker_seed = int(base_seed) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32 - 1))
    torch.manual_seed(worker_seed)


def _resolve_class_weights(
    samples: list[dict],
    *,
    num_classes: int,
    requested: bool | str,
) -> tuple[torch.Tensor | None, str]:
    if isinstance(requested, str):
        normalized = requested.strip().lower()
        enabled = normalized not in {"", "0", "false", "off", "none", "disabled"}
        mode = normalized or "off"
    else:
        enabled = bool(requested)
        mode = "balanced" if enabled else "off"
    if not enabled:
        return None, "off"
    return _build_class_weights(samples, num_classes), mode if mode != "true" else "balanced"


def _resolve_class_weight_multipliers(
    labels: list[str],
    requested: dict | None,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if not isinstance(requested, dict) or not requested:
        return None, {}
    values = np.ones(len(labels), dtype=np.float32)
    payload: dict[str, float] = {}
    label_to_index = {str(label): index for index, label in enumerate(labels)}
    for label, raw_multiplier in requested.items():
        label_key = str(label or "").strip()
        if label_key not in label_to_index:
            continue
        try:
            multiplier = float(raw_multiplier)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(multiplier):
            continue
        multiplier = float(np.clip(multiplier, 0.25, 8.0))
        values[label_to_index[label_key]] = multiplier
        if abs(multiplier - 1.0) > 1e-6:
            payload[label_key] = round(multiplier, 6)
    if not payload:
        return None, {}
    return torch.tensor(values, dtype=torch.float32), payload


def _build_loss_function(
    *,
    loss_name: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
    label_smoothing: float,
) -> tuple[nn.Module, str]:
    normalized_loss = str(loss_name or "cross_entropy").strip().lower()
    smoothing = max(0.0, min(float(label_smoothing), 0.999))
    if normalized_loss in {"focal", "focal_loss"}:
        return (
            FocalLoss(
                weight=class_weights,
                gamma=max(float(focal_gamma), 0.0),
                label_smoothing=smoothing,
            ),
            "focal",
        )
    return (
        nn.CrossEntropyLoss(
            weight=class_weights,
            reduction="none",
            label_smoothing=smoothing,
        ),
        "cross_entropy",
    )


def _build_balanced_sampler(
    samples: list[dict],
    *,
    num_classes: int,
    requested: bool | str,
    distribution: dict,
    generator: torch.Generator | None,
    class_weight_multipliers: torch.Tensor | None = None,
) -> tuple[WeightedRandomSampler | None, str]:
    enabled = False
    mode = "off"
    if isinstance(requested, str):
        normalized = requested.strip().lower()
        if normalized in {"1", "true", "yes", "on", "balanced"}:
            enabled = True
            mode = "balanced"
        elif normalized in {"auto", "default"}:
            imbalance_ratio = distribution.get("imbalance_ratio")
            enabled = (
                imbalance_ratio is not None
                and float(imbalance_ratio) >= float(distribution.get("ratio_warn", 5.0) or 5.0)
            )
            mode = "auto_on" if enabled else "auto_off"
        else:
            mode = "off"
    else:
        enabled = bool(requested)
        mode = "balanced" if enabled else "off"

    if not enabled:
        return None, mode

    sample_weights = _build_sample_weights(
        samples,
        num_classes=num_classes,
        class_weight_multipliers=class_weight_multipliers,
    )
    if sample_weights is None:
        return None, f"{mode}_unavailable"
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    return sampler, mode


def _build_sample_weights(
    samples: list[dict],
    *,
    num_classes: int,
    class_weight_multipliers: torch.Tensor | None = None,
) -> torch.Tensor | None:
    counts = np.zeros(num_classes, dtype=np.float64)
    label_indices: list[int] = []
    for sample in samples:
        label_idx = _safe_label_index(sample.get("label_idx"), num_classes=num_classes)
        label_indices.append(label_idx)
        if label_idx >= 0:
            counts[label_idx] += 1.0
    nonzero_classes = int(np.count_nonzero(counts))
    if nonzero_classes <= 1:
        return None
    class_weights = np.zeros(num_classes, dtype=np.float64)
    total = float(counts.sum())
    for class_index, count in enumerate(counts):
        if count > 0:
            class_weights[class_index] = total / (count * nonzero_classes)
    if class_weight_multipliers is not None:
        multiplier_values = class_weight_multipliers.detach().cpu().numpy().astype(np.float64, copy=False)
        if multiplier_values.shape[0] >= num_classes:
            class_weights = class_weights * multiplier_values[:num_classes]
    sample_weights = []
    for sample, label_idx in zip(samples, label_indices, strict=False):
        base_weight = class_weights[label_idx] if label_idx >= 0 else 0.0
        try:
            sample_multiplier = float(sample.get("sample_weight", 1.0) or 1.0)
        except (TypeError, ValueError):
            sample_multiplier = 1.0
        sample_weights.append(base_weight * max(sample_multiplier, 0.0))
    return torch.tensor(sample_weights, dtype=torch.double)


def _mean_loss(loss_values: torch.Tensor) -> torch.Tensor:
    if loss_values.ndim == 0:
        return loss_values
    return loss_values.mean()


def _sum_loss(loss_values: torch.Tensor, *, batch_size: int) -> float:
    if loss_values.ndim == 0:
        return float(loss_values.detach().item()) * batch_size
    return float(loss_values.detach().sum().item())


def _tensor_to_float_list(values: torch.Tensor | None) -> list[float] | None:
    if values is None:
        return None
    return [round(float(value), 6) for value in values.detach().cpu().tolist()]


def _label_name(labels: list[str] | None, index: int) -> str:
    if labels is not None and 0 <= index < len(labels):
        return labels[index]
    return str(index)


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: str,
    train: bool,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    scaler,
    grad_clip_norm: float,
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_items = 0

    for pose, mask, labels in loader:
        pose = pose.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
            logits = model(pose, mask)
            loss_values = criterion(logits, labels)
            loss = _mean_loss(loss_values)

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += _sum_loss(loss_values, batch_size=batch_size)
        total_items += batch_size

    return total_loss / max(total_items, 1)


@torch.inference_mode()
def _evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int,
    label_names: list[str] | None,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_cross_entropy_loss = 0.0
    total_items = 0
    true_confidence_total = 0.0
    pred_confidence_total = 0.0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    misclassified: list[dict] = []
    dataset_samples = getattr(loader.dataset, "samples", [])
    sample_offset = 0

    for pose, mask, labels in loader:
        pose = pose.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
            logits = model(pose, mask)
            loss_values = criterion(logits, labels)
        logits_for_metrics = logits.float()
        cross_entropy_loss_values = F.cross_entropy(logits_for_metrics, labels, reduction="none")
        probabilities = F.softmax(logits_for_metrics, dim=1)
        preds = logits.argmax(dim=1)

        batch_size = int(labels.shape[0])
        total_loss += _sum_loss(loss_values, batch_size=batch_size)
        total_cross_entropy_loss += float(cross_entropy_loss_values.detach().sum().item())
        true_confidence_total += float(
            probabilities.gather(1, labels.view(-1, 1)).detach().sum().item()
        )
        pred_confidence_total += float(probabilities.max(dim=1).values.detach().sum().item())
        total_items += batch_size

        true_np = labels.detach().cpu().numpy()
        pred_np = preds.detach().cpu().numpy()
        for batch_index, (true_label, pred_label) in enumerate(zip(true_np, pred_np, strict=False)):
            confusion[int(true_label), int(pred_label)] += 1
            if int(true_label) == int(pred_label) or len(misclassified) >= MAX_MISCLASSIFIED_EXAMPLES:
                continue
            sample_index = sample_offset + batch_index
            sample = dataset_samples[sample_index] if sample_index < len(dataset_samples) else {}
            true_name = _label_name(label_names, int(true_label))
            pred_name = _label_name(label_names, int(pred_label))
            misclassified.append(
                {
                    "sample_index": sample_index,
                    "item_id": sample.get("item_id"),
                    "target_label": sample.get("target_label") or true_name,
                    "predicted_label": pred_name,
                    "true_index": int(true_label),
                    "predicted_index": int(pred_label),
                    "pose_path": sample.get("pose_path"),
                    "video_path": sample.get("video_path"),
                }
            )
        sample_offset += batch_size

    accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))
    macro_f1, per_class = _compute_f1(confusion, labels=label_names)
    classification_summary = _summarize_classification_metrics(per_class)
    return {
        "loss": total_loss / max(total_items, 1),
        "cross_entropy_loss": total_cross_entropy_loss / max(total_items, 1),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        **classification_summary,
        "mean_true_confidence": true_confidence_total / max(total_items, 1),
        "mean_pred_confidence": pred_confidence_total / max(total_items, 1),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "misclassified_examples": misclassified,
    }


def _build_validation_error_analysis(final_metrics: dict, *, labels: list[str]) -> dict:
    confusion = np.asarray(final_metrics.get("confusion_matrix") or [], dtype=np.int64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    label_names = list(labels)
    per_class = final_metrics.get("per_class") if isinstance(final_metrics.get("per_class"), list) else []
    examples = [
        dict(example)
        for example in final_metrics.get("misclassified_examples", [])
        if isinstance(example, dict)
    ]

    class_summary = []
    for index, label in enumerate(label_names):
        if index >= confusion.shape[0]:
            break
        tp = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        predicted = int(confusion[:, index].sum())
        row = per_class[index] if index < len(per_class) and isinstance(per_class[index], dict) else {}
        class_summary.append(
            {
                "label": label,
                "class_index": index,
                "support": support,
                "predicted": predicted,
                "correct": tp,
                "false_negatives": max(support - tp, 0),
                "false_positives": max(predicted - tp, 0),
                "precision": row.get("precision"),
                "recall": row.get("recall"),
                "f1": row.get("f1"),
            }
        )

    pair_examples: dict[tuple[int, int], list[dict]] = {}
    false_negative_examples: dict[str, list[dict]] = {label: [] for label in label_names}
    for example in examples:
        true_index = _safe_label_index(example.get("true_index"), num_classes=len(label_names))
        predicted_index = _safe_label_index(example.get("predicted_index"), num_classes=len(label_names))
        if true_index < 0 or predicted_index < 0 or true_index == predicted_index:
            continue
        pair_key = (true_index, predicted_index)
        pair_bucket = pair_examples.setdefault(pair_key, [])
        if len(pair_bucket) < MAX_ERROR_EXAMPLES_PER_GROUP:
            pair_bucket.append(example)
        true_label = _label_name(label_names, true_index)
        label_bucket = false_negative_examples.setdefault(true_label, [])
        if len(label_bucket) < MAX_ERROR_EXAMPLES_PER_GROUP:
            label_bucket.append(example)

    confusion_pairs = []
    for true_index in range(confusion.shape[0]):
        for predicted_index in range(confusion.shape[1]):
            if true_index == predicted_index:
                continue
            count = int(confusion[true_index, predicted_index])
            if count <= 0:
                continue
            pair_key = (true_index, predicted_index)
            confusion_pairs.append(
                {
                    "true_label": _label_name(label_names, true_index),
                    "predicted_label": _label_name(label_names, predicted_index),
                    "true_index": true_index,
                    "predicted_index": predicted_index,
                    "count": count,
                    "examples": pair_examples.get(pair_key, []),
                }
            )
    confusion_pairs.sort(key=lambda item: (-int(item["count"]), item["true_label"], item["predicted_label"]))

    low_recall_classes = sorted(
        [
            row
            for row in class_summary
            if int(row.get("support") or 0) > 0 and float(row.get("recall") or 0.0) < 0.3
        ],
        key=lambda row: (float(row.get("recall") or 0.0), -int(row.get("support") or 0)),
    )
    missing_prediction_classes = [
        row["label"]
        for row in class_summary
        if int(row.get("support") or 0) > 0 and int(row.get("predicted") or 0) == 0
    ]
    filtered_false_negatives = {
        label: rows
        for label, rows in false_negative_examples.items()
        if rows
    }
    return {
        "summary": {
            "low_recall_classes": low_recall_classes[:8],
            "top_confusion_pairs": [
                {key: value for key, value in pair.items() if key != "examples"}
                for pair in confusion_pairs[:8]
            ],
            "missing_prediction_classes": missing_prediction_classes,
            "misclassified_example_count": len(examples),
        },
        "class_summary": class_summary,
        "false_negative_examples": filtered_false_negatives,
        "confusion_pair_examples": confusion_pairs[:20],
    }


def _compute_f1(confusion: np.ndarray, *, labels: list[str] | None = None) -> tuple[float, list[dict]]:
    metrics = []
    f1_scores = []
    for class_index in range(confusion.shape[0]):
        tp = float(confusion[class_index, class_index])
        fp = float(confusion[:, class_index].sum() - tp)
        fn = float(confusion[class_index, :].sum() - tp)
        support = int(confusion[class_index, :].sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        metrics.append(
            {
                "class_index": class_index,
                "label": _label_name(labels, class_index),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": support,
            }
        )
        f1_scores.append(f1)
    macro_f1 = float(sum(f1_scores) / max(len(f1_scores), 1))
    return macro_f1, metrics


def _summarize_classification_metrics(per_class: list[dict]) -> dict:
    supported_rows = [
        row for row in per_class
        if int(row.get("support", 0) or 0) > 0
    ]
    zero_support_labels = [
        str(row.get("label") or row.get("class_index"))
        for row in per_class
        if int(row.get("support", 0) or 0) <= 0
    ]
    supported_f1 = [float(row.get("f1", 0.0) or 0.0) for row in supported_rows]
    supported_recall = [float(row.get("recall", 0.0) or 0.0) for row in supported_rows]
    return {
        "macro_f1_supported": float(sum(supported_f1) / max(len(supported_f1), 1)),
        "balanced_accuracy": float(sum(supported_recall) / max(len(supported_recall), 1)),
        "zero_support_labels": zero_support_labels,
        "supported_class_count": len(supported_rows),
    }


def _load_compatible_state_dict(model: nn.Module, checkpoint_state: dict) -> tuple[int, list[str]]:
    model_state = model.state_dict()
    compatible_state = {}
    skipped_keys: list[str] = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            skipped_keys.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_keys.append(key)
            continue
        compatible_state[key] = value

    if compatible_state:
        model.load_state_dict(compatible_state, strict=False)
    return len(compatible_state), skipped_keys


def _build_dataloader(
    *,
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    sampler,
    num_workers: int,
    pin_memory: bool,
    pin_memory_device: str | None,
    prefetch_factor: int,
    persistent_workers: bool,
    generator: torch.Generator | None,
    worker_init_fn,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
        kwargs["shuffle"] = False
    else:
        kwargs["shuffle"] = shuffle
    if generator is not None and sampler is None:
        kwargs["generator"] = generator
    if worker_init_fn is not None:
        kwargs["worker_init_fn"] = worker_init_fn
    if pin_memory and pin_memory_device and DATALOADER_SUPPORTS_PIN_MEMORY_DEVICE:
        kwargs["pin_memory_device"] = pin_memory_device
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(dataset, **kwargs)


def _is_dataloader_worker_permission_error(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}"
    return (
        isinstance(exc, PermissionError)
        or "WinError 5" in message
        or "Access is denied" in message
        or "access is denied" in message
        or "액세스가 거부" in message
    )


def _uses_cuda(device: str) -> bool:
    return str(device).strip().lower().startswith("cuda")


def _resolve_num_workers(value: int | str, *, batch_size: int) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "auto", "default"}:
            requested = None
        else:
            requested = max(int(normalized), 0)
    else:
        requested = max(int(value), 0)

    if requested is not None:
        return requested

    cpu_count = os.cpu_count() or 2
    auto_workers = min(8, max(2, cpu_count // 2))
    return min(auto_workers, max(int(batch_size), 1))


def _resolve_eval_batch_size(value: int | None, *, train_batch_size: int) -> int:
    if value is None:
        requested = 0
    else:
        requested = int(value)
    if requested > 0:
        return requested
    return max(train_batch_size, train_batch_size * 2)


def _resolve_dataset_cache_size(value: int, *, num_workers: int) -> int:
    requested = max(int(value), 0)
    if requested == 0:
        return 0
    if num_workers <= 0:
        return requested
    return max(128, requested // (num_workers + 1))


def _resolve_pin_memory(value: bool | str, *, use_cuda: bool) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "auto", "default"}:
            return bool(use_cuda)
        return normalized in {"1", "true", "yes", "on"}
    return bool(value and use_cuda)


def _resolve_pin_memory_device(*, use_pin_memory: bool, device: str) -> str | None:
    if not use_pin_memory:
        return None
    if not _uses_cuda(device):
        return None
    if device.strip().lower() == "cuda":
        return "cuda"
    return str(device).strip()


def _configure_training_acceleration(*, device: str, use_cuda: bool, deterministic: bool) -> None:
    if not use_cuda:
        return
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        if hasattr(torch.backends.cudnn, "deterministic"):
            torch.backends.cudnn.deterministic = bool(deterministic)
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _create_grad_scaler(*, enabled: bool):
    if not enabled:
        return None
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        try:
            return amp_module.GradScaler("cuda", enabled=True)
        except TypeError:
            return amp_module.GradScaler(enabled=True)
    cuda_amp = getattr(torch.cuda, "amp", None)
    if cuda_amp is not None and hasattr(cuda_amp, "GradScaler"):
        return cuda_amp.GradScaler(enabled=True)
    return None


def _resolve_amp_dtype(
    *,
    device: str,
    enabled: bool,
    requested_dtype: str | None,
) -> tuple[torch.dtype | None, str]:
    if not enabled or not _uses_cuda(device):
        return None, "disabled"

    normalized = str(requested_dtype or "auto").strip().lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16, "float16"
    if normalized in {"bfloat16", "bf16"}:
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_bf16_supported) and is_bf16_supported():
            return torch.bfloat16, "bfloat16"
        return torch.float16, "float16"

    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported) and is_bf16_supported():
        return torch.bfloat16, "bfloat16"
    return torch.float16, "float16"


def _autocast_context(*, device: str, enabled: bool, amp_dtype: torch.dtype | None):
    if not enabled:
        return nullcontext()
    device_type = "cuda" if _uses_cuda(device) else "cpu"
    if hasattr(torch, "autocast"):
        kwargs = {"device_type": device_type, "enabled": True}
        if amp_dtype is not None:
            kwargs["dtype"] = amp_dtype
        try:
            return torch.autocast(**kwargs)
        except TypeError:
            kwargs.pop("dtype", None)
            return torch.autocast(**kwargs)
    amp_module = getattr(torch.cuda, "amp", None)
    if amp_module is not None and hasattr(amp_module, "autocast"):
        kwargs = {"enabled": True}
        if amp_dtype is not None:
            kwargs["dtype"] = amp_dtype
        try:
            return amp_module.autocast(**kwargs)
        except TypeError:
            kwargs.pop("dtype", None)
            return amp_module.autocast(**kwargs)
    return nullcontext()


def _write_progress(progress_path: Path, payload: dict) -> None:
    content = {
        **payload,
        "updated_at": datetime.now(UTC).astimezone().isoformat(),
    }
    write_json_atomic(progress_path, content)


def _save_torch_checkpoint_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temp_path)
        _replace_path_with_retries(temp_path, path)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def _replace_path_with_retries(
    source_path: Path,
    target_path: Path,
    *,
    retries: int = 30,
    delay_seconds: float = 0.1,
) -> None:
    last_error: OSError | None = None
    for attempt in range(max(int(retries), 1)):
        try:
            source_path.replace(target_path)
            return
        except PermissionError as exc:
            last_error = exc
        except OSError as exc:
            last_error = exc
        if attempt + 1 < retries:
            time.sleep(float(delay_seconds))
    if last_error is not None:
        raise last_error
    source_path.replace(target_path)
