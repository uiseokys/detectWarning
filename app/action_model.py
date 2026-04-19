from __future__ import annotations

import json
import os
import platform
from collections import OrderedDict
from contextlib import nullcontext
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainingArtifacts:
    best_model_path: Path
    metrics_path: Path
    labels_path: Path
    history: list[dict]


class PoseSequenceDataset(Dataset):
    def __init__(self, manifest_path: Path, *, cache_size: int = 0) -> None:
        self.samples = []
        self.cache_size = max(int(cache_size), 0)
        self._cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))
        if not self.samples:
            raise RuntimeError(f"학습용 샘플이 없습니다: {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        cache_key = str(sample["pose_path"])
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
        label_idx = int(sample["label_idx"])
        return (
            torch.from_numpy(pose),
            torch.from_numpy(mask),
            torch.tensor(label_idx, dtype=torch.long),
        )


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
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    num_workers: int | str = 0,
    device: str = "cuda",
    amp: bool = True,
    compile_model: bool = False,
    compile_backend: str | None = None,
    dataset_cache_size: int = 2048,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    progress_path: Path | None = None,
    resume_from: Path | None = None,
) -> TrainingArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.json"
    metrics_path = output_dir / "metrics.json"
    best_model_path = output_dir / "best_action_model.pt"

    train_dataset = PoseSequenceDataset(train_manifest, cache_size=dataset_cache_size)
    val_dataset = PoseSequenceDataset(val_manifest, cache_size=dataset_cache_size)

    use_cuda = _uses_cuda(device)
    use_amp = bool(amp and use_cuda and torch.cuda.is_available())
    resolved_num_workers = _resolve_num_workers(num_workers, batch_size=batch_size)
    resolved_prefetch_factor = max(int(prefetch_factor), 1)
    use_persistent_workers = bool(persistent_workers and resolved_num_workers > 0)

    _configure_training_acceleration(device=device, use_cuda=use_cuda)

    train_loader = _build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=resolved_num_workers,
        pin_memory=use_cuda,
        prefetch_factor=resolved_prefetch_factor,
        persistent_workers=use_persistent_workers,
    )
    val_loader = _build_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=resolved_num_workers,
        pin_memory=use_cuda,
        prefetch_factor=resolved_prefetch_factor,
        persistent_workers=use_persistent_workers,
    )

    first_pose, _first_mask, _first_label = train_dataset[0]
    base_model = TemporalPoseClassifier(
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
            base_model.load_state_dict(checkpoint_state, strict=False)
            resumed_from_checkpoint = True
            resume_mode = "full"
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

    class_weights = _build_class_weights(train_dataset.samples, len(labels)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = _create_grad_scaler(enabled=use_amp)

    history: list[dict] = []
    best_val_f1 = -1.0
    best_epoch = 0
    train_sample_count = len(train_dataset)
    val_sample_count = len(val_dataset)

    print(
        "[train] acceleration "
        f"device={device} "
        f"amp={'on' if use_amp else 'off'} "
        f"compile={'on' if compiled_model else 'off'} "
        f"compile_backend={compiled_backend} "
        f"resume={resume_mode} "
        f"cache={dataset_cache_size} "
        f"workers={resolved_num_workers} "
        f"prefetch={resolved_prefetch_factor if resolved_num_workers > 0 else 0} "
        f"persistent={'on' if use_persistent_workers else 'off'}"
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
                "amp_enabled": use_amp,
                "compile_enabled": compiled_model,
                "compile_backend": compiled_backend,
                "dataset_cache_size": dataset_cache_size,
                "num_workers": resolved_num_workers,
                "prefetch_factor": resolved_prefetch_factor if resolved_num_workers > 0 else 0,
                "persistent_workers": use_persistent_workers,
            },
        )

    for epoch in range(1, epochs + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_metrics = _evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=len(labels),
            use_amp=use_amp,
        )
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
            "learning_rate": round(current_lr, 8),
        }
        history.append(epoch_metrics)
        epoch_progress = epoch / max(epochs, 1)
        print(
            "[train] "
            f"{epoch_progress * 100:.1f}% "
            f"(epoch {epoch}/{epochs}) "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] >= best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": base_model.state_dict(),
                    "labels": labels,
                    "num_joints": int(first_pose.shape[1]),
                    "input_dim": int(first_pose.shape[2]),
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                },
                best_model_path,
            )

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
                    "amp_enabled": use_amp,
                    "compile_enabled": compiled_model,
                    "compile_backend": compiled_backend,
                    "num_workers": resolved_num_workers,
                    "prefetch_factor": resolved_prefetch_factor if resolved_num_workers > 0 else 0,
                    "persistent_workers": use_persistent_workers,
                },
            )

    with labels_path.open("w", encoding="utf-8") as handle:
        json.dump({"labels": labels}, handle, ensure_ascii=False, indent=2)

    final_metrics = _evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=len(labels),
        use_amp=use_amp,
    )
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "history": history,
                "final_validation": final_metrics,
                "labels": labels,
                "best_epoch": best_epoch,
                "resumed_from_checkpoint": resumed_from_checkpoint,
                "resume_mode": resume_mode,
                "amp_enabled": use_amp,
                "compile_enabled": compiled_model,
                "compile_backend": compiled_backend,
                "num_workers": resolved_num_workers,
                "prefetch_factor": resolved_prefetch_factor if resolved_num_workers > 0 else 0,
                "persistent_workers": use_persistent_workers,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    if progress_path is not None:
        _write_progress(
            progress_path,
            {
                "state": "completed",
                "device": device,
                "epochs_total": epochs,
                "epochs_completed": epochs,
                "best_val_macro_f1": round(best_val_f1, 6),
                "best_epoch": best_epoch,
                "latest": history[-1] if history else None,
                "history": history,
                "labels": labels,
                "final_validation": final_metrics,
                "resumed_from_checkpoint": resumed_from_checkpoint,
                "resume_mode": resume_mode,
                "train_samples": train_sample_count,
                "val_samples": val_sample_count,
                "amp_enabled": use_amp,
                "compile_enabled": compiled_model,
                "compile_backend": compiled_backend,
                "num_workers": resolved_num_workers,
                "prefetch_factor": resolved_prefetch_factor if resolved_num_workers > 0 else 0,
                "persistent_workers": use_persistent_workers,
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
        counts[int(sample["label_idx"])] += 1.0
    weights = counts.sum() / (counts * len(counts))
    return torch.tensor(weights, dtype=torch.float32)


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: str,
    train: bool,
    use_amp: bool,
    scaler,
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

        with _autocast_context(device=device, enabled=use_amp):
            logits = model(pose, mask)
            loss = criterion(logits, labels)

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
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
    use_amp: bool,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_items = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pose, mask, labels in loader:
        pose = pose.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _autocast_context(device=device, enabled=use_amp):
            logits = model(pose, mask)
            loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        true_np = labels.detach().cpu().numpy()
        pred_np = preds.detach().cpu().numpy()
        for true_label, pred_label in zip(true_np, pred_np):
            confusion[int(true_label), int(pred_label)] += 1

    accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))
    macro_f1, per_class = _compute_f1(confusion)
    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def _compute_f1(confusion: np.ndarray) -> tuple[float, list[dict]]:
    metrics = []
    f1_scores = []
    for class_index in range(confusion.shape[0]):
        tp = float(confusion[class_index, class_index])
        fp = float(confusion[:, class_index].sum() - tp)
        fn = float(confusion[class_index, :].sum() - tp)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        metrics.append(
            {
                "class_index": class_index,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            }
        )
        f1_scores.append(f1)
    macro_f1 = float(sum(f1_scores) / max(len(f1_scores), 1))
    return macro_f1, metrics


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
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(dataset, **kwargs)


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

    if requested is not None and requested > 0:
        return requested

    cpu_count = os.cpu_count() or 2
    auto_workers = min(8, max(2, cpu_count // 2))
    return min(auto_workers, max(int(batch_size), 1))


def _configure_training_acceleration(*, device: str, use_cuda: bool) -> None:
    if not use_cuda:
        return
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
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


def _autocast_context(*, device: str, enabled: bool):
    if not enabled:
        return nullcontext()
    device_type = "cuda" if _uses_cuda(device) else "cpu"
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device_type, enabled=True)
    amp_module = getattr(torch.cuda, "amp", None)
    if amp_module is not None and hasattr(amp_module, "autocast"):
        return amp_module.autocast(enabled=True)
    return nullcontext()


def _write_progress(progress_path: Path, payload: dict) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        **payload,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    with progress_path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2)
