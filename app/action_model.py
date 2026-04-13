from __future__ import annotations

import json
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
    def __init__(self, manifest_path: Path) -> None:
        self.samples = []
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
        pose_path = Path(sample["pose_path"])
        loaded = np.load(pose_path, allow_pickle=False)
        pose = loaded["pose"].astype(np.float32)
        mask = loaded["mask"].astype(np.float32)
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
    num_workers: int = 0,
    device: str = "cuda",
    progress_path: Path | None = None,
) -> TrainingArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.json"
    metrics_path = output_dir / "metrics.json"
    best_model_path = output_dir / "best_action_model.pt"

    train_dataset = PoseSequenceDataset(train_manifest)
    val_dataset = PoseSequenceDataset(val_manifest)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    first_pose, _first_mask, _first_label = train_dataset[0]
    model = TemporalPoseClassifier(
        num_joints=first_pose.shape[1],
        input_dim=first_pose.shape[2],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=len(labels),
        dropout=dropout,
    ).to(device)

    class_weights = _build_class_weights(train_dataset.samples, len(labels)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    history: list[dict] = []
    best_val_f1 = -1.0
    best_epoch = 0

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
            },
        )

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )
        val_metrics = _evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=len(labels),
        )
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
        }
        history.append(epoch_metrics)

        if val_metrics["macro_f1"] >= best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
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
    )
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "history": history,
                "final_validation": final_metrics,
                "labels": labels,
                "best_epoch": best_epoch,
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
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_items = 0

    for pose, mask, labels in loader:
        pose = pose.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(pose, mask)
        loss = criterion(logits, labels)

        if train:
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
) -> dict:
    model.eval()
    total_loss = 0.0
    total_items = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pose, mask, labels in loader:
        pose = pose.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

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


def _write_progress(progress_path: Path, payload: dict) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        **payload,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    with progress_path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2)
