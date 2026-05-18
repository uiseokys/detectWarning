from __future__ import annotations

import pickle
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ensemble_action_models import apply_class_bias, metrics_from_probabilities
from hybrid_pose_ensemble import build_feature_matrix, build_feature_models, read_manifest_rows
from reporting import write_json_atomic


DEFAULT_SPECIALIZED_MODELS = (
    "extra_trees_fast",
    "extra_trees_leaf2_fast",
    "extra_trees_accuracy",
    "random_forest_leaf2",
    "hist_gradient",
    "logreg",
)


def run_specialized_action_tasks(
    *,
    config: dict,
    paths: dict,
    manifests: dict[str, Path],
    labels: list[str],
) -> dict:
    task_config = config.get("training_tasks") if isinstance(config.get("training_tasks"), dict) else {}
    if not bool(task_config.get("enabled", True)):
        return {"enabled": False, "reason": "disabled"}

    normal_label = str(task_config.get("normal_label") or "normal").strip() or "normal"
    output_dir = paths["artifacts_dir"] / "specialized_tasks"
    output_dir.mkdir(parents=True, exist_ok=True)
    danger_labels = [label for label in labels if str(label) != normal_label]
    clear_stale_specialized_metrics(
        output_dir=output_dir,
        normal_label=normal_label,
        danger_labels=danger_labels,
    )

    rows_by_split = {
        split_name: read_manifest_rows(path)
        for split_name, path in manifests.items()
        if isinstance(path, Path) and path.exists()
    }
    model_names = _resolve_model_names(task_config)
    summary = {
        "schema_version": 1,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "normal_label": normal_label,
        "models": model_names,
        "tasks": {},
    }

    if bool(task_config.get("detection_enabled", True)):
        detection_min_recall = float(task_config.get("detection_min_recall", 0.95) or 0.95)
        detection_max_false_alarm_rate = float(
            task_config.get("detection_max_false_alarm_rate", 0.55) or 0.55
        )
        summary["tasks"]["detection"] = train_specialized_feature_task(
            task_name="detection",
            task_label="Abnormal detection",
            rows_by_split=rows_by_split,
            labels=["normal", "abnormal"],
            output_dir=output_dir / "detection",
            model_names=model_names,
            row_mapper=lambda row: _map_detection_row(row, normal_label=normal_label),
            target_metric=str(task_config.get("detection_target_metric") or "detection_balanced"),
            detection_min_recall=detection_min_recall,
            detection_max_false_alarm_rate=detection_max_false_alarm_rate,
        )

    if bool(task_config.get("classification_enabled", True)):
        summary["tasks"]["classification"] = train_specialized_feature_task(
            task_name="classification",
            task_label="Abnormal type classification",
            rows_by_split=rows_by_split,
            labels=danger_labels,
            output_dir=output_dir / "classification",
            model_names=model_names,
            row_mapper=lambda row: _map_classification_row(row, labels=danger_labels),
            target_metric=str(task_config.get("classification_target_metric") or "macro_f1_supported"),
            detection_min_recall=0.0,
            detection_max_false_alarm_rate=1.0,
            classification_bias_multipliers=_resolve_bias_multipliers(task_config),
        )
        if bool(task_config.get("pose_classification_enabled", True)):
            summary["tasks"]["pose_classification"] = train_pose_abnormal_classifier(
                config=config,
                paths=paths,
                rows_by_split=rows_by_split,
                labels=danger_labels,
                output_dir=output_dir / "pose_classification",
                task_config=task_config,
            )

    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def clear_stale_specialized_metrics(*, output_dir: Path, normal_label: str, danger_labels: list[str]) -> None:
    now = _utc_now_iso()
    placeholders = {
        output_dir / "classification" / "metrics.json": {
            "task": "classification",
            "name": "Abnormal type classification",
            "available": False,
            "reason": "refreshing_for_current_label_set",
            "labels": danger_labels,
            "created_at": now,
            "best_result": {},
        },
        output_dir / "pose_classification" / "metrics.json": {
            "task": "pose_classification",
            "name": "Pose abnormal type classification",
            "available": False,
            "reason": "refreshing_for_current_label_set",
            "labels": danger_labels,
            "created_at": now,
            "best_result": {},
        },
        output_dir / "detection" / "metrics.json": {
            "task": "detection",
            "name": "Abnormal detection",
            "available": False,
            "reason": "refreshing_for_current_label_set",
            "labels": [normal_label, "abnormal"],
            "created_at": now,
            "best_result": {},
        },
    }
    for path, payload in placeholders.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, payload)


def train_pose_abnormal_classifier(
    *,
    config: dict,
    paths: dict,
    rows_by_split: dict[str, list[dict]],
    labels: list[str],
    output_dir: Path,
    task_config: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifests = {
        split_name: materialize_pose_classification_manifest(
            rows,
            labels=labels,
            output_path=manifest_dir / f"{split_name}.jsonl",
        )
        for split_name, rows in rows_by_split.items()
        if split_name in {"train", "val", "test"}
    }
    train_count = count_jsonl_lines(manifests.get("train"))
    val_count = count_jsonl_lines(manifests.get("val"))
    payload: dict[str, Any] = {
        "task": "pose_classification",
        "name": "Pose abnormal type classification",
        "labels": labels,
        "created_at": _utc_now_iso(),
        "train_samples": train_count,
        "val_samples": val_count,
        "test_samples": count_jsonl_lines(manifests.get("test")),
    }
    if len(labels) < 2 or train_count <= 0 or val_count <= 0:
        payload.update({"available": False, "reason": "not_enough_pose_samples", "best_result": {}})
        write_json_atomic(output_dir / "metrics.json", payload)
        return payload

    training = config.get("training") if isinstance(config.get("training"), dict) else {}
    from action_model import train_action_classifier

    pose_multipliers = dict(task_config.get("pose_classification_class_weight_multipliers") or {})
    if "abduction" in labels:
        pose_multipliers["abduction"] = max(float(pose_multipliers.get("abduction") or 1.0), 2.5)
    try:
        artifacts = train_action_classifier(
            train_manifest=manifests["train"],
            val_manifest=manifests["val"],
            output_dir=output_dir,
            labels=labels,
            epochs=int(task_config.get("pose_classification_epochs") or training.get("epochs") or 30),
            batch_size=int(training.get("batch_size", 48)),
            eval_batch_size=int(training.get("eval_batch_size", 96) or 0),
            learning_rate=float(task_config.get("pose_classification_learning_rate") or training.get("learning_rate") or 3e-4),
            weight_decay=float(task_config.get("pose_classification_weight_decay") or training.get("weight_decay") or 0.005),
            hidden_dim=int(training.get("hidden_dim", 128)),
            num_layers=int(training.get("num_layers", 2)),
            dropout=float(task_config.get("pose_classification_dropout") or training.get("dropout") or 0.45),
            temporal_pooling=str(training.get("temporal_pooling", "mean")),
            label_smoothing=float(task_config.get("pose_classification_label_smoothing") or 0.0),
            loss_name=str(task_config.get("pose_classification_loss") or training.get("loss") or "focal"),
            focal_gamma=float(task_config.get("pose_classification_focal_gamma") or training.get("focal_gamma") or 1.0),
            class_weight=training.get("class_weight", "balanced"),
            class_weight_multipliers=pose_multipliers,
            balanced_sampler=training.get("balanced_sampler", True),
            grad_clip_norm=float(training.get("grad_clip_norm", 1.0)),
            seed=training.get("seed", config.get("split", {}).get("seed", 42)),
            deterministic=bool(training.get("deterministic", False)),
            num_workers=training.get("num_workers", "auto"),
            device=str(training.get("device", "cuda")),
            amp=bool(training.get("amp", True)),
            amp_dtype=str(training.get("amp_dtype", "auto")),
            compile_model=bool(training.get("compile_model", True)),
            compile_backend=str(training.get("compile_backend", "auto")),
            dataset_cache_size=int(training.get("dataset_cache_size", 2048)),
            prefetch_factor=int(training.get("prefetch_factor", 2)),
            persistent_workers=bool(training.get("persistent_workers", True)),
            pin_memory=training.get("pin_memory", "auto"),
            early_stopping_patience=int(task_config.get("pose_classification_early_stopping_patience") or training.get("early_stopping_patience") or 8),
            early_stopping_min_delta=float(
                task_config.get("pose_classification_early_stopping_min_delta")
                or training.get("early_stopping_min_delta", 0.0015)
            ),
            selection_metric=str(task_config.get("pose_classification_selection_metric") or training.get("selection_metric") or "macro_f1"),
            overfit_guard_enabled=bool(task_config.get("pose_classification_overfit_guard_enabled", True)),
            overfit_guard_min_epoch=int(task_config.get("pose_classification_overfit_guard_min_epoch") or 6),
            overfit_guard_loss_gap=float(task_config.get("pose_classification_overfit_guard_loss_gap") or 1.25),
            overfit_guard_patience=int(task_config.get("pose_classification_overfit_guard_patience") or 2),
            imbalance_warn_min_samples=int(training.get("imbalance_warn_min_samples", 8)),
            imbalance_warn_ratio=float(training.get("imbalance_warn_ratio", 3.0)),
            progress_path=output_dir / "training_progress.json",
            resume_from=None,
            max_duplicate_pose_label_samples=int(task_config.get("pose_classification_max_duplicate_pose_label_samples") or 0),
        )
    except Exception as exc:
        payload.update({"available": False, "reason": str(exc), "best_result": {}})
        write_json_atomic(output_dir / "metrics.json", payload)
        return payload

    metrics = read_json_file(artifacts.metrics_path)
    final_validation = metrics.get("final_validation") if isinstance(metrics, dict) else {}
    if isinstance(final_validation, dict):
        final_validation["classification"] = build_multiclass_task_metrics(final_validation, labels=labels)
    payload.update(
        {
            "available": True,
            "best_result": {
                "model": "pose_lstm",
                "score": float((final_validation or {}).get("macro_f1_supported") or (final_validation or {}).get("macro_f1") or 0.0),
                "validation": final_validation,
                "metrics_path": str(artifacts.metrics_path),
                "model_path": str(artifacts.best_model_path),
                "class_weight_multipliers": pose_multipliers,
            },
        }
    )
    write_json_atomic(output_dir / "metrics.json", payload)
    return payload


def materialize_pose_classification_manifest(rows: list[dict], *, labels: list[str], output_path: Path) -> Path:
    label_to_idx = {label: index for index, label in enumerate(labels)}
    seen: set[tuple[str, str, str]] = set()
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            label = str(row.get("target_label") or "").strip()
            pose_path = str(row.get("pose_path") or "").strip()
            if label not in label_to_idx or not pose_path or not Path(pose_path).exists():
                continue
            key = (pose_path, label, str(row.get("item_id") or ""))
            if key in seen:
                continue
            seen.add(key)
            next_row = dict(row)
            next_row["source_target_label"] = label
            next_row["target_label"] = label
            next_row["label"] = label
            next_row["label_idx"] = label_to_idx[label]
            handle.write(json.dumps(next_row, ensure_ascii=False) + "\n")
    return output_path


def count_jsonl_lines(path: Path | None) -> int:
    if not isinstance(path, Path) or not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def read_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def train_specialized_feature_task(
    *,
    task_name: str,
    task_label: str,
    rows_by_split: dict[str, list[dict]],
    labels: list[str],
    output_dir: Path,
    model_names: list[str],
    row_mapper,
    target_metric: str,
    detection_min_recall: float,
    detection_max_false_alarm_rate: float = 1.0,
    classification_bias_multipliers: list[float] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapped_rows = {
        split_name: _map_rows(rows, row_mapper=row_mapper)
        for split_name, rows in rows_by_split.items()
    }
    train_rows = mapped_rows.get("train") or []
    val_rows = mapped_rows.get("val") or []
    test_rows = mapped_rows.get("test") or []
    payload: dict[str, Any] = {
        "task": task_name,
        "name": task_label,
        "labels": labels,
        "target_metric": target_metric,
        "created_at": _utc_now_iso(),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "test_samples": len(test_rows),
        "train_distribution": dict(Counter(str(row.get("target_label")) for row in train_rows)),
        "val_distribution": dict(Counter(str(row.get("target_label")) for row in val_rows)),
        "test_distribution": dict(Counter(str(row.get("target_label")) for row in test_rows)),
        "candidates": [],
    }
    if len(labels) < 2 or not _has_multiple_labels(train_rows) or not _has_multiple_labels(val_rows):
        payload.update(
            {
                "available": False,
                "reason": "not_enough_classes",
                "best_result": {},
            }
        )
        write_json_atomic(output_dir / "metrics.json", payload)
        return payload

    x_train, y_train = build_feature_matrix(train_rows)
    x_val, y_val = build_feature_matrix(val_rows)
    x_test, y_test = build_feature_matrix(test_rows) if test_rows else (None, None)
    models = build_feature_models()
    best: dict | None = None
    for model_name in model_names:
        model = models.get(model_name)
        if model is None:
            continue
        print(f"[specialized][{task_name}] training {model_name}", flush=True)
        try:
            model.fit(x_train, y_train)
        except Exception as exc:
            payload["candidates"].append(
                {
                    "model": model_name,
                    "available": False,
                    "reason": str(exc),
                }
            )
            print(f"[specialized][{task_name}] skipped {model_name}: {exc}", flush=True)
            continue
        val_probabilities = _predict_full_probabilities(model, x_val, num_classes=len(labels))
        val_metrics = metrics_from_probabilities(
            val_probabilities,
            true_labels=y_val,
            samples=val_rows,
            labels=labels,
        )
        threshold = None
        if task_name == "detection":
            threshold_result = select_detection_threshold(
                val_probabilities,
                true_labels=y_val,
                labels=labels,
                target_metric=target_metric,
                min_recall=detection_min_recall,
                max_false_alarm_rate=detection_max_false_alarm_rate,
            )
            threshold = threshold_result["threshold"]
            val_probabilities = detection_probabilities_at_threshold(
                val_probabilities,
                threshold=threshold,
            )
            val_metrics = metrics_from_probabilities(
                val_probabilities,
                true_labels=y_val,
                samples=val_rows,
                labels=labels,
            )
            val_metrics["detection"] = threshold_result["metrics"]
            val_metrics["detection"]["threshold"] = threshold
        else:
            bias_result = select_multiclass_class_bias(
                val_probabilities,
                true_labels=y_val,
                samples=val_rows,
                labels=labels,
                multipliers=classification_bias_multipliers,
                target_metric=target_metric,
            )
            val_probabilities = bias_result["probabilities"]
            val_metrics = metrics_from_probabilities(
                val_probabilities,
                true_labels=y_val,
                samples=val_rows,
                labels=labels,
            )
            val_metrics["classification"] = build_multiclass_task_metrics(val_metrics, labels=labels)
            val_metrics["classification"]["class_bias"] = bias_result["class_bias"]
            val_metrics["classification"]["class_bias_name"] = bias_result["class_bias_name"]
        test_metrics = None
        if x_test is not None and y_test is not None:
            test_probabilities = _predict_full_probabilities(model, x_test, num_classes=len(labels))
            if task_name == "detection" and threshold is not None:
                test_probabilities = detection_probabilities_at_threshold(
                    test_probabilities,
                    threshold=threshold,
                )
            elif task_name == "classification":
                test_probabilities = apply_class_bias(
                    test_probabilities,
                    bias_result["class_bias"] if "bias_result" in locals() else {},
                    labels=labels,
                )
            test_metrics = metrics_from_probabilities(
                test_probabilities,
                true_labels=y_test,
                samples=test_rows,
                labels=labels,
            )
            if task_name == "detection":
                test_metrics["detection"] = build_detection_metrics(test_metrics, labels=labels)
                test_metrics["detection"]["threshold"] = threshold
            else:
                test_metrics["classification"] = build_multiclass_task_metrics(test_metrics, labels=labels)
                test_metrics["classification"]["class_bias"] = bias_result["class_bias"] if "bias_result" in locals() else {}
                test_metrics["classification"]["class_bias_name"] = (
                    bias_result["class_bias_name"] if "bias_result" in locals() else "none"
                )
        result = {
            "model": model_name,
            "threshold": threshold,
            "class_bias": bias_result["class_bias"] if task_name == "classification" and "bias_result" in locals() else {},
            "class_bias_name": (
                bias_result["class_bias_name"] if task_name == "classification" and "bias_result" in locals() else "none"
            ),
            "validation": val_metrics,
            "holdout_test": test_metrics,
            "score": _score_result(val_metrics, target_metric=target_metric),
        }
        payload["candidates"].append(_compact_candidate(result))
        if best is None or result["score"] > best["score"]:
            best = result
            with (output_dir / "best_feature_model.pkl").open("wb") as handle:
                pickle.dump({"model": model, "model_name": model_name, "labels": labels, "task": task_name}, handle)

    payload["available"] = best is not None
    payload["best_result"] = _compact_candidate(best) if best else {}
    write_json_atomic(output_dir / "metrics.json", payload)
    return payload


def build_detection_metrics(metrics: dict, *, labels: list[str]) -> dict:
    try:
        normal_index = labels.index("normal")
    except ValueError:
        normal_index = 0
    confusion = np.asarray(metrics.get("confusion_matrix") or [], dtype=np.int64)
    if confusion.ndim != 2 or confusion.shape[0] <= normal_index or confusion.shape[1] <= normal_index:
        return {}
    total = int(confusion.sum())
    tn = int(confusion[normal_index, normal_index])
    fp = int(confusion[normal_index, :].sum() - tn)
    fn = int(confusion[:, normal_index].sum() - tn)
    tp = int(total - tn - fp - fn)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    specificity = tn / max(tn + fp, 1)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_alarm_rate": fp / max(fp + tn, 1),
        "miss_rate": fn / max(fn + tp, 1),
    }


def select_detection_threshold(
    probabilities: np.ndarray,
    *,
    true_labels: np.ndarray,
    labels: list[str],
    target_metric: str,
    min_recall: float = 0.95,
    max_false_alarm_rate: float = 1.0,
) -> dict:
    if probabilities.shape[1] < 2:
        threshold = 0.5
        metrics = detection_metrics_from_predictions(true_labels, np.zeros_like(true_labels), labels=labels)
        return {"threshold": threshold, "metrics": metrics, "score": 0.0}
    abnormal_scores = probabilities[:, 1]
    candidate_thresholds = sorted(
        {
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            *np.linspace(0.05, 0.95, 37).round(6).tolist(),
            *np.quantile(abnormal_scores, np.linspace(0.02, 0.98, 49)).round(6).tolist(),
        }
    )
    best: dict | None = None
    recall_only_best: dict | None = None
    fallback: dict | None = None
    max_false_alarm_rate = max(0.0, min(float(max_false_alarm_rate), 1.0))
    for threshold in candidate_thresholds:
        predicted = (abnormal_scores >= float(threshold)).astype(np.int64)
        metrics = detection_metrics_from_predictions(true_labels, predicted, labels=labels)
        score = _score_detection_metrics(metrics, target_metric=target_metric)
        candidate = {
            "threshold": round(float(threshold), 6),
            "metrics": metrics,
            "score": score,
        }
        if fallback is None or score > fallback["score"]:
            fallback = candidate
        if metrics.get("recall", 0.0) < min_recall:
            continue
        if recall_only_best is None or score > recall_only_best["score"]:
            recall_only_best = candidate
        if metrics.get("false_alarm_rate", 1.0) > max_false_alarm_rate:
            continue
        if best is None or score > best["score"]:
            best = candidate
    selected = best or recall_only_best or fallback or {"threshold": 0.5, "metrics": {}, "score": 0.0}
    if isinstance(selected, dict):
        selected["constraints"] = {
            "min_recall": float(min_recall),
            "max_false_alarm_rate": float(max_false_alarm_rate),
            "met_recall": float((selected.get("metrics") or {}).get("recall") or 0.0) >= float(min_recall),
            "met_false_alarm_rate": float((selected.get("metrics") or {}).get("false_alarm_rate") or 1.0)
            <= float(max_false_alarm_rate),
        }
    return selected


def detection_probabilities_at_threshold(probabilities: np.ndarray, *, threshold: float) -> np.ndarray:
    abnormal_scores = np.asarray(probabilities[:, 1], dtype=np.float64)
    predicted_abnormal = abnormal_scores >= float(threshold)
    adjusted = np.zeros((probabilities.shape[0], 2), dtype=np.float64)
    adjusted[:, 1] = predicted_abnormal.astype(np.float64)
    adjusted[:, 0] = 1.0 - adjusted[:, 1]
    return adjusted


def detection_metrics_from_predictions(true_labels: np.ndarray, predicted_labels: np.ndarray, *, labels: list[str]) -> dict:
    true_labels = np.asarray(true_labels, dtype=np.int64)
    predicted_labels = np.asarray(predicted_labels, dtype=np.int64)
    normal_index = 0
    tp = int(((true_labels != normal_index) & (predicted_labels != normal_index)).sum())
    tn = int(((true_labels == normal_index) & (predicted_labels == normal_index)).sum())
    fp = int(((true_labels == normal_index) & (predicted_labels != normal_index)).sum())
    fn = int(((true_labels != normal_index) & (predicted_labels == normal_index)).sum())
    total = tp + tn + fp + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    specificity = tn / max(tn + fp, 1)
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "false_alarm_rate": fp / max(fp + tn, 1),
        "miss_rate": fn / max(fn + tp, 1),
    }


def build_multiclass_task_metrics(metrics: dict, *, labels: list[str]) -> dict:
    per_class = metrics.get("per_class") if isinstance(metrics.get("per_class"), list) else []
    rows = [row for row in per_class if isinstance(row, dict) and int(row.get("support") or 0) > 0]
    if not rows:
        return {}
    supports = np.asarray([int(row.get("support") or 0) for row in rows], dtype=np.float64)
    f1_values = np.asarray([float(row.get("f1") or 0.0) for row in rows], dtype=np.float64)
    precision_values = np.asarray([float(row.get("precision") or 0.0) for row in rows], dtype=np.float64)
    recall_values = np.asarray([float(row.get("recall") or 0.0) for row in rows], dtype=np.float64)
    total_support = max(float(supports.sum()), 1.0)
    return {
        "macro_precision": float(precision_values.mean()),
        "macro_recall": float(recall_values.mean()),
        "macro_f1": float(f1_values.mean()),
        "min_f1": float(f1_values.min()),
        "weighted_f1": float((f1_values * supports).sum() / total_support),
        "support": int(supports.sum()),
        "classes": [str(row.get("label") or labels[int(row.get("class_index") or 0)]) for row in rows],
    }


def select_multiclass_class_bias(
    probabilities: np.ndarray,
    *,
    true_labels: np.ndarray,
    samples: list[dict],
    labels: list[str],
    multipliers: list[float] | None,
    target_metric: str,
) -> dict:
    candidates = build_class_bias_candidates(labels, multipliers=multipliers)
    best: dict | None = None
    for name, bias in candidates:
        biased = apply_class_bias(probabilities, bias, labels=labels)
        metrics = metrics_from_probabilities(
            biased,
            true_labels=true_labels,
            samples=samples,
            labels=labels,
        )
        task_metrics = build_multiclass_task_metrics(metrics, labels=labels)
        score = score_multiclass_bias(metrics, task_metrics, target_metric=target_metric)
        candidate = {
            "class_bias_name": name,
            "class_bias": bias,
            "probabilities": biased,
            "metrics": metrics,
            "task_metrics": task_metrics,
            "score": score,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best or {
        "class_bias_name": "none",
        "class_bias": {},
        "probabilities": probabilities,
        "metrics": {},
        "task_metrics": {},
        "score": 0.0,
    }


def build_class_bias_candidates(labels: list[str], *, multipliers: list[float] | None) -> list[tuple[str, dict[str, float]]]:
    values = multipliers or [0.65, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.15, 1.25, 1.35, 1.4, 1.6, 1.75, 1.9, 2.1, 2.3]
    candidates: list[tuple[str, dict[str, float]]] = [("none", {})]
    for label in labels:
        for value in values:
            if abs(float(value) - 1.0) < 1e-9:
                continue
            candidates.append((f"{label}_{value:g}", {label: float(value)}))
    for up_label in labels:
        for down_label in labels:
            if up_label == down_label:
                continue
            for up_value, down_value in (
                (1.15, 0.95),
                (1.25, 0.9),
                (1.35, 0.85),
                (1.4, 0.85),
                (1.6, 0.8),
                (1.75, 0.75),
                (1.9, 0.75),
                (2.1, 0.7),
            ):
                candidates.append(
                    (
                        f"{up_label}_{up_value:g}_{down_label}_{down_value:g}",
                        {up_label: up_value, down_label: down_value},
                    )
                )
    if "abduction" in labels:
        for value in (1.5, 1.8, 2.1, 2.5, 3.0):
            candidates.append((f"abduction_{value:g}", {"abduction": value}))
    return candidates


def score_multiclass_bias(metrics: dict, task_metrics: dict, *, target_metric: str) -> float:
    macro_f1 = float(task_metrics.get("macro_f1") or metrics.get("macro_f1_supported") or metrics.get("macro_f1") or 0.0)
    min_f1 = float(task_metrics.get("min_f1") or 0.0)
    balanced = float(metrics.get("balanced_accuracy") or 0.0)
    accuracy = float(metrics.get("accuracy") or 0.0)
    target = str(target_metric or "").strip().lower()
    if target == "accuracy":
        return (0.55 * accuracy) + (0.25 * macro_f1) + (0.15 * balanced) + (0.05 * min_f1)
    return (0.55 * macro_f1) + (0.25 * balanced) + (0.15 * min_f1) + (0.05 * accuracy)


def _map_rows(rows: list[dict], *, row_mapper) -> list[dict]:
    mapped: list[dict] = []
    for row in rows:
        next_row = row_mapper(row)
        if next_row is not None:
            mapped.append(next_row)
    return mapped


def _map_detection_row(row: dict, *, normal_label: str) -> dict:
    original_label = str(row.get("target_label") or "").strip()
    mapped_label = "normal" if original_label == normal_label else "abnormal"
    next_row = dict(row)
    next_row["source_target_label"] = original_label
    next_row["target_label"] = mapped_label
    next_row["label"] = mapped_label
    next_row["label_idx"] = 0 if mapped_label == "normal" else 1
    return next_row


def _map_classification_row(row: dict, *, labels: list[str]) -> dict | None:
    original_label = str(row.get("target_label") or "").strip()
    if original_label not in labels:
        return None
    next_row = dict(row)
    next_row["source_target_label"] = original_label
    next_row["target_label"] = original_label
    next_row["label"] = original_label
    next_row["label_idx"] = labels.index(original_label)
    return next_row


def _resolve_model_names(task_config: dict) -> list[str]:
    configured = task_config.get("feature_models")
    if isinstance(configured, list):
        names = [str(name).strip() for name in configured if str(name).strip()]
    else:
        names = list(DEFAULT_SPECIALIZED_MODELS)
    return names or list(DEFAULT_SPECIALIZED_MODELS)


def _resolve_bias_multipliers(task_config: dict) -> list[float]:
    values = task_config.get("classification_class_bias_multipliers")
    if not isinstance(values, list):
        return [0.65, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.9, 2.3]
    parsed = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    return parsed or [0.65, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.9, 2.3]


def _has_multiple_labels(rows: list[dict]) -> bool:
    return len({int(row.get("label_idx", -1)) for row in rows}) >= 2


def _score_result(metrics: dict, *, target_metric: str) -> float:
    if target_metric == "abnormal_f1":
        detection = metrics.get("detection") if isinstance(metrics.get("detection"), dict) else {}
        return float(detection.get("f1") or metrics.get("macro_f1_supported") or 0.0)
    if target_metric == "detection_balanced":
        detection = metrics.get("detection") if isinstance(metrics.get("detection"), dict) else {}
        return _score_detection_metrics(detection, target_metric=target_metric)
    if target_metric == "abnormal_recall":
        detection = metrics.get("detection") if isinstance(metrics.get("detection"), dict) else {}
        return float(detection.get("recall") or 0.0)
    return float(metrics.get(target_metric) or metrics.get("macro_f1_supported") or metrics.get("macro_f1") or 0.0)


def _score_detection_metrics(metrics: dict, *, target_metric: str) -> float:
    recall = float(metrics.get("recall") or 0.0)
    specificity = float(metrics.get("specificity") or 0.0)
    f1 = float(metrics.get("f1") or 0.0)
    balanced_accuracy = float(metrics.get("balanced_accuracy") or 0.0)
    if target_metric == "abnormal_recall":
        return recall
    if target_metric == "abnormal_f1":
        return f1
    if target_metric == "detection_practical":
        precision = float(metrics.get("precision") or 0.0)
        false_alarm = float(metrics.get("false_alarm_rate") or 0.0)
        false_alarm_penalty = max(0.0, false_alarm - 0.45)
        return (
            (0.35 * f1)
            + (0.25 * specificity)
            + (0.20 * precision)
            + (0.15 * recall)
            + (0.05 * balanced_accuracy)
            - (0.35 * false_alarm_penalty)
        )
    return (0.45 * recall) + (0.35 * specificity) + (0.20 * balanced_accuracy)


def _predict_full_probabilities(model, features: np.ndarray, *, num_classes: int) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = getattr(model, "classes_", None)
    if classes is None or len(classes) == num_classes:
        return probabilities
    aligned = np.zeros((probabilities.shape[0], num_classes), dtype=np.float64)
    for source_index, class_index in enumerate(classes):
        try:
            target_index = int(class_index)
        except (TypeError, ValueError):
            continue
        if 0 <= target_index < num_classes:
            aligned[:, target_index] = probabilities[:, source_index]
    row_sums = aligned.sum(axis=1, keepdims=True)
    return np.divide(aligned, np.maximum(row_sums, 1e-12))


def _compact_candidate(result: dict | None) -> dict:
    if not result:
        return {}
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    holdout = result.get("holdout_test") if isinstance(result.get("holdout_test"), dict) else {}
    return {
        "model": result.get("model"),
        "score": result.get("score"),
        "threshold": result.get("threshold"),
        "class_bias": result.get("class_bias") or {},
        "class_bias_name": result.get("class_bias_name") or "none",
        "validation": validation,
        "holdout_test": holdout,
    }


def _utc_now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat()
