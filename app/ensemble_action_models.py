from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from action_model import (
    PoseSequenceDataset,
    TemporalPoseClassifier,
    _build_validation_error_analysis,
    _compute_f1,
    _summarize_classification_metrics,
)
from action_training_pipeline import (
    get_continual_config,
    get_target_labels,
    load_config,
    load_training_manifests,
    resolve_paths,
)
from auto_tune_action_training import objective_score_for_metric, safe_float
from reporting import read_json, write_json_atomic


DEFAULT_TOP_K = 12
DEFAULT_MAX_SIZE = 5
MAX_MISCLASSIFIED_EXAMPLES = 200
DEFAULT_TEMPERATURE_VALUES = (0.75, 0.9, 1.0, 1.1, 1.25, 1.5)
DEFAULT_PROBABILITY_POWER_VALUES = (0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)
DEFAULT_CLASS_BIAS_MULTIPLIERS = (0.7, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.3)
PROMOTED_ENSEMBLE_FILES = (
    "best_action_ensemble.json",
    "ensemble_metrics.json",
    "ensemble_summary.json",
)


@dataclass(frozen=True)
class EnsembleCandidate:
    name: str
    model_path: Path
    metrics_path: Path
    output_dir: Path
    score: float
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    loss: float
    trial_number: int | None = None


@dataclass(frozen=True)
class LoadedMember:
    candidate: EnsembleCandidate
    probabilities: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepared pose manifest와 auto-tune trial checkpoint를 재사용해 validation 앙상블을 탐색합니다. "
            "새 전처리는 수행하지 않습니다."
        )
    )
    parser.add_argument("--config", default="configs/action_training.aihub_shell.example.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument(
        "--target-metric",
        choices=("balanced", "accuracy", "macro_f1"),
        default=None,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--no-promote", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    paths = resolve_paths(config, config_path.parent)
    labels = get_target_labels(config)
    continual_config = get_continual_config(config)
    training_manifests = load_training_manifests(
        config,
        paths,
        continual_enabled=continual_config["enabled"],
    )
    auto_tune_config = config.get("auto_tune") if isinstance(config.get("auto_tune"), dict) else {}
    target_metric = str(args.target_metric or auto_tune_config.get("target_metric") or "accuracy")
    output_dir = resolve_ensemble_output_dir(
        paths,
        config_path.parent,
        args.output_dir,
    )
    promote = bool(args.promote or (not args.no_promote and auto_tune_config.get("promote_ensemble", True)))
    result = run_ensemble_search(
        config=config,
        paths=paths,
        training_manifests=training_manifests,
        labels=labels,
        output_dir=output_dir,
        target_metric=target_metric,
        top_k=int(args.top_k or auto_tune_config.get("ensemble_top_k", DEFAULT_TOP_K)),
        max_size=int(args.max_size or auto_tune_config.get("ensemble_max_size", DEFAULT_MAX_SIZE)),
        device=str(args.device or config.get("training", {}).get("device") or "cuda"),
        promote=promote,
    )
    print(
        "[ensemble] best "
        f"accuracy={result['best_result']['metrics']['final_validation']['accuracy']:.4f} "
        f"macro_f1={result['best_result']['metrics']['final_validation']['macro_f1']:.4f} "
        f"members={len(result['best_result']['members'])} "
        f"promoted={result['promoted']}"
    )
    print(f"[ensemble] summary: {result['summary_path']}")


def run_ensemble_search(
    *,
    config: dict,
    paths: dict,
    training_manifests: dict[str, Path],
    labels: list[str],
    output_dir: Path,
    target_metric: str,
    top_k: int = DEFAULT_TOP_K,
    max_size: int = DEFAULT_MAX_SIZE,
    device: str = "cuda",
    promote: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = collect_ensemble_candidates(
        paths["artifacts_dir"],
        labels=labels,
        target_metric=target_metric,
        top_k=top_k,
    )
    if not candidates:
        raise RuntimeError("앙상블 후보 checkpoint를 찾지 못했습니다.")

    resolved_device = resolve_device(device)
    val_dataset = PoseSequenceDataset(training_manifests["val"], cache_size=0)
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.get("training", {}).get("eval_batch_size", 96) or 96),
        shuffle=False,
        num_workers=0,
    )
    val_labels = np.asarray([int(sample["label_idx"]) for sample in val_dataset.samples], dtype=np.int64)
    loaded_members = [
        load_member_probabilities(candidate, val_loader, labels=labels, device=resolved_device)
        for candidate in candidates
    ]
    auto_tune_config = config.get("auto_tune") if isinstance(config.get("auto_tune"), dict) else {}

    best_result = search_best_ensemble(
        loaded_members,
        true_labels=val_labels,
        samples=val_dataset.samples,
        labels=labels,
        target_metric=target_metric,
        max_size=max_size,
        temperature_values=normalize_float_grid(
            auto_tune_config.get("ensemble_temperature_values"),
            default=DEFAULT_TEMPERATURE_VALUES,
        ),
        probability_power_values=normalize_float_grid(
            auto_tune_config.get("ensemble_probability_power_values"),
            default=DEFAULT_PROBABILITY_POWER_VALUES,
        ),
        bias_multipliers=normalize_float_grid(
            auto_tune_config.get("ensemble_class_bias_multipliers"),
            default=DEFAULT_CLASS_BIAS_MULTIPLIERS,
        ),
    )
    if best_result is None:
        raise RuntimeError("앙상블 후보 평가에 실패했습니다.")

    test_metrics = None
    if "test" in training_manifests and Path(training_manifests["test"]).exists():
        test_metrics = evaluate_saved_ensemble_on_manifest(
            best_result,
            manifest_path=training_manifests["test"],
            labels=labels,
            config=config,
            device=resolved_device,
        )
        best_result["test_validation"] = test_metrics

    write_ensemble_artifacts(
        output_dir,
        best_result=best_result,
        candidates=candidates,
        target_metric=target_metric,
        test_metrics=test_metrics,
    )
    summary_path = output_dir / "ensemble_summary.json"
    summary = read_json(summary_path) or {}
    promoted = bool(promote and should_promote_ensemble(best_result, paths["artifacts_dir"], target_metric=target_metric))
    summary["promoted"] = promoted
    summary["updated_at"] = utc_now_iso()
    write_json_atomic(summary_path, summary)
    if promoted:
        promote_ensemble_artifacts(output_dir, paths["artifacts_dir"])
    return {
        "summary_path": str(summary_path),
        "output_dir": str(output_dir),
        "promoted": promoted,
        "best_result": best_result,
    }


def collect_ensemble_candidates(
    artifacts_dir: Path,
    *,
    labels: list[str],
    target_metric: str,
    top_k: int,
) -> list[EnsembleCandidate]:
    auto_tune_dir = artifacts_dir / "auto_tune"
    candidates: dict[str, EnsembleCandidate] = {}
    for metrics_path in auto_tune_dir.glob("runs/*/metrics.json"):
        candidate = candidate_from_metrics_path(metrics_path, labels=labels, target_metric=target_metric)
        if candidate is not None:
            candidates[str(candidate.model_path.resolve()).lower()] = candidate
    main_candidate = candidate_from_metrics_path(
        artifacts_dir / "metrics.json",
        labels=labels,
        target_metric=target_metric,
        fallback_name="current_main",
    )
    if main_candidate is not None:
        candidates[str(main_candidate.model_path.resolve()).lower()] = main_candidate

    sorted_candidates = sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate.score,
            candidate.accuracy,
            candidate.macro_f1,
            candidate.balanced_accuracy,
        ),
        reverse=True,
    )
    return sorted_candidates[: max(int(top_k), 1)]


def candidate_from_metrics_path(
    metrics_path: Path,
    *,
    labels: list[str],
    target_metric: str,
    fallback_name: str | None = None,
) -> EnsembleCandidate | None:
    if not metrics_path.exists():
        return None
    model_path = metrics_path.parent / "best_action_model.pt"
    if not model_path.exists():
        return None
    metrics = read_json(metrics_path)
    if not isinstance(metrics, dict):
        return None
    model_type = str(metrics.get("model_type") or "").lower()
    if model_type in {"ensemble", "hybrid_ensemble"}:
        return None
    metrics_labels = metrics.get("labels") if isinstance(metrics.get("labels"), list) else labels
    if list(metrics_labels) != list(labels):
        return None
    final_validation = metrics.get("final_validation") if isinstance(metrics.get("final_validation"), dict) else {}
    accuracy = safe_float(final_validation.get("accuracy")) or 0.0
    macro_f1 = safe_float(final_validation.get("macro_f1")) or 0.0
    balanced_accuracy = safe_float(final_validation.get("balanced_accuracy")) or 0.0
    loss = safe_float(final_validation.get("cross_entropy_loss")) or safe_float(final_validation.get("loss")) or 0.0
    trial = metrics.get("auto_tune_trial") if isinstance(metrics.get("auto_tune_trial"), dict) else {}
    name = str(trial.get("name") or fallback_name or metrics_path.parent.name)
    trial_number = None
    if "_trial_" in metrics_path.parent.name:
        try:
            trial_number = int(metrics_path.parent.name.split("_trial_", 1)[1][:2])
        except ValueError:
            trial_number = None
    return EnsembleCandidate(
        name=name,
        model_path=model_path,
        metrics_path=metrics_path,
        output_dir=metrics_path.parent,
        score=objective_score_for_metric(metrics, target_metric=target_metric),
        accuracy=accuracy,
        macro_f1=macro_f1,
        balanced_accuracy=balanced_accuracy,
        loss=loss,
        trial_number=trial_number,
    )


def load_member_probabilities(
    candidate: EnsembleCandidate,
    loader: DataLoader,
    *,
    labels: list[str],
    device: str,
) -> LoadedMember:
    model = load_checkpoint_model(candidate.model_path, labels=labels, device=device)
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for pose, mask, _batch_labels in loader:
            pose = pose.to(device, non_blocking=False)
            mask = mask.to(device, non_blocking=False)
            logits = model(pose, mask).float()
            probabilities.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return LoadedMember(candidate=candidate, probabilities=np.concatenate(probabilities, axis=0))


def load_checkpoint_model(model_path: Path, *, labels: list[str], device: str) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    checkpoint_labels = list(checkpoint.get("labels", []))
    if checkpoint_labels and checkpoint_labels != list(labels):
        raise RuntimeError(f"checkpoint label 순서가 다릅니다: {model_path}")
    hidden_dim = int(checkpoint.get("hidden_dim", 128))
    state_dict = checkpoint.get("model_state_dict", {}) or {}
    classifier_weight = state_dict.get("classifier.1.weight")
    classifier_input_dim = int(classifier_weight.shape[1]) if classifier_weight is not None else hidden_dim * 2
    if classifier_input_dim == hidden_dim * 4:
        from mean_max_pooling import MeanMaxTemporalPoseClassifier

        model_class = MeanMaxTemporalPoseClassifier
    else:
        model_class = TemporalPoseClassifier
    model = model_class(
        num_joints=int(checkpoint.get("num_joints", 17)),
        input_dim=int(checkpoint.get("input_dim", 3)),
        hidden_dim=hidden_dim,
        num_layers=int(checkpoint.get("num_layers", 2)),
        num_classes=len(labels),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model


def search_best_ensemble(
    members: list[LoadedMember],
    *,
    true_labels: np.ndarray,
    samples: list[dict],
    labels: list[str],
    target_metric: str,
    max_size: int,
    temperature_values: list[float] | None = None,
    probability_power_values: list[float] | None = None,
    bias_multipliers: list[float] | None = None,
) -> dict | None:
    if not members:
        return None
    upper_size = max(1, min(int(max_size), len(members)))
    best: dict | None = None
    for size in range(1, upper_size + 1):
        for combo_indices in itertools.combinations(range(len(members)), size):
            combo_members = [members[index] for index in combo_indices]
            for weight_mode in weight_modes_for_size(size):
                weights = build_member_weights(combo_members, mode=weight_mode)
                base_probabilities = weighted_average_probabilities(combo_members, weights)
                for transform_name, transformed_probabilities in probability_transform_variants(
                    base_probabilities,
                    temperature_values=temperature_values,
                    probability_power_values=probability_power_values,
                ):
                    for bias_name, bias in class_bias_variants(labels, multipliers=bias_multipliers):
                        probabilities = apply_class_bias(transformed_probabilities, bias, labels=labels)
                        metrics = metrics_from_probabilities(
                            probabilities,
                            true_labels=true_labels,
                            samples=samples,
                            labels=labels,
                        )
                        payload = build_ensemble_result_payload(
                            combo_members,
                            weights=weights,
                            weight_mode=weight_mode,
                            probability_transform=transform_name,
                            class_bias=bias,
                            class_bias_name=bias_name,
                            metrics=metrics,
                            target_metric=target_metric,
                            labels=labels,
                        )
                        if best is None or ensemble_result_sort_key(payload) > ensemble_result_sort_key(best):
                            best = payload
    return best


def weight_modes_for_size(size: int) -> list[str]:
    if size <= 1:
        return ["single"]
    modes = [
        "uniform",
        "score",
        "score_squared",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "loss_inverse",
        "rank_decay_0.75",
        "rank_decay_0.6",
        "best_heavy",
    ]
    if size == 2:
        modes.extend(f"alpha_{value:g}" for value in (0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85))
    return modes


def build_member_weights(members: list[LoadedMember], *, mode: str) -> np.ndarray:
    if mode in {"single", "uniform"}:
        return np.full(len(members), 1.0 / max(len(members), 1), dtype=np.float64)
    if mode.startswith("alpha_") and len(members) == 2:
        try:
            alpha = float(mode.split("_", 1)[1])
        except ValueError:
            alpha = 0.5
        alpha = min(max(alpha, 0.0), 1.0)
        return np.asarray([alpha, 1.0 - alpha], dtype=np.float64)
    if mode.startswith("rank_decay_"):
        try:
            decay = float(mode.rsplit("_", 1)[1])
        except ValueError:
            decay = 0.75
        raw = np.asarray([decay**index for index in range(len(members))], dtype=np.float64)
        return raw / raw.sum()
    if mode == "best_heavy":
        raw = np.full(len(members), 0.5 / max(len(members) - 1, 1), dtype=np.float64)
        raw[0] = 0.5 if len(members) > 1 else 1.0
        return raw / raw.sum()
    raw_values = []
    for member in members:
        candidate = member.candidate
        if mode == "accuracy":
            raw_values.append(candidate.accuracy)
        elif mode == "macro_f1":
            raw_values.append(candidate.macro_f1)
        elif mode == "balanced_accuracy":
            raw_values.append(candidate.balanced_accuracy)
        elif mode == "loss_inverse":
            raw_values.append(1.0 / max(candidate.loss, 1e-6))
        elif mode == "score_squared":
            raw_values.append(max(candidate.score, 1e-6) ** 2)
        else:
            raw_values.append(candidate.score)
    raw = np.asarray(raw_values, dtype=np.float64)
    raw = np.maximum(raw - raw.min() + 1e-4, 1e-4)
    return raw / raw.sum()


def weighted_average_probabilities(members: list[LoadedMember], weights: np.ndarray) -> np.ndarray:
    stacked = np.stack([member.probabilities for member in members], axis=0)
    averaged = np.tensordot(weights.astype(np.float64), stacked.astype(np.float64), axes=(0, 0))
    return normalize_probabilities(averaged)


def class_bias_variants(
    labels: list[str],
    *,
    multipliers: list[float] | None = None,
) -> list[tuple[str, dict[str, float]]]:
    variants: list[tuple[str, dict[str, float]]] = [("none", {})]
    bias_values = multipliers or list(DEFAULT_CLASS_BIAS_MULTIPLIERS)
    for label in labels:
        for multiplier in bias_values:
            variants.append((f"{label}_{multiplier:g}", {label: multiplier}))
    pair_values = [value for value in bias_values if value not in {0.95, 1.05}]
    for up_label, down_label in itertools.permutations(labels, 2):
        for up_multiplier in pair_values:
            if up_multiplier <= 1.0:
                continue
            for down_multiplier in pair_values:
                if down_multiplier >= 1.0:
                    continue
                variants.append(
                    (
                        f"{up_label}_{up_multiplier:g}_{down_label}_{down_multiplier:g}",
                        {up_label: up_multiplier, down_label: down_multiplier},
                    )
                )
    return variants


def probability_transform_variants(
    probabilities: np.ndarray,
    *,
    temperature_values: list[float] | None,
    probability_power_values: list[float] | None,
) -> list[tuple[str, np.ndarray]]:
    variants: dict[str, np.ndarray] = {"raw": probabilities}
    for temperature in temperature_values or list(DEFAULT_TEMPERATURE_VALUES):
        if temperature <= 0 or abs(temperature - 1.0) < 1e-9:
            continue
        variants[f"temperature_{temperature:g}"] = apply_probability_power(probabilities, 1.0 / temperature)
    for power in probability_power_values or list(DEFAULT_PROBABILITY_POWER_VALUES):
        if power <= 0 or abs(power - 1.0) < 1e-9:
            continue
        variants[f"power_{power:g}"] = apply_probability_power(probabilities, power)
    return list(variants.items())


def apply_named_probability_transform(probabilities: np.ndarray, name: str | None) -> np.ndarray:
    normalized_name = str(name or "raw")
    if normalized_name.startswith("temperature_"):
        try:
            temperature = float(normalized_name.split("_", 1)[1])
        except ValueError:
            return probabilities
        if temperature > 0:
            return apply_probability_power(probabilities, 1.0 / temperature)
    if normalized_name.startswith("power_"):
        try:
            power = float(normalized_name.split("_", 1)[1])
        except ValueError:
            return probabilities
        if power > 0:
            return apply_probability_power(probabilities, power)
    return probabilities


def apply_probability_power(probabilities: np.ndarray, power: float) -> np.ndarray:
    adjusted = np.power(np.clip(probabilities, 1e-12, 1.0), float(power))
    return normalize_probabilities(adjusted)


def apply_class_bias(probabilities: np.ndarray, bias: dict[str, float], *, labels: list[str]) -> np.ndarray:
    if not bias:
        return probabilities
    adjusted = probabilities.copy()
    for class_index, label in enumerate(labels[: adjusted.shape[1]]):
        if label in bias:
            adjusted[:, class_index] *= float(bias[label])
    return normalize_probabilities(adjusted)


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    return probabilities / probabilities.sum(axis=1, keepdims=True).clip(min=1e-12)


def metrics_from_probabilities(
    probabilities: np.ndarray,
    *,
    true_labels: np.ndarray,
    samples: list[dict],
    labels: list[str],
) -> dict:
    preds = probabilities.argmax(axis=1)
    num_classes = len(labels)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    misclassified: list[dict] = []
    true_confidence_total = 0.0
    pred_confidence_total = 0.0
    ce_loss_total = 0.0
    for sample_index, (true_label, pred_label) in enumerate(zip(true_labels, preds, strict=False)):
        true_index = int(true_label)
        pred_index = int(pred_label)
        confusion[true_index, pred_index] += 1
        true_probability = float(probabilities[sample_index, true_index])
        true_confidence_total += true_probability
        pred_confidence_total += float(probabilities[sample_index, pred_index])
        ce_loss_total += -math.log(max(true_probability, 1e-12))
        if true_index == pred_index or len(misclassified) >= MAX_MISCLASSIFIED_EXAMPLES:
            continue
        sample = samples[sample_index] if sample_index < len(samples) else {}
        misclassified.append(
            {
                "sample_index": sample_index,
                "item_id": sample.get("item_id"),
                "target_label": sample.get("target_label") or labels[true_index],
                "predicted_label": labels[pred_index],
                "true_index": true_index,
                "predicted_index": pred_index,
                "pose_path": sample.get("pose_path"),
                "video_path": sample.get("video_path"),
            }
        )
    total = max(int(true_labels.shape[0]), 1)
    accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))
    macro_f1, per_class = _compute_f1(confusion, labels=labels)
    classification_summary = _summarize_classification_metrics(per_class)
    return {
        "loss": ce_loss_total / total,
        "cross_entropy_loss": ce_loss_total / total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        **classification_summary,
        "mean_true_confidence": true_confidence_total / total,
        "mean_pred_confidence": pred_confidence_total / total,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "misclassified_examples": misclassified,
    }


def build_ensemble_result_payload(
    members: list[LoadedMember],
    *,
    weights: np.ndarray,
    weight_mode: str,
    probability_transform: str,
    class_bias: dict[str, float],
    class_bias_name: str,
    metrics: dict,
    target_metric: str,
    labels: list[str],
) -> dict:
    metric_payload = {
        "final_validation": metrics,
        "validation_error_analysis": _build_validation_error_analysis(metrics, labels=labels).get("summary", {}),
    }
    score = objective_score_for_metric(metric_payload, target_metric=target_metric)
    return {
        "score": score,
        "target_metric": target_metric,
        "weight_mode": weight_mode,
        "probability_transform": probability_transform,
        "class_bias_name": class_bias_name,
        "class_bias": class_bias,
        "weights": [round(float(value), 8) for value in weights.tolist()],
        "members": [candidate_to_payload(member.candidate) for member in members],
        "metrics": {
            "final_validation": metrics,
        },
    }


def ensemble_result_sort_key(result: dict) -> tuple:
    final_validation = result.get("metrics", {}).get("final_validation", {})
    return (
        safe_float(result.get("score")) or 0.0,
        safe_float(final_validation.get("accuracy")) or 0.0,
        safe_float(final_validation.get("macro_f1")) or 0.0,
        safe_float(final_validation.get("balanced_accuracy")) or 0.0,
        -len(result.get("members") or []),
    )


def candidate_to_payload(candidate: EnsembleCandidate) -> dict:
    return {
        "name": candidate.name,
        "trial_number": candidate.trial_number,
        "model_path": str(candidate.model_path),
        "metrics_path": str(candidate.metrics_path),
        "output_dir": str(candidate.output_dir),
        "score": round(candidate.score, 6),
        "accuracy": candidate.accuracy,
        "macro_f1": candidate.macro_f1,
        "balanced_accuracy": candidate.balanced_accuracy,
        "loss": candidate.loss,
    }


def write_ensemble_artifacts(
    output_dir: Path,
    *,
    best_result: dict,
    candidates: list[EnsembleCandidate],
    target_metric: str,
    test_metrics: dict | None,
) -> None:
    final_validation = best_result["metrics"]["final_validation"]
    labels = [row["label"] for row in final_validation.get("per_class", [])]
    error_analysis = _build_validation_error_analysis(final_validation, labels=labels)
    metrics_payload = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "model_type": "ensemble",
        "target_metric": target_metric,
        "history": [],
        "final_validation": final_validation,
        "validation_error_analysis": error_analysis.get("summary", {}),
        "validation_error_analysis_path": str(output_dir / "validation_error_analysis.json"),
        "false_negative_examples_path": str(output_dir / "false_negative_examples.json"),
        "confusion_pair_examples_path": str(output_dir / "confusion_pair_examples.json"),
        "labels": labels,
        "best_val_macro_f1": round(float(final_validation.get("macro_f1", 0.0)), 6),
        "best_epoch": None,
        "ensemble": {
            "score": best_result["score"],
            "weight_mode": best_result["weight_mode"],
            "probability_transform": best_result.get("probability_transform", "raw"),
            "class_bias": best_result["class_bias"],
            "class_bias_name": best_result["class_bias_name"],
            "weights": best_result["weights"],
            "members": best_result["members"],
        },
    }
    if test_metrics is not None:
        metrics_payload["holdout_test"] = test_metrics

    ensemble_payload = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "target_metric": target_metric,
        "labels": labels,
        "best_result": {
            key: value
            for key, value in best_result.items()
            if key not in {"metrics", "test_validation"}
        },
        "metrics_path": str(output_dir / "ensemble_metrics.json"),
    }
    summary_payload = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "target_metric": target_metric,
        "candidate_count": len(candidates),
        "candidates": [candidate_to_payload(candidate) for candidate in candidates],
        "best_result": {
            **ensemble_payload["best_result"],
            "metrics": {
                "accuracy": final_validation.get("accuracy"),
                "macro_f1": final_validation.get("macro_f1"),
                "balanced_accuracy": final_validation.get("balanced_accuracy"),
                "confusion_matrix": final_validation.get("confusion_matrix"),
            },
            "holdout_test": test_metrics,
        },
        "promoted": False,
    }
    write_json_atomic(output_dir / "ensemble_metrics.json", metrics_payload)
    write_json_atomic(output_dir / "best_action_ensemble.json", ensemble_payload)
    write_json_atomic(output_dir / "validation_error_analysis.json", error_analysis)
    write_json_atomic(output_dir / "false_negative_examples.json", error_analysis.get("false_negative_examples", {}))
    write_json_atomic(output_dir / "confusion_pair_examples.json", error_analysis.get("confusion_pair_examples", []))
    write_json_atomic(output_dir / "ensemble_summary.json", summary_payload)


def evaluate_saved_ensemble_on_manifest(
    best_result: dict,
    *,
    manifest_path: Path,
    labels: list[str],
    config: dict,
    device: str,
) -> dict:
    dataset = PoseSequenceDataset(manifest_path, cache_size=0)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("eval_batch_size", 96) or 96),
        shuffle=False,
        num_workers=0,
    )
    true_labels = np.asarray([int(sample["label_idx"]) for sample in dataset.samples], dtype=np.int64)
    members = []
    for member_payload in best_result["members"]:
        candidate = EnsembleCandidate(
            name=str(member_payload["name"]),
            model_path=Path(member_payload["model_path"]),
            metrics_path=Path(member_payload["metrics_path"]),
            output_dir=Path(member_payload["output_dir"]),
            score=float(member_payload.get("score", 0.0) or 0.0),
            accuracy=float(member_payload.get("accuracy", 0.0) or 0.0),
            macro_f1=float(member_payload.get("macro_f1", 0.0) or 0.0),
            balanced_accuracy=float(member_payload.get("balanced_accuracy", 0.0) or 0.0),
            loss=float(member_payload.get("loss", 0.0) or 0.0),
            trial_number=member_payload.get("trial_number"),
        )
        members.append(load_member_probabilities(candidate, loader, labels=labels, device=device))
    probabilities = weighted_average_probabilities(members, np.asarray(best_result["weights"], dtype=np.float64))
    probabilities = apply_named_probability_transform(probabilities, best_result.get("probability_transform"))
    probabilities = apply_class_bias(probabilities, best_result.get("class_bias") or {}, labels=labels)
    return metrics_from_probabilities(probabilities, true_labels=true_labels, samples=dataset.samples, labels=labels)


def should_promote_ensemble(best_result: dict, artifacts_dir: Path, *, target_metric: str) -> bool:
    current_metrics = read_json(artifacts_dir / "metrics.json")
    if not isinstance(current_metrics, dict):
        return True
    current_score = objective_score_for_metric(current_metrics, target_metric=target_metric)
    ensemble_score = safe_float(best_result.get("score")) or 0.0
    current_accuracy = safe_float(current_metrics.get("final_validation", {}).get("accuracy")) or 0.0
    ensemble_accuracy = safe_float(best_result.get("metrics", {}).get("final_validation", {}).get("accuracy")) or 0.0
    return (ensemble_score, ensemble_accuracy) > (current_score, current_accuracy)


def promote_ensemble_artifacts(output_dir: Path, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name in PROMOTED_ENSEMBLE_FILES:
        source = output_dir / name
        if source.exists():
            shutil.copy2(source, artifacts_dir / name)
    metrics_source = output_dir / "ensemble_metrics.json"
    if metrics_source.exists():
        shutil.copy2(metrics_source, artifacts_dir / "metrics.json")
    for name in (
        "validation_error_analysis.json",
        "false_negative_examples.json",
        "confusion_pair_examples.json",
    ):
        source = output_dir / name
        if source.exists():
            shutil.copy2(source, artifacts_dir / name)


def resolve_ensemble_output_dir(paths: dict, base_dir: Path, configured: str | None) -> Path:
    if configured:
        candidate = Path(str(configured)).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    run_id = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = paths["artifacts_dir"] / "auto_tune" / "ensembles" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_device(device: str) -> str:
    requested = str(device or "cuda").strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def normalize_float_grid(values, *, default: tuple[float, ...]) -> list[float]:
    if not isinstance(values, list):
        return list(default)
    parsed: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            parsed.append(number)
    unique = sorted({round(value, 8) for value in parsed})
    return unique or list(default)


def utc_now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat()


if __name__ == "__main__":
    main()
