from __future__ import annotations

import argparse
import itertools
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from action_model import (
    MAX_DUPLICATE_POSE_LABEL_SAMPLES,
    PoseSequenceDataset,
    _build_validation_error_analysis,
    _filter_conflicting_pose_label_samples,
    _limit_duplicate_pose_label_samples,
)
from action_training_pipeline import (
    get_continual_config,
    get_target_labels,
    load_config,
    load_training_manifests,
    resolve_paths,
)
from ensemble_action_models import (
    EnsembleCandidate,
    apply_class_bias,
    apply_named_probability_transform,
    load_member_probabilities,
    metrics_from_probabilities,
    normalize_probabilities,
    probability_transform_variants,
    normalize_float_grid,
    weighted_average_probabilities,
)
from reporting import read_json, write_json_atomic

DEFAULT_HYBRID_FEATURE_WEIGHT_MAX = 0.75
DEFAULT_HYBRID_FEATURE_WEIGHT_STEP = 0.01
DEFAULT_HYBRID_TEMPERATURE_VALUES = (0.8, 0.9, 1.0, 1.1, 1.25)
DEFAULT_HYBRID_CLASS_BIAS_MULTIPLIERS = (0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3)
DEFAULT_HYBRID_AUTO_FEATURE_WEIGHTING = {
    "enabled": True,
    "min_rgb_ready_ratio": 0.8,
    "low_fallback_ratio": 0.05,
    "medium_fallback_ratio": 0.15,
    "high_fallback_ratio": 0.3,
    "low_feature_weight_max": 0.6,
    "medium_feature_weight_max": 0.7,
    "high_feature_weight_max": 0.85,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepared pose 파일에서 즉석 feature 모델을 학습하고 기존 neural ensemble과 확률을 섞어 "
            "validation 성능을 탐색합니다."
        )
    )
    parser.add_argument("--config", default="configs/action_training.aihub_shell.example.json")
    parser.add_argument("--output-dir", default=None)
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
    manifests = load_training_manifests(
        config,
        paths,
        continual_enabled=continual_config["enabled"],
    )
    output_dir = resolve_output_dir(paths, config_path.parent, args.output_dir)
    promote = bool(args.promote or (not args.no_promote and config.get("auto_tune", {}).get("promote_hybrid", True)))
    result = run_hybrid_search(
        config=config,
        paths=paths,
        manifests=manifests,
        labels=labels,
        output_dir=output_dir,
        promote=promote,
    )
    final_validation = result["best_result"]["metrics"]["final_validation"]
    print(
        "[hybrid] best "
        f"accuracy={final_validation['accuracy']:.4f} "
        f"macro_f1={final_validation['macro_f1']:.4f} "
        f"feature_model={result['best_result']['feature_model']} "
        f"feature_weight={result['best_result']['feature_weight']:.2f} "
        f"promoted={result['promoted']}"
    )
    print(f"[hybrid] summary: {result['summary_path']}")


def run_hybrid_search(
    *,
    config: dict,
    paths: dict,
    manifests: dict[str, Path],
    labels: list[str],
    output_dir: Path,
    promote: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_manifest_rows(manifests["train"])
    val_rows_all = read_manifest_rows(manifests["val"])
    test_rows_all = read_manifest_rows(manifests["test"]) if "test" in manifests and Path(manifests["test"]).exists() else []
    val_neural_rows = filter_pose_ready_rows(val_rows_all)
    test_neural_rows = filter_pose_ready_rows(test_rows_all)
    if len(val_neural_rows) != len(val_rows_all) or len(test_neural_rows) != len(test_rows_all):
        print(
            "[hybrid] RGB/I3D-only rows will use feature-only fallback "
            f"val={len(val_rows_all) - len(val_neural_rows)} "
            f"test={len(test_rows_all) - len(test_neural_rows)}"
        )
    print("[hybrid][progress] 5% building feature matrices", flush=True)
    x_train, y_train = build_feature_matrix(train_rows)
    x_val, y_val = build_feature_matrix(val_rows_all)
    x_test, y_test = build_feature_matrix(test_rows_all) if test_rows_all else (None, None)

    feature_models = build_feature_models()
    feature_results = []
    total_models = max(len(feature_models), 1)
    for model_index, (name, model) in enumerate(feature_models.items(), start=1):
        start_percent = 10 + int((model_index - 1) * 45 / total_models)
        print(
            f"[hybrid][progress] {start_percent}% feature model {model_index}/{total_models} start: {name}",
            flush=True,
        )
        model.fit(x_train, y_train)
        val_probabilities = model.predict_proba(x_val)
        test_probabilities = model.predict_proba(x_test) if x_test is not None else None
        feature_results.append(
            {
                "name": name,
                "model": model,
                "val_probabilities": val_probabilities,
                "test_probabilities": test_probabilities,
                "val_metrics": metrics_from_probabilities(
                    val_probabilities,
                    true_labels=y_val,
                    samples=val_rows_all,
                    labels=labels,
                ),
                "test_metrics": metrics_from_probabilities(
                    test_probabilities,
                    true_labels=y_test,
                    samples=test_rows_all,
                    labels=labels,
                )
                if test_probabilities is not None
                else None,
            }
        )
        done_percent = 10 + int(model_index * 45 / total_models)
        val_metrics = feature_results[-1]["val_metrics"]
        print(
            f"[hybrid][progress] {done_percent}% feature model {model_index}/{total_models} done: {name} "
            f"acc={val_metrics['accuracy']:.4f} f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )

    print("[hybrid][progress] 60% loading neural ensemble probabilities", flush=True)
    neural_val, neural_test, neural_members = load_neural_ensemble_probabilities(
        paths,
        manifests=manifests,
        labels=labels,
    )
    print("[hybrid][progress] 75% searching hybrid weights", flush=True)
    best_result = select_best_hybrid(
        feature_results,
        neural_val=neural_val,
        neural_test=neural_test,
        y_val=y_val,
        y_test=y_test,
        train_rows=train_rows,
        val_rows=val_rows_all,
        test_rows=test_rows_all,
        val_neural_rows=val_neural_rows,
        test_neural_rows=test_neural_rows,
        labels=labels,
        neural_members=neural_members,
        auto_tune_config=config.get("auto_tune") if isinstance(config.get("auto_tune"), dict) else {},
    )
    if best_result is None:
        raise RuntimeError("하이브리드 앙상블 후보를 찾지 못했습니다.")

    print("[hybrid][progress] 95% writing hybrid artifacts", flush=True)
    write_hybrid_artifacts(
        output_dir,
        best_result=best_result,
        feature_results=feature_results,
        labels=labels,
    )
    promoted = bool(promote and should_promote(best_result, paths["artifacts_dir"]))
    summary = read_json(output_dir / "hybrid_summary.json") or {}
    summary["promoted"] = promoted
    summary["updated_at"] = utc_now_iso()
    write_json_atomic(output_dir / "hybrid_summary.json", summary)
    if promoted:
        promote_hybrid(output_dir, paths["artifacts_dir"])
    print("[hybrid][progress] 100% hybrid search complete", flush=True)
    return {
        "summary_path": str(output_dir / "hybrid_summary.json"),
        "output_dir": str(output_dir),
        "promoted": promoted,
        "best_result": best_result,
    }


def build_feature_models() -> dict[str, Any]:
    return {
        "extra_trees_fast": ExtraTreesClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=43,
            max_features="sqrt",
            n_jobs=1,
        ),
        "extra_trees_leaf2_fast": ExtraTreesClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=52,
            max_features=0.7,
            min_samples_leaf=2,
            n_jobs=1,
        ),
        "extra_trees_accuracy": ExtraTreesClassifier(
            n_estimators=900,
            class_weight="balanced",
            random_state=73,
            max_features=0.85,
            min_samples_leaf=1,
            n_jobs=1,
        ),
        "random_forest_fast": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=42,
            max_features="sqrt",
            n_jobs=1,
        ),
        "random_forest_leaf2": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            random_state=77,
            max_features=0.65,
            min_samples_leaf=2,
            n_jobs=1,
        ),
        "hist_gradient": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=220,
            l2_regularization=0.02,
            random_state=42,
        ),
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1200, C=1.0, class_weight="balanced"),
        ),
        "gaussian_nb": GaussianNB(),
    }


def resolve_adaptive_feature_weight_max(
    auto_tune_config: dict,
    *,
    rows: list[dict],
    requested_max: float,
) -> tuple[float, dict]:
    auto_config = auto_tune_config.get("hybrid_auto_feature_weighting")
    if not isinstance(auto_config, dict):
        auto_config = DEFAULT_HYBRID_AUTO_FEATURE_WEIGHTING
    else:
        auto_config = {**DEFAULT_HYBRID_AUTO_FEATURE_WEIGHTING, **auto_config}

    requested_max = min(max(float(requested_max), 0.0), 1.0)
    total = len(rows)
    fallback_count = 0
    rgb_ready_count = 0
    rgb_path_exists_cache: dict[str, bool] = {}
    for row in rows:
        pose_fallback_mode = str(row.get("pose_fallback_mode") or "").strip()
        valid_frames = safe_float(row.get("valid_frames"))
        if row.get("rgb_only_fallback") or pose_fallback_mode or valid_frames <= 0:
            fallback_count += 1
        feature_path = str(row.get("rgb_feature_path") or "").strip()
        if feature_path and cached_path_exists(feature_path, rgb_path_exists_cache):
            rgb_ready_count += 1

    fallback_ratio = fallback_count / max(total, 1)
    rgb_ready_ratio = rgb_ready_count / max(total, 1)
    payload = {
        "enabled": bool(auto_config.get("enabled", True)),
        "requested_feature_weight_max": round(requested_max, 4),
        "resolved_feature_weight_max": round(requested_max, 4),
        "fallback_rows": fallback_count,
        "rgb_ready_rows": rgb_ready_count,
        "total_rows": total,
        "fallback_ratio": round(fallback_ratio, 6),
        "rgb_ready_ratio": round(rgb_ready_ratio, 6),
        "reason": "disabled",
    }
    if not payload["enabled"]:
        return requested_max, payload

    min_rgb_ready_ratio = float(auto_config.get("min_rgb_ready_ratio", 0.8) or 0.8)
    if rgb_ready_ratio < min_rgb_ready_ratio:
        payload["reason"] = "rgb_features_not_ready"
        return requested_max, payload

    resolved_max = requested_max
    reason = "base"
    thresholds = [
        (
            float(auto_config.get("high_fallback_ratio", 0.3) or 0.3),
            float(auto_config.get("high_feature_weight_max", 0.85) or 0.85),
            "high_pose_fallback_ratio",
        ),
        (
            float(auto_config.get("medium_fallback_ratio", 0.15) or 0.15),
            float(auto_config.get("medium_feature_weight_max", 0.7) or 0.7),
            "medium_pose_fallback_ratio",
        ),
        (
            float(auto_config.get("low_fallback_ratio", 0.05) or 0.05),
            float(auto_config.get("low_feature_weight_max", 0.6) or 0.6),
            "low_pose_fallback_ratio",
        ),
    ]
    for ratio_threshold, target_max, target_reason in thresholds:
        if fallback_ratio >= ratio_threshold:
            resolved_max = max(resolved_max, target_max)
            reason = target_reason
            break
    resolved_max = min(max(resolved_max, 0.0), 1.0)
    payload["resolved_feature_weight_max"] = round(resolved_max, 4)
    payload["reason"] = reason
    return resolved_max, payload


def cached_path_exists(path_text: str, cache: dict[str, bool]) -> bool:
    if path_text not in cache:
        cache[path_text] = Path(path_text).exists()
    return cache[path_text]


def select_best_hybrid(
    feature_results: list[dict],
    *,
    neural_val: np.ndarray,
    neural_test: np.ndarray | None,
    y_val: np.ndarray,
    y_test: np.ndarray | None,
    train_rows: list[dict],
    val_rows: list[dict],
    test_rows: list[dict],
    val_neural_rows: list[dict],
    test_neural_rows: list[dict],
    labels: list[str],
    neural_members: list[dict],
    auto_tune_config: dict,
) -> dict | None:
    best = None
    requested_feature_weight_max = float(
        auto_tune_config.get("hybrid_feature_weight_max", DEFAULT_HYBRID_FEATURE_WEIGHT_MAX)
        or DEFAULT_HYBRID_FEATURE_WEIGHT_MAX
    )
    feature_weight_step = float(
        auto_tune_config.get("hybrid_feature_weight_step", DEFAULT_HYBRID_FEATURE_WEIGHT_STEP)
        or DEFAULT_HYBRID_FEATURE_WEIGHT_STEP
    )
    feature_weight_max = min(max(requested_feature_weight_max, 0.0), 1.0)
    feature_weight_max, feature_weight_auto = resolve_adaptive_feature_weight_max(
        auto_tune_config,
        rows=[*train_rows, *val_rows],
        requested_max=feature_weight_max,
    )
    feature_weight_step = min(max(feature_weight_step, 0.005), 0.25)
    feature_weights = np.arange(0.0, feature_weight_max + feature_weight_step / 2.0, feature_weight_step)
    temperature_values = normalize_float_grid(
        auto_tune_config.get("hybrid_temperature_values"),
        default=DEFAULT_HYBRID_TEMPERATURE_VALUES,
    )
    bias_multipliers = normalize_float_grid(
        auto_tune_config.get("hybrid_class_bias_multipliers"),
        default=DEFAULT_HYBRID_CLASS_BIAS_MULTIPLIERS,
    )
    neural_variants = probability_transform_variants(
        neural_val,
        temperature_values=temperature_values,
        probability_power_values=[1.0],
    )
    bias_variants = hybrid_bias_variants(labels, multipliers=bias_multipliers)
    total_steps = max(
        len(feature_results)
        * len(neural_variants)
        * len(neural_variants)
        * max(len(feature_weights), 1)
        * len(bias_variants),
        1,
    )
    completed_steps = 0
    for feature_result in feature_results:
        feature_variants = probability_transform_variants(
            feature_result["val_probabilities"],
            temperature_values=temperature_values,
            probability_power_values=[1.0],
        )
        for neural_transform, neural_probabilities in neural_variants:
            for feature_transform, feature_probabilities in feature_variants:
                neural_probabilities_full = align_neural_probabilities(
                    all_rows=val_rows,
                    neural_rows=val_neural_rows,
                    neural_probabilities=neural_probabilities,
                    fallback_probabilities=feature_probabilities,
                )
                for feature_weight in feature_weights:
                    mixed = normalize_probabilities(
                        ((1.0 - feature_weight) * neural_probabilities_full)
                        + (feature_weight * feature_probabilities)
                    )
                    for bias_name, bias in bias_variants:
                        completed_steps += 1
                        if completed_steps == 1 or completed_steps % 500 == 0 or completed_steps == total_steps:
                            percent = 75 + int(completed_steps * 20 / total_steps)
                            print(
                                f"[hybrid][progress] {percent}% hybrid search "
                                f"{completed_steps}/{total_steps}",
                                flush=True,
                            )
                        biased = apply_class_bias(mixed, bias, labels=labels)
                        metrics = metrics_from_probabilities(
                            biased,
                            true_labels=y_val,
                            samples=val_rows,
                            labels=labels,
                        )
                        test_metrics = None
                        if (
                            neural_test is not None
                            and feature_result["test_probabilities"] is not None
                            and y_test is not None
                        ):
                            test_neural = apply_hybrid_temperature(neural_test, neural_transform)
                            test_feature = apply_hybrid_temperature(
                                feature_result["test_probabilities"],
                                feature_transform,
                            )
                            test_neural = align_neural_probabilities(
                                all_rows=test_rows,
                                neural_rows=test_neural_rows,
                                neural_probabilities=test_neural,
                                fallback_probabilities=test_feature,
                            )
                            test_mixed = normalize_probabilities(
                                ((1.0 - feature_weight) * test_neural)
                                + (feature_weight * test_feature)
                            )
                            test_mixed = apply_class_bias(test_mixed, bias, labels=labels)
                            test_metrics = metrics_from_probabilities(
                                test_mixed,
                                true_labels=y_test,
                                samples=test_rows,
                                labels=labels,
                            )
                        result = {
                            "feature_model": feature_result["name"],
                            "feature_weight": round(float(feature_weight), 4),
                            "neural_weight": round(float(1.0 - feature_weight), 4),
                            "feature_weight_auto": feature_weight_auto,
                            "neural_transform": neural_transform,
                            "feature_transform": feature_transform,
                            "class_bias_name": bias_name,
                            "class_bias": bias,
                            "neural_members": neural_members,
                            "metrics": {"final_validation": metrics},
                            "test_validation": test_metrics,
                            "feature_val_metrics": feature_result["val_metrics"],
                            "feature_test_metrics": feature_result["test_metrics"],
                        }
                        if best is None or hybrid_sort_key(result) > hybrid_sort_key(best):
                            best = result
    return best


def align_neural_probabilities(
    *,
    all_rows: list[dict],
    neural_rows: list[dict],
    neural_probabilities: np.ndarray,
    fallback_probabilities: np.ndarray,
) -> np.ndarray:
    if len(all_rows) == len(neural_rows) == int(neural_probabilities.shape[0]):
        return neural_probabilities
    aligned = np.asarray(fallback_probabilities, dtype=np.float64).copy()
    row_index = {row_identity(row): index for index, row in enumerate(all_rows)}
    assigned = 0
    for neural_index, row in enumerate(neural_rows):
        target_index = row_index.get(row_identity(row))
        if target_index is None or neural_index >= len(neural_probabilities):
            continue
        aligned[target_index] = neural_probabilities[neural_index]
        assigned += 1
    if assigned != len(neural_rows):
        print(
            "[hybrid] warning: neural probability alignment was partial "
            f"assigned={assigned}/{len(neural_rows)}",
            flush=True,
        )
    return normalize_probabilities(aligned)


def row_identity(row: dict) -> tuple:
    return (
        str(row.get("item_id") or ""),
        str(row.get("source_video") or row.get("video_path") or row.get("video") or ""),
        str(row.get("clip_start_seconds") or ""),
        str(row.get("clip_end_seconds") or ""),
        str(row.get("target_label") or row.get("label_idx") or ""),
        str(row.get("pose_path") or ""),
        str(row.get("rgb_feature_path") or ""),
    )


def apply_hybrid_temperature(probabilities: np.ndarray, transform_name: str) -> np.ndarray:
    if not transform_name.startswith("temperature_"):
        return probabilities
    try:
        temperature = float(transform_name.split("_", 1)[1])
    except ValueError:
        return probabilities
    if temperature <= 0:
        return probabilities
    adjusted = np.power(np.clip(probabilities, 1e-12, 1.0), 1.0 / temperature)
    return normalize_probabilities(adjusted)


def hybrid_sort_key(result: dict) -> tuple:
    metrics = result["metrics"]["final_validation"]
    test_metrics = result.get("test_validation") if isinstance(result.get("test_validation"), dict) else {}
    per_class = metrics.get("per_class") if isinstance(metrics.get("per_class"), list) else []
    supported_recalls = [
        float(row.get("recall") or 0.0)
        for row in per_class
        if isinstance(row, dict) and int(row.get("support") or 0) > 0
    ]
    min_supported_recall = min(supported_recalls, default=0.0)
    return (
        float(metrics.get("macro_f1_supported", 0.0)),
        float(metrics.get("macro_f1", 0.0)),
        float(metrics.get("balanced_accuracy", 0.0)),
        min_supported_recall,
        float(metrics.get("accuracy", 0.0)),
        float(test_metrics.get("macro_f1_supported", 0.0) or 0.0),
        float(result.get("feature_weight", 0.0)),
    )


def hybrid_bias_variants(labels: list[str], *, multipliers: list[float] | None = None) -> list[tuple[str, dict[str, float]]]:
    variants = [("none", {})]
    bias_values = multipliers or list(DEFAULT_HYBRID_CLASS_BIAS_MULTIPLIERS)
    for label in labels:
        for multiplier in bias_values:
            variants.append((f"{label}_{multiplier:g}", {label: multiplier}))
    for up_label, down_label in itertools.permutations(labels, 2):
        pair_values = [(1.1, 0.9), (1.2, 0.8), (1.3, 0.7)]
        if up_label == "normal":
            pair_values.extend([(1.5, 0.7), (1.8, 0.65), (2.0, 0.6)])
        if up_label == "abduction":
            pair_values.extend([(1.5, 0.8), (1.8, 0.75)])
        for up_multiplier, down_multiplier in pair_values:
            variants.append(
                (
                    f"{up_label}_{up_multiplier:g}_{down_label}_{down_multiplier:g}",
                    {up_label: up_multiplier, down_label: down_multiplier},
                )
            )
    return variants


def load_neural_ensemble_probabilities(
    paths: dict,
    *,
    manifests: dict[str, Path],
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray | None, list[dict]]:
    ensemble_path = paths["artifacts_dir"] / "best_action_ensemble.json"
    ensemble = read_json(ensemble_path)
    if not isinstance(ensemble, dict):
        raise RuntimeError(f"기존 neural ensemble 파일을 찾지 못했습니다: {ensemble_path}")
    best = ensemble.get("best_result") if isinstance(ensemble.get("best_result"), dict) else {}
    member_payloads = best.get("members") if isinstance(best.get("members"), list) else []
    weights = np.asarray(best.get("weights") or [], dtype=np.float64)
    if not member_payloads or weights.size != len(member_payloads):
        raise RuntimeError(f"neural ensemble 구성이 올바르지 않습니다: {ensemble_path}")
    val_dataset = PoseSequenceDataset(manifests["val"], cache_size=0)
    val_loader = DataLoader(val_dataset, batch_size=192, shuffle=False, num_workers=0)
    test_dataset = PoseSequenceDataset(manifests["test"], cache_size=0) if "test" in manifests else None
    test_loader = DataLoader(test_dataset, batch_size=192, shuffle=False, num_workers=0) if test_dataset else None
    val_members = []
    test_members = []
    for payload in member_payloads:
        candidate = EnsembleCandidate(
            name=str(payload["name"]),
            model_path=Path(payload["model_path"]),
            metrics_path=Path(payload["metrics_path"]),
            output_dir=Path(payload["output_dir"]),
            score=float(payload.get("score", 0.0) or 0.0),
            accuracy=float(payload.get("accuracy", 0.0) or 0.0),
            macro_f1=float(payload.get("macro_f1", 0.0) or 0.0),
            balanced_accuracy=float(payload.get("balanced_accuracy", 0.0) or 0.0),
            loss=float(payload.get("loss", 0.0) or 0.0),
            trial_number=payload.get("trial_number"),
        )
        val_members.append(load_member_probabilities(candidate, val_loader, labels=labels, device="cpu"))
        if test_loader is not None:
            test_members.append(load_member_probabilities(candidate, test_loader, labels=labels, device="cpu"))
    val_probabilities = weighted_average_probabilities(val_members, weights)
    val_probabilities = apply_named_probability_transform(val_probabilities, best.get("probability_transform"))
    val_probabilities = apply_class_bias(val_probabilities, best.get("class_bias") or {}, labels=labels)
    test_probabilities = weighted_average_probabilities(test_members, weights) if test_members else None
    if test_probabilities is not None:
        test_probabilities = apply_named_probability_transform(test_probabilities, best.get("probability_transform"))
        test_probabilities = apply_class_bias(test_probabilities, best.get("class_bias") or {}, labels=labels)
    return val_probabilities, test_probabilities, member_payloads


def build_feature_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = [extract_pose_features(row) for row in rows]
    labels = [int(row["label_idx"]) for row in rows]
    return stack_feature_vectors(features).astype(np.float32), np.asarray(labels, dtype=np.int64)


def filter_pose_ready_rows(rows: list[dict]) -> list[dict]:
    pose_ready = [row for row in rows if row_has_existing_pose(row)]
    filtered, skipped = _filter_conflicting_pose_label_samples(pose_ready)
    if skipped:
        print(f"[hybrid] filtered conflicting pose-label rows for neural mixing: {skipped}", flush=True)
    filtered, skipped_duplicates = _limit_duplicate_pose_label_samples(filtered)
    if skipped_duplicates:
        print(
            "[hybrid] capped duplicate pose-label rows for neural mixing: "
            f"max_per_pose_label={MAX_DUPLICATE_POSE_LABEL_SAMPLES} skipped={skipped_duplicates}",
            flush=True,
        )
    return filtered


def row_has_existing_pose(row: dict) -> bool:
    if row.get("pose_array") is not None and row.get("pose_mask") is not None:
        return True
    pose_path = str(row.get("pose_path") or "").strip()
    return bool(pose_path and Path(pose_path).exists())


def extract_pose_features(row: dict) -> np.ndarray:
    if row_has_existing_pose(row):
        if row.get("pose_array") is not None and row.get("pose_mask") is not None:
            pose = np.asarray(row["pose_array"], dtype=np.float32)
            mask = np.asarray(row["pose_mask"], dtype=np.float32)
        else:
            with np.load(row["pose_path"], allow_pickle=False) as loaded:
                pose = np.asarray(loaded["pose"], dtype=np.float32)
                mask = np.asarray(loaded["mask"], dtype=np.float32)
        confidence = pose[..., 2:3]
        xy = pose[..., :2]
        frame_mean = np.where(confidence > 0.05, xy, np.nan).mean(axis=1)
        frame_mean = np.nan_to_num(frame_mean, nan=0.0)
        xy_centered = xy - frame_mean[:, None, :]
        velocity = np.diff(xy_centered, axis=0)
        acceleration = np.diff(velocity, axis=0)
        xy_masked = np.where(confidence > 0.05, xy_centered, np.nan)
        velocity_masked = np.where(confidence[1:] > 0.05, velocity, np.nan)
        acceleration_masked = np.where(confidence[2:] > 0.05, acceleration, np.nan)
        speed = np.linalg.norm(np.nan_to_num(velocity_masked, nan=0.0), axis=-1)
        bbox = build_bbox_series(xy, confidence)
        parts = [
            feature_stats(xy_masked),
            feature_stats(velocity_masked),
            feature_stats(acceleration_masked),
            np.asarray(
                [
                    mask.mean(),
                    mask.sum(),
                    confidence.mean(),
                    confidence.std(),
                    speed.mean(),
                    speed.std(),
                    speed.max(),
                ],
                dtype=np.float32,
            ),
            feature_stats(bbox),
            feature_stats(xy_centered),
        ]
        feature_vector = np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        feature_vector = np.zeros(0, dtype=np.float32)
    rgb_vector = load_rgb_feature_vector(row)
    if rgb_vector is not None:
        feature_vector = np.concatenate([feature_vector, rgb_vector.astype(np.float32, copy=False)])
    action_vector = build_aux_action_vector(row)
    if action_vector.size:
        feature_vector = np.concatenate([feature_vector, action_vector])
    metadata_vector = build_row_metadata_vector(row)
    if metadata_vector.size:
        feature_vector = np.concatenate([feature_vector, metadata_vector])
    return feature_vector.astype(np.float32, copy=False)


def load_rgb_feature_vector(row: dict) -> np.ndarray | None:
    if row.get("rgb_feature") is not None:
        return summarize_rgb_feature_vector(np.asarray(row["rgb_feature"], dtype=np.float32))
    feature_path = str(row.get("rgb_feature_path") or "").strip()
    if not feature_path:
        return None
    path = Path(feature_path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as loaded:
            for key in ("feature", "features", "embedding", "rgb_feature"):
                if key in loaded:
                    values = np.asarray(loaded[key], dtype=np.float32)
                    return summarize_rgb_feature_vector(values)
    except Exception:
        return None
    return None


def summarize_rgb_feature_vector(values: np.ndarray, *, max_bins: int = 512) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    if values.ndim >= 2:
        flattened = values.reshape(-1, values.shape[-1])
        summary = np.concatenate(
            [
                flattened.mean(axis=0),
                flattened.std(axis=0),
                flattened.min(axis=0),
                flattened.max(axis=0),
            ]
        )
    else:
        summary = values.ravel()
    summary = np.asarray(summary, dtype=np.float32).ravel()
    if summary.size > max_bins:
        summary = pool_1d_feature(summary, max_bins=max_bins)
    global_stats = np.asarray(
        [
            values.mean(),
            values.std(),
            values.min(),
            values.max(),
            float(values.size),
            float(values.ndim),
        ],
        dtype=np.float32,
    )
    return np.concatenate([summary, global_stats]).astype(np.float32, copy=False)


def pool_1d_feature(values: np.ndarray, *, max_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).ravel()
    max_bins = max(int(max_bins), 1)
    if values.size <= max_bins:
        return values
    chunks = np.array_split(values, max_bins)
    return np.asarray([chunk.mean(dtype=np.float32) for chunk in chunks], dtype=np.float32)


def stack_feature_vectors(features: list[np.ndarray]) -> np.ndarray:
    if not features:
        return np.zeros((0, 0), dtype=np.float32)
    max_dim = max(int(feature.size) for feature in features)
    stacked = np.zeros((len(features), max_dim), dtype=np.float32)
    for index, feature in enumerate(features):
        values = np.asarray(feature, dtype=np.float32).ravel()
        stacked[index, : min(values.size, max_dim)] = values[:max_dim]
    return stacked


def build_aux_action_vector(row: dict, *, buckets: int = 32) -> np.ndarray:
    actions = row.get("aux_actions")
    if not isinstance(actions, list) or not actions:
        return np.zeros(buckets, dtype=np.float32)
    vector = np.zeros(buckets, dtype=np.float32)
    for action in actions:
        text = str(action or "").strip().lower()
        if not text:
            continue
        bucket = sum(ord(ch) for ch in text) % buckets
        vector[bucket] += 1.0
    if vector.max() > 0:
        vector /= max(float(vector.max()), 1.0)
    return vector


def build_row_metadata_vector(row: dict) -> np.ndarray:
    frame_stats = row.get("frame_stats") if isinstance(row.get("frame_stats"), dict) else {}
    rejection_reasons = (
        frame_stats.get("rejection_reasons")
        if isinstance(frame_stats.get("rejection_reasons"), dict)
        else {}
    )
    clip_start = safe_float(row.get("clip_start_seconds"))
    clip_end = safe_float(row.get("clip_end_seconds"))
    values = [
        safe_float(row.get("valid_frames")),
        safe_float(row.get("confirmed_frames")),
        safe_float(row.get("fallback_frames")),
        safe_float(row.get("total_valid_keypoints")),
        safe_float(row.get("avg_pose_confidence")),
        safe_float(frame_stats.get("sampled_frames")),
        safe_float(frame_stats.get("detection_frames")),
        safe_float(frame_stats.get("candidate_frames")),
        safe_float(frame_stats.get("accepted_frames")),
        safe_float(frame_stats.get("fallback_frames")),
        safe_float(frame_stats.get("rejected_candidates")),
        safe_float(rejection_reasons.get("unstable_track")),
        safe_float(rejection_reasons.get("too_few_valid_kpts")),
        max(clip_end - clip_start, 0.0),
        safe_float(row.get("sample_weight"), default=1.0),
    ]
    return np.asarray(values, dtype=np.float32)


def safe_float(value, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def feature_stats(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    stat_shape = values.shape[1:]
    if values.shape[0] == 0:
        return np.zeros(int(np.prod(stat_shape, dtype=np.int64)) * 4, dtype=np.float32)

    valid = np.isfinite(values)
    safe_values = np.where(valid, values, 0.0)
    counts = valid.sum(axis=0)
    counts_safe = np.maximum(counts, 1)

    mean = safe_values.sum(axis=0) / counts_safe
    squared_mean = np.square(safe_values).sum(axis=0) / counts_safe
    std = np.sqrt(np.maximum(squared_mean - np.square(mean), 0.0))

    minimum = np.where(valid, values, np.inf).min(axis=0)
    maximum = np.where(valid, values, -np.inf).max(axis=0)
    minimum = np.where(counts > 0, minimum, 0.0)
    maximum = np.where(counts > 0, maximum, 0.0)

    return np.nan_to_num(
        np.concatenate([mean.ravel(), std.ravel(), minimum.ravel(), maximum.ravel()]),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)


def build_bbox_series(xy: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    rows = []
    for frame_index in range(xy.shape[0]):
        points = xy[frame_index, confidence[frame_index, :, 0] > 0.05]
        if len(points):
            minimum = points.min(axis=0)
            maximum = points.max(axis=0)
            width, height = maximum - minimum
            rows.append([width, height, width * height, height / max(width, 1e-4)])
        else:
            rows.append([0.0, 0.0, 0.0, 0.0])
    return np.asarray(rows, dtype=np.float32)


def read_manifest_rows(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_hybrid_artifacts(
    output_dir: Path,
    *,
    best_result: dict,
    feature_results: list[dict],
    labels: list[str],
) -> None:
    final_validation = best_result["metrics"]["final_validation"]
    error_analysis = _build_validation_error_analysis(final_validation, labels=labels)
    payload = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "model_type": "hybrid_ensemble",
        "labels": labels,
        "history": [],
        "final_validation": final_validation,
        "validation_error_analysis": error_analysis.get("summary", {}),
        "validation_error_analysis_path": str(output_dir / "validation_error_analysis.json"),
        "false_negative_examples_path": str(output_dir / "false_negative_examples.json"),
        "confusion_pair_examples_path": str(output_dir / "confusion_pair_examples.json"),
        "best_val_macro_f1": round(float(final_validation.get("macro_f1", 0.0)), 6),
        "best_epoch": None,
        "hybrid": compact_hybrid_result(best_result),
    }
    if best_result.get("test_validation") is not None:
        payload["holdout_test"] = best_result["test_validation"]
    write_json_atomic(output_dir / "hybrid_metrics.json", payload)
    write_json_atomic(output_dir / "validation_error_analysis.json", error_analysis)
    write_json_atomic(output_dir / "false_negative_examples.json", error_analysis.get("false_negative_examples", {}))
    write_json_atomic(output_dir / "confusion_pair_examples.json", error_analysis.get("confusion_pair_examples", []))
    summary = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "best_result": compact_hybrid_result(best_result),
        "feature_candidates": [
            {
                "name": result["name"],
                "val_accuracy": result["val_metrics"]["accuracy"],
                "val_macro_f1": result["val_metrics"]["macro_f1"],
                "test_accuracy": result["test_metrics"]["accuracy"] if result.get("test_metrics") else None,
            }
            for result in feature_results
        ],
        "promoted": False,
    }
    write_json_atomic(output_dir / "hybrid_summary.json", summary)
    model = next(result["model"] for result in feature_results if result["name"] == best_result["feature_model"])
    with (output_dir / "best_pose_feature_model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "feature_model": best_result["feature_model"],
                "labels": labels,
                "feature_weight": best_result["feature_weight"],
                "neural_weight": best_result["neural_weight"],
                "feature_weight_auto": best_result.get("feature_weight_auto", {}),
                "neural_transform": best_result["neural_transform"],
                "feature_transform": best_result["feature_transform"],
                "class_bias": best_result["class_bias"],
            },
            handle,
        )


def compact_hybrid_result(result: dict) -> dict:
    final_validation = result["metrics"]["final_validation"]
    test_validation = result.get("test_validation") if isinstance(result.get("test_validation"), dict) else {}
    return {
        "feature_model": result["feature_model"],
        "feature_weight": result["feature_weight"],
        "neural_weight": result["neural_weight"],
        "feature_weight_auto": result.get("feature_weight_auto", {}),
        "neural_transform": result.get("neural_transform", "raw"),
        "feature_transform": result.get("feature_transform", "raw"),
        "class_bias": result["class_bias"],
        "class_bias_name": result["class_bias_name"],
        "val_accuracy": final_validation.get("accuracy"),
        "macro_f1": final_validation.get("macro_f1"),
        "balanced_accuracy": final_validation.get("balanced_accuracy"),
        "macro_f1_supported": final_validation.get("macro_f1_supported"),
        "test_accuracy": test_validation.get("accuracy"),
        "test_macro_f1": test_validation.get("macro_f1"),
        "test_macro_f1_supported": test_validation.get("macro_f1_supported"),
        "neural_members": result["neural_members"],
    }


def should_promote(best_result: dict, artifacts_dir: Path) -> bool:
    current_metrics = read_json(artifacts_dir / "metrics.json")
    if not isinstance(current_metrics, dict):
        return True
    current = current_metrics.get("final_validation") if isinstance(current_metrics.get("final_validation"), dict) else {}
    candidate = best_result["metrics"]["final_validation"]
    return (
        float(candidate.get("accuracy", 0.0)),
        float(candidate.get("macro_f1", 0.0)),
    ) > (
        float(current.get("accuracy", 0.0) or 0.0),
        float(current.get("macro_f1", 0.0) or 0.0),
    )


def promote_hybrid(output_dir: Path, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    copies = {
        "hybrid_metrics.json": "metrics.json",
        "hybrid_summary.json": "hybrid_summary.json",
        "best_pose_feature_model.pkl": "best_pose_feature_model.pkl",
        "validation_error_analysis.json": "validation_error_analysis.json",
        "false_negative_examples.json": "false_negative_examples.json",
        "confusion_pair_examples.json": "confusion_pair_examples.json",
    }
    for source_name, target_name in copies.items():
        source = output_dir / source_name
        if source.exists():
            import shutil

            shutil.copy2(source, artifacts_dir / target_name)


def resolve_output_dir(paths: dict, base_dir: Path, configured: str | None) -> Path:
    if configured:
        candidate = Path(str(configured)).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    run_id = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = paths["artifacts_dir"] / "auto_tune" / "hybrid" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def utc_now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat()


if __name__ == "__main__":
    main()
