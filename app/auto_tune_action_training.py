from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from action_training_pipeline import (
    build_stage_timing_entry,
    current_timestamp_iso,
    get_continual_config,
    get_target_labels,
    load_config,
    load_training_manifests,
    resolve_paths,
    validate_training_class_coverage,
    write_pipeline_status,
)
from gpu_autotune import apply_gpu_auto_tune
from reporting import read_json, write_json_atomic


PROMOTED_ARTIFACT_NAMES = (
    "best_action_model.pt",
    "metrics.json",
    "labels.json",
    "training_progress.json",
    "validation_error_analysis.json",
    "false_negative_examples.json",
    "confusion_pair_examples.json",
)
DEFAULT_MAX_TRIALS = 6
DEFAULT_TARGET_RECALL = 0.35
DEFAULT_TARGET_METRIC = "balanced"
TUNABLE_TRAINING_KEYS = {
    "learning_rate",
    "weight_decay",
    "hidden_dim",
    "num_layers",
    "dropout",
    "temporal_pooling",
    "label_smoothing",
    "loss",
    "focal_gamma",
    "class_weight",
    "class_weight_multipliers",
    "balanced_sampler",
    "early_stopping_patience",
    "early_stopping_min_delta",
}


@dataclass(frozen=True)
class TrialSpec:
    name: str
    params: dict[str, Any]
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "이미 추출된 active/cumulative prepared pose manifest만 사용해 여러 학습 설정을 "
            "자동으로 비교하고 best trial을 승격합니다."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/action_training.aihub_shell.example.json",
        help="학습 설정 JSON 경로",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help=f"실행할 최대 trial 수. 기본값은 설정 auto_tune.max_trials 또는 {DEFAULT_MAX_TRIALS}",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="trial 결과 저장 경로. 기본값은 workspace/artifacts/auto_tune",
    )
    parser.add_argument(
        "--resume-from-best",
        action="store_true",
        help="현재 best_action_model.pt에서 warm-start합니다. 기본은 fresh 학습입니다.",
    )
    parser.add_argument(
        "--no-promote-best",
        action="store_true",
        help="best trial을 메인 artifacts로 복사하지 않고 결과만 저장합니다.",
    )
    parser.add_argument(
        "--target-metric",
        choices=("balanced", "accuracy", "macro_f1"),
        default=None,
        help="best trial을 고를 목적 함수. 기본값은 설정 auto_tune.target_metric 또는 balanced",
    )
    parser.add_argument(
        "--no-update-config",
        action="store_true",
        help="best trial의 JSON 값을 config 파일에 반영하지 않습니다.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 학습 없이 생성될 trial 설정만 출력합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    config = apply_gpu_auto_tune(config, stage="train", logger=print)
    paths = resolve_paths(config, config_path.parent)
    labels = get_target_labels(config)
    continual_config = get_continual_config(config)
    training_manifests = load_training_manifests(
        config,
        paths,
        continual_enabled=continual_config["enabled"],
    )
    validate_training_class_coverage(config, training_manifests, labels=labels)

    auto_tune_config = config.get("auto_tune", {}) if isinstance(config.get("auto_tune"), dict) else {}
    max_trials = max(
        int(args.trials if args.trials is not None else auto_tune_config.get("max_trials", DEFAULT_MAX_TRIALS)),
        1,
    )
    output_dir = resolve_auto_tune_dir(
        paths,
        config_path.parent,
        args.output_dir or auto_tune_config.get("output_dir"),
    )
    promote_best = not args.no_promote_best and bool(auto_tune_config.get("promote_best", True))
    update_config_with_best = (
        not args.no_update_config
        and bool(auto_tune_config.get("update_config_with_best", False))
    )
    target_metric = str(args.target_metric or auto_tune_config.get("target_metric") or DEFAULT_TARGET_METRIC)
    target_recall = float(auto_tune_config.get("target_recall", DEFAULT_TARGET_RECALL))

    previous_metrics = read_json(paths["artifacts_dir"] / "metrics.json") or {}
    trial_queue = build_initial_trial_specs(
        labels=labels,
        base_training=config.get("training", {}),
        previous_metrics=previous_metrics if isinstance(previous_metrics, dict) else {},
        target_recall=target_recall,
    )
    trial_queue.extend(
        build_exploration_trial_specs(
            labels=labels,
            base_training=config.get("training", {}),
            previous_metrics=previous_metrics if isinstance(previous_metrics, dict) else {},
            auto_tune_config=auto_tune_config,
            target_recall=target_recall,
        )
    )
    trial_queue = dedupe_trial_specs(trial_queue)

    if args.dry_run:
        write_json_atomic(
            output_dir / "dry_run_trials.json",
            {
                "created_at": utc_now_iso(),
                "max_trials": max_trials,
                "target_metric": target_metric,
                "update_config_with_best": update_config_with_best,
                "trials": [trial_to_payload(trial) for trial in trial_queue[:max_trials]],
            },
        )
        print(f"[auto-tune] dry-run trials: {output_dir / 'dry_run_trials.json'}")
        return

    run_id = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    trials_log_path = output_dir / "trials.jsonl"
    status_path = output_dir / "status.json"
    started_at = current_timestamp_iso()
    stage_timings: dict[str, dict] = {}
    results: list[dict[str, Any]] = []
    seen_signatures = {trial_signature(trial.params) for trial in trial_queue}

    write_pipeline_status(
        paths,
        stage="auto_tune",
        state="running",
        message=f"자동 학습 개선을 시작합니다. 최대 {max_trials}개 trial을 비교합니다.",
        pipeline_started_at=started_at,
        stage_started_at=started_at,
        stage_timings=stage_timings,
        stage_progress=0.0,
        auto_tune={
            "run_id": run_id,
            "output_dir": str(output_dir),
            "max_trials": max_trials,
        },
    )

    try:
        trial_index = 0
        while trial_index < max_trials and trial_index < len(trial_queue):
            trial = trial_queue[trial_index]
            trial_number = trial_index + 1
            trial_started_at = current_timestamp_iso()
            trial_output_dir = runs_dir / f"{run_id}_trial_{trial_number:02d}_{slugify(trial.name)}"
            trial_progress_path = trial_output_dir / "training_progress.json"
            stage_timings[f"trial_{trial_number:02d}"] = {
                "started_at": trial_started_at,
                "finished_at": None,
                "duration_seconds": None,
            }
            write_auto_tune_status(
                status_path,
                run_id=run_id,
                state="running",
                current_trial=trial_number,
                max_trials=max_trials,
                trial=trial,
                results=results,
            )
            write_pipeline_status(
                paths,
                stage="auto_tune",
                state="running",
                message=f"자동 학습 개선 trial {trial_number}/{max_trials}: {trial.name}",
                pipeline_started_at=started_at,
                stage_started_at=trial_started_at,
                stage_timings=stage_timings,
                stage_progress=round((trial_number - 1) / max_trials, 4),
                auto_tune={
                    "run_id": run_id,
                    "output_dir": str(output_dir),
                    "trial": trial_to_payload(trial),
                    "completed_trials": len(results),
                    "max_trials": max_trials,
                },
            )

            print(f"[auto-tune] trial {trial_number}/{max_trials}: {trial.name} | {trial.reason}")
            metrics = run_trial(
                config=config,
                paths=paths,
                training_manifests=training_manifests,
                labels=labels,
                trial=trial,
                trial_output_dir=trial_output_dir,
                trial_progress_path=trial_progress_path,
                resume_from_best=bool(args.resume_from_best),
            )
            result = build_trial_result(
                trial_number=trial_number,
                trial=trial,
                output_dir=trial_output_dir,
                metrics=metrics,
                started_at=trial_started_at,
                finished_at=current_timestamp_iso(),
                target_metric=target_metric,
            )
            results.append(result)
            append_jsonl(trials_log_path, result)
            stage_timings[f"trial_{trial_number:02d}"] = build_stage_timing_entry(
                trial_started_at,
                result["finished_at"],
            )

            next_trial = build_feedback_trial(
                labels=labels,
                base_training=config.get("training", {}),
                result=result,
                target_recall=target_recall,
                trial_number=trial_number + 1,
            )
            if next_trial is not None:
                signature = trial_signature(next_trial.params)
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    trial_queue.append(next_trial)

            write_json_atomic(
                summary_path,
                build_summary_payload(
                    run_id=run_id,
                    state="running",
                    output_dir=output_dir,
                    results=results,
                    promoted=False,
                    target_metric=target_metric,
                ),
            )
            trial_index += 1

        best_result = select_best_result(results)
        promoted = False
        if best_result is not None and promote_best:
            promote_trial_artifacts(Path(best_result["output_dir"]), paths["artifacts_dir"])
            promoted = True
        best_config_path = None
        config_updated = False
        if best_result is not None:
            best_config_path = write_best_training_config(
                output_dir=output_dir,
                config=config,
                best_result=best_result,
                target_metric=target_metric,
            )
            if update_config_with_best:
                update_source_config_with_best(
                    config_path,
                    best_result=best_result,
                    best_config_path=best_config_path,
                    target_metric=target_metric,
                )
                config_updated = True

        ensemble_result = None
        ensemble_error = None
        if best_result is not None and bool(auto_tune_config.get("ensemble_enabled", True)):
            try:
                from ensemble_action_models import run_ensemble_search

                ensemble_result = run_ensemble_search(
                    config=config,
                    paths=paths,
                    training_manifests=training_manifests,
                    labels=labels,
                    output_dir=output_dir / "ensembles" / run_id,
                    target_metric=target_metric,
                    top_k=int(auto_tune_config.get("ensemble_top_k", 12) or 12),
                    max_size=int(auto_tune_config.get("ensemble_max_size", 5) or 5),
                    device=str(config.get("training", {}).get("device", "cuda")),
                    promote=bool(auto_tune_config.get("promote_ensemble", True)),
                )
                print(f"[auto-tune] ensemble summary: {ensemble_result.get('summary_path')}")
            except Exception as exc:
                ensemble_error = str(exc)
                print(f"[auto-tune] ensemble search skipped after error: {exc}")

        hybrid_result = None
        hybrid_error = None
        if best_result is not None and bool(auto_tune_config.get("hybrid_enabled", True)):
            try:
                from hybrid_pose_ensemble import run_hybrid_search

                hybrid_result = run_hybrid_search(
                    config=config,
                    paths=paths,
                    manifests=training_manifests,
                    labels=labels,
                    output_dir=output_dir / "hybrid" / run_id,
                    promote=bool(auto_tune_config.get("promote_hybrid", True)),
                )
                print(f"[auto-tune] hybrid summary: {hybrid_result.get('summary_path')}")
            except Exception as exc:
                hybrid_error = str(exc)
                print(f"[auto-tune] hybrid search skipped after error: {exc}")

        finished_at = current_timestamp_iso()
        stage_timings["total"] = build_stage_timing_entry(started_at, finished_at)
        summary_payload = build_summary_payload(
            run_id=run_id,
            state="completed",
            output_dir=output_dir,
            results=results,
            promoted=promoted,
            target_metric=target_metric,
            best_config_path=best_config_path,
            config_updated=config_updated,
            ensemble_result=ensemble_result,
            ensemble_error=ensemble_error,
            hybrid_result=hybrid_result,
            hybrid_error=hybrid_error,
        )
        write_json_atomic(summary_path, summary_payload)
        write_auto_tune_status(
            status_path,
            run_id=run_id,
            state="completed",
            current_trial=len(results),
            max_trials=max_trials,
            trial=None,
            results=results,
        )
        write_pipeline_status(
            paths,
            stage="completed",
            state="completed",
            message=build_completion_message(best_result, promoted=promoted),
            pipeline_started_at=started_at,
            stage_started_at=started_at,
            stage_timings=stage_timings,
            total_duration_seconds=stage_timings["total"]["duration_seconds"],
            stage_progress=1.0,
            auto_tune={
                "run_id": run_id,
                "output_dir": str(output_dir),
                "summary_path": str(summary_path),
                "best_result": best_result,
                "promoted": promoted,
                "best_config_path": str(best_config_path) if best_config_path else None,
                "config_updated": config_updated,
                "ensemble_result": compact_ensemble_result(ensemble_result),
                "ensemble_error": ensemble_error,
                "hybrid_result": compact_hybrid_result(hybrid_result),
                "hybrid_error": hybrid_error,
            },
        )
        print(f"[auto-tune] summary: {summary_path}")
    except Exception as exc:
        finished_at = current_timestamp_iso()
        stage_timings["total"] = build_stage_timing_entry(started_at, finished_at)
        write_auto_tune_status(
            status_path,
            run_id=run_id,
            state="error",
            current_trial=len(results) + 1,
            max_trials=max_trials,
            trial=trial_queue[len(results)] if len(results) < len(trial_queue) else None,
            results=results,
            error=str(exc),
        )
        write_pipeline_status(
            paths,
            stage="error",
            state="error",
            message=f"자동 학습 개선 중 오류가 발생했습니다: {exc}",
            pipeline_started_at=started_at,
            stage_started_at=started_at,
            stage_timings=stage_timings,
            total_duration_seconds=stage_timings["total"]["duration_seconds"],
            stage_progress=1.0,
            auto_tune={
                "run_id": run_id,
                "output_dir": str(output_dir),
                "completed_trials": len(results),
            },
        )
        raise


def run_trial(
    *,
    config: dict,
    paths: dict,
    training_manifests: dict[str, Path],
    labels: list[str],
    trial: TrialSpec,
    trial_output_dir: Path,
    trial_progress_path: Path,
    resume_from_best: bool,
) -> dict:
    from action_model import train_action_classifier

    training = merged_training_config(config.get("training", {}), trial.params)
    resume_from = paths["artifacts_dir"] / "best_action_model.pt" if resume_from_best else None
    if resume_from is not None and not resume_from.exists():
        resume_from = None
    write_json_atomic(
        trial_output_dir / "trial_config.json",
        {
            "created_at": utc_now_iso(),
            "trial": trial_to_payload(trial),
            "effective_training": sanitize_training_payload(training),
            "resume_from": str(resume_from) if resume_from is not None else None,
        },
    )

    artifacts = train_action_classifier(
        train_manifest=training_manifests["train"],
        val_manifest=training_manifests["val"],
        output_dir=trial_output_dir,
        labels=labels,
        epochs=int(training.get("epochs", 20)),
        batch_size=int(training.get("batch_size", 16)),
        eval_batch_size=int(training.get("eval_batch_size", 0) or 0),
        learning_rate=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-3)),
        hidden_dim=int(training.get("hidden_dim", 128)),
        num_layers=int(training.get("num_layers", 2)),
        dropout=float(training.get("dropout", 0.2)),
        temporal_pooling=str(training.get("temporal_pooling", "mean")),
        label_smoothing=float(training.get("label_smoothing", 0.05)),
        loss_name=str(training.get("loss", "cross_entropy")),
        focal_gamma=float(training.get("focal_gamma", 2.0)),
        class_weight=training.get("class_weight", "balanced"),
        class_weight_multipliers=training.get("class_weight_multipliers") or {},
        balanced_sampler=training.get("balanced_sampler", "auto"),
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
        early_stopping_patience=int(training.get("early_stopping_patience", 5)),
        early_stopping_min_delta=float(training.get("early_stopping_min_delta", 0.001)),
        imbalance_warn_min_samples=int(training.get("imbalance_warn_min_samples", 8)),
        imbalance_warn_ratio=float(training.get("imbalance_warn_ratio", 5.0)),
        progress_path=trial_progress_path,
        resume_from=resume_from,
    )
    metrics = read_json(artifacts.metrics_path)
    if not isinstance(metrics, dict):
        raise RuntimeError(f"trial metrics를 읽지 못했습니다: {artifacts.metrics_path}")
    metrics["auto_tune_trial"] = trial_to_payload(trial)
    metrics["auto_tune_effective_training"] = sanitize_training_payload(training)
    write_json_atomic(artifacts.metrics_path, metrics)
    return metrics


def build_initial_trial_specs(
    *,
    labels: list[str],
    base_training: dict,
    previous_metrics: dict,
    target_recall: float,
) -> list[TrialSpec]:
    feedback = extract_metric_feedback(previous_metrics, labels=labels, target_recall=target_recall)
    missing_or_low = feedback["missing_or_low_recall_labels"]
    overpredicted = feedback["overpredicted_label"]

    base_multipliers = normalize_multiplier_map(base_training.get("class_weight_multipliers"), labels=labels)
    recovery_mild = build_recovery_multipliers(
        labels=labels,
        base=base_multipliers,
        boost_labels=missing_or_low,
        suppress_label=overpredicted,
        boost=2.0,
        suppress=0.8,
    )
    recovery_strong = build_recovery_multipliers(
        labels=labels,
        base=base_multipliers,
        boost_labels=missing_or_low,
        suppress_label=overpredicted,
        boost=2.5,
        suppress=0.7,
    )

    return [
        TrialSpec(
            name="baseline_fresh",
            reason="현재 설정을 fresh 기준점으로 다시 측정합니다.",
            params={
                "resume_from_best": False,
                "class_weight_multipliers": base_multipliers,
            },
        ),
        TrialSpec(
            name="recall_recovery_mild",
            reason="예측되지 않거나 recall이 낮은 클래스를 살리고 과예측 클래스를 살짝 낮춥니다.",
            params={
                "resume_from_best": False,
                "class_weight_multipliers": recovery_mild,
                "balanced_sampler": True,
                "loss": "focal",
                "focal_gamma": 1.0,
                "label_smoothing": min(float(base_training.get("label_smoothing", 0.03) or 0.03), 0.03),
                "early_stopping_patience": max(int(base_training.get("early_stopping_patience", 5) or 5), 6),
            },
        ),
        TrialSpec(
            name="recall_recovery_strong",
            reason="missing prediction 클래스에 더 강한 비용을 주고 label smoothing을 줄입니다.",
            params={
                "resume_from_best": False,
                "class_weight_multipliers": recovery_strong,
                "balanced_sampler": True,
                "loss": "focal",
                "focal_gamma": 0.5,
                "label_smoothing": 0.0,
                "early_stopping_patience": max(int(base_training.get("early_stopping_patience", 5) or 5), 8),
            },
        ),
        TrialSpec(
            name="sampler_off_weighted",
            reason="balanced sampler와 class weight의 이중 보정이 예측 쏠림을 만드는지 확인합니다.",
            params={
                "resume_from_best": False,
                "class_weight_multipliers": recovery_mild,
                "balanced_sampler": False,
                "class_weight": "balanced",
                "loss": "focal",
                "focal_gamma": 1.0,
            },
        ),
        TrialSpec(
            name="cross_entropy_probe",
            reason="focal loss가 어려운 클래스를 과도하게 흔드는지 cross entropy로 비교합니다.",
            params={
                "resume_from_best": False,
                "class_weight_multipliers": recovery_mild,
                "balanced_sampler": True,
                "loss": "cross_entropy",
                "label_smoothing": min(float(base_training.get("label_smoothing", 0.03) or 0.03), 0.03),
            },
        ),
    ]


def build_feedback_trial(
    *,
    labels: list[str],
    base_training: dict,
    result: dict,
    target_recall: float,
    trial_number: int,
) -> TrialSpec | None:
    feedback = extract_metric_feedback(result.get("metrics") or {}, labels=labels, target_recall=target_recall)
    boost_labels = feedback["missing_or_low_recall_labels"]
    if not boost_labels:
        return None
    previous_params = result.get("params") if isinstance(result.get("params"), dict) else {}
    previous_multipliers = normalize_multiplier_map(
        previous_params.get("class_weight_multipliers") or base_training.get("class_weight_multipliers"),
        labels=labels,
    )
    next_multipliers = build_recovery_multipliers(
        labels=labels,
        base=previous_multipliers,
        boost_labels=boost_labels,
        suppress_label=feedback["overpredicted_label"],
        boost=1.35,
        suppress=0.85,
        max_multiplier=3.0,
    )
    return TrialSpec(
        name=f"feedback_recovery_{trial_number:02d}",
        reason="직전 trial의 낮은 recall/missing prediction 클래스를 기준으로 비용을 재조정합니다.",
        params={
            "resume_from_best": False,
            "class_weight_multipliers": next_multipliers,
            "balanced_sampler": previous_params.get("balanced_sampler", True),
            "loss": previous_params.get("loss", "focal"),
            "focal_gamma": max(float(previous_params.get("focal_gamma", base_training.get("focal_gamma", 1.0)) or 1.0), 0.5),
            "label_smoothing": max(float(previous_params.get("label_smoothing", 0.0) or 0.0) * 0.5, 0.0),
            "early_stopping_patience": max(int(base_training.get("early_stopping_patience", 5) or 5), 8),
        },
    )


def build_exploration_trial_specs(
    *,
    labels: list[str],
    base_training: dict,
    previous_metrics: dict,
    auto_tune_config: dict,
    target_recall: float,
) -> list[TrialSpec]:
    if not bool(auto_tune_config.get("exploration_enabled", True)):
        return []

    feedback = extract_metric_feedback(previous_metrics, labels=labels, target_recall=target_recall)
    low_labels = feedback["missing_or_low_recall_labels"]
    overpredicted = feedback["overpredicted_label"]
    base_multipliers = normalize_multiplier_map(base_training.get("class_weight_multipliers"), labels=labels)
    balanced_multipliers = build_recovery_multipliers(
        labels=labels,
        base=base_multipliers,
        boost_labels=low_labels,
        suppress_label=overpredicted,
        boost=1.35,
        suppress=0.85,
        max_multiplier=3.0,
    )
    stronger_multipliers = build_recovery_multipliers(
        labels=labels,
        base=balanced_multipliers,
        boost_labels=low_labels,
        suppress_label=overpredicted,
        boost=1.2,
        suppress=0.9,
        max_multiplier=3.0,
    )

    learning_rate = float(base_training.get("learning_rate", 5e-4) or 5e-4)
    weight_decay = float(base_training.get("weight_decay", 0.005) or 0.005)
    dropout = float(base_training.get("dropout", 0.45) or 0.45)
    hidden_dim = int(base_training.get("hidden_dim", 128) or 128)
    patience = max(int(base_training.get("early_stopping_patience", 6) or 6), 8)
    seed_values = normalize_seed_list(
        auto_tune_config.get("seeds"),
        default=[7, 13, 21, 42, 77, 123, 2026],
    )

    trials = [
        TrialSpec(
            name="json_best_low_lr",
            reason="현재 best JSON에 가까운 상태에서 learning_rate를 낮춰 안정성을 봅니다.",
            params={
                "class_weight_multipliers": base_multipliers,
                "balanced_sampler": True,
                "loss": "focal",
                "focal_gamma": float(base_training.get("focal_gamma", 1.0) or 1.0),
                "label_smoothing": float(base_training.get("label_smoothing", 0.0) or 0.0),
                "learning_rate": round(max(learning_rate * 0.6, 1e-5), 8),
                "early_stopping_patience": patience,
            },
        ),
        TrialSpec(
            name="json_best_high_lr",
            reason="현재 best JSON에 가까운 상태에서 learning_rate를 높여 더 빠른 탈출을 시도합니다.",
            params={
                "class_weight_multipliers": base_multipliers,
                "balanced_sampler": True,
                "loss": "focal",
                "focal_gamma": float(base_training.get("focal_gamma", 1.0) or 1.0),
                "label_smoothing": float(base_training.get("label_smoothing", 0.0) or 0.0),
                "learning_rate": round(min(learning_rate * 1.6, 0.003), 8),
                "early_stopping_patience": patience,
            },
        ),
        TrialSpec(
            name="less_regularized",
            reason="현재 데이터 규모에서 dropout/weight_decay가 과한지 줄여봅니다.",
            params={
                "class_weight_multipliers": balanced_multipliers,
                "balanced_sampler": True,
                "loss": "focal",
                "focal_gamma": 1.0,
                "label_smoothing": 0.0,
                "dropout": round(max(dropout - 0.15, 0.15), 4),
                "weight_decay": round(max(weight_decay * 0.4, 0.0001), 8),
                "early_stopping_patience": patience,
            },
        ),
        TrialSpec(
            name="more_regularized",
            reason="validation 변동이 큰 경우를 대비해 regularization을 더 줍니다.",
            params={
                "class_weight_multipliers": balanced_multipliers,
                "balanced_sampler": True,
                "loss": "focal",
                "focal_gamma": 1.0,
                "label_smoothing": 0.02,
                "dropout": round(min(dropout + 0.1, 0.6), 4),
                "weight_decay": round(min(weight_decay * 1.6, 0.02), 8),
                "early_stopping_patience": patience,
            },
        ),
        TrialSpec(
            name="larger_hidden",
            reason="pose 패턴 표현력이 부족한지 hidden dimension을 키워 확인합니다.",
            params={
                "class_weight_multipliers": balanced_multipliers,
                "balanced_sampler": True,
                "loss": "focal",
                "focal_gamma": 1.0,
                "label_smoothing": 0.0,
                "hidden_dim": min(max(hidden_dim, 128) * 2, 384),
                "dropout": round(max(dropout - 0.05, 0.2), 4),
                "early_stopping_patience": patience,
            },
        ),
        TrialSpec(
            name="gentle_class_balance",
            reason="class multiplier가 과하게 흔들리는지 완만한 보정으로 비교합니다.",
            params={
                "class_weight_multipliers": shrink_multipliers_toward_one(balanced_multipliers, factor=0.55),
                "balanced_sampler": True,
                "loss": "focal",
                "focal_gamma": 1.0,
                "label_smoothing": 0.0,
                "early_stopping_patience": patience,
            },
        ),
        TrialSpec(
            name="mean_pooling_control",
            reason="mean+max pooling과 기존 mean pooling을 같은 설정에서 비교합니다.",
            params={
                "class_weight_multipliers": base_multipliers,
                "balanced_sampler": True,
                "loss": "focal",
                "focal_gamma": float(base_training.get("focal_gamma", 1.0) or 1.0),
                "label_smoothing": float(base_training.get("label_smoothing", 0.0) or 0.0),
                "learning_rate": learning_rate,
                "temporal_pooling": "mean",
                "early_stopping_patience": patience,
            },
        ),
        TrialSpec(
            name="mean_max_pooling_probe",
            reason="짧고 강한 움직임이 평균 pooling에서 희석되는지 mean+max pooling으로 확인합니다.",
            params={
                "class_weight_multipliers": base_multipliers,
                "balanced_sampler": True,
                "loss": "focal",
                "focal_gamma": float(base_training.get("focal_gamma", 1.0) or 1.0),
                "label_smoothing": float(base_training.get("label_smoothing", 0.0) or 0.0),
                "learning_rate": learning_rate,
                "temporal_pooling": "mean_max",
                "dropout": dropout,
                "early_stopping_patience": patience,
            },
        ),
        TrialSpec(
            name="mean_max_low_lr",
            reason="mean+max pooling에서 낮은 learning rate가 안정적인지 확인합니다.",
            params={
                "class_weight_multipliers": base_multipliers,
                "balanced_sampler": True,
                "loss": "focal",
                "focal_gamma": float(base_training.get("focal_gamma", 1.0) or 1.0),
                "label_smoothing": 0.0,
                "learning_rate": round(max(learning_rate * 0.6, 1e-5), 8),
                "temporal_pooling": "mean_max",
                "early_stopping_patience": patience,
            },
        ),
        TrialSpec(
            name="stronger_class_balance",
            reason="낮은 recall 클래스가 계속 남으면 class multiplier를 조금 더 밀어봅니다.",
            params={
                "class_weight_multipliers": stronger_multipliers,
                "balanced_sampler": True,
                "loss": "focal",
                "focal_gamma": 0.8,
                "label_smoothing": 0.0,
                "early_stopping_patience": patience,
            },
        ),
        TrialSpec(
            name="sampler_only",
            reason="loss class weight를 빼고 sampler만으로 균형을 잡는 쪽을 비교합니다.",
            params={
                "class_weight_multipliers": {},
                "balanced_sampler": True,
                "class_weight": False,
                "loss": "focal",
                "focal_gamma": 1.0,
                "label_smoothing": 0.0,
                "early_stopping_patience": patience,
            },
        ),
        TrialSpec(
            name="weighted_no_sampler",
            reason="sampler 없이 loss weight만 사용하는 쪽을 비교합니다.",
            params={
                "class_weight_multipliers": balanced_multipliers,
                "balanced_sampler": False,
                "class_weight": "balanced",
                "loss": "focal",
                "focal_gamma": 1.0,
                "label_smoothing": 0.0,
                "early_stopping_patience": patience,
            },
        ),
        TrialSpec(
            name="focal_low_gamma_long_patience",
            reason="focal gamma를 낮추고 patience를 늘려 validation peak를 더 오래 기다립니다.",
            params={
                "class_weight_multipliers": balanced_multipliers,
                "balanced_sampler": True,
                "loss": "focal",
                "focal_gamma": 0.35,
                "label_smoothing": 0.0,
                "learning_rate": round(max(learning_rate * 0.8, 1e-5), 8),
                "early_stopping_patience": max(patience, 12),
                "early_stopping_min_delta": 0.001,
            },
        ),
        TrialSpec(
            name="focal_high_gamma_hard_examples",
            reason="어려운 샘플에 더 집중하는 focal gamma 상향 조합을 확인합니다.",
            params={
                "class_weight_multipliers": stronger_multipliers,
                "balanced_sampler": True,
                "loss": "focal",
                "focal_gamma": 1.6,
                "label_smoothing": 0.0,
                "learning_rate": round(max(learning_rate * 0.75, 1e-5), 8),
                "early_stopping_patience": max(patience, 12),
            },
        ),
        TrialSpec(
            name="cross_entropy_no_smoothing",
            reason="검증 정확도만 보면 focal보다 plain cross entropy가 더 안정적인지 확인합니다.",
            params={
                "class_weight_multipliers": balanced_multipliers,
                "balanced_sampler": True,
                "loss": "cross_entropy",
                "label_smoothing": 0.0,
                "learning_rate": round(max(learning_rate * 0.9, 1e-5), 8),
                "early_stopping_patience": max(patience, 12),
            },
        ),
        TrialSpec(
            name="cross_entropy_smoothing",
            reason="라벨 노이즈가 있을 때를 대비해 약한 label smoothing을 둔 cross entropy를 비교합니다.",
            params={
                "class_weight_multipliers": balanced_multipliers,
                "balanced_sampler": True,
                "loss": "cross_entropy",
                "label_smoothing": 0.05,
                "learning_rate": learning_rate,
                "early_stopping_patience": max(patience, 12),
            },
        ),
        TrialSpec(
            name="wide_three_layer",
            reason="표현력이 부족한 경우를 대비해 hidden dimension과 recurrent layer를 함께 늘립니다.",
            params={
                "class_weight_multipliers": balanced_multipliers,
                "balanced_sampler": True,
                "loss": "focal",
                "focal_gamma": 1.0,
                "label_smoothing": 0.0,
                "hidden_dim": min(max(hidden_dim, 128) * 2, 384),
                "num_layers": 3,
                "dropout": round(min(max(dropout, 0.35), 0.55), 4),
                "learning_rate": round(max(learning_rate * 0.7, 1e-5), 8),
                "early_stopping_patience": max(patience, 12),
            },
        ),
        TrialSpec(
            name="narrow_low_dropout",
            reason="데이터가 작아 큰 모델이 불안정한 경우를 대비해 더 작은 모델도 비교합니다.",
            params={
                "class_weight_multipliers": shrink_multipliers_toward_one(balanced_multipliers, factor=0.7),
                "balanced_sampler": True,
                "loss": "focal",
                "focal_gamma": 0.8,
                "label_smoothing": 0.0,
                "hidden_dim": max(min(hidden_dim, 128), 96),
                "num_layers": 2,
                "dropout": round(max(dropout - 0.2, 0.1), 4),
                "weight_decay": round(max(weight_decay * 0.3, 0.00005), 8),
                "early_stopping_patience": max(patience, 12),
            },
        ),
    ]
    if bool(auto_tune_config.get("seed_sweep_enabled", True)):
        seed_base_params = {
            "class_weight_multipliers": base_multipliers,
            "balanced_sampler": True,
            "loss": "focal",
            "focal_gamma": float(base_training.get("focal_gamma", 1.0) or 1.0),
            "label_smoothing": float(base_training.get("label_smoothing", 0.0) or 0.0),
            "learning_rate": float(base_training.get("learning_rate", learning_rate) or learning_rate),
            "temporal_pooling": str(base_training.get("temporal_pooling", "mean") or "mean"),
            "early_stopping_patience": patience,
        }
        low_lr_seed_params = {
            **seed_base_params,
            "learning_rate": round(max(float(seed_base_params["learning_rate"]) * 0.6, 1e-5), 8),
        }
        for seed in seed_values:
            trials.append(
                TrialSpec(
                    name=f"seed_sweep_{seed}",
                    reason="같은 JSON 설정에서 초기화/샘플 순서 seed만 바꿔 validation 정확도 변동을 탐색합니다.",
                    params={**seed_base_params, "seed": int(seed)},
                )
            )
            trials.append(
                TrialSpec(
                    name=f"seed_sweep_low_lr_{seed}",
                    reason="best에 가까운 낮은 learning rate 설정에서 seed 변동을 탐색합니다.",
                    params={**low_lr_seed_params, "seed": int(seed)},
                )
            )
    return trials


def extract_metric_feedback(metrics: dict, *, labels: list[str], target_recall: float) -> dict:
    final_validation = metrics.get("final_validation") if isinstance(metrics.get("final_validation"), dict) else {}
    per_class = final_validation.get("per_class") if isinstance(final_validation.get("per_class"), list) else []
    confusion = final_validation.get("confusion_matrix") if isinstance(final_validation.get("confusion_matrix"), list) else []
    low_recall_labels: list[str] = []
    missing_prediction_labels: list[str] = []
    predicted_counts = predicted_counts_from_confusion(confusion, labels=labels)
    support_counts = support_counts_from_confusion(confusion, labels=labels)

    by_label = {str(row.get("label")): row for row in per_class if isinstance(row, dict)}
    for label in labels:
        row = by_label.get(label) or {}
        recall = safe_float(row.get("recall"))
        predicted = predicted_counts.get(label, 0)
        support = int(row.get("support", support_counts.get(label, 0)) or 0)
        if support > 0 and predicted <= 0:
            missing_prediction_labels.append(label)
        if support > 0 and (recall is None or recall < target_recall):
            low_recall_labels.append(label)

    if not low_recall_labels and not missing_prediction_labels:
        analysis = metrics.get("validation_error_analysis")
        if isinstance(analysis, dict):
            for label in analysis.get("missing_prediction_classes") or []:
                if label in labels:
                    missing_prediction_labels.append(label)
            for row in analysis.get("low_recall_classes") or []:
                label = str(row.get("label") or "")
                if label in labels:
                    low_recall_labels.append(label)

    overpredicted_label = None
    overpredicted_ratio = 0.0
    for label in labels:
        support = max(int(support_counts.get(label, 0) or 0), 1)
        predicted = int(predicted_counts.get(label, 0) or 0)
        ratio = predicted / support
        if ratio > overpredicted_ratio:
            overpredicted_label = label
            overpredicted_ratio = ratio

    missing_or_low = []
    for label in [*missing_prediction_labels, *low_recall_labels]:
        if label not in missing_or_low:
            missing_or_low.append(label)
    return {
        "missing_prediction_labels": missing_prediction_labels,
        "low_recall_labels": low_recall_labels,
        "missing_or_low_recall_labels": missing_or_low,
        "overpredicted_label": overpredicted_label if overpredicted_ratio >= 1.5 else None,
        "predicted_counts": predicted_counts,
        "support_counts": support_counts,
    }


def build_recovery_multipliers(
    *,
    labels: list[str],
    base: dict[str, float],
    boost_labels: list[str],
    suppress_label: str | None,
    boost: float,
    suppress: float,
    max_multiplier: float = 3.0,
) -> dict[str, float]:
    multipliers = {label: round(float(base.get(label, 1.0) or 1.0), 4) for label in labels}
    for label in boost_labels:
        if label in multipliers:
            multipliers[label] = round(min(max(multipliers[label] * boost, boost), max_multiplier), 4)
    if suppress_label in multipliers and suppress_label not in boost_labels:
        multipliers[suppress_label] = round(max(multipliers[suppress_label] * suppress, 0.5), 4)
    return {label: value for label, value in multipliers.items() if abs(value - 1.0) > 1e-6}


def build_trial_result(
    *,
    trial_number: int,
    trial: TrialSpec,
    output_dir: Path,
    metrics: dict,
    started_at: str,
    finished_at: str,
    target_metric: str = DEFAULT_TARGET_METRIC,
) -> dict[str, Any]:
    final_validation = metrics.get("final_validation") if isinstance(metrics.get("final_validation"), dict) else {}
    per_class = final_validation.get("per_class") if isinstance(final_validation.get("per_class"), list) else []
    missing_prediction_classes = []
    analysis = metrics.get("validation_error_analysis")
    if isinstance(analysis, dict):
        missing_prediction_classes = [str(label) for label in analysis.get("missing_prediction_classes") or []]
    min_recall = min(
        [float(row.get("recall")) for row in per_class if isinstance(row, dict) and safe_float(row.get("recall")) is not None],
        default=0.0,
    )
    score = objective_score_for_metric(metrics, target_metric=target_metric)
    return {
        "trial_number": trial_number,
        "name": trial.name,
        "reason": trial.reason,
        "params": trial.params,
        "score": score,
        "target_metric": target_metric,
        "best_val_macro_f1": safe_float(metrics.get("best_val_macro_f1")) or 0.0,
        "val_accuracy": safe_float(final_validation.get("accuracy")) or 0.0,
        "balanced_accuracy": safe_float(final_validation.get("balanced_accuracy")) or 0.0,
        "macro_f1": safe_float(final_validation.get("macro_f1")) or 0.0,
        "min_recall": round(min_recall, 6),
        "missing_prediction_classes": missing_prediction_classes,
        "best_epoch": metrics.get("best_epoch"),
        "stopped_early": bool(metrics.get("stopped_early")),
        "started_at": started_at,
        "finished_at": finished_at,
        "output_dir": str(output_dir),
        "metrics_path": str(output_dir / "metrics.json"),
        "model_path": str(output_dir / "best_action_model.pt"),
        "metrics": metrics,
    }


def objective_score(metrics: dict) -> float:
    return objective_score_for_metric(metrics, target_metric=DEFAULT_TARGET_METRIC)


def objective_score_for_metric(metrics: dict, *, target_metric: str = DEFAULT_TARGET_METRIC) -> float:
    final_validation = metrics.get("final_validation") if isinstance(metrics.get("final_validation"), dict) else {}
    macro_f1 = safe_float(final_validation.get("macro_f1")) or safe_float(metrics.get("best_val_macro_f1")) or 0.0
    accuracy = safe_float(final_validation.get("accuracy")) or 0.0
    balanced_accuracy = safe_float(final_validation.get("balanced_accuracy")) or 0.0
    per_class = final_validation.get("per_class") if isinstance(final_validation.get("per_class"), list) else []
    recalls = [
        float(row.get("recall"))
        for row in per_class
        if isinstance(row, dict) and safe_float(row.get("recall")) is not None
    ]
    min_recall = min(recalls, default=0.0)
    missing_prediction_count = 0
    analysis = metrics.get("validation_error_analysis")
    if isinstance(analysis, dict):
        missing_prediction_count = len(analysis.get("missing_prediction_classes") or [])
    normalized_target = str(target_metric or DEFAULT_TARGET_METRIC).strip().lower()
    if normalized_target == "accuracy":
        score = (0.55 * accuracy) + (0.25 * macro_f1) + (0.15 * balanced_accuracy) + (0.05 * min_recall)
    elif normalized_target == "macro_f1":
        score = (0.70 * macro_f1) + (0.20 * balanced_accuracy) + (0.10 * min_recall)
    else:
        score = (0.50 * macro_f1) + (0.30 * balanced_accuracy) + (0.15 * min_recall) + (0.05 * accuracy)
    score -= 0.05 * missing_prediction_count
    return round(score, 6)


def select_best_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    return max(
        results,
        key=lambda result: (
            safe_float(result.get("score")) or 0.0,
            safe_float(result.get("macro_f1")) or 0.0,
            safe_float(result.get("balanced_accuracy")) or 0.0,
        ),
    )


def promote_trial_artifacts(trial_output_dir: Path, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name in PROMOTED_ARTIFACT_NAMES:
        source = trial_output_dir / name
        if not source.exists():
            continue
        shutil.copy2(source, artifacts_dir / name)


def write_best_training_config(
    *,
    output_dir: Path,
    config: dict,
    best_result: dict,
    target_metric: str,
) -> Path:
    path = output_dir / "best_training_config.json"
    params = best_result.get("params") if isinstance(best_result.get("params"), dict) else {}
    effective_training = best_result.get("metrics", {}).get("auto_tune_effective_training")
    if not isinstance(effective_training, dict):
        effective_training = sanitize_training_payload(merged_training_config(config.get("training", {}), params))
    update = build_training_update_payload(params)
    payload = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "target_metric": target_metric,
        "best_result": compact_result(best_result),
        "training_update": update,
        "effective_training": effective_training,
        "notes": [
            "training_update는 기존 config JSON에 반영할 최소 변경값입니다.",
            "effective_training은 trial 실행 시 실제로 사용된 병합 설정입니다.",
        ],
    }
    write_json_atomic(path, payload)
    return path


def update_source_config_with_best(
    config_path: Path,
    *,
    best_result: dict,
    best_config_path: Path,
    target_metric: str,
) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    if not isinstance(raw_config, dict):
        raise RuntimeError(f"config JSON 최상위 구조가 객체가 아닙니다: {config_path}")

    params = best_result.get("params") if isinstance(best_result.get("params"), dict) else {}
    training_update = build_training_update_payload(params)
    training = raw_config.setdefault("training", {})
    if not isinstance(training, dict):
        training = {}
        raw_config["training"] = training
    for key, value in training_update.items():
        training[key] = copy.deepcopy(value)

    adaptive = training.setdefault("adaptive_class_weighting", {})
    if isinstance(adaptive, dict):
        adaptive["enabled"] = False
    else:
        training["adaptive_class_weighting"] = {"enabled": False}

    continual = raw_config.setdefault("continual_learning", {})
    if isinstance(continual, dict):
        continual["resume_from_best"] = False

    auto_tune = raw_config.setdefault("auto_tune", {})
    if isinstance(auto_tune, dict):
        auto_tune["target_metric"] = target_metric
        auto_tune["last_best_config_path"] = str(best_config_path)
        auto_tune["last_best_trial"] = compact_result(best_result)

    write_json_atomic(config_path, raw_config)


def build_training_update_payload(params: dict) -> dict:
    update = {}
    for key, value in (params or {}).items():
        if key in TUNABLE_TRAINING_KEYS:
            update[key] = copy.deepcopy(value)
    return update


def build_summary_payload(
    *,
    run_id: str,
    state: str,
    output_dir: Path,
    results: list[dict[str, Any]],
    promoted: bool,
    target_metric: str = DEFAULT_TARGET_METRIC,
    best_config_path: Path | None = None,
    config_updated: bool = False,
    ensemble_result: dict | None = None,
    ensemble_error: str | None = None,
    hybrid_result: dict | None = None,
    hybrid_error: str | None = None,
) -> dict:
    compact_results = [compact_result(result) for result in results]
    best_result = select_best_result(results)
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "state": state,
        "updated_at": utc_now_iso(),
        "output_dir": str(output_dir),
        "target_metric": target_metric,
        "promoted": promoted,
        "best_config_path": str(best_config_path) if best_config_path else None,
        "config_updated": config_updated,
        "best_result": compact_result(best_result) if best_result else None,
        "trials": compact_results,
    }
    if ensemble_result is not None:
        payload["ensemble_result"] = compact_ensemble_result(ensemble_result)
    if ensemble_error:
        payload["ensemble_error"] = ensemble_error
    if hybrid_result is not None:
        payload["hybrid_result"] = compact_hybrid_result(hybrid_result)
    if hybrid_error:
        payload["hybrid_error"] = hybrid_error
    return payload


def compact_result(result: dict[str, Any] | None) -> dict | None:
    if not result:
        return None
    return {
        key: result.get(key)
        for key in (
            "trial_number",
            "name",
            "reason",
            "score",
            "target_metric",
            "best_val_macro_f1",
            "macro_f1",
            "balanced_accuracy",
            "val_accuracy",
            "min_recall",
            "missing_prediction_classes",
            "best_epoch",
            "stopped_early",
            "started_at",
            "finished_at",
            "output_dir",
            "metrics_path",
            "model_path",
            "params",
        )
    }


def compact_ensemble_result(result: dict[str, Any] | None) -> dict | None:
    if not isinstance(result, dict):
        return None
    best = result.get("best_result") if isinstance(result.get("best_result"), dict) else {}
    final_validation = best.get("metrics", {}).get("final_validation", {})
    return {
        "summary_path": result.get("summary_path"),
        "output_dir": result.get("output_dir"),
        "promoted": bool(result.get("promoted")),
        "score": best.get("score"),
        "target_metric": best.get("target_metric"),
        "weight_mode": best.get("weight_mode"),
        "class_bias": best.get("class_bias"),
        "weights": best.get("weights"),
        "members": best.get("members"),
        "val_accuracy": final_validation.get("accuracy"),
        "macro_f1": final_validation.get("macro_f1"),
        "balanced_accuracy": final_validation.get("balanced_accuracy"),
        "test_accuracy": best.get("test_validation", {}).get("accuracy")
        if isinstance(best.get("test_validation"), dict)
        else None,
    }


def compact_hybrid_result(result: dict[str, Any] | None) -> dict | None:
    if not isinstance(result, dict):
        return None
    best = result.get("best_result") if isinstance(result.get("best_result"), dict) else {}
    final_validation = best.get("metrics", {}).get("final_validation", {})
    test_validation = best.get("test_validation") if isinstance(best.get("test_validation"), dict) else {}
    return {
        "summary_path": result.get("summary_path"),
        "output_dir": result.get("output_dir"),
        "promoted": bool(result.get("promoted")),
        "feature_model": best.get("feature_model"),
        "feature_weight": best.get("feature_weight"),
        "neural_weight": best.get("neural_weight"),
        "class_bias": best.get("class_bias"),
        "val_accuracy": final_validation.get("accuracy"),
        "macro_f1": final_validation.get("macro_f1"),
        "balanced_accuracy": final_validation.get("balanced_accuracy"),
        "test_accuracy": test_validation.get("accuracy"),
    }


def write_auto_tune_status(
    path: Path,
    *,
    run_id: str,
    state: str,
    current_trial: int,
    max_trials: int,
    trial: TrialSpec | None,
    results: list[dict[str, Any]],
    error: str | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "state": state,
        "updated_at": utc_now_iso(),
        "current_trial": current_trial,
        "max_trials": max_trials,
        "trial": trial_to_payload(trial) if trial else None,
        "best_result": compact_result(select_best_result(results)),
        "completed_trials": [compact_result(result) for result in results],
    }
    if error:
        payload["error"] = error
    write_json_atomic(path, payload)


def build_completion_message(best_result: dict | None, *, promoted: bool) -> str:
    if not best_result:
        return "자동 학습 개선이 완료됐지만 성공한 trial이 없습니다."
    action = "메인 모델로 승격했습니다" if promoted else "결과를 저장했습니다"
    return (
        f"자동 학습 개선 완료: best trial {best_result.get('trial_number')} "
        f"{best_result.get('name')} score={best_result.get('score')} "
        f"macro_f1={best_result.get('macro_f1'):.4f}. {action}."
    )


def merged_training_config(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base) if isinstance(base, dict) else {}
    for key, value in (override or {}).items():
        if key == "resume_from_best":
            continue
        merged[key] = copy.deepcopy(value)
    return merged


def normalize_multiplier_map(value: Any, *, labels: list[str]) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    normalized = {}
    label_set = set(labels)
    for key, raw in value.items():
        label = str(key).strip()
        if label not in label_set:
            continue
        try:
            multiplier = float(raw)
        except (TypeError, ValueError):
            continue
        if multiplier > 0 and abs(multiplier - 1.0) > 1e-6:
            normalized[label] = round(multiplier, 4)
    return normalized


def shrink_multipliers_toward_one(multipliers: dict[str, float], *, factor: float) -> dict[str, float]:
    factor = max(0.0, min(float(factor), 1.0))
    shrunk = {}
    for label, raw in (multipliers or {}).items():
        value = float(raw)
        next_value = 1.0 + ((value - 1.0) * factor)
        if abs(next_value - 1.0) > 1e-6:
            shrunk[str(label)] = round(next_value, 4)
    return shrunk


def normalize_seed_list(value: Any, *, default: list[int]) -> list[int]:
    raw_values = value if isinstance(value, list) else default
    seeds = []
    seen = set()
    for raw in raw_values:
        try:
            seed = int(raw)
        except (TypeError, ValueError):
            continue
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return seeds or list(default)


def sanitize_training_payload(training: dict) -> dict:
    payload = {}
    for key, value in (training or {}).items():
        if key in TUNABLE_TRAINING_KEYS or key in {
            "epochs",
            "batch_size",
            "eval_batch_size",
            "device",
            "amp",
            "amp_dtype",
            "compile_model",
            "compile_backend",
            "dataset_cache_size",
            "num_workers",
            "pin_memory",
            "prefetch_factor",
            "persistent_workers",
            "grad_clip_norm",
            "seed",
            "deterministic",
        }:
            payload[key] = copy.deepcopy(value)
    return payload


def predicted_counts_from_confusion(confusion: list, *, labels: list[str]) -> dict[str, int]:
    counts = {label: 0 for label in labels}
    for row in confusion:
        if not isinstance(row, list):
            continue
        for index, value in enumerate(row):
            if index < len(labels):
                counts[labels[index]] += int(value or 0)
    return counts


def support_counts_from_confusion(confusion: list, *, labels: list[str]) -> dict[str, int]:
    counts = {label: 0 for label in labels}
    for index, row in enumerate(confusion):
        if index < len(labels) and isinstance(row, list):
            counts[labels[index]] = sum(int(value or 0) for value in row)
    return counts


def dedupe_trial_specs(trials: list[TrialSpec]) -> list[TrialSpec]:
    deduped = []
    seen = set()
    for trial in trials:
        signature = trial_signature(trial.params)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(trial)
    return deduped


def trial_signature(params: dict) -> str:
    normalized = json.dumps(params or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return normalized


def trial_to_payload(trial: TrialSpec | None) -> dict | None:
    if trial is None:
        return None
    return {
        "name": trial.name,
        "reason": trial.reason,
        "params": trial.params,
    }


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str))
        handle.write("\n")


def resolve_auto_tune_dir(paths: dict, base_dir: Path, configured: str | None) -> Path:
    if configured:
        candidate = Path(str(configured)).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    default_dir = paths["artifacts_dir"] / "auto_tune"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir


def slugify(value: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized[:80] or "trial"


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def utc_now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat()


if __name__ == "__main__":
    main()
