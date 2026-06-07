from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from action_training_pipeline import load_config, resolve_paths
from reporting import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple fight BiLSTM seeds and promote the best checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reuse-split", action="store_true")
    return parser.parse_args()


def resolve_seed_list(config: dict) -> list[int]:
    bilstm = config.get("fight_bilstm") if isinstance(config.get("fight_bilstm"), dict) else {}
    raw = bilstm.get("seeds")
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list):
        values = raw
    else:
        values = [42, 43, 44]
    seeds: list[int] = []
    for value in values:
        try:
            seed = int(value)
        except (TypeError, ValueError):
            continue
        if seed not in seeds:
            seeds.append(seed)
    return seeds or [42, 43, 44]


def metric_score(metrics: dict) -> tuple[float, float, float, float]:
    positive_label = str(metrics.get("positive_label") or "violence")
    if str(metrics.get("recommended_result") or "") == "fusion":
        holdout = metrics.get("recommended_holdout_test") if isinstance(metrics.get("recommended_holdout_test"), dict) else {}
        validation = metrics.get("recommended_validation") if isinstance(metrics.get("recommended_validation"), dict) else {}
    else:
        holdout = metrics.get("holdout_test") if isinstance(metrics.get("holdout_test"), dict) else {}
        validation = metrics.get("final_validation") if isinstance(metrics.get("final_validation"), dict) else {}
    if not holdout and isinstance(metrics.get("fusion_holdout_test"), dict):
        holdout = metrics.get("fusion_holdout_test") or {}
    if not validation and isinstance(metrics.get("fusion_validation"), dict):
        validation = metrics.get("fusion_validation") or {}
    normal_f1 = class_metric(holdout, "normal", "f1")
    positive_f1 = class_metric(holdout, positive_label, "f1")
    return (
        float(holdout.get("macro_f1") or 0.0),
        float(normal_f1),
        float(positive_f1),
        float(validation.get("macro_f1") or 0.0),
    )


def class_metric(metrics: dict, label: str, key: str) -> float:
    for row in metrics.get("per_class") or []:
        if isinstance(row, dict) and str(row.get("label") or "") == label:
            try:
                return float(row.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def write_seed_config(base_config: dict, *, seed: int, output_dir: Path, path: Path) -> None:
    config = json.loads(json.dumps(base_config))
    config.setdefault("paths", {})["artifacts_dir"] = str(output_dir)
    config.setdefault("fight_bilstm", {})["seed"] = int(seed)
    config["fight_bilstm"].pop("seeds", None)
    write_json_atomic(path, config)


def copy_tree_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name.startswith("seed_"):
            continue
        target = destination / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def run_seed(config_path: Path, *, reuse_split: bool) -> None:
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(Path(__file__).resolve().parent / "train_fight_bilstm.py"),
        "--config",
        str(config_path),
    ]
    if reuse_split:
        command.append("--reuse-split")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    paths = resolve_paths(config, config_path.parent)
    output_dir = paths["artifacts_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = resolve_seed_list(config)
    runtime_dir = paths["workspace_dir"] / "runtime_configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    best_run: dict | None = None

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_config_path = runtime_dir / f"{config_path.stem}_seed_{seed}.json"
        write_seed_config(config, seed=seed, output_dir=seed_dir, path=seed_config_path)
        print(f"[bilstm-sweep] seed {seed} start -> {seed_dir}", flush=True)
        run_seed(seed_config_path, reuse_split=args.reuse_split)
        metrics_path = seed_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig")) if metrics_path.exists() else {}
        run = {
            "seed": seed,
            "artifacts_dir": str(seed_dir),
            "metrics_path": str(metrics_path),
            "score": metric_score(metrics),
            "holdout_test": metrics.get("holdout_test", {}),
            "fusion_holdout_test": metrics.get("fusion_holdout_test", {}),
            "recommended_result": metrics.get("recommended_result"),
            "recommended_holdout_test": metrics.get("recommended_holdout_test", {}),
            "final_validation": metrics.get("final_validation", {}),
        }
        runs.append(run)
        if best_run is None or tuple(run["score"]) > tuple(best_run["score"]):
            best_run = run

    if best_run is None:
        raise RuntimeError("No successful BiLSTM seed run was produced.")
    best_dir = Path(str(best_run["artifacts_dir"]))
    copy_tree_contents(best_dir, output_dir)
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "seeds": seeds,
        "best_seed": best_run["seed"],
        "best_score": best_run["score"],
        "best_artifacts_dir": str(best_dir),
        "runs": runs,
    }
    write_json_atomic(output_dir / "seed_sweep_summary.json", summary)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        metrics["seed_sweep"] = summary
        write_json_atomic(metrics_path, metrics)
    print(f"[bilstm-sweep] best seed {best_run['seed']} score={best_run['score']}", flush=True)


if __name__ == "__main__":
    main()
