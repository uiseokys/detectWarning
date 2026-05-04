from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_WORKSPACE = Path("../training_data/action_pipeline_aihub")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except OSError:
        return []
    return rows


def count_labels(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        label = row.get("target_label") or row.get("label") or row.get("label_name")
        if label in (None, ""):
            label_idx = row.get("label_idx")
            label = f"idx:{label_idx}" if label_idx is not None else "unknown"
        counts[str(label)] += 1
    return counts


def summarize_manifest_set(workspace: Path) -> dict[str, Any]:
    manifests_dir = workspace / "manifests"
    manifest_specs = {
        "raw": manifests_dir / "cumulative_raw_items.jsonl",
        "train": manifests_dir / "cumulative_split_train.jsonl",
        "val": manifests_dir / "cumulative_split_val.jsonl",
        "test": manifests_dir / "cumulative_split_test.jsonl",
        "prepared_train": manifests_dir / "cumulative_prepared_train.jsonl",
        "prepared_val": manifests_dir / "cumulative_prepared_val.jsonl",
        "prepared_test": manifests_dir / "cumulative_prepared_test.jsonl",
    }
    summary: dict[str, Any] = {}
    for split, path in manifest_specs.items():
        rows = read_jsonl(path)
        summary[split] = {
            "path": str(path),
            "total": len(rows),
            "by_label": dict(sorted(count_labels(rows).items())),
        }
    return summary


def build_label_retention(summary: dict[str, Any]) -> list[dict[str, Any]]:
    raw_counts = Counter(summary.get("raw", {}).get("by_label") or {})
    prepared_counts: Counter[str] = Counter()
    for split in ("prepared_train", "prepared_val", "prepared_test"):
        prepared_counts.update(summary.get(split, {}).get("by_label") or {})
    labels = sorted(set(raw_counts) | set(prepared_counts))
    rows = []
    for label in labels:
        raw_count = int(raw_counts.get(label, 0) or 0)
        prepared_count = int(prepared_counts.get(label, 0) or 0)
        retention = prepared_count / raw_count if raw_count > 0 else None
        rows.append(
            {
                "label": label,
                "raw": raw_count,
                "prepared": prepared_count,
                "retention": round(retention, 4) if retention is not None else None,
            }
        )
    return rows


def best_and_latest(metrics: dict[str, Any]) -> dict[str, Any]:
    history = metrics.get("history")
    history = history if isinstance(history, list) else []
    latest = history[-1] if history else {}
    best_epoch = metrics.get("best_epoch")
    best_history = {}
    for row in history:
        if isinstance(row, dict) and row.get("epoch") == best_epoch:
            best_history = row
            break
    final_validation = metrics.get("final_validation")
    final_validation = final_validation if isinstance(final_validation, dict) else {}
    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": metrics.get("best_val_macro_f1"),
        "best_history": best_history,
        "latest_epoch": latest.get("epoch"),
        "latest": latest,
        "final_validation_accuracy": final_validation.get("accuracy"),
        "final_validation_macro_f1": final_validation.get("macro_f1"),
        "final_validation_loss": final_validation.get("loss"),
    }


def build_confusion_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    final_validation = metrics.get("final_validation")
    final_validation = final_validation if isinstance(final_validation, dict) else {}
    labels = metrics.get("labels") or []
    matrix = final_validation.get("confusion_matrix") or []
    pairs = []
    if isinstance(labels, list) and isinstance(matrix, list):
        for true_idx, row in enumerate(matrix):
            if not isinstance(row, list):
                continue
            true_label = labels[true_idx] if true_idx < len(labels) else str(true_idx)
            for pred_idx, count in enumerate(row):
                if true_idx == pred_idx:
                    continue
                try:
                    count_int = int(count or 0)
                except (TypeError, ValueError):
                    count_int = 0
                if count_int <= 0:
                    continue
                pred_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
                pairs.append({"from": true_label, "to": pred_label, "count": count_int})
    pairs.sort(key=lambda item: item["count"], reverse=True)
    return {"top_confusions": pairs[:10]}


def build_audit(workspace: Path) -> dict[str, Any]:
    artifacts_dir = workspace / "artifacts"
    metrics = read_json(artifacts_dir / "metrics.json")
    progress = read_json(artifacts_dir / "training_progress.json")
    manifest_summary = summarize_manifest_set(workspace)
    error_analysis_path = metrics.get("validation_error_analysis_path") or progress.get("validation_error_analysis_path")
    false_negative_path = metrics.get("false_negative_examples_path") or progress.get("false_negative_examples_path")
    confusion_pair_path = metrics.get("confusion_pair_examples_path") or progress.get("confusion_pair_examples_path")
    return {
        "workspace": str(workspace),
        "best_vs_latest": best_and_latest(metrics or progress),
        "confusion_summary": build_confusion_summary(metrics or progress),
        "label_retention": build_label_retention(manifest_summary),
        "manifests": manifest_summary,
        "artifact_paths": {
            "metrics": str(artifacts_dir / "metrics.json"),
            "training_progress": str(artifacts_dir / "training_progress.json"),
            "validation_error_analysis": str(error_analysis_path or artifacts_dir / "validation_error_analysis.json"),
            "false_negative_examples": str(false_negative_path or artifacts_dir / "false_negative_examples.json"),
            "confusion_pair_examples": str(confusion_pair_path or artifacts_dir / "confusion_pair_examples.json"),
        },
    }


def print_audit(audit: dict[str, Any]) -> None:
    best = audit.get("best_vs_latest") or {}
    print("\n== Best vs latest ==")
    print(f"best_epoch: {best.get('best_epoch')}")
    print(f"best_val_macro_f1: {best.get('best_val_macro_f1')}")
    print(f"latest_epoch: {best.get('latest_epoch')}")
    print(f"final_validation_accuracy: {best.get('final_validation_accuracy')}")
    print(f"final_validation_macro_f1: {best.get('final_validation_macro_f1')}")
    print(f"final_validation_loss: {best.get('final_validation_loss')}")

    print("\n== Top confusions ==")
    for row in (audit.get("confusion_summary") or {}).get("top_confusions") or []:
        print(f"{row['from']} -> {row['to']}: {row['count']}")

    print("\n== Label retention ==")
    for row in audit.get("label_retention") or []:
        retention = row.get("retention")
        retention_text = "-" if retention is None else f"{retention * 100:.1f}%"
        print(f"{row['label']}: raw={row['raw']} prepared={row['prepared']} retention={retention_text}")

    print("\n== Useful artifact paths ==")
    for key, value in (audit.get("artifact_paths") or {}).items():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize action training results, dataset retention, and confusion pairs.")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="Action training workspace directory")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    audit = build_audit(workspace)
    print_audit(audit)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(audit, handle, ensure_ascii=False, indent=2)
        print(f"\nwrote: {output_path}")


if __name__ == "__main__":
    main()
