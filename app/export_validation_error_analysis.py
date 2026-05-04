from __future__ import annotations

import argparse
import json
from pathlib import Path

from action_model import _build_validation_error_analysis
from reporting import write_json_atomic


DEFAULT_METRICS_PATH = Path("training_data/action_pipeline_aihub/artifacts/metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export validation false-negative and confusion-pair analysis from an existing metrics.json."
    )
    parser.add_argument(
        "--metrics",
        default=str(DEFAULT_METRICS_PATH),
        help="Path to metrics.json. Defaults to the AIHub training workspace metrics file.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for exported JSON files. Defaults to the metrics file directory.",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics).resolve()
    if not metrics_path.exists():
        raise SystemExit(f"metrics.json not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if not isinstance(metrics, dict):
        raise SystemExit(f"metrics.json must contain a JSON object: {metrics_path}")

    labels = [str(label) for label in metrics.get("labels") or [] if str(label or "").strip()]
    final_validation = metrics.get("final_validation") if isinstance(metrics.get("final_validation"), dict) else {}
    if not labels:
        labels = [
            str(row.get("label"))
            for row in final_validation.get("per_class", [])
            if isinstance(row, dict) and row.get("label") is not None
        ]
    if not labels:
        raise SystemExit("Could not resolve labels from metrics.json.")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else metrics_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    error_analysis_path = output_dir / "validation_error_analysis.json"
    false_negative_path = output_dir / "false_negative_examples.json"
    confusion_pair_path = output_dir / "confusion_pair_examples.json"

    error_analysis = _build_validation_error_analysis(final_validation, labels=labels)
    write_json_atomic(error_analysis_path, error_analysis)
    write_json_atomic(false_negative_path, error_analysis.get("false_negative_examples", {}))
    write_json_atomic(confusion_pair_path, error_analysis.get("confusion_pair_examples", []))

    print(f"validation_error_analysis={error_analysis_path}")
    print(f"false_negative_examples={false_negative_path}")
    print(f"confusion_pair_examples={confusion_pair_path}")


if __name__ == "__main__":
    main()
