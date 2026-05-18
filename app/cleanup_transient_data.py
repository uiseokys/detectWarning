from __future__ import annotations

import argparse
from pathlib import Path

from action_training_pipeline import cleanup_transient_job_data, load_config, resolve_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="원본/임시 다운로드 데이터를 정리하고 누적 feature/manifest는 유지합니다.")
    parser.add_argument("--config", default="configs/action_training.aihub_shell.example.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    paths = resolve_paths(config, config_path.parent)
    cleanup_transient_job_data(paths)
    print("[cleanup] raw/imported/extracted/current manifests cleaned")


if __name__ == "__main__":
    main()
