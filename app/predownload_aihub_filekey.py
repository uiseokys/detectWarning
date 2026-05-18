from __future__ import annotations

import argparse
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from action_training_pipeline import (
    aihub_predownload_marker,
    build_aihub_predownload_cache_dir,
    build_aihub_shell_command,
    resolve_aihub_api_key,
    resolve_aihub_shell_path,
    resolve_paths,
)
from reporting import write_json_atomic
from training_config import load_action_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one AIHub filekey into the predownload cache.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_action_training_config(config_path)
    paths = resolve_paths(config, config_path.parent)
    shell_config = config.get("aihub_shell", {}) if isinstance(config.get("aihub_shell"), dict) else {}
    shell_path = resolve_aihub_shell_path(shell_config)
    api_key = resolve_aihub_api_key(shell_config)
    mode = str(shell_config.get("mode", "d")).strip()
    datasetkey = shell_config.get("datasetkey")
    datapackagekey = shell_config.get("datapackagekey")
    filekey = shell_config.get("filekey")
    if not datasetkey or not filekey:
        raise RuntimeError("datasetkey and filekey are required for predownload.")

    cache_dir = build_aihub_predownload_cache_dir(paths, filekey=filekey)
    marker_path = aihub_predownload_marker(cache_dir)
    if marker_path.exists():
        print(f"[predownload] cache already complete: filekey={filekey} dir={cache_dir}", flush=True)
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    if any(path.name != marker_path.name for path in cache_dir.iterdir()):
        print(f"[predownload] resume incomplete cache: filekey={filekey} dir={cache_dir}", flush=True)
    command = build_aihub_shell_command(shell_path, api_key, mode=mode)
    command.extend(["-datasetkey", str(datasetkey)])
    if datapackagekey is not None:
        command.extend(["-datapckagekey", str(datapackagekey)])
    command.extend(["-filekey", str(filekey)])

    started_at = datetime.now(UTC).astimezone().isoformat()
    print(f"[predownload] start datasetkey={datasetkey} filekey={filekey} dir={cache_dir}", flush=True)
    print(f"[predownload] command: {' '.join(str(part) for part in command[:4])} ...", flush=True)
    result = subprocess.run(command, cwd=str(cache_dir), check=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)

    files = [path for path in cache_dir.rglob("*") if path.is_file() and path.name != marker_path.name]
    write_json_atomic(
        marker_path,
        {
            "datasetkey": str(datasetkey),
            "filekey": str(filekey),
            "started_at": started_at,
            "finished_at": datetime.now(UTC).astimezone().isoformat(),
            "file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files if path.exists()),
        },
    )
    print(f"[predownload] complete filekey={filekey} files={len(files)}", flush=True)


if __name__ == "__main__":
    main()
