from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from action_model import train_action_classifier
from detector import FaceDetector, PersonDetector
from person_classifier import PersonPresenceFilter
from tracker import PersonTracker


CONFIRMED_PERSON_STATES = {"full_body_person", "upper_body_person"}


@dataclass
class DownloadedItem:
    item_id: str
    source_label: str
    target_label: str
    video_path: Path
    download_url: str
    metadata: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="API에서 행동 영상을 받아 pose 기반 행동 분류 학습까지 자동으로 수행합니다."
    )
    parser.add_argument(
        "--config",
        default="configs/action_training.example.json",
        help="학습 파이프라인 설정 JSON 경로",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=("all", "download", "prepare", "train"),
        help="실행할 단계",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    validate_source_config(config, config_path)
    paths = resolve_paths(config, config_path.parent)
    continual_config = get_continual_config(config)
    write_pipeline_status(
        paths,
        stage="starting",
        state="running",
        message="학습 파이프라인을 시작합니다.",
        config_path=str(config_path),
        stage_progress=0.0,
    )

    try:
        downloaded: list[DownloadedItem] = []
        split_manifests: dict[str, Path] | None = None
        training_manifests: dict[str, Path] | None = None

        if args.stage in {"all", "download"}:
            write_pipeline_status(
                paths,
                stage="download",
                state="running",
                message="API에서 영상을 다운로드하는 중입니다.",
                stage_progress=0.05,
            )
            downloaded = download_dataset(config, paths)
            split_manifests = split_dataset(downloaded, config, paths)
        elif args.stage == "prepare":
            split_manifests = load_existing_current_split_manifests(paths)

        if args.stage in {"all", "prepare"}:
            if split_manifests is None:
                split_manifests = load_existing_current_split_manifests(paths)
            write_pipeline_status(
                paths,
                stage="prepare",
                state="running",
                message="영상에서 pose 시퀀스를 추출하는 중입니다.",
                stage_progress=0.55,
            )
            prepared_manifests = prepare_pose_dataset(config, paths, split_manifests)
            training_manifests = update_cumulative_manifests(config, paths, split_manifests, prepared_manifests)
        elif args.stage == "train":
            training_manifests = load_training_manifests(paths, continual_enabled=continual_config["enabled"])

        if args.stage in {"all", "train"}:
            if training_manifests is None:
                training_manifests = load_training_manifests(paths, continual_enabled=continual_config["enabled"])
            write_pipeline_status(
                paths,
                stage="train",
                state="running",
                message="행동 분류 모델을 학습하는 중입니다.",
                stage_progress=0.8,
            )
            labels = get_target_labels(config)
            train_manifest = training_manifests["train"]
            val_manifest = training_manifests["val"]
            resume_from = None
            if continual_config["enabled"] and continual_config["resume_from_best"]:
                candidate_checkpoint = paths["artifacts_dir"] / "best_action_model.pt"
                if candidate_checkpoint.exists():
                    resume_from = candidate_checkpoint
            artifacts = train_action_classifier(
                train_manifest=train_manifest,
                val_manifest=val_manifest,
                output_dir=paths["artifacts_dir"],
                labels=labels,
                epochs=int(config.get("training", {}).get("epochs", 20)),
                batch_size=int(config.get("training", {}).get("batch_size", 16)),
                learning_rate=float(config.get("training", {}).get("learning_rate", 1e-3)),
                hidden_dim=int(config.get("training", {}).get("hidden_dim", 128)),
                num_layers=int(config.get("training", {}).get("num_layers", 2)),
                dropout=float(config.get("training", {}).get("dropout", 0.2)),
                num_workers=int(config.get("training", {}).get("num_workers", 0)),
                device=str(config.get("training", {}).get("device", "cuda")),
                progress_path=paths["training_progress"],
                resume_from=resume_from,
            )
            print(f"[train] best model: {artifacts.best_model_path}")
            print(f"[train] metrics: {artifacts.metrics_path}")
            print(f"[train] labels: {artifacts.labels_path}")

            if args.stage == "all" and continual_config["cleanup_raw_after_job"]:
                cleanup_transient_job_data(paths)

        write_pipeline_status(
            paths,
            stage="completed",
            state="completed",
            message="학습 파이프라인이 완료되었습니다.",
            stage_progress=1.0,
        )
    except Exception as exc:
        write_pipeline_status(
            paths,
            stage="error",
            state="error",
            message=str(exc),
            stage_progress=1.0,
        )
        raise


def resolve_paths(config: dict, base_dir: Path) -> dict:
    workspace_dir = (base_dir / config.get("paths", {}).get("workspace_dir", "training_data/action_pipeline")).resolve()
    raw_dir = workspace_dir / "raw_videos"
    import_dir = workspace_dir / "imported_dataset"
    extracted_dir = workspace_dir / "extracted_dataset"
    manifests_dir = workspace_dir / "manifests"
    prepared_dir = workspace_dir / "prepared_pose"
    artifacts_dir = workspace_dir / "artifacts"
    for path in (workspace_dir, raw_dir, import_dir, extracted_dir, manifests_dir, prepared_dir, artifacts_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "workspace_dir": workspace_dir,
        "raw_dir": raw_dir,
        "import_dir": import_dir,
        "extracted_dir": extracted_dir,
        "manifests_dir": manifests_dir,
        "prepared_dir": prepared_dir,
        "artifacts_dir": artifacts_dir,
        "pipeline_status": workspace_dir / "pipeline_status.json",
        "training_progress": artifacts_dir / "training_progress.json",
        "raw_manifest": manifests_dir / "cumulative_raw_items.jsonl",
        "split_train": manifests_dir / "cumulative_split_train.jsonl",
        "split_val": manifests_dir / "cumulative_split_val.jsonl",
        "split_test": manifests_dir / "cumulative_split_test.jsonl",
        "prepared_train": manifests_dir / "cumulative_prepared_train.jsonl",
        "prepared_val": manifests_dir / "cumulative_prepared_val.jsonl",
        "prepared_test": manifests_dir / "cumulative_prepared_test.jsonl",
        "current_raw_manifest": manifests_dir / "current_raw_items.jsonl",
        "current_split_train": manifests_dir / "current_split_train.jsonl",
        "current_split_val": manifests_dir / "current_split_val.jsonl",
        "current_split_test": manifests_dir / "current_split_test.jsonl",
        "current_prepared_train": manifests_dir / "current_prepared_train.jsonl",
        "current_prepared_val": manifests_dir / "current_prepared_val.jsonl",
        "current_prepared_test": manifests_dir / "current_prepared_test.jsonl",
        "continual_state": manifests_dir / "continual_state.json",
    }


def write_pipeline_status(paths: dict, *, stage: str, state: str, message: str, **extra) -> None:
    payload = {
        "stage": stage,
        "state": state,
        "message": message,
        "workspace_dir": str(paths["workspace_dir"]),
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        **extra,
    }
    with paths["pipeline_status"].open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def download_dataset(config: dict, paths: dict) -> list[DownloadedItem]:
    source_mode = str(config.get("dataset_source", "json_api")).strip().lower()
    if source_mode == "aihub_shell":
        return download_dataset_via_aihub_shell(config, paths)

    api_config = config["api"]
    dataset_config = config["dataset"]
    raw_manifest_path = paths["current_raw_manifest"]

    session = requests.Session()
    headers = build_headers(api_config)
    session.headers.update(headers)

    label_mapping = dataset_config.get("label_mapping", {})
    max_items_per_class = int(dataset_config.get("max_items_per_class", 0))
    per_class_counts: dict[str, int] = defaultdict(int)

    items = fetch_api_items(session, api_config)
    downloaded: list[DownloadedItem] = []

    with raw_manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for item in items:
            source_label = str(extract_field(item, api_config["fields"]["label"])).strip()
            target_label = label_mapping.get(source_label)
            if not target_label:
                continue

            if max_items_per_class > 0 and per_class_counts[target_label] >= max_items_per_class:
                continue

            download_url = build_download_url(item, api_config)
            if not download_url:
                continue

            item_id = str(extract_field(item, api_config["fields"]["id"]))
            filename = build_filename(item, api_config, download_url, item_id)
            safe_target_label = slugify(target_label)
            target_dir = paths["raw_dir"] / safe_target_label
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / filename

            download_to_file(session, download_url, target_path, timeout=float(api_config.get("timeout_seconds", 60.0)))

            downloaded_item = DownloadedItem(
                item_id=item_id,
                source_label=source_label,
                target_label=target_label,
                video_path=target_path.resolve(),
                download_url=download_url,
                metadata=item,
            )
            downloaded.append(downloaded_item)
            per_class_counts[target_label] += 1

            manifest_handle.write(
                json.dumps(
                    {
                        "item_id": downloaded_item.item_id,
                        "source_label": downloaded_item.source_label,
                        "target_label": downloaded_item.target_label,
                        "video_path": str(downloaded_item.video_path),
                        "download_url": downloaded_item.download_url,
                        "metadata": downloaded_item.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"[download] saved {len(downloaded)} items -> {raw_manifest_path}")
    return downloaded


def download_dataset_via_aihub_shell(config: dict, paths: dict) -> list[DownloadedItem]:
    shell_config = config.get("aihub_shell", {})
    shell_path = resolve_aihub_shell_path(shell_config)
    api_key = resolve_aihub_api_key(shell_config)
    mode = str(shell_config.get("mode", "d")).strip()
    datasetkey = shell_config.get("datasetkey")
    datapackagekey = shell_config.get("datapackagekey")
    filekey = shell_config.get("filekey")
    import_dir = paths["import_dir"]
    raw_manifest_path = paths["current_raw_manifest"]

    command = build_aihub_shell_command(shell_path, api_key, mode=mode)
    if datasetkey is not None and filekey:
        validate_aihub_filekeys(shell_path, api_key, datasetkey=datasetkey, requested_filekeys=filekey)
    if datasetkey is not None:
        command.extend(["-datasetkey", str(datasetkey)])
    if datapackagekey is not None:
        command.extend(["-datapckagekey", str(datapackagekey)])
    if filekey:
        if isinstance(filekey, list):
            command.extend(["-filekey", "{" + ",".join(str(item) for item in filekey) + "}"])
        else:
            command.extend(["-filekey", str(filekey)])

    write_pipeline_status(
        paths,
        stage="download",
        state="running",
        message="AIHub에서 분할 ZIP 데이터를 다운로드하는 중입니다.",
        stage_progress=0.08,
    )
    subprocess.run(command, cwd=str(import_dir), check=True)
    write_pipeline_status(
        paths,
        stage="download",
        state="running",
        message="다운로드한 분할 ZIP 조각을 병합하는 중입니다.",
        stage_progress=0.22,
    )
    merge_split_archives(import_dir)
    write_pipeline_status(
        paths,
        stage="download",
        state="running",
        message="병합된 ZIP 파일을 압축 해제하는 중입니다.",
        stage_progress=0.38,
    )
    source_root = extract_archives(import_dir, paths["extracted_dir"])
    write_pipeline_status(
        paths,
        stage="download",
        state="running",
        message="압축 해제된 영상 파일을 스캔하고 라벨을 정리하는 중입니다.",
        stage_progress=0.5,
    )
    downloaded = scan_local_video_dataset(config, paths, source_root=source_root)
    write_pipeline_status(
        paths,
        stage="download",
        state="running",
        message=f"압축 해제 영상 스캔이 완료되었습니다. {len(downloaded)}개 샘플을 찾았습니다.",
        stage_progress=0.55,
        discovered_items=len(downloaded),
    )
    if not downloaded:
        raise RuntimeError(
            "다운로드 후 학습용 영상 파일을 찾지 못했습니다.\n"
            "확인할 것:\n"
            "1. 입력한 filekey가 실제 승인된 분할 파일인지\n"
            "2. AIHub에서 해당 데이터셋 다운로드 승인이 완료되었는지\n"
            "3. 분할 압축 파일이 .zip.part* 형태로 정상 저장되었는지\n"
            "4. label_mapping의 한글 라벨명이 압축 해제 폴더명과 일치하는지"
        )

    with raw_manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for item in downloaded:
            manifest_handle.write(
                json.dumps(
                    {
                        "item_id": item.item_id,
                        "source_label": item.source_label,
                        "target_label": item.target_label,
                        "video_path": str(item.video_path),
                        "download_url": item.download_url,
                        "metadata": item.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"[download] aihubshell imported {len(downloaded)} videos -> {raw_manifest_path}")
    return downloaded


def merge_split_archives(import_dir: Path) -> list[Path]:
    part_groups: dict[Path, list[Path]] = defaultdict(list)
    pattern = re.compile(r"^(?P<base>.+\.zip)\.part(?P<part>.+)$", re.IGNORECASE)

    for path in import_dir.rglob("*"):
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        base_name = match.group("base")
        base_path = path.with_name(base_name)
        part_groups[base_path].append(path)

    merged_archives: list[Path] = []
    for base_path, parts in sorted(part_groups.items(), key=lambda item: str(item[0])):
        sorted_parts = sorted(parts, key=split_part_sort_key)
        if not sorted_parts:
            continue

        needs_merge = True
        if base_path.exists() and base_path.stat().st_size > 0:
            latest_part_mtime = max(part.stat().st_mtime for part in sorted_parts)
            if base_path.stat().st_mtime >= latest_part_mtime:
                needs_merge = False

        if needs_merge:
            with base_path.open("wb") as merged_handle:
                for part_path in sorted_parts:
                    with part_path.open("rb") as part_handle:
                        shutil.copyfileobj(part_handle, merged_handle, length=1024 * 1024)
            if base_path.stat().st_size == 0:
                raise RuntimeError(
                    f"분할 압축 병합 결과가 0바이트입니다: {base_path}\n"
                    "AIHub 안내처럼 filekey와 폴더 경로가 맞는지 다시 확인해 주세요."
                )

        merged_archives.append(base_path)

    return merged_archives


def split_part_sort_key(path: Path):
    match = re.search(r"\.part(.+)$", path.name, re.IGNORECASE)
    part_token = match.group(1) if match else path.name
    if part_token.isdigit():
        return (0, int(part_token))
    numeric = re.sub(r"[^0-9]", "", part_token)
    if numeric.isdigit():
        return (1, int(numeric), part_token)
    return (2, part_token)


def build_aihub_shell_command(shell_path: str, api_key: str, *, mode: str) -> list[str]:
    shell_candidate = Path(shell_path)
    if os.name == "nt" and is_probably_shell_script(shell_candidate):
        bash_path = resolve_windows_bash()
        if not bash_path:
            raise RuntimeError(
                "현재 aihubshell 파일이 Windows 실행 파일이 아니라 bash 스크립트입니다.\n"
                "확인할 것:\n"
                "1. Git Bash를 설치해서 bash.exe 를 사용할 수 있는지\n"
                "2. 또는 Windows용 aihubshell.exe 가 있는지\n"
                "3. 프로젝트 루트의 aihubshell 파일이 macOS/Linux용 스크립트가 아닌지"
            )
        return [bash_path, str(shell_candidate), "-mode", mode, "-aihubapikey", api_key]

    return [shell_path, "-mode", mode, "-aihubapikey", api_key]


def is_probably_shell_script(path: Path) -> bool:
    if path.suffix.lower() in {".exe", ".bat", ".cmd", ".com"}:
        return False
    try:
        header = path.read_bytes()[:128]
    except OSError:
        return False
    return header.startswith(b"#!") or b"/bin/bash" in header or b"/bin/sh" in header


def resolve_windows_bash() -> str | None:
    candidates = [
        shutil.which("bash"),
        shutil.which("bash.exe"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return str(candidate_path)
    return None


def validate_aihub_filekeys(shell_path: str, api_key: str, *, datasetkey, requested_filekeys) -> None:
    requested = normalize_requested_filekeys(requested_filekeys)
    if not requested:
        return

    try:
        payload = fetch_aihub_file_tree(datasetkey=datasetkey)
    except Exception as exc:
        print(f"[aihubshell] filekey 목록 검증을 건너뜁니다: {exc}")
        return

    available_entries = collect_aihub_file_entries(payload)
    if not available_entries:
        print("[aihubshell] filekey 목록이 비어 있어 검증을 건너뜁니다.")
        return

    available_keys = {entry["filekey"] for entry in available_entries}
    missing = [filekey for filekey in requested if filekey not in available_keys]
    if not missing:
        return

    examples = ", ".join(entry["filekey"] for entry in available_entries[:10])
    matched_names = "\n".join(
        f"- {entry['filekey']}: {entry.get('name', '-')}"
        for entry in available_entries[:10]
    )
    raise RuntimeError(
        "입력한 filekey가 AIHub 파일 목록에 없습니다.\n"
        f"- datasetkey: {datasetkey}\n"
        f"- 요청 filekey: {', '.join(missing)}\n"
        f"- 예시 filekey: {examples}\n"
        "아래 목록을 먼저 확인해 주세요:\n"
        f"{matched_names}"
    )


def normalize_requested_filekeys(requested_filekeys) -> list[str]:
    if requested_filekeys is None:
        return []
    if isinstance(requested_filekeys, list):
        values = requested_filekeys
    else:
        values = [requested_filekeys]
    normalized = []
    for value in values:
        text = str(value).strip()
        if text and text.lower() != "all":
            normalized.append(text)
    return normalized


def fetch_aihub_file_tree(*, datasetkey) -> dict | list:
    filetree_url = f"https://api.aihub.or.kr/info/{datasetkey}.do"
    response = requests.get(filetree_url, timeout=60)
    response.raise_for_status()
    merged_output = response.text.strip()
    payload_text = extract_json_payload(merged_output)
    if not payload_text:
        raise RuntimeError(
            "AIHub 파일 목록 조회 결과를 해석하지 못했습니다.\n"
            f"- datasetkey: {datasetkey}\n"
            f"- raw output: {merged_output[:1000] if merged_output else '(empty)'}\n"
            "AIHub 파일 목록 응답 형식이 예상과 다를 수 있습니다."
        )
    return json.loads(payload_text)


def extract_json_payload(text: str) -> str:
    candidates = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start >= 0 and end > start:
            candidates.append(text[start:end + 1])
    if not candidates:
        return ""
    return max(candidates, key=len)


def collect_aihub_file_entries(payload) -> list[dict]:
    entries: list[dict] = []

    def walk(node):
        if isinstance(node, dict):
            filekey = None
            for key in ("fileSn", "filesn", "fileKey", "filekey"):
                if key in node and node[key] not in (None, ""):
                    filekey = str(node[key]).strip()
                    break
            if filekey:
                entries.append(
                    {
                        "filekey": filekey,
                        "name": str(
                            node.get("fileNm")
                            or node.get("fileName")
                            or node.get("filePath")
                            or node.get("path")
                            or node.get("name")
                            or ""
                        ).strip(),
                    }
                )
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)

    unique: dict[str, dict] = {}
    for entry in entries:
        unique.setdefault(entry["filekey"], entry)
    return sorted(unique.values(), key=lambda item: item["filekey"])


def scan_local_video_dataset(config: dict, paths: dict, source_root: Path | None = None) -> list[DownloadedItem]:
    dataset_config = config["dataset"]
    import_dir = source_root or paths["import_dir"]
    raw_dir = paths["raw_dir"]
    label_mapping = dataset_config.get("label_mapping", {})
    extensions = tuple(
        ext.lower()
        for ext in dataset_config.get("video_extensions", [".mp4", ".avi", ".mov", ".mkv", ".wmv"])
    )

    scanned: list[DownloadedItem] = []
    unlabeled_examples: list[str] = []
    candidate_video_count = 0
    for video_path in sorted(import_dir.rglob("*")):
        if not video_path.is_file():
            continue
        if video_path.suffix.lower() not in extensions:
            continue
        candidate_video_count += 1

        source_label, target_label = infer_label_from_path(video_path, label_mapping)
        if not target_label:
            if len(unlabeled_examples) < 12:
                unlabeled_examples.append(str(video_path))
            continue

        destination_dir = raw_dir / slugify(target_label)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / sanitize_filename(video_path.name)
        if not destination_path.exists():
            shutil.copy2(video_path, destination_path)

        relative_id = str(video_path.relative_to(import_dir)).replace("\\", "/")
        scanned.append(
            DownloadedItem(
                item_id=slugify(relative_id),
                source_label=source_label,
                target_label=target_label,
                video_path=destination_path.resolve(),
                download_url="aihubshell://local-import",
                metadata={
                    "source_path": str(video_path.resolve()),
                    "relative_path": relative_id,
                },
            )
        )
    if candidate_video_count and not scanned:
        print("[scan] 영상 파일은 찾았지만 label_mapping과 경로가 맞지 않아 학습 데이터로 분류되지 않았습니다.")
        print(f"[scan] candidate videos: {candidate_video_count}")
        if unlabeled_examples:
            print("[scan] unlabeled examples:")
            for sample_path in unlabeled_examples:
                print(f"  - {sample_path}")
    elif scanned:
        print(f"[scan] labeled videos: {len(scanned)} / candidates: {candidate_video_count}")
    return scanned


def extract_archives(import_dir: Path, extracted_dir: Path) -> Path:
    zip_files = sorted(
        path for path in import_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".zip"
    )
    if not zip_files:
        return import_dir

    extracted_any = False
    for zip_path in zip_files:
        target_dir = extracted_dir / sanitize_filename(zip_path.stem)
        marker = target_dir / ".extracted_ok"
        if marker.exists():
            extracted_any = True
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(target_dir)
        marker.write_text("ok", encoding="utf-8")
        extracted_any = True

    return extracted_dir if extracted_any else import_dir


def split_dataset(downloaded: list[DownloadedItem], config: dict, paths: dict) -> dict[str, Path]:
    split_config = config.get("split", {})
    train_ratio = float(split_config.get("train_ratio", 0.7))
    val_ratio = float(split_config.get("val_ratio", 0.15))
    test_ratio = float(split_config.get("test_ratio", 0.15))
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise RuntimeError("split 비율 합계는 1.0 이어야 합니다.")

    rng = random.Random(int(split_config.get("seed", 42)))
    by_label: dict[str, list[DownloadedItem]] = defaultdict(list)
    for item in downloaded:
        by_label[item.target_label].append(item)

    split_items = {"train": [], "val": [], "test": []}
    for label, items in by_label.items():
        rng.shuffle(items)
        total = len(items)
        train_end = max(1, int(total * train_ratio))
        val_end = train_end + max(1, int(total * val_ratio)) if total >= 3 else train_end
        split_items["train"].extend(items[:train_end])
        split_items["val"].extend(items[train_end:val_end])
        split_items["test"].extend(items[val_end:])

    split_paths = {
        "train": paths["current_split_train"],
        "val": paths["current_split_val"],
        "test": paths["current_split_test"],
    }
    for split_name, target_path in split_paths.items():
        with target_path.open("w", encoding="utf-8") as handle:
            for item in split_items[split_name]:
                handle.write(
                    json.dumps(
                        {
                            "item_id": item.item_id,
                            "source_label": item.source_label,
                            "target_label": item.target_label,
                            "video_path": str(item.video_path),
                            "download_url": item.download_url,
                            "metadata": item.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"[split] {split_name}: {len(split_items[split_name])} -> {target_path}")

    if not split_items["val"] and split_items["train"]:
        moved = split_items["train"].pop()
        split_items["val"].append(moved)
        _rewrite_split_manifest(split_paths["train"], split_items["train"])
        _rewrite_split_manifest(split_paths["val"], split_items["val"])
        print("[split] validation 샘플이 없어 train에서 1개를 val로 이동했습니다.")

    return split_paths


def load_existing_current_split_manifests(paths: dict) -> dict[str, Path]:
    split_paths = {
        "train": paths["current_split_train"],
        "val": paths["current_split_val"],
        "test": paths["current_split_test"],
    }
    for split_name, path in split_paths.items():
        if not path.exists():
            raise RuntimeError(f"기존 split manifest를 찾지 못했습니다: {split_name} -> {path}")
    return split_paths


def _rewrite_split_manifest(path: Path, items: list[DownloadedItem]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(
                json.dumps(
                    {
                        "item_id": item.item_id,
                        "source_label": item.source_label,
                        "target_label": item.target_label,
                        "video_path": str(item.video_path),
                        "download_url": item.download_url,
                        "metadata": item.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def prepare_pose_dataset(config: dict, paths: dict, split_manifests: dict[str, Path]) -> dict[str, Path]:
    preprocess_config = config.get("preprocess", {})
    device = str(preprocess_config.get("device", "cuda:0"))
    person_detector = PersonDetector(
        score_threshold=float(preprocess_config.get("person_score_threshold", 0.25)),
        resize_width=int(preprocess_config.get("person_imgsz", 640)),
        device=device,
    )
    face_detector = FaceDetector()

    target_labels = get_target_labels(config)
    label_to_idx = {label: index for index, label in enumerate(target_labels)}
    min_frames_with_person = int(preprocess_config.get("min_frames_with_person", 4))

    prepared_paths = {
        "train": paths["current_prepared_train"],
        "val": paths["current_prepared_val"],
        "test": paths["current_prepared_test"],
    }

    manifest_rows: dict[str, list[dict]] = {}
    overall_total = 0
    for split_name, manifest_path in split_manifests.items():
        rows: list[dict] = []
        with manifest_path.open("r", encoding="utf-8") as source_handle:
            for line in source_handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        manifest_rows[split_name] = rows
        overall_total += len(rows)

    processed_total = 0

    for split_name, manifest_path in split_manifests.items():
        target_manifest_path = prepared_paths[split_name]
        rows = manifest_rows.get(split_name, [])
        split_total = len(rows)
        with target_manifest_path.open("w", encoding="utf-8") as target_handle:
            kept = 0
            skipped = 0
            for split_index, sample in enumerate(rows, start=1):
                video_path = Path(sample["video_path"])
                target_label = sample["target_label"]
                processed_total += 1
                if (
                    processed_total == 1
                    or processed_total == overall_total
                    or processed_total % 5 == 0
                ):
                    prepare_ratio = processed_total / max(overall_total, 1)
                    write_pipeline_status(
                        paths,
                        stage="prepare",
                        state="running",
                        message=f"{split_name} split에서 pose 시퀀스를 추출하는 중입니다.",
                        stage_progress=round(0.55 + (0.25 * prepare_ratio), 4),
                        processed_items=processed_total,
                        total_items=overall_total,
                        current_split=split_name,
                        split_index=split_index,
                        split_total=split_total,
                        current_video=video_path.name,
                        kept_items=kept,
                        skipped_items=skipped,
                    )
                try:
                    sequence = extract_pose_sequence(
                        video_path=video_path,
                        person_detector=person_detector,
                        face_detector=face_detector,
                        sequence_length=int(preprocess_config.get("sequence_length", 48)),
                        max_frames_to_scan=int(preprocess_config.get("max_frames_to_scan", 160)),
                    )
                except Exception as exc:
                    skipped += 1
                    print(f"[prepare] skip unreadable video: {video_path} ({exc})")
                    continue
                if sequence["valid_frames"] < min_frames_with_person:
                    continue

                pose_output_dir = paths["prepared_dir"] / split_name / slugify(target_label)
                pose_output_dir.mkdir(parents=True, exist_ok=True)
                pose_path = pose_output_dir / f"{video_path.stem}_{sample['item_id']}.npz"
                np.savez_compressed(
                    pose_path,
                    pose=sequence["pose"],
                    mask=sequence["mask"],
                    label_idx=np.int64(label_to_idx[target_label]),
                )

                target_handle.write(
                    json.dumps(
                        {
                            **sample,
                            "pose_path": str(pose_path.resolve()),
                            "label_idx": label_to_idx[target_label],
                            "valid_frames": sequence["valid_frames"],
                            "confirmed_frames": sequence["confirmed_frames"],
                            "chosen_track_id": sequence["chosen_track_id"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1
            write_pipeline_status(
                paths,
                stage="prepare",
                state="running",
                message=f"{split_name} split pose 추출이 완료되었습니다.",
                stage_progress=round(0.55 + (0.25 * (processed_total / max(overall_total, 1))), 4) if overall_total else 0.8,
                processed_items=processed_total,
                total_items=overall_total,
                current_split=split_name,
                split_index=split_total,
                split_total=split_total,
                kept_items=kept,
                skipped_items=skipped,
            )
        print(f"[prepare] {split_name}: {kept} samples ({skipped} skipped) -> {target_manifest_path}")

    return prepared_paths


def load_existing_current_prepared_manifests(paths: dict) -> dict[str, Path]:
    prepared_paths = {
        "train": paths["current_prepared_train"],
        "val": paths["current_prepared_val"],
        "test": paths["current_prepared_test"],
    }
    for split_name, path in prepared_paths.items():
        if split_name in {"train", "val"} and not path.exists():
            raise RuntimeError(f"기존 prepared manifest를 찾지 못했습니다: {split_name} -> {path}")
    return prepared_paths


def get_continual_config(config: dict) -> dict:
    continual = config.get("continual_learning", {})
    return {
        "enabled": bool(continual.get("enabled", True)),
        "resume_from_best": bool(continual.get("resume_from_best", True)),
        "cleanup_raw_after_job": bool(continual.get("cleanup_raw_after_job", True)),
    }


def load_training_manifests(paths: dict, *, continual_enabled: bool) -> dict[str, Path]:
    if continual_enabled and paths["prepared_train"].exists() and paths["prepared_val"].exists():
        return {
            "train": paths["prepared_train"],
            "val": paths["prepared_val"],
            "test": paths["prepared_test"],
        }
    return {
        "train": paths["current_prepared_train"],
        "val": paths["current_prepared_val"],
        "test": paths["current_prepared_test"],
    }


def update_cumulative_manifests(
    config: dict,
    paths: dict,
    split_manifests: dict[str, Path],
    prepared_manifests: dict[str, Path],
) -> dict[str, Path]:
    continual_config = get_continual_config(config)
    if not continual_config["enabled"]:
        return prepared_manifests

    shell_config = config.get("aihub_shell", {})
    job_meta = {
        "job_datasetkey": str(shell_config.get("datasetkey", "")).strip() or None,
        "job_filekey": normalize_requested_filekeys(shell_config.get("filekey")),
        "job_added_at": datetime.now(timezone.utc).astimezone().isoformat(),
    }

    merge_jsonl_entries(
        paths["current_raw_manifest"],
        paths["raw_manifest"],
        extra_fields=job_meta,
    )
    merge_jsonl_entries(
        split_manifests["train"],
        paths["split_train"],
        extra_fields=job_meta,
    )
    merge_jsonl_entries(
        split_manifests["val"],
        paths["split_val"],
        extra_fields=job_meta,
    )
    merge_jsonl_entries(
        split_manifests["test"],
        paths["split_test"],
        extra_fields=job_meta,
    )
    merge_jsonl_entries(
        prepared_manifests["train"],
        paths["prepared_train"],
        extra_fields=job_meta,
    )
    merge_jsonl_entries(
        prepared_manifests["val"],
        paths["prepared_val"],
        extra_fields=job_meta,
    )
    merge_jsonl_entries(
        prepared_manifests["test"],
        paths["prepared_test"],
        extra_fields=job_meta,
    )

    with paths["continual_state"].open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "updated_at": job_meta["job_added_at"],
                "datasetkey": job_meta["job_datasetkey"],
                "filekeys": job_meta["job_filekey"],
                "raw_total": count_manifest_lines(paths["raw_manifest"]),
                "prepared_train_total": count_manifest_lines(paths["prepared_train"]),
                "prepared_val_total": count_manifest_lines(paths["prepared_val"]),
                "prepared_test_total": count_manifest_lines(paths["prepared_test"]),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "[continual] cumulative prepared samples "
        f"train={count_manifest_lines(paths['prepared_train'])}, "
        f"val={count_manifest_lines(paths['prepared_val'])}, "
        f"test={count_manifest_lines(paths['prepared_test'])}"
    )
    return {
        "train": paths["prepared_train"],
        "val": paths["prepared_val"],
        "test": paths["prepared_test"],
    }


def merge_jsonl_entries(source_path: Path, target_path: Path, *, extra_fields: dict | None = None) -> int:
    source_entries = read_jsonl_entries(source_path)
    if not source_entries:
        return 0

    existing_entries = read_jsonl_entries(target_path)
    seen = {build_manifest_unique_key(entry) for entry in existing_entries}
    merged_entries = list(existing_entries)
    added = 0

    for entry in source_entries:
        merged_entry = dict(entry)
        if extra_fields:
            merged_entry.update(extra_fields)
        unique_key = build_manifest_unique_key(merged_entry)
        if unique_key in seen:
            continue
        merged_entries.append(merged_entry)
        seen.add(unique_key)
        added += 1

    write_jsonl_entries(target_path, merged_entries)
    return added


def read_jsonl_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def write_jsonl_entries(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_manifest_unique_key(entry: dict) -> str:
    metadata = entry.get("metadata") or {}
    pose_path = entry.get("pose_path")
    if pose_path:
        return f"pose::{pose_path}"
    relative_path = metadata.get("relative_path")
    if relative_path:
        return f"relative::{relative_path}"
    source_path = metadata.get("source_path")
    if source_path:
        return f"source::{source_path}"
    item_id = entry.get("item_id")
    target_label = entry.get("target_label")
    if item_id:
        return f"item::{item_id}::{target_label}"
    return json.dumps(entry, sort_keys=True, ensure_ascii=False)


def count_manifest_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def cleanup_transient_job_data(paths: dict) -> None:
    for key in ("raw_dir", "import_dir", "extracted_dir"):
        target = paths.get(key)
        if isinstance(target, Path) and target.exists():
            shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)

    for key in (
        "current_raw_manifest",
        "current_split_train",
        "current_split_val",
        "current_split_test",
        "current_prepared_train",
        "current_prepared_val",
        "current_prepared_test",
    ):
        target = paths.get(key)
        if isinstance(target, Path) and target.exists():
            target.unlink()


def extract_pose_sequence(
    *,
    video_path: Path,
    person_detector: PersonDetector,
    face_detector: FaceDetector,
    sequence_length: int,
    max_frames_to_scan: int,
) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"영상 파일을 열지 못했습니다: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_indices = build_frame_indices(total_frames, sequence_length, max_frames_to_scan)

    tracker = PersonTracker()
    presence_filter = PersonPresenceFilter(debug=False)
    track_frames: dict[int, dict[int, dict]] = defaultdict(dict)

    for time_index, frame_index in enumerate(frame_indices):
        frame = read_frame_at(capture, frame_index)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = person_detector.detect(frame)
        tracked = tracker.update(detections)
        faces = face_detector.detect(frame)
        evaluated = presence_filter.evaluate(tracked, faces, gray, frame.shape)
        for candidate in evaluated:
            if candidate.get("person_state") == "rejected":
                continue
            candidate_id = int(candidate["id"])
            previous = track_frames[candidate_id].get(time_index)
            if previous is None or candidate.get("person_score", 0) >= previous.get("person_score", 0):
                track_frames[candidate_id][time_index] = candidate

    capture.release()

    if not track_frames:
        return {
            "pose": np.zeros((sequence_length, 17, 3), dtype=np.float32),
            "mask": np.zeros((sequence_length,), dtype=np.float32),
            "valid_frames": 0,
            "confirmed_frames": 0,
            "chosen_track_id": -1,
        }

    chosen_track_id = choose_best_track(track_frames)
    chosen_frames = track_frames[chosen_track_id]
    pose = np.zeros((sequence_length, 17, 3), dtype=np.float32)
    mask = np.zeros((sequence_length,), dtype=np.float32)
    valid_frames = 0
    confirmed_frames = 0

    for time_index in range(sequence_length):
        candidate = chosen_frames.get(time_index)
        if candidate is None:
            continue
        pose[time_index] = normalize_pose(candidate.get("keypoints", []), candidate["bbox"])
        mask[time_index] = 1.0
        valid_frames += 1
        if candidate.get("person_state") in CONFIRMED_PERSON_STATES:
            confirmed_frames += 1

    return {
        "pose": pose,
        "mask": mask,
        "valid_frames": valid_frames,
        "confirmed_frames": confirmed_frames,
        "chosen_track_id": chosen_track_id,
    }


def build_frame_indices(total_frames: int, sequence_length: int, max_frames_to_scan: int) -> list[int]:
    if total_frames > 0:
        effective_total = min(total_frames, max_frames_to_scan)
        if effective_total <= sequence_length:
            return list(range(effective_total))
        return np.linspace(0, effective_total - 1, num=sequence_length, dtype=int).tolist()
    return list(range(sequence_length))


def read_frame_at(capture: cv2.VideoCapture, frame_index: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        return None
    return frame


def choose_best_track(track_frames: dict[int, dict[int, dict]]) -> int:
    best_track_id = -1
    best_score = None
    for track_id, frames in track_frames.items():
        if not frames:
            continue
        confirmed_count = sum(
            1 for frame in frames.values() if frame.get("person_state") in CONFIRMED_PERSON_STATES
        )
        avg_score = sum(float(frame.get("person_score", 0.0)) for frame in frames.values()) / max(len(frames), 1)
        score = confirmed_count * 100.0 + len(frames) * 10.0 + avg_score
        if best_score is None or score > best_score:
            best_score = score
            best_track_id = track_id
    if best_track_id < 0:
        return next(iter(track_frames))
    return best_track_id


def normalize_pose(keypoints: list[dict], bbox) -> np.ndarray:
    x, y, w, h = bbox
    normalized = np.zeros((17, 3), dtype=np.float32)
    for index in range(min(len(keypoints), 17)):
        point = keypoints[index]
        conf = float(point.get("confidence", 0.0))
        if conf <= 0.0:
            continue
        normalized[index, 0] = float((float(point.get("x", 0.0)) - x) / max(w, 1))
        normalized[index, 1] = float((float(point.get("y", 0.0)) - y) / max(h, 1))
        normalized[index, 2] = conf
    return normalized


def fetch_api_items(session: requests.Session, api_config: dict) -> list[dict]:
    list_url = api_config["list_url"]
    page_param = api_config.get("page_param")
    page_start = int(api_config.get("page_start", 1))
    static_params = dict(api_config.get("params", {}))
    max_pages = int(api_config.get("max_pages", 0))

    results = []
    page = page_start
    fetched_pages = 0

    while True:
        params = dict(static_params)
        if page_param:
            params[page_param] = page
        try:
            response = session.get(
                list_url,
                params=params,
                timeout=float(api_config.get("timeout_seconds", 60.0)),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "API 목록 조회에 실패했습니다.\n"
                f"- 요청 주소: {list_url}\n"
                f"- 에러: {exc}\n\n"
                "확인할 것:\n"
                "1. config의 api.list_url 이 실제 주소로 바뀌었는지\n"
                "2. items_path / next_path / fields 설정이 API 응답 구조와 맞는지\n"
                "3. 비공개 API라면 auth_required / auth_token_env 설정이 맞는지"
            ) from exc
        payload = response.json()
        items = extract_field(payload, api_config.get("items_path", "")) if api_config.get("items_path") else payload
        if not isinstance(items, list):
            raise RuntimeError("API items_path 결과가 리스트가 아닙니다.")
        results.extend(items)

        fetched_pages += 1
        if max_pages > 0 and fetched_pages >= max_pages:
            break

        next_value = extract_field(payload, api_config.get("next_path", "")) if api_config.get("next_path") else None
        if next_value:
            if isinstance(next_value, str) and next_value.startswith("http"):
                list_url = next_value
                page_param = None
            else:
                page += 1
            continue

        if page_param and len(items) > 0:
            page += 1
            continue
        break

    return results


def build_headers(api_config: dict) -> dict:
    headers = dict(api_config.get("headers", {}))
    auth_env = api_config.get("auth_token_env")
    auth_header = api_config.get("auth_header", "Authorization")
    auth_required = bool(api_config.get("auth_required", False))
    if auth_env:
        value = os.environ.get(auth_env)
        if not value:
            if auth_required:
                raise RuntimeError(
                    f"환경변수 {auth_env} 가 설정되지 않았습니다. "
                    "비공개 API라면 토큰을 export 하거나, 공개 API라면 config에서 "
                    "`auth_required: false` 또는 `auth_token_env: \"\"` 로 설정해 주세요."
                )
            return headers
        if auth_header.lower() == "authorization" and not value.lower().startswith("bearer "):
            value = f"Bearer {value}"
        headers[auth_header] = value
    return headers


def validate_source_config(config: dict, config_path: Path) -> None:
    source_mode = str(config.get("dataset_source", "json_api")).strip().lower()
    if source_mode == "aihub_shell":
        validate_aihub_shell_config(config.get("aihub_shell", {}), config_path)
        return
    validate_api_config(config.get("api", {}), config_path)


def validate_api_config(api_config: dict, config_path: Path) -> None:
    list_url = str(api_config.get("list_url", "")).strip()
    if not list_url:
        raise RuntimeError(
            f"API 설정이 비어 있습니다: {config_path}\n"
            "config의 api.list_url 에 실제 데이터 목록 API 주소를 넣어 주세요."
        )

    parsed = urlparse(list_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(
            f"api.list_url 형식이 올바르지 않습니다: {list_url}\n"
            "예: https://example.com/api/videos"
        )

    placeholder_hosts = {
        "your-dataset-api.example.com",
        "example.com",
    }
    if parsed.netloc in placeholder_hosts or "your-dataset-api" in parsed.netloc:
        raise RuntimeError(
            "예제용 placeholder API 주소가 그대로 들어 있습니다.\n"
            f"- 현재 주소: {list_url}\n"
            f"- 설정 파일: {config_path}\n\n"
            "해야 할 일:\n"
            "1. configs/action_training.example.json 의 api.list_url 을 실제 API 주소로 변경\n"
            "2. API 응답에 맞게 items_path / fields.id / fields.label / fields.download_url 수정\n"
            "3. 비공개 API라면 auth_required 와 auth_token_env 도 함께 설정"
        )


def validate_aihub_shell_config(shell_config: dict, config_path: Path) -> None:
    if shell_config.get("datasetkey") in (None, "") and shell_config.get("datapackagekey") in (None, ""):
        raise RuntimeError(
            f"AIHub shell 설정이 부족합니다: {config_path}\n"
            "aihub_shell.datasetkey 또는 aihub_shell.datapackagekey 중 하나는 필요합니다."
        )
    resolve_aihub_shell_path(shell_config)
    resolve_aihub_api_key(shell_config)


def resolve_aihub_shell_path(shell_config: dict) -> str:
    configured = str(shell_config.get("path", "")).strip()
    if configured:
        raw_candidate = Path(configured).expanduser()
        candidates = [raw_candidate]
        if raw_candidate.suffix == "":
            candidates.extend(
                [
                    raw_candidate.with_suffix(".exe"),
                    raw_candidate.with_suffix(".bat"),
                    raw_candidate.with_suffix(".cmd"),
                ]
            )
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        raise RuntimeError(
            f"aihubshell 경로를 찾지 못했습니다: {configured}\n"
            "확인할 것:\n"
            "1. config의 aihub_shell.path 가 현재 PC 기준 경로인지\n"
            "2. Windows라면 aihubshell.exe 인지\n"
            "3. 프로젝트 루트에 있다면 path를 'aihubshell' 로 둘 수 있는지"
        )

    discovered = shutil.which("aihubshell")
    if discovered:
        return discovered

    raise RuntimeError(
        "aihubshell 실행 파일을 찾지 못했습니다.\n"
        "AIHub 공식 안내처럼 aihubshell을 설치한 뒤,\n"
        "1. PATH에 등록하거나\n"
        "2. config의 aihub_shell.path 에 실행 파일 경로를 넣어 주세요."
    )


def resolve_aihub_api_key(shell_config: dict) -> str:
    direct_key = str(shell_config.get("api_key", "")).strip()
    if direct_key:
        return direct_key

    env_name = str(shell_config.get("api_key_env", "AIHUB_API_KEY")).strip()
    value = os.environ.get(env_name)
    if value:
        return value

    raise RuntimeError(
        f"AIHub API 키를 찾지 못했습니다.\n"
        f"- 환경변수 {env_name} 를 export 하거나\n"
        "- config의 aihub_shell.api_key 에 직접 넣어 주세요.\n"
        "또한 AIHub 데이터셋은 승인 완료 후 다운로드 가능합니다."
    )


def infer_label_from_path(video_path: Path, label_mapping: dict) -> tuple[str, str | None]:
    relative_text = str(video_path).replace("\\", "/")
    relative_text_lower = relative_text.lower()
    for source_label, target_label in label_mapping.items():
        source_text = str(source_label).strip()
        if not source_text:
            continue
        if source_text in relative_text or source_text.lower() in relative_text_lower:
            return str(source_label), str(target_label)
    return "", None


def build_download_url(item: dict, api_config: dict) -> str:
    direct_key = api_config["fields"].get("download_url")
    if direct_key:
        direct_url = extract_field(item, direct_key)
        if direct_url:
            return str(direct_url)

    template = api_config.get("download_url_template")
    if template:
        item_id = extract_field(item, api_config["fields"]["id"])
        return template.format(item_id=item_id)
    return ""


def build_filename(item: dict, api_config: dict, download_url: str, item_id: str) -> str:
    filename_key = api_config["fields"].get("filename")
    if filename_key:
        filename = extract_field(item, filename_key)
        if filename:
            return sanitize_filename(str(filename))

    parsed = urlparse(download_url)
    name = Path(parsed.path).name
    if name:
        return sanitize_filename(name)
    return f"{slugify(item_id)}.mp4"


def download_to_file(session: requests.Session, url: str, target_path: Path, timeout: float) -> None:
    if target_path.exists() and target_path.stat().st_size > 0:
        return

    with session.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with target_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def extract_field(data, path: str):
    if not path:
        return data
    current = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^0-9a-zA-Z가-힣._-]+", "_", value)
    return value.strip("._-") or "item"


def sanitize_filename(filename: str) -> str:
    filename = filename.replace("\\", "_").replace("/", "_")
    if "." not in filename:
        filename += ".mp4"
    return slugify(filename.rsplit(".", 1)[0]) + "." + filename.rsplit(".", 1)[1]


def get_target_labels(config: dict) -> list[str]:
    labels = config.get("dataset", {}).get("target_labels")
    if labels:
        return list(labels)

    mapped = set(config.get("dataset", {}).get("label_mapping", {}).values())
    if not mapped:
        raise RuntimeError("dataset.target_labels 또는 dataset.label_mapping 이 필요합니다.")
    return sorted(mapped)


if __name__ == "__main__":
    main()
