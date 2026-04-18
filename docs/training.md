# 행동 학습 파이프라인

## 핵심 엔트리

- 관리자 대시보드: `python app/training_dashboard.py --config configs/action_training.aihub_shell.example.json --port 8010`
- 전체 파이프라인 직접 실행: `python app/action_training_pipeline.py --config configs/action_training.aihub_shell.example.json --stage all`
- Windows 통합 런처: `run_training_share.cmd`

## 입력 개념

- `datasetkey`: AIHub 데이터셋 키
- `filekey`: 분할 ZIP 파일 키

## 주요 상태 파일

- `training_data/action_pipeline_aihub/pipeline_status.json`
- `training_data/action_pipeline_aihub/training_progress.json`
- `training_data/action_pipeline_aihub/launcher_history.json`
- `training_data/action_pipeline_aihub/manifests/*.jsonl`

## 운영 팁

- 클래스 체계를 바꿀 때는 기존 prepared pose를 최대한 재사용하고, 모델만 다시 학습하는 흐름을 우선 고려합니다.
- 누적 학습 중 파일이 많이 쌓이면 `prepared_*`, `cumulative_*`, `active_*` manifest의 역할을 먼저 확인합니다.
- Pages 공유를 쓸 때는 `pages_sync.pages_dir`와 `DETECTWARNING_LIVE_URL` 값을 분리해서 관리하는 것이 안전합니다.
