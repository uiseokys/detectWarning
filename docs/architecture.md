# detectWarning Architecture

## Runtime Surfaces

- `app/main.py`: 로컬 카메라 또는 영상 파일을 입력으로 받아 위험 상황을 분석하는 실행 진입점입니다.
- `app/inference_server.py`: 원격 추론 API와 관리 화면을 제공합니다.
- `app/camera_uploader.py`: 클라이언트 장치에서 프레임을 원격 서버로 업로드합니다.
- `app/training_dashboard.py`: 학습 큐 제어, 상태 모니터링, 자동 추천, Pages 동기화를 담당합니다.
- `app/action_training_pipeline.py`: AIHub 다운로드, split, pose 전처리, 행동 모델 학습을 수행합니다.
- `app/update_pages_site.py`: 정적 Pages 리포트와 live 상태 파일을 생성합니다.

## Shared Modules

- `app/action_model.py`: pose 시퀀스 기반 행동 분류 모델, 학습 루프, 평가 지표, 오류 분석 저장 로직입니다.
- `app/training_insights.py`: accuracy, macro F1, loss, class report, confusion matrix를 rule-based로 해석합니다.
- `app/gpu_autotune.py`: GPU VRAM/사용률을 읽어 안전한 범위에서 batch와 전처리 설정을 자동 조정합니다.
- `app/reporting.py`: metrics, manifest, job summary를 공통 형식으로 정리합니다.
- `app/training_config.py`: 학습 config 기본값 병합과 Pages sync 설정 해석을 담당합니다.
- `app/dashboard_runtime.py`: job lifecycle, 로그 tail, Pages sync helper를 제공합니다.
- `app/pipeline_prepare.py`: 영상 샘플링과 pose prepare 보조 로직입니다.
- `app/risk_analyzer.py`: 영상/음성 위험 점수를 합산하는 rule-based 분석 로직입니다.
- `app/detector.py`: YOLO 기반 사람/객체 검출 로직입니다.

## Training State Files

학습 파이프라인은 `training_data/action_pipeline_aihub/` 아래에 JSON/JSONL 상태 파일을 저장합니다.

- `pipeline_status.json`: 현재 파이프라인 단계, 메시지, 진행률
- `artifacts/training_progress.json`: epoch별 학습 진행 상태
- `artifacts/metrics.json`: 최종 validation 지표, class report, confusion matrix
- `artifacts/best_action_model.pt`: best validation macro F1 기준 체크포인트
- `artifacts/validation_error_analysis.json`: 낮은 recall 클래스와 혼동쌍 요약
- `artifacts/false_negative_examples.json`: 클래스별 false negative 예시
- `artifacts/confusion_pair_examples.json`: 자주 혼동되는 클래스 쌍과 샘플 예시
- `manifests/current_*`: 현재 filekey 작업 기준 raw/split/prepared manifest
- `manifests/cumulative_*`: 누적 raw/split/prepared manifest
- `manifests/active_prepared_*`: 실제 학습에 투입되는 필터링된 manifest
- `manifests/current_skipped_videos.json`: 현재 작업의 broken/skipped 통계
- `manifests/cumulative_skipped_videos.json`: 누적 broken/skipped 통계
- `launcher_history.json`: 대시보드에서 관리하는 완료 작업 이력

## Pages Share Flow

1. `training_dashboard.py` 또는 `update_pages_site.py`가 `latest-result.json`, `live-status.json`을 생성합니다.
2. Pages 디렉터리에 반영되면 정적 리포트가 갱신됩니다.
3. live endpoint가 열려 있으면 Pages가 live 상태를 probe해서 현재 학습 화면으로 연결할 수 있습니다.

## Design Constraints

- 학습 상태는 파일 기반으로 유지되므로 JSON/JSONL 읽기 실패에 대한 방어 코드가 중요합니다.
- checkpoint, label mapping, manifest schema는 학습 재개와 평가 표시가 함께 의존하므로 임의 변경을 피해야 합니다.
- 자동 추천은 성능 진단을 참고하지만, 이미 많은 클래스가 반복 추천되지 않도록 train/validation 분포 균형을 우선합니다.
- GPU 자동 튜닝은 batch, detector batch, prefetch worker처럼 호환성 위험이 낮은 값만 조정합니다.
