# detectWarning Architecture

## Runtime surfaces

- `app/main.py`
  로컬 단일 머신 데모 실행 진입점
- `app/inference_server.py`
  원격 추론 서버 및 관제 UI
- `app/camera_uploader.py`
  노트북/원격 카메라 업로더
- `app/training_dashboard.py`
  학습 큐 제어, 상태 모니터링, Pages 동기화
- `app/action_training_pipeline.py`
  다운로드 → split → prepare → train 오케스트레이션
- `app/update_pages_site.py`
  오프라인 리포트와 live 상태 파일 생성

## Shared modules

- `app/reporting.py`
  metrics/manifest/job 기록을 공통 형태로 정규화하는 모듈
- `app/training_config.py`
  학습 config 기본값 주입, 공통 설정 정규화, Pages sync 해석
- `app/risk_analyzer.py`
  STT 및 비디오 맥락을 점수화하는 규칙 기반 분석기
- `app/detector.py`
  YOLO 사람/포즈 검출

## State files

학습 파이프라인 워크스페이스 아래에 JSON/JSONL 상태 파일이 저장됩니다.

- `pipeline_status.json`
  현재 파이프라인 단계와 메시지
- `artifacts/training_progress.json`
  epoch별 학습 기록
- `manifests/current_*`
  현재 filekey 기준 raw/split/prepared
- `manifests/cumulative_*`
  누적 raw/split/prepared
- `manifests/active_prepared_*`
  현재 클래스 체계로 재매핑된 학습용 manifest
- `manifests/current_skipped_videos.json`
  현재 작업 broken/skipped 요약
- `manifests/cumulative_skipped_videos.json`
  누적 broken/skipped 요약
- `launcher_history.json`
  대시보드 큐/완료 작업 이력

## Pages share flow

1. `training_dashboard.py` 또는 `update_pages_site.py`가 `latest-result.json`, `live-status.json`을 생성
2. Pages 저장소에 반영되면 팀원용 정적 리포트가 갱신
3. live가 열려 있으면 Pages가 live endpoint를 probe해서 실시간 보기로 이동

## Design constraints

- Quick Tunnel + Pages 조합은 완전한 실시간 동기화가 아니라 eventually consistent 구조입니다.
- 학습과 리포트 생성은 파일 기반 상태 저장을 사용하므로, 상태 파일 읽기 실패에 대한 방어 로직이 중요합니다.
- 위험 규칙은 `configs/risk_rules.json`으로 외부화되어 있으며, 코드와 룰셋을 분리해 유지보수합니다.
