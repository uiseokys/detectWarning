# detectWarning

영상과 음성 신호를 함께 사용해 위험 상황을 감지하고, 로컬/원격 추론과 행동 분류 학습 대시보드를 제공하는 프로젝트입니다.

현재 학습 파이프라인은 AIHub 분할 ZIP `filekey`를 받아 영상 데이터를 내려받고, pose 시퀀스를 전처리한 뒤 행동 분류 모델을 학습합니다. 대시보드는 학습 상태, 큐, GPU 상태, 성능 지표, 자동 진단 결과를 보여줍니다.

## 빠른 시작

### 설치

```bash
python -m venv .venv
pip install -r requirements.txt
```

개발 및 테스트 도구까지 설치하려면 다음을 추가로 실행합니다.

```bash
pip install -r requirements-dev.txt
```

### 로컬 추론

```bash
python app/main.py --source 0 --stt --stt-language ko-KR
```

### 원격 추론 서버

```powershell
python app\inference_server.py --host 0.0.0.0 --port 8000 --yolo-device cuda:0 --stt-device cuda --stt-compute-type float16
```

### 학습 대시보드

```powershell
python app\training_dashboard.py --config configs/action_training.aihub_shell.example.json --port 8010
```

### 전처리 재사용 자동 튜닝

이미 만들어진 `manifests/active_prepared_*` 또는 `cumulative_prepared_*`만 사용해서 여러 학습 설정을 비교하려면 아래처럼 실행합니다. 다운로드와 pose 전처리는 다시 하지 않습니다.

```powershell
python app\auto_tune_action_training.py --config configs/action_training.aihub_shell.example.json --trials 16
```

Auto-tune also runs an optional ensemble search when `auto_tune.ensemble_enabled=true`. It reuses saved trial checkpoints, searches weighted probability ensembles and class-bias calibration on the prepared validation manifest, writes `artifacts/best_action_ensemble.json` and `artifacts/ensemble_metrics.json`, and promotes the ensemble metrics when they beat the best single checkpoint. To run only the ensemble pass:

```powershell
python app\ensemble_action_models.py --config configs/action_training.aihub_shell.example.json --top-k 12 --max-size 5 --target-metric accuracy --promote
```

trial 결과는 `training_data/action_pipeline_aihub/artifacts/auto_tune/` 아래에 따로 저장되고, 기본 설정에서는 가장 좋은 trial의 `best_action_model.pt`, `metrics.json`, 분석 파일을 메인 `artifacts/`로 승격해 대시보드가 바로 표시합니다. 또한 `auto_tune.update_config_with_best=true`이면 best trial의 학습 JSON 값이 설정 파일에 반영되고, 전체 effective 설정은 `artifacts/auto_tune/best_training_config.json`에 저장됩니다. 후보만 확인하려면 `--dry-run`, 승격 없이 비교만 하려면 `--no-promote-best`, config 갱신을 막으려면 `--no-update-config`를 사용하세요.

## 주요 진입점

- `app/main.py`: 로컬 카메라/영상 기반 위험 감지 실행
- `app/inference_server.py`: 원격 추론 서버와 관리 UI
- `app/camera_uploader.py`: 원격 서버로 프레임을 업로드하는 클라이언트
- `app/action_training_pipeline.py`: AIHub 다운로드, split, pose 전처리, 행동 모델 학습 파이프라인
- `app/auto_tune_action_training.py`: 기존 전처리 결과를 재사용해 반복 학습 설정을 자동 비교/승격
- `app/training_dashboard.py`: 학습 큐, 상태 모니터링, 자동 추천, 학습 진단 대시보드
- `app/action_model.py`: pose 기반 행동 분류 모델, 학습 루프, 평가/분석 저장
- `app/training_insights.py`: 학습 지표를 rule-based로 해석하는 자동 진단 로직
- `app/export_validation_error_analysis.py`: 기존 `metrics.json`에서 false negative/confusion pair 분석 파일 생성

## 학습 결과 파일

기본 AIHub 학습 워크스페이스는 `training_data/action_pipeline_aihub/`입니다.

- `pipeline_status.json`: 현재 파이프라인 단계와 진행률
- `artifacts/training_progress.json`: epoch별 학습 진행 상태
- `artifacts/metrics.json`: 최종 성능 지표, class report, confusion matrix
- `artifacts/best_action_model.pt`: best validation macro F1 기준 체크포인트
- `artifacts/validation_error_analysis.json`: 낮은 recall 클래스와 혼동쌍 요약
- `artifacts/false_negative_examples.json`: 클래스별 false negative 예시
- `artifacts/confusion_pair_examples.json`: 자주 혼동되는 클래스 쌍과 예시
- `manifests/current_*`: 현재 filekey 작업의 raw/split/prepared manifest
- `manifests/cumulative_*`: 누적 학습 데이터 manifest
- `manifests/active_prepared_*`: 실제 학습에 사용되는 필터링된 manifest

## 테스트

```bash
python -m unittest discover -s tests
```

일부 Windows 환경에서는 `%TEMP%` 권한 문제로 임시 파일을 만드는 테스트가 실패할 수 있습니다. 이 경우 작업 디렉터리 안의 쓰기 가능한 임시 폴더를 사용하도록 환경변수를 조정한 뒤 다시 실행하세요.
