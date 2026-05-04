# 행동 학습 파이프라인

이 문서는 AIHub 행동 분류 학습 흐름을 빠르게 확인하기 위한 운영 메모입니다.

## 실행 진입점

- 학습 대시보드: `python app/training_dashboard.py --config configs/action_training.aihub_shell.example.json --port 8010`
- 전체 파이프라인 직접 실행: `python app/action_training_pipeline.py --config configs/action_training.aihub_shell.example.json --stage all`
- Windows 실행 도우미: `run_training_share.cmd`

## 입력 개념

- `datasetkey`: AIHub 데이터셋 키
- `filekey`: AIHub 파일 목록에 있는 분할 ZIP 파일 키
- `outsidedoor_*`, `insidedoor_*`, `inside_croki_*`: ZIP 이름 기준 장소/촬영 유형 그룹

## 파이프라인 단계

1. `download`: AIHub filekey의 분할 ZIP을 다운로드하고 병합/압축 해제합니다.
2. `split`: 영상 목록을 train/validation/test로 나눕니다.
3. `prepare`: 영상에서 사람 pose 시퀀스를 추출하고 prepared manifest를 만듭니다.
4. `train`: 누적 prepared manifest를 기준으로 행동 분류 모델을 학습합니다.
5. `report`: metrics, best checkpoint, error analysis 파일을 저장하고 대시보드가 이를 표시합니다.

## 주요 상태 파일

- `training_data/action_pipeline_aihub/pipeline_status.json`
- `training_data/action_pipeline_aihub/artifacts/training_progress.json`
- `training_data/action_pipeline_aihub/artifacts/metrics.json`
- `training_data/action_pipeline_aihub/launcher_history.json`
- `training_data/action_pipeline_aihub/manifests/*.jsonl`

## 성능 분석 파일

재학습 없이 기존 `metrics.json`에서 validation 오류 분석 파일을 다시 만들 수 있습니다.

```powershell
python app\export_validation_error_analysis.py --metrics training_data\action_pipeline_aihub\artifacts\metrics.json
```

생성되는 파일:

- `validation_error_analysis.json`
- `false_negative_examples.json`
- `confusion_pair_examples.json`

## 운영 주의점

- 클래스 체계를 바꿀 때는 기존 prepared pose를 재사용할 수 있는지 먼저 확인하세요.
- `normal`을 target label에 포함하면 normal 데이터도 반드시 train/validation에 있어야 합니다.
- 단일 클래스 filekey만 누적하면 모델이 한 클래스로 치우칠 수 있으므로, 자동 추천은 부족한 클래스의 outside filekey를 우선 큐에 넣도록 구성되어 있습니다.
- 대시보드 코드 변경 후에는 대시보드 서버를 재시작해야 UI와 추천 로직이 반영됩니다.
