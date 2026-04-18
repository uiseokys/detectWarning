# detectWarning

카메라 영상과 음성을 이용해 위험 상황을 감지하고, 로컬/원격 추론과 행동 학습 리포트를 함께 운영하는 프로젝트입니다.

## 빠른 시작

### 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

개발용 도구까지 같이 설치하려면:

```bash
pip install -r requirements-dev.txt
```

### 로컬 추론

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR
```

### 원격 추론 서버

```powershell
python app\inference_server.py --host 0.0.0.0 --port 8000 --yolo-device cuda:0 --stt-device cuda --stt-compute-type float16
```

### 학습 대시보드

```bash
python3 app/training_dashboard.py --config configs/action_training.aihub_shell.example.json --port 8010
```

## 문서

- 원격 추론/업로더: [docs/remote-inference.md](/Users/jung-uiseok/Desktop/detectWarning/docs/remote-inference.md)
- 행동 학습 파이프라인: [docs/training.md](/Users/jung-uiseok/Desktop/detectWarning/docs/training.md)
- Pages 공유 리포트: [docs/pages-share.md](/Users/jung-uiseok/Desktop/detectWarning/docs/pages-share.md)
- 구조 개요와 상태 파일: [docs/architecture.md](/Users/jung-uiseok/Desktop/detectWarning/docs/architecture.md)

## 주요 엔트리 포인트

- `app/main.py`: 로컬 추론 실행
- `app/inference_server.py`: 원격 추론 서버 + 웹 관제 화면
- `app/camera_uploader.py`: 노트북 업로더
- `app/training_dashboard.py`: 학습 관리자 대시보드
- `app/action_training_pipeline.py`: 학습 파이프라인 엔트리
- `app/update_pages_site.py`: Pages 리포트 내보내기

## 운영 메모

- 학습 결과와 운영 상태 파일은 `training_data/`와 `logs/` 아래에 생성됩니다.
- Pages 공유를 사용할 때는 `pages_sync` 설정과 live URL 환경 변수를 함께 관리해야 합니다.
- 민감한 키와 내부 경로는 커밋하지 않도록 `.gitignore`와 환경 변수를 사용합니다.

## 테스트

```bash
python3 -m unittest discover -s tests
```
