# detectWarning

카메라 영상과 마이크 음성을 이용해 사람을 찾고, 위험할 수 있는 상황을 실시간으로 보여주는 프로젝트입니다.

이 프로젝트는 다음 정보를 화면에 보여줍니다.

- 사람
- 얼굴
- 음성 인식 결과
- 위험도

또한 팀 프로젝트용으로, 노트북 카메라 영상을 데스크탑 서버로 보내서 분석하는 방식도 지원합니다.

## 이 프로젝트가 하는 일

쉽게 말하면 아래 흐름으로 동작합니다.

1. 카메라에서 사람을 찾습니다.
2. 사람의 관절점과 얼굴을 확인합니다.
3. 마이크 음성을 글자로 바꿉니다.
4. 영상과 음성을 함께 보고 위험도를 계산합니다.
5. 결과를 화면이나 웹에서 보여줍니다.

## 설치

처음 한 번만 하면 됩니다.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 가장 추천하는 실행 방법

혼자서 바로 테스트할 때는 이 명령어 하나면 충분합니다.

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR
```

이렇게 실행하면:

- 웹캠 화면이 열리고
- 사람과 얼굴이 표시되고
- 마이크 음성이 인식되고
- 위험도가 함께 표시됩니다

종료는 `q` 또는 `Esc` 키입니다.

## 팀 프로젝트용 실행 방법

팀원이 각자 노트북 카메라를 쓰고,
분석은 한 대의 데스크탑에서 하고 싶을 때 사용하는 방법입니다.

### 1. 데스크탑에서 서버 실행

Windows PowerShell 기준:

```powershell
.venv\Scripts\activate
python app\inference_server.py --host 0.0.0.0 --port 8000 --yolo-device cuda:0 --stt-device cuda --stt-compute-type float16
```

이 명령어는 데스크탑 GPU를 써서 사람 인식과 음성 인식을 처리합니다.

### 2. 노트북에서 카메라와 마이크 전송

macOS 기준:

```bash
source .venv/bin/activate
python3 app/camera_uploader.py --source 0 --server-url http://100.x.x.x:8000 --stt
```

여기서 `100.x.x.x`는 데스크탑의 주소입니다.
보통 Tailscale로 연결한 뒤 받은 IP를 넣으면 됩니다.

이렇게 실행하면:

- 노트북은 카메라와 마이크를 서버로 보내고
- 무거운 분석은 데스크탑이 하고
- 노트북 부담은 줄어듭니다

### 3. 웹에서 결과 보기

노트북 브라우저에서 아래 주소를 열면 됩니다.

```text
http://100.x.x.x:8000
```

이 웹 화면에서 볼 수 있는 정보:

- 현재 카메라 화면
- 사람 수
- 얼굴 수
- 음성 인식 결과
- 위험도
- 데스크탑 GPU / CPU / 메모리 상태

## 실행 전에 확인할 것

- 카메라 권한 허용
- 마이크 권한 허용
- 다른 앱이 카메라나 마이크를 쓰고 있지 않은지 확인

특히 macOS에서는 아래 경로에서 권한을 확인하면 됩니다.

- `시스템 설정 > 개인정보 보호 및 보안 > 카메라`
- `시스템 설정 > 개인정보 보호 및 보안 > 마이크`

## 화면에서 보이는 것

- `사람`: 현재 사람으로 인정된 대상 수
- `얼굴`: 현재 감지된 얼굴 수
- `음성 인식`: 지금 마이크가 잘 들어오는지와 인식 결과
- `위험도`: 현재 상황이 얼마나 위험해 보이는지 점수로 표시

## 로그 파일

위험도가 높게 올라가면 로그가 저장됩니다.

- 저장 위치: `logs/warnings.jsonl`

이 파일은 나중에 서버 전송이나 웹 대시보드 기능으로 확장할 때 그대로 사용할 수 있습니다.

## 잘 안 될 때 가장 먼저 볼 것

### 웹캠이 안 열릴 때

- 카메라 권한 확인
- `--source 0` 대신 다른 카메라 번호가 필요한지 확인
- Zoom, Meet, FaceTime 같은 앱 종료

### 마이크가 안 될 때

- 마이크 권한 확인
- 입력 장치가 올바른지 확인

### 노트북에서 웹 접속이 안 될 때

- 데스크탑 서버가 켜져 있는지 확인
- 데스크탑과 노트북이 같은 Tailscale 네트워크에 있는지 확인
- 주소의 `100.x.x.x`가 맞는지 다시 확인

## 한 줄 정리

가장 자주 쓰는 명령어는 아래 3개입니다.

로컬에서 바로 실행:

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR
```

데스크탑 서버 실행:

```powershell
python app\inference_server.py --host 0.0.0.0 --port 8000 --yolo-device cuda:0 --stt-device cuda --stt-compute-type float16
```

노트북에서 업로더 실행:

```bash
python3 app/camera_uploader.py --source 0 --server-url http://100.x.x.x:8000 --stt
```

## 행동 학습 자동화

행동 학습은 AIHub `aihubshell` 설정 파일을 기준으로 실행하는 것을 추천합니다.

설정 파일:

[`configs/action_training.aihub_shell.example.json`](/Users/jung-uiseok/Desktop/detectWarning/configs/action_training.aihub_shell.example.json)

`aihubshell` 파일이 프로젝트 루트에 있으면 예제 설정 그대로 써도 됩니다.
Windows에서 파일명이 `aihubshell.exe`여도 자동으로 찾도록 되어 있습니다.
AIHub 분할 압축 파일(`.zip.part*`)은 학습 파이프라인에서 자동으로 병합한 뒤 압축 해제합니다.
입력한 `filekey`가 실제 데이터셋 파일 목록에 없으면 다운로드 전에 먼저 검증해서 대시보드 로그에 안내합니다.

중요:

- `datasetkey` 는 AIHub 데이터셋 전체의 키입니다.
- 대시보드에 넣는 `filekey` 는 `insidedoor_01.zip | key: 49825` 같은 분할 ZIP의 key 값입니다.
- 예를 들어 `key: 49825` 는 보통 `filekey` 이고, `datasetkey` 로 넣는 값이 아닙니다.

학습 대시보드 실행:

```bash
python3 app/training_dashboard.py --config configs/action_training.aihub_shell.example.json --port 8010
```

그 다음 브라우저에서 `http://127.0.0.1:8010` 을 열고:

1. `datasetkey`를 입력합니다.
2. 분할 ZIP의 `filekey`를 입력합니다.
3. 필요하면 대시보드에서 `AIHub API 키`를 직접 입력합니다.
4. `대기열에 추가` 버튼을 누릅니다.
5. 여러 `filekey`를 넣으면 하나가 끝난 뒤 다음 작업이 자동으로 이어집니다.
6. 같은 화면에서 다운로드, 압축 해제, pose 전처리, 학습 진행률을 확인합니다.

현재 기본 동작은 누적 학습입니다.

- 각 `filekey` 작업이 끝나면 원본 다운로드/압축해제 데이터는 정리합니다.
- 대신 이전 작업에서 만든 `prepared pose` 데이터와 `best_action_model.pt` 는 유지합니다.
- 다음 `filekey` 학습 때는 새 데이터와 이전 prepared 데이터를 함께 복습하면서 이어서 학습합니다.
- 즉 저장 공간은 아끼면서도, 이전에 학습한 내용을 계속 누적하는 방식입니다.

노트북에서 확인하고 싶다면:

1. 데스크탑에서 위 대시보드를 실행합니다.
2. 노트북 브라우저에서 `http://데스크탑_Tailscale_IP:8010` 으로 접속합니다.
3. 같은 대시보드를 원격으로 보면서 현재 작업, 대기열, 최근 완료 작업을 확인할 수 있습니다.

터미널에서 바로 전체 파이프라인을 실행하고 싶다면:

```bash
python3 app/action_training_pipeline.py --config configs/action_training.aihub_shell.example.json --stage all
```
