# detectWarning

파이썬기반응용프로그래밍 4조

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행

웹캠 사용:

```bash
python3 app/main.py --source 0
```

영상과 함께 마이크 음성 인식(STT) 사용:

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR
```

경고 감지용 권장 균형 설정:

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR --stt-model base --stt-phrase-seconds 1.8 --stt-beam-size 1
```

정확도를 더 높이고 싶을 때:

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR --stt-model small --stt-phrase-seconds 2.8 --stt-beam-size 4 --stt-best-of 4
```

지연을 더 줄이고 싶을 때:

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR --stt-model tiny --stt-phrase-seconds 1.5 --stt-beam-size 1
```

로컬 영상 파일 사용:

```bash
python3 app/main.py --source /path/to/video.mp4
```

종료는 `q` 또는 `Esc` 키로 할 수 있습니다.

화면에는 다음 정보가 표시됩니다:

- 감지된 사람을 `Person 1` 같은 추적 ID와 함께 초록색 박스로 표시
- 감지된 얼굴을 파란색 박스로 표시
- 마이크에서 인식한 STT 상태와 최근 음성 인식 결과 표시
- 최근 음성 키워드, 오디오 크기, 사람/얼굴 감지 결과를 합친 실시간 위험 점수 표시

## 위험 점수

- 위험 점수는 0~100 범위의 휴리스틱 값이며, 안전 인증을 받은 분류기는 아닙니다.
- `살려줘`, `도와줘`, `하지마`, `불이야` 같은 긴급 표현이 인식되면 더 민감하게 반응합니다.
- 큰 소리와 사람/얼굴 감지가 함께 나타나면 점수가 더 올라가도록 설계되어 있습니다.

## 참고 사항

- 영상 입력이 로컬 파일이어도 STT는 마이크를 사용합니다.
- 새로운 음성 인식 의존성은 `pip install -r requirements.txt`로 설치할 수 있습니다.
- STT는 현재 `faster-whisper`를 사용한 로컬 Whisper 추론으로 동작합니다.
- 처음 모델을 로드할 때는 Whisper 가중치 다운로드가 필요할 수 있어, 최초 1회는 인터넷 연결이 필요할 수 있습니다.
- 기본 STT 설정은 경고 감지에 맞춘 균형형 설정입니다. `base` 모델과 짧은 구간, 빠른 디코딩을 사용합니다.
- `small`, `medium` 같은 더 큰 모델은 보통 더 정확하지만, CPU/GPU 자원을 더 많이 사용합니다.
- `tiny`는 더 빠르지만, 한국어 인식 품질까지 고려하면 보통 `base`가 더 좋은 균형점입니다.
- macOS에서는 OpenCV와 Whisper 의존성 간 FFmpeg 충돌을 피하기 위해 STT를 별도 프로세스로 실행합니다.
