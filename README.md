# detectWarning

파이썬기반응용프로그래밍 4조 프로젝트입니다.

웹캠 또는 영상 파일에서 사람과 얼굴을 감지하고, 마이크 입력 음성과 함께 분석하여 실시간 위험도를 계산합니다.

현재 시스템은 다음 흐름으로 동작합니다.

- 영상: 사람 감지, 얼굴 감지, 간단한 움직임/추적 분석
- 음성: Whisper 기반 STT, 음량 분석, 상대 음량 급상승 감지
- 위험도: 영상 + 음성 + 키워드 + 큰 소리 패턴을 결합한 휴리스틱 점수 계산
- 이벤트 로그: 위험 점수가 일정 기준 이상이면 JSONL 형태로 저장
- 표시: 한국어 UI로 사람 수, 얼굴 수, STT 결과, 위험도를 화면에 오버레이
- 원격 추론: 팀원 노트북 카메라 영상을 데스크탑 서버로 보내 사람/얼굴 분석 가능

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

팀원 노트북 카메라 영상을 데스크탑에서 분석하려면 먼저 데스크탑에서 서버를 실행합니다.

데스크탑 서버 실행:

```bash
python3 app/inference_server.py --host 0.0.0.0 --port 8000
```

팀원 노트북 실행:

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR --server-url http://100.x.x.x:8000
```

`100.x.x.x`는 Tailscale로 연결된 데스크탑의 IP 주소입니다.

경고 감지용 권장 균형 설정:

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR --stt-model base --stt-phrase-seconds 1.8 --stt-beam-size 1
```

사람 감지를 좀 더 민감하게 하고 싶을 때:

```bash
python3 app/main.py --source 0 --person-score-threshold 0.1 --person-nms-threshold 0.4
```

사람 감지 정확도를 더 높이고 싶을 때:

```bash
python3 app/main.py --source 0 --person-imgsz 1280 --person-score-threshold 0.2
```

FPS를 더 높이고 싶을 때:

```bash
python3 app/main.py --source 0 --person-imgsz 640 --person-detect-interval 3
```

STT 정확도를 더 높이고 싶을 때:

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR --stt-model small --stt-phrase-seconds 2.8 --stt-beam-size 4 --stt-best-of 4
```

STT 지연을 더 줄이고 싶을 때:

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR --stt-model tiny --stt-phrase-seconds 1.5 --stt-beam-size 1
```

로컬 영상 파일 사용:

```bash
python3 app/main.py --source /path/to/video.mp4
```

위험 이벤트 로그까지 함께 저장:

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR --warning-log-path logs/warnings.jsonl --warning-log-min-score 60
```

원격 추론과 함께 로그도 저장:

```bash
python3 app/main.py --source 0 --stt --stt-language ko-KR --server-url http://100.x.x.x:8000 --warning-log-path logs/warnings.jsonl
```

종료는 `q` 또는 `Esc` 키로 할 수 있습니다.

화면 표시 정보

- 감지된 사람을 `사람 1` 같은 추적 ID와 함께 초록색 박스로 표시
- 감지된 얼굴을 파란색 박스로 표시
- 상단에 `사람: n | 얼굴: n` 표시
- `음성 인식`, `인식 내용`, `위험도`, `FPS`를 한국어로 표시
- 원격 추론 사용 시 `원격 추론 | 서버 지연: nms` 상태를 표시
- 최근 음성 키워드, 오디오 크기, 사람/얼굴 감지 결과를 합친 실시간 위험 점수 표시

## 위험 점수

- 위험 점수는 0~100 범위의 휴리스틱 값이며, 안전 인증을 받은 분류기는 아닙니다.
- 위험 단계는 `낮음`, `주의`, `경계`, `위험`으로 표시됩니다.
- 단일 신호 하나만 보기보다 영상과 음성 신호를 함께 사용해 점수를 계산합니다.

현재 반영된 기준은 다음과 같습니다.

- 음성 키워드:
  `살려줘`, `도와줘`, `하지 마`, `왜 이러세요`, `이러지 마세요`, `놔 주세요`, `불이야`, `경찰`, `신고해` 등
- 음성 강도:
  절대 음량이 큰 경우 가중치 부여
- 상대 음량:
  최근 몇 초간의 평균보다 갑자기 크게 올라간 소리를 `음량 급상승`, `고성 급상승`으로 반영
- 반복 패턴:
  큰 소리가 반복되면 `반복적 고성`
- 매우 큰 소리:
  `비명 의심`

PDF 기준을 반영한 현재 위험 평가 항목:

- 영상 분석 기준:
  `넘어짐 의심`, `몸싸움 의심`, `달리며 추격`, `장시간 쓰러짐`
- 음성 분석 기준:
  `비명 의심`, `반복적 고성`, `위협적 음성 패턴`

조합 가중치도 들어가 있습니다.

- 키워드 + 사람
- 키워드 + 얼굴
- 키워드 + 큰 소리
- 큰 소리 + 사람

## 위험 이벤트 로그

- 위험 점수가 기본 `60` 이상이면 `logs/warnings.jsonl`에 이벤트를 저장합니다.
- 같은 이벤트가 짧은 시간에 반복될 때는 중복 로그를 줄이기 위해 쿨다운이 적용됩니다.
- 저장 형식은 서버 전송을 염두에 둔 JSON Lines(`.jsonl`)입니다.

예시 스키마:

```json
{
  "event_id": "evt_20260401_143218_123456",
  "timestamp": "2026-04-01T14:32:18+09:00",
  "source": "0",
  "score": 78,
  "level": "HIGH",
  "categories": ["도움 요청", "반복적 고성", "사람:1"],
  "matched_keywords": ["도와줘"],
  "transcript": "도와주세요",
  "people_count": 1,
  "face_count": 1,
  "audio_level": 0.1832
}
```

## 참고 사항

- 영상 입력이 로컬 파일이어도 STT는 마이크를 사용합니다.
- 새로운 음성 인식 의존성은 `pip install -r requirements.txt`로 설치할 수 있습니다.
- STT는 현재 `faster-whisper`를 사용한 로컬 Whisper 추론으로 동작합니다.
- 사람 감지는 현재 Ultralytics `YOLO26n` 모델을 사용합니다.
- 처음 모델을 로드할 때는 Whisper 가중치 다운로드가 필요할 수 있어, 최초 1회는 인터넷 연결이 필요할 수 있습니다.
- 처음 사람 감지를 실행할 때도 YOLO26 가중치 다운로드가 필요할 수 있습니다.
- 기본 STT 설정은 경고 감지에 맞춘 균형형 설정입니다. `base` 모델과 짧은 구간, 빠른 디코딩을 사용합니다.
- `small`, `medium` 같은 더 큰 모델은 보통 더 정확하지만, CPU/GPU 자원을 더 많이 사용합니다.
- `tiny`는 더 빠르지만, 한국어 인식 품질까지 고려하면 보통 `base`가 더 좋은 균형점입니다.
- macOS에서는 OpenCV와 Whisper 의존성 간 FFmpeg 충돌을 피하기 위해 STT를 별도 프로세스로 실행합니다.
- 사람 감지가 느리면 `--person-imgsz` 값을 낮추고, 더 정확하게 보고 싶으면 값을 높여볼 수 있습니다.
- FPS가 부족하면 `--person-detect-interval` 값을 2 또는 3으로 높여 사람 감지를 덜 자주 수행할 수 있습니다.
- 서버 연동 전 단계로 `logs/warnings.jsonl` 파일을 그대로 읽어 전송 계층에 연결할 수 있습니다.
- `--server-url`을 주면 노트북은 로컬 YOLO/얼굴 추론을 하지 않고 JPEG 프레임만 서버로 전송합니다.
- 원격 추론 서버는 `client_id`별로 사람 추적 상태를 따로 유지합니다.
- 팀 프로젝트에서는 Tailscale로 데스크탑과 팀원 노트북을 같은 tailnet에 연결하는 방식을 권장합니다.

## 원격 추론 구조

- 팀원 노트북:
  카메라 입력, STT, 위험도 계산, UI 표시, 로그 저장
- 데스크탑 서버:
  사람 감지, 얼굴 감지, 사람 추적
- 통신 방식:
  노트북이 JPEG 프레임을 HTTP로 전송하고, 서버가 사람/얼굴 결과 JSON을 반환

노트북 쪽 주요 옵션:

- `--server-url`
  원격 추론 서버 주소
- `--server-client-id`
  팀원별 고유 식별자. 비우면 자동 생성
- `--server-timeout-seconds`
  서버 응답 대기 시간
- `--server-jpeg-quality`
  전송용 JPEG 품질. 낮출수록 빠르지만 화질이 떨어짐

서버 쪽 주요 옵션:

- `--host`
  바인드 주소. 팀원 접속을 받으려면 보통 `0.0.0.0`
- `--port`
  서버 포트
- `--person-score-threshold`
  서버 사람 감지 민감도
- `--person-imgsz`
  서버 YOLO 입력 크기
- `--client-session-ttl`
  팀원별 추적 상태 유지 시간

## 오탐 감소 로직

- 옷걸이에 걸린 옷, 마네킹, 정지된 전신 형태 같은 오탐을 줄이기 위해
  `얼굴이 없고 오랫동안 거의 움직이지 않는 사람 박스`는 화면에서 숨기도록 처리했습니다.
- 따라서 얼굴이 없고 장시간 정지된 객체는 사람으로 감지되더라도 표시되지 않을 수 있습니다.

## 성능 튜닝 팁

- 더 빠르게:
  `--person-imgsz 640 --person-detect-interval 3`
- 더 정확하게:
  `--person-imgsz 1280 --person-score-threshold 0.2`
- 사람을 더 민감하게 잡고 싶을 때:
  `--person-score-threshold 0.1`
- STT를 더 정확하게:
  `--stt-model small --stt-beam-size 4 --stt-best-of 4`
- STT를 더 빠르게:
  `--stt-model tiny --stt-phrase-seconds 1.5`
