# detectWarning 발표용 추가 시각화 자료

이 문서는 발표 때 기존 시스템 아키텍처 도표와 함께 쓰기 좋은 보조 시각화 자료를 정리한 것이다.  
PNG 자료는 `python docs\render_presentation_visuals.py`로 생성한다.

## 1. 음성 사용 전/후 위험 점수 차이

발표 포인트: “영상만으로는 애매했던 상황이 음성 위험 신호와 결합되면서 더 높은 위험도로 확정된다.”

```mermaid
flowchart LR
  A["영상만 분석<br/>videoOnlyScore 62<br/>주의 후보"] --> B["CLOVA STT<br/>도움 요청·위협·통증 표현"]
  B --> C["음성 위험 점수<br/>audioScore 31"]
  A --> D["영상+음성 결합<br/>riskScore 84<br/>위험 확정"]
  C --> D
  D --> E["Audio gain +22<br/>음성으로 위험 판단 강화"]
```

## 2. 위험 판단 상태 머신

발표 포인트: “한 번 튄 점수로 바로 위험을 띄우지 않고, 클래스별 threshold와 반복 확인을 거친다.”

```mermaid
stateDiagram-v2
  [*] --> Normal
  Normal --> Suspicious: score >= warning_min
  Suspicious --> Confirmed: danger_min 충족 또는 confirm_hits 충족
  Suspicious --> Normal: window 안에 반복 실패
  Confirmed --> Cooldown: 이벤트 전송 + 알림 후보
  Cooldown --> Normal: cooldown 종료

  note right of Suspicious
    violence: 2회/8초
    fall: 1회/8초
    loitering: 2회/14초
    audio_risk: 2회/20초
  end note
```

## 3. 영상·음성 결합 판단 매트릭스

발표 포인트: “음성만으로 무조건 danger를 만들지 않고, 영상 근거와 결합될 때 강하게 올린다.”

```mermaid
flowchart TB
  subgraph Audio["음성 신호"]
    A0["없음"]
    A1["약함<br/>큰 소리/애매한 단어"]
    A2["강함<br/>도움 요청/위협/통증"]
  end
  subgraph Video["영상 신호"]
    V0["정상/약함"]
    V1["주의 후보"]
    V2["강함"]
  end
  V0 --> R0["관찰 또는 의심<br/>audio_only_cap 적용"]
  V1 --> R1["주의 후보<br/>반복 확인"]
  V2 --> R2["영상만 위험 후보<br/>video_only_cap 적용"]
  A2 --> R3["영상+음성 확정<br/>audioVideoGain 상승"]
  R1 --> R3
  R2 --> R3
```

## 4. 클래스별 오탐 억제 장치

발표 포인트: “폭력, 쓰러짐, 배회는 오탐 패턴이 다르기 때문에 같은 threshold를 쓰지 않는다.”

```mermaid
flowchart LR
  V["violence"] --> V1["2명 이상 / 접촉 / 빠른 움직임"]
  V --> V2["음성 확인 시 가산<br/>비명·도움 요청·충격음"]
  V --> V3["영상만이면 cap 적용"]

  C["collapse"] --> C1["단일 프레임 억제"]
  C --> C2["temporal abnormal / 자세 근거"]
  C --> C3["통증·도움 요청·충격음 보강"]

  L["loitering"] --> L1["8초 이상 지속"]
  L --> L2["반복 hit 필요"]
  L --> L3["기다림·정지 패턴 suppress"]
```

## 5. 개인정보 보호 경계

발표 포인트: “음성 인식을 쓰지만 STT 원문은 앱 백엔드로 보내지 않는다.”

```mermaid
flowchart LR
  Mic["마이크/영상 음성"] --> STT["추론 서버<br/>CLOVA STT"]
  STT --> T["transcript<br/>서버 내부 전용"]
  T --> R["위험 신호 추출<br/>audioScore/eventClasses"]
  R --> M["메타데이터만 전송<br/>audioRiskSignalDetected<br/>audioScore<br/>audioVideoGain"]
  M --> B["앱 백엔드/PWA"]
  T -. "전송 금지" .-> B
```

## 6. 시연 흐름 스토리보드

발표 포인트: “카메라 입력부터 앱 알림과 위험 클립까지 실제 서비스 흐름으로 이어진다.”

```mermaid
sequenceDiagram
  participant C as 카메라/테스트 영상
  participant I as 추론 서버
  participant R as RiskAnalyzer
  participant B as 앱 백엔드
  participant P as PWA/iPhone

  C->>I: 영상 프레임 + 음성 조각
  I->>R: 행동 분석 + 음성 위험 신호
  R-->>I: videoOnlyScore / audioScore / riskScore
  I->>B: CCTV 상태 + 위험 이벤트
  B-->>P: SSE 실시간 점수 갱신
  B-->>P: Web Push 위험 알림
  P->>I: WebRTC direct 분석 영상 요청
  I-->>P: 분석 프레임 스트림
  P->>B: 위험 기록 상세 조회
  B-->>P: clipUrl / riskEvidence
```

