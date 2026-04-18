# Pages 공유 리포트

## 목적

- 서버가 꺼져 있어도 최신 학습 결과를 팀원이 볼 수 있게 합니다.
- live 대시보드가 열릴 수 있을 때는 실시간 화면으로 전환합니다.

## 주요 파일

- `pages_site/index.html`
- `pages_site/latest-result.json`
- `pages_site/live-status.json`
- `app/update_pages_site.py`

## 동작 흐름

1. 학습 결과가 나오면 `latest-result.json`을 갱신합니다.
2. live 대시보드가 켜지면 `live-status.json`을 `online`으로 갱신합니다.
3. Pages 화면은 offline 리포트를 기본으로 보여주고, live가 확인되면 실시간 뷰로 연결합니다.

## 운영 팁

- Quick Tunnel은 주소가 바뀔 수 있으므로 상태 전환이 완벽하게 즉시 맞지 않을 수 있습니다.
- 정확한 online/offline 판정이 중요하면 고정 live URL 구조로 옮기는 것이 맞습니다.
- Pages 리포트에는 내부 경로와 작업 이력이 들어갈 수 있으니 공개 범위를 꼭 확인합니다.
