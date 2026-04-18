# Security Policy

## Supported Scope

이 저장소는 학습 대시보드, 원격 추론 서버, Pages 리포트, 로컬/원격 런처를 포함합니다.
현재는 최신 `main` 기준만 보안 수정 대상으로 봅니다.

## 주요 주의사항

- `AIHUB_API_KEY`, `DETECTWARNING_LIVE_URL`, `DETECTWARNING_WORKSPACE_DIR` 같은 환경 변수는 절대 로그나 커밋에 남기지 않습니다.
- `training_data/`, `logs/`, 생성된 `latest-result.json`, `live-status.json`에는 내부 경로와 운영 정보가 포함될 수 있으므로 공개 저장소에 그대로 올리지 않습니다.
- Cloudflare Tunnel이나 외부 네트워크로 대시보드를 노출할 때는 허용 origin을 명시적으로 제한하는 것을 권장합니다.
- Pages 화면이나 대시보드에서 표시하는 로그/메시지는 외부 입력을 그대로 HTML로 주입하지 않도록 escaping을 유지해야 합니다.

## 취약점 제보

보안 문제를 발견하면 공개 이슈 대신 프로젝트 담당자에게 직접 전달하는 것을 권장합니다.
제보에는 아래를 포함해 주세요.

- 재현 단계
- 영향 범위
- 예상되는 공격 시나리오
- 관련 로그나 스크린샷

확인 후 필요하면 영향 범위와 수정 계획을 문서화해 반영합니다.
