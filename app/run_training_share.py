from __future__ import annotations

import argparse
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


URL_PATTERN = re.compile(r"https://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+")
LIVE_URL_PATTERNS = (
    re.compile(r"^https://[A-Za-z0-9-]+\.trycloudflare\.com/?$"),
    re.compile(r"^https://[A-Za-z0-9.-]+\.workers\.dev/?$"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="training_dashboard와 cloudflared tunnel을 한 번에 실행합니다."
    )
    parser.add_argument(
        "--config",
        default="configs/action_training.aihub_shell.example.json",
        help="training_dashboard에 넘길 설정 파일 경로",
    )
    parser.add_argument("--dashboard-host", default="0.0.0.0", help="dashboard bind host")
    parser.add_argument("--dashboard-port", type=int, default=8010, help="dashboard port")
    parser.add_argument(
        "--tunnel-url",
        default="http://127.0.0.1:8010",
        help="cloudflared가 연결할 로컬 dashboard URL",
    )
    parser.add_argument(
        "--cloudflared",
        default="cloudflared",
        help="cloudflared 실행 파일 경로 또는 명령",
    )
    parser.add_argument(
        "--live-url-env",
        default="DETECTWARNING_LIVE_URL",
        help="training_dashboard에 전달할 live URL 환경변수 이름",
    )
    return parser.parse_args()


def resolve_cloudflared_path(cloudflared_arg: str, project_root: Path) -> str:
    candidates: list[str] = []

    raw = str(cloudflared_arg or "").strip()
    if raw:
        candidates.append(raw)

    local_candidates = [
        project_root / "cloudflared.exe",
        project_root / "cloudflared",
        project_root / "tools" / "cloudflared.exe",
        project_root / "tools" / "cloudflared",
    ]
    candidates.extend(str(path) for path in local_candidates)

    common_windows = [
        r"C:\Program Files\cloudflared\cloudflared.exe",
        r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
    ]
    candidates.extend(common_windows)

    if raw:
        if raw.lower().endswith(".exe"):
            candidates.append(raw[:-4])
        else:
            candidates.append(f"{raw}.exe")

    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isabs(candidate) or "\\" in candidate or "/" in candidate:
            if Path(candidate).exists():
                return str(Path(candidate))
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    raise RuntimeError(
        "cloudflared 실행 파일을 찾지 못했습니다.\n"
        "확인할 것:\n"
        "1. cloudflared가 Windows에 설치되어 있는지\n"
        "2. PATH에 등록되어 있는지\n"
        "3. 또는 프로젝트 폴더에 cloudflared.exe가 있는지"
    )


def stop_processes_on_port(port: int) -> list[int]:
    stopped: list[int] = []
    if os.name != "nt":
        return stopped

    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception:
        return stopped

    seen: set[int] = set()
    port_suffix = f":{port}"
    for line in result.stdout.splitlines():
        if port_suffix not in line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        local_addr = parts[1]
        pid_text = parts[-1]
        if not local_addr.endswith(port_suffix):
            continue
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        if pid <= 0 or pid in seen:
            continue
        seen.add(pid)
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            stopped.append(pid)
        except Exception:
            continue
    return stopped


def stop_stale_detectwarning_python_processes(project_root: Path) -> list[int]:
    stopped: list[int] = []
    if os.name != "nt":
        return stopped

    project_hint = str(project_root).replace("\\", "\\\\").replace("'", "''").lower()
    current_pid = os.getpid()
    ps_script = rf"""
$project = '{project_hint}'
$selfPid = {current_pid}
Get-CimInstance Win32_Process |
  Where-Object {{
    $_.ProcessId -ne $selfPid -and
    $_.Name -match '^python(?:\.exe)?$' -and
    $_.CommandLine -and
    $_.CommandLine.ToLower().Contains($project) -and
    (
      $_.CommandLine -match 'training_dashboard\.py' -or
      $_.CommandLine -match 'action_training_pipeline\.py' -or
      $_.CommandLine -match 'run_training_share\.py'
    )
  }} |
  ForEach-Object {{
    try {{
      Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
      Write-Output $_.ProcessId
    }} catch {{}}
  }}
"""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception:
        return stopped

    for line in result.stdout.splitlines():
        pid_text = line.strip()
        if pid_text.isdigit():
            stopped.append(int(pid_text))
    return stopped


def clear_local_pycache(project_root: Path) -> None:
    for target in [
        project_root / "app" / "__pycache__",
        project_root / "__pycache__",
    ]:
        if not target.exists():
            continue
        try:
            shutil.rmtree(target)
        except Exception:
            pass


def start_tunnel(cloudflared_cmd: str, tunnel_url: str) -> tuple[subprocess.Popen, str]:
    process = subprocess.Popen(
        [cloudflared_cmd, "tunnel", "--url", tunnel_url],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    lines: list[str] = []
    live_url: str | None = None
    deadline = time.time() + 45

    while time.time() < deadline:
        if process.poll() is not None:
            output = "".join(lines[-40:])
            raise RuntimeError(
                "cloudflared가 live URL을 만들기 전에 종료되었습니다.\n"
                f"{output}"
            )
        line = process.stdout.readline() if process.stdout is not None else ""
        if not line:
            time.sleep(0.1)
            continue
        lines.append(line)
        for match in URL_PATTERN.finditer(line):
            candidate = match.group(0).rstrip(")")
            if any(pattern.match(candidate) for pattern in LIVE_URL_PATTERNS):
                live_url = candidate
                break
        if live_url:
            break

    if not live_url:
        output = "".join(lines[-60:])
        process.terminate()
        raise RuntimeError(
            "cloudflared 출력에서 live URL을 찾지 못했습니다.\n"
            f"{output}"
        )

    return process, live_url


def stream_process(prefix: str, process: subprocess.Popen) -> threading.Thread:
    def _reader() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            print(f"[{prefix}] {line.rstrip()}")

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return thread


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=8)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    dashboard_script = Path(__file__).resolve().with_name("training_dashboard.py")
    cloudflared_cmd = resolve_cloudflared_path(args.cloudflared, project_root)

    tunnel_process: subprocess.Popen | None = None
    dashboard_process: subprocess.Popen | None = None

    def cleanup(*_args) -> None:
        stop_process(dashboard_process)
        stop_process(tunnel_process)

    try:
        stale_pids = stop_stale_detectwarning_python_processes(project_root)
        if stale_pids:
            print(
                "[launcher] 이전 detectWarning Python 프로세스를 정리했습니다: "
                + ", ".join(str(pid) for pid in stale_pids)
            )
            time.sleep(1.0)
        stopped_pids = stop_processes_on_port(args.dashboard_port)
        if stopped_pids:
            print(
                "[launcher] 기존 dashboard 포트 점유 프로세스를 정리했습니다: "
                + ", ".join(str(pid) for pid in stopped_pids)
            )
            time.sleep(1.0)
        clear_local_pycache(project_root)
        print(f"[launcher] cloudflared = {cloudflared_cmd}")
        print("[launcher] cloudflared tunnel을 시작합니다...")
        tunnel_process, live_url = start_tunnel(cloudflared_cmd, args.tunnel_url)
        print(f"[launcher] live_url = {live_url}")

        env = os.environ.copy()
        env[args.live_url_env] = live_url

        dashboard_cmd = [
            sys.executable,
            str(dashboard_script),
            "--config",
            args.config,
            "--host",
            args.dashboard_host,
            "--port",
            str(args.dashboard_port),
        ]

        print("[launcher] training_dashboard를 시작합니다...")
        dashboard_process = subprocess.Popen(
            dashboard_cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        stream_process("cloudflared", tunnel_process)
        stream_process("dashboard", dashboard_process)
        print(f"[launcher] dashboard pid = {dashboard_process.pid}")
        time.sleep(2.0)
        if dashboard_process.poll() is not None:
            cleanup()
            raise RuntimeError(
                "training_dashboard가 바로 종료되었습니다. "
                "위의 [dashboard] 로그를 확인해 주세요."
            )

        def handle_signal(signum, _frame):
            print(f"[launcher] signal {signum} 수신, 종료합니다...")
            cleanup()
            raise SystemExit(0)

        signal.signal(signal.SIGINT, handle_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handle_signal)

        exit_code = dashboard_process.wait()
        cleanup()
        raise SystemExit(exit_code)

    except KeyboardInterrupt:
        cleanup()
        raise SystemExit(0)
    except Exception as exc:
        cleanup()
        print(f"[launcher] 오류: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
