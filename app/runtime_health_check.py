from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import urlopen


def fetch_json(url: str, timeout: float) -> dict:
    with urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8", errors="replace")
    data = json.loads(payload)
    return data if isinstance(data, dict) else {}


def summarize_backend(health: dict) -> str:
    return (
        f"backend ok webrtc={health.get('webrtcConnections', 0)} "
        f"sse_drop={health.get('realtimeDroppedMessages', 0)} "
        f"sse_sub={health.get('realtimeSubscribers', 0)}"
    )


def summarize_inference(health: dict) -> str:
    post = health.get("app_backend_post") if isinstance(health.get("app_backend_post"), dict) else {}
    circuit = post.get("circuit_breaker") if isinstance(post.get("circuit_breaker"), dict) else {}
    webrtc = health.get("webrtc") if isinstance(health.get("webrtc"), dict) else {}
    last_webrtc_error = str(webrtc.get("lastError") or "")
    return (
        f"inference ok sessions={health.get('sessions', 0)} "
        f"webrtc={health.get('webrtc_connections', 0)} "
        f"webrtc_err={last_webrtc_error or '-'} "
        f"backend_q={post.get('queue_size', 0)} "
        f"circuit={'OPEN' if circuit.get('open') else 'closed'} "
        f"fail={circuit.get('failure_count', 0)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll detectWarning runtime health endpoints.")
    parser.add_argument("--inference-url", default="http://127.0.0.1:8002")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--duration-seconds", type=float, default=1800.0)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    args = parser.parse_args()

    deadline = time.monotonic() + max(1.0, args.duration_seconds)
    while time.monotonic() <= deadline:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        parts = [timestamp]
        for label, base_url, summarizer in (
            ("inference", args.inference_url.rstrip("/"), summarize_inference),
            ("backend", args.backend_url.rstrip("/"), summarize_backend),
        ):
            try:
                parts.append(summarizer(fetch_json(f"{base_url}/health", args.timeout_seconds)))
            except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                parts.append(f"{label} error {type(exc).__name__}: {str(exc)[:120]}")
        print(" | ".join(parts), flush=True)
        time.sleep(max(0.5, args.interval_seconds))


if __name__ == "__main__":
    main()
