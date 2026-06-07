# detectWarning Stability Maintenance Notes

## Safe Runtime Defaults

- Desktop inference server: `http://127.0.0.1:8002`
- Local app backend: `http://127.0.0.1:8000`
- Shared runtime defaults live in `run_inference_common.cmd`.
- Local CLOVA credentials should live in `clova.env`; commit only `clova.env.example`.

## Large File Split Plan

The following files are intentionally left functional-first for now. Split them only in small, tested steps.

1. `app/training_dashboard.py`
   - Move queue/job orchestration into `training_dashboard_jobs.py`.
   - Move AIHub filekey policy/recommendation logic into `training_dashboard_filekeys.py`.
   - Keep FastAPI route registration thin.

2. `app/inference_server.py`
   - Keep `create_app()` and route wiring in this file.
   - Move dashboard HTML/JS into a dedicated template/static module.
   - Keep already split services (`stt_service.py`, `backend_bridge.py`, `event_clip_service.py`, `webrtc_stream.py`, `runtime_state.py`, `system_monitoring.py`) as the stable boundary.

3. `app/training_dashboard_view.py`
   - Extract JavaScript into a static file after a browser regression check.
   - Preserve IDs and API paths exactly during the split.

## Verification Checklist

Run these before a demo or after runtime changes:

```bat
run_tests.cmd
.venv\Scripts\python.exe -m py_compile app\inference_server.py app\camera_uploader.py app\webrtc_stream.py
.venv\Scripts\python.exe app\runtime_health_check.py --duration-seconds 10
```

If WebRTC fails, check `/health` for:

- `webrtc.lastError`
- `webrtc.lastOfferAgeSeconds`
- `webrtc.lastSuccessAgeSeconds`
- `app_backend_post.circuit_breaker`
