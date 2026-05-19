@echo off
setlocal

chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=1
set PYTHONUNBUFFERED=1

cd /d "%~dp0"

set PYTHON_EXE=.venv\Scripts\python.exe
set DASHBOARD_URL=http://127.0.0.1:8000/
set ACCEL_DEVICE=cpu
set YOLO_DEVICE=cpu
set STT_DEVICE=cpu
set STT_COMPUTE_TYPE=int8
set PERSON_IMGSZ=640
set PERSON_SCORE=0.25

if not exist "%PYTHON_EXE%" (
  echo [launcher] Python virtual environment was not found: %PYTHON_EXE%
  echo [launcher] Please create .venv and install requirements first.
  pause
  exit /b 1
)

echo [launcher] Starting detectWarning server for MacBook camera upload...
echo [launcher] Windows PC will run AI processing only.
echo [launcher] MacBook should send camera frames to http://WINDOWS_PC_IP:8000

for /f %%D in ('"%PYTHON_EXE%" -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2^>nul') do set ACCEL_DEVICE=%%D
if "%ACCEL_DEVICE%"=="cuda" (
  set YOLO_DEVICE=cuda:0
  set STT_DEVICE=cuda
  set STT_COMPUTE_TYPE=float16
  set PERSON_IMGSZ=960
  set PERSON_SCORE=0.20
)

echo [launcher] Device: %ACCEL_DEVICE%
echo [launcher] Local dashboard: %DASHBOARD_URL%
echo [launcher] Possible Windows PC IP addresses:
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /c:"IPv4"') do echo   %%A
echo.
echo [launcher] MacBook command example:
echo python3 app/camera_uploader.py --source iphone --server-url http://WINDOWS_PC_IP:8000 --max-fps 12 --frame-width 840 --jpeg-quality 65 --stt --stt-phrase-seconds 1.2 --stt-silence-seconds 0.25 --select-audio-device
echo [launcher] MacBook camera scan:
echo python3 app/camera_uploader.py --list-video-devices --server-url http://WINDOWS_PC_IP:8000
echo [launcher] CLOVA STT only: set NCLOUD_CLOVA_CLIENT_ID and NCLOUD_CLOVA_CLIENT_SECRET before running this file.
echo.

start "" "%DASHBOARD_URL%"

"%PYTHON_EXE%" -X utf8 "app\inference_server.py" --host 0.0.0.0 --port 8000 --yolo-device %YOLO_DEVICE% --person-imgsz %PERSON_IMGSZ% --person-score-threshold %PERSON_SCORE% --person-detect-interval 2 --stt-provider clova --clova-timeout-seconds 2.0 --stt-model medium --stt-device %STT_DEVICE% --stt-compute-type %STT_COMPUTE_TYPE% --stt-beam-size 5 --stt-best-of 5 --stt-no-speech-threshold 0.45 --action-artifacts-dir "training_data\action_pipeline_aihub\artifacts" --action-rgb-model i3d_r50 --action-clip-seconds 4 --action-interval-seconds 2 --action-normal-threshold 0.78 --action-min-confidence 0.45 --action-collapse-static-motion-threshold 4 --action-collapse-static-abnormal-threshold 0.90
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo [launcher] Server stopped with exit code=%EXIT_CODE%
  pause
)

exit /b %EXIT_CODE%
