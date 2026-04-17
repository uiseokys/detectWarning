@echo off
setlocal

chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=1
set PYTHONUNBUFFERED=1

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [launcher] .venv\Scripts\python.exe 를 찾을 수 없습니다.
  echo [launcher] 먼저 가상환경이 있는지 확인해 주세요.
  pause
  exit /b 1
)

echo [launcher] detectWarning 공유 런처를 시작합니다...
".venv\Scripts\python.exe" -X utf8 "app\run_training_share.py" --config "configs\action_training.aihub_shell.example.json"
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo [launcher] 실행이 종료되었습니다. exit code=%EXIT_CODE%
  pause
)

exit /b %EXIT_CODE%
