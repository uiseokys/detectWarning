@echo off
setlocal

chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"

set PYTHON_EXE=.venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" (
  echo [tests] Python virtual environment was not found: %PYTHON_EXE%
  exit /b 1
)

echo [tests] Running unit tests...
"%PYTHON_EXE%" -m unittest discover -s tests -p "test_*.py"
exit /b %ERRORLEVEL%
