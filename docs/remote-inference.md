# 원격 추론 서버와 업로더

## 서버 실행

```powershell
python app\inference_server.py --host 0.0.0.0 --port 8000 --yolo-device cuda:0 --stt-device cuda --stt-compute-type float16
```

## 노트북 업로더 실행

```bash
python3 app/camera_uploader.py --source 0 --server-url http://100.x.x.x:8000 --stt
```

## 확인 포인트

- 서버는 GPU/CPU/메모리 상태와 클라이언트별 위험도, 사람 수, 얼굴 수, STT 상태를 보여줍니다.
- 업로더는 카메라 프레임과 오디오를 서버로 보내고, 서버는 클라이언트별 세션을 유지합니다.
- 네트워크가 불안정하면 업로더 timeout과 JPEG 품질을 먼저 조정합니다.

## 자주 보는 파일

- `app/inference_server.py`
- `app/camera_uploader.py`
- `app/remote_inference.py`
- `app/audio_detector.py`
