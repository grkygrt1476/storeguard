# storeguard
Real-time unmanned store CCTV anomaly detection (ONNX + TensorRT).무인매장 CCTV 영상을 실시간 스트림처럼 처리하고 이상행동 감지 프로젝트

무인가게(또는 CCTV 환경)에서 발생할 수 있는 이상행동을 **실시간 영상 파이프라인**으로 탐지/표시하는 미니 데모입니다.  
5일 안에 “돌아가는 증거”를 남기는 것이 목표이며, 이후 상용 최적화 루트(ONNX Runtime/TensorRT)까지 확장 가능한 구조로 설계합니다.

## Goals
- 영상 입력(loop) → 추론 → 오버레이 출력까지 **엔드투엔드 파이프라인 증명**
- baseline vs 최적화(ONNX Runtime → TensorRT)로 **FPS/latency 비교 지표** 남기기
- 결과물: 실행 방법, 스크린샷, 성능 로그, 간단한 아키텍처/설명(README)

## Scope (5 days)
- D1: Video loop + FPS overlay + screenshot
- D2: ROI intrusion event (YOLO person + head-point rule)
- D3: TensorRT FP16 engine build + evidence log + benchmark
- D4: ORT(CUDA) vs TRT(FP16) 비교 + E2E FPS/p95 측정
- D5: (Optional) Triton serving + 포트폴리오 패키징

## Demo (D1-2 Video Loop MVP)
```bash
python scripts/demo_video.py
```
- q: quit
- s: save screenshot to assets/images/storeguard_d1_video_loop.jpg

![D1 video loop](assets/images/storeguard_d1_video_loop.jpg)

## Demo (D2 Intrusion Event)
```bash
python scripts/demo_video.py
```
- q: quit
- s: save screenshot (e.g., assets/images/storeguard_d2_intrusion.jpg)

Intrusion rule: if "head-point" (bbox center at 20% height) enters ROI rectangle, trigger event with cooldown.
![D2 intrusion](assets/images/storeguard_d2_intrusion.jpg)

## ONNX Smoke Test (WSL GPU)
```bash
./scripts/run_onnx.sh scripts/onnx_smoke.py
./scripts/run_onnx.sh -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
## NGC (nvcr.io) Login (for TensorRT container)
- Username: `$oauthtoken`
- Password: NGC API Key 

## TensorRT FP16 (D3)

We export YOLOv8n to ONNX and build a TensorRT FP16 engine using `trtexec` (via NGC Docker image).
This leaves reproducible evidence logs and produces an `.engine` for future Triton serving.
- Input ONNX: `outputs/models/yolov8n_320.onnx`
- Output Engine: `outputs/engine/yolov8n_320_fp16.engine`
- Evidence log: `outputs/logs/d3_trt_build_fp16.log`
- Benchmark (trtexec summary): Throughput ~742 qps, mean latency ~1.27 ms (RTX 4080 SUPER, FP16)

### Performance (single-batch, 320x320, RTX 4080 SUPER)
| Runtime | Precision | Mean Latency (ms) | Throughput (qps) | Evidence |
|---|---:|---:|---:|---|
| TensorRT | FP16 | 1.27 | 741.8 | outputs/logs/d3_trt_build_fp16.log |

### 1) Run TensorRT container (WSL/GPU)
We use the NGC TensorRT container so local TensorRT installation is not required.

### 2) Build FP16 engine (ONNX → .engine)
```bash
mkdir -p outputs/engine outputs/logs

trtexec_trt \
  --onnx=outputs/models/yolov8n_320.onnx \
  --saveEngine=outputs/engine/yolov8n_320_fp16.engine \
  --fp16 \
  2>&1 | tee outputs/logs/d3_trt_build_fp16.log
```
### Environment stamp (D3)
- Host: RTX 4080 SUPER (WSL2), see `nvidia-smi` in `outputs/logs/d3_trt_stamp.log`
- Docker: see `outputs/logs/d3_trt_stamp.log`
- TensorRT container: `nvcr.io/nvidia/tensorrt:25.12-py3` (TensorRT 10.14.1)

### Reproducible TensorRT commands (no shell functions)
- `scripts/trt.sh`: runs commands inside the TensorRT NGC container
- `scripts/trtexec_trt.sh`: convenience wrapper for `trtexec`

Example:
```bash
./scripts/trtexec_trt.sh --help | head
```

## 15 완료 판정
- 새 셸(함수 없는 상태)에서 아래가 되면 끝
```bash
# 1) trtexec 동작 확인
./scripts/trtexec_trt.sh -h | head

# 2) TensorRT 버전 스탬프
./scripts/trt.sh python -c "import tensorrt as trt; print(trt.__version__)"

```