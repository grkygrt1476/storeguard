## Environment (Common)

- Host: WSL2 (Ubuntu 24.04), RTX 4080 SUPER
- Evidence:
  - `outputs/logs/d3_triton_nvidia_smi.log`
  - `outputs/logs/d3_triton_server_env.log`

---

## Environment (D3) — Model inference baselines

- Model: YOLOv8n (ONNX, 320x320)
- TensorRT: FP16 engine built via `trtexec` inside container
- Evidence:
  - TRT build/bench log: `outputs/logs/d3_trt_build_fp16.log`
  - ORT CUDA smoke: `outputs/logs/onnx_smoke_cuda.txt`
  - Engine output: `outputs/engine/yolov8n_320_fp16.engine` *(not committed)*

---

## Environment (D4) — Triton E2E pipeline (real-time)

- Serving: Triton Server (TensorRT backend, FP16 engine)
- Pipeline: preprocess → infer (Triton) → encode/overlay
- Evidence:
  - Triton readiness: `outputs/logs/d4_triton_ready.http`
  - Triton server log: `outputs/logs/d4_triton_server_trt.log`
  - E2E run log: `outputs/logs/d4_demo_run_trt.log`
  - E2E metrics: `outputs/logs/d4_e2e_metrics_trt.json`

> Notes:
> - `drops` = real-time pipeline에서 “처리 지연 때문에 버린 프레임 수”
>   - `0`이면 “프레임 드랍 없이” 끝까지 처리했다는 뜻

---

## Environment (D5) — Classifier ONNX export + ORT (TAP/TT)

- Model: MobileNetV3 Small + temporal head (T=16, 224x224)
- Export: ONNX opset=18
- ORT providers: CUDAExecutionProvider (fallback CPU)
- Evidence (TAP):
  - SHA256 stamp: `artifacts/onnx/d5_cls_tap_seed43.sha256`
  - IO check: `outputs/logs/d5_cls_tap_onnx_io.log`
  - CUDA smoke: `outputs/logs/d5_cls_tap_onnx_smoke.log`

---

## Method

### Common settings
- Input shape: `1x3x320x320`
- Batch: 1
- Goal: capture **mean latency** and a simple throughput number

### TensorRT (trtexec)
- Build: ONNX → `.engine` with FP16
- Bench: trtexec performance summary (mean latency / throughput)

---

## Results

> Conventions
> - **Mean (ms)** and **p95 (ms)** always refer to the measured unit in each table.
> - **Rate (1/s)** is always shown as “per second”, but the **unit type** differs:
>   - Inference: **infer/s** (≈ qps)
>   - End-to-end: **frames/s** (fps)
> - `drops` = number of frames dropped by the real-time pipeline (0 means no drops)

### Inference benchmarks (unit = one model inference)

| Stage | Runtime | Precision | Mean (ms) | p95 (ms) | Rate (1/s) | Rate type | Notes | Evidence |
|---|---|---|---:|---:|---:|---|---|---|
| D3 | TensorRT (trtexec) | FP16 | 1.27 | - | 741.80 | infer/s | single-stream; trtexec summary | `outputs/logs/d3_trt_build_fp16.log` |
| D3 | ONNXRuntime (ORT) | FP32 | 3.63 | - | 275.48 | infer/s | CUDAExecutionProvider; 20 runs | `outputs/logs/onnx_smoke_cuda.txt` |
| D5 | Classifier (ORT) | FP32 | 2.96 | 4.91 | 337.70 | infer/s | TAP ONNX; CUDAExecutionProvider; 50 iters | `outputs/logs/d5_cls_tap_onnx_smoke.log` |
| D5 | Classifier (ORT) | FP32 | 2.92 | 3.07 | 342.70 | infer/s | TT tiny ONNX; CUDAExecutionProvider; 50 iters | `outputs/logs/d5_cls_tt_tiny_onnx_smoke.log` |

### End-to-end pipeline benchmarks (unit = one video frame through full pipeline)

| Stage | Runtime | Precision | Mean (ms) | p95 (ms) | Rate (1/s) | Rate type | Drops | Notes | Evidence |
|---|---|---|---:|---:|---:|---|---:|---|---|
| D4 | Triton (pipeline) | FP16 | 9.07 | 11.17 | 102.90 | frames/s | 0 | total=preprocess+infer+encode | `outputs/logs/d4_e2e_metrics_trt.json` |




---

## Commands (D3)

### Build TensorRT FP16 engine
```bash
mkdir -p outputs/engine outputs/logs

trtexec_trt \
  --onnx=outputs/models/yolov8n_320.onnx \
  --saveEngine=outputs/engine/yolov8n_320_fp16.engine \
  --fp16 \
  2>&1 | tee outputs/logs/d3_trt_build_fp16.log

```
## Next
- Measure ONNXRuntime (CUDA) latency with same input shape (1x3x320x320)
- Add end-to-end FPS/p95 latency for demo pipeline
- (Optional) Triton serving benchmark (concurrency)
