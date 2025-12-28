# Performance Notes

This file tracks benchmark results across stages (ONNXRuntime → TensorRT FP16 → Triton).

---

## Environment (D3)

- Host: WSL2 (Ubuntu 24.04), RTX 4080 SUPER
- Docker: `nvcr.io/nvidia/tensorrt:25.12-py3`
- TensorRT: 10.14.1 (inside container)
- Model: YOLOv8n (exported to ONNX, 320x320)

Evidence:
- TRT build/bench log: `outputs/logs/d3_trt_build_fp16.log`
- Engine output: `outputs/engine/yolov8n_320_fp16.engine` *(not committed)*

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

| Stage | Runtime | Precision | Mean Latency (ms) | Throughput (qps) | Notes | Evidence |
|---|---|---:|---:|---:|---|---|
| D3 | TensorRT (trtexec) | FP16 | 1.270 | 741.8 | single-stream, trtexec summary | `outputs/logs/d3_trt_build_fp16.log` |
| D3 | ONNXRuntime | CUDA | TBD | TBD | to be measured (Task #11) | (to add) |
| D4 | Triton | FP16 | TBD | TBD | to be measured | (to add) |

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
