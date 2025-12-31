# Reproduction Guide

This document contains **copy-paste runnable commands** to reproduce the benchmark evidence and artifacts in this repo.

## What you can reproduce
- **D3**: YOLOv8n ONNXRuntime (CUDA EP) smoke + TensorRT FP16 engine build/bench (trtexec)
- **D4**: Triton (pipeline) end-to-end run + metrics JSON generation
- **D5**: Classifier (TAP / TT) ONNX export + ORT IO check + CUDA smoke latency

## Preconditions
- Repo root: run all commands from the repository root (`~/storeguard`)
- Python: `python3.10+` with `.venv` activated
- GPU: NVIDIA GPU + CUDA driver installed (WSL2 is supported)
- Optional (D3 TensorRT / D4 Triton): Docker + NVIDIA Container Toolkit

## Evidence locations
- Logs: `outputs/logs/` (see `outputs/logs/INDEX.md`)
- Non-committed heavy artifacts (expected):
  - TensorRT engine: `outputs/engine/*.engine`
  - Large model binaries: `outputs/models/*.onnx` (kept local; use SHA stamps in `artifacts/onnx/`)

> Notes
> - Commands are grouped by **Day (D3/D4/D5)** to match the project checklist.
> - If a command produces a log file, the log path is shown in the command block via `tee`.

## Commands

### D3 — TensorRT FP16 engine build (trtexec)
```bash
mkdir -p outputs/engine outputs/logs

trtexec_trt \
  --onnx=outputs/models/yolov8n_320.onnx \
  --saveEngine=outputs/engine/yolov8n_320_fp16.engine \
  --fp16 \
  2>&1 | tee outputs/logs/d3_trt_build_fp16.log

```

### D3 — ONNXRuntime CUDA smoke (YOLO, 320x320)
```
  # ORT CUDA EP로 20 runs 평균 latency 기록
  python scripts/onnx_smoke.py \
    2>&1 | tee outputs/logs/onnx_smoke_cuda.txt
```

### D4 — Triton E2E metrics evidence
```
  # Triton readiness (HTTP 200 등)
  cat outputs/logs/d4_triton_ready.http

  # Triton server log
  tail -n 200 outputs/logs/d4_triton_server_trt.log

  # Demo run log
  cat outputs/logs/d4_demo_run_trt.log

  # E2E metrics (mean/p95/max, fps_effective, drops 등)
  cat outputs/logs/d4_e2e_metrics_trt.json
```
### D5 — Classifier export + IO check + smoke (TAP)
```
  # export
  python scripts/d5_cls_onnx.py export \
    --run-dir outputs/modeling/run_006_seed_sweep_seed43 \
    --temporal tap \
    --out outputs/models/cls_tap_seed43.onnx \
    2>&1 | tee outputs/logs/d5_cls_tap_onnx_export.log

  # io
  python scripts/d5_cls_onnx.py io \
    --onnx outputs/models/cls_tap_seed43.onnx \
    2>&1 | tee outputs/logs/d5_cls_tap_onnx_io.log

  # smoke (CUDA)
  python scripts/d5_cls_onnx.py smoke \
    --onnx outputs/models/cls_tap_seed43.onnx \
    --T 16 --iters 50 --device cuda \
    2>&1 | tee outputs/logs/d5_cls_tap_onnx_smoke.log
```

### D5 — Classifier smoke (TT tiny, optional ablation evidence)
```
  python scripts/d5_cls_onnx.py io \
    --onnx outputs/models/cls_tt_tiny_seed43.onnx \
    2>&1 | tee outputs/logs/d5_cls_tt_tiny_onnx_io.log

  python scripts/d5_cls_onnx.py smoke \
    --onnx outputs/models/cls_tt_tiny_seed43.onnx \
    --T 16 --iters 50 --device cuda \
    2>&1 | tee outputs/logs/d5_cls_tt_tiny_onnx_smoke.log

```