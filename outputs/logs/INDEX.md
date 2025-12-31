# outputs/logs — Evidence Index

> Purpose: keep reproducible evidence (logs/metrics/http stamps) without committing heavy binaries
> (e.g., engines/models/videos). This folder is the “paper trail” for performance + serving steps.

## Naming convention
- `d{N}_*`: day-based work log set
  - D3: TensorRT build + Triton 1-shot
  - D4: End-to-end demo (Triton pipeline)
  - D5: Classifier ONNX export + ORT smoke
- `.log`: human-readable run log
- `.json`: metrics artifact (machine-readable)
- `.http`: readiness probe output
- `.txt`: notes / environment / fallback memo / quick smoke

---
## D5 — Classifier ONNX (TAP / TT tiny)

### TAP
- `d5_cls_tap_onnx_io.log` — ONNX opset + I/O names/shapes (ORT session)
- `d5_cls_tap_onnx_smoke.log` — ORT latency smoke (CUDA EP) + sample output

### TT tiny
- `d5_cls_tt_tiny_onnx_io.log` — ONNX opset + I/O names/shapes (ORT session)
- `d5_cls_tt_tiny_onnx_smoke.log` — ORT latency smoke (CUDA EP) + sample output

### D5 (local-only / noisy)
- `d5_cls_onnx_export.log` — early export log (kept for debugging)
- `d5_cls_onnx_io.log` — early ONNX I/O check (kept for debugging)

> Related (outside this folder)
> - `artifacts/onnx/d5_cls_tap_seed43.sha256` — SHA256 stamp for the exported TAP ONNX
> - `artifacts/onnx/d5_cls_tt_tiny_seed43.sha256` — SHA256 stamp for the exported TT tiny ONNX


---

## D4 — E2E Demo (Triton pipeline + TRT)
- `d4_demo_run_trt.log` — baseline demo run trace
- `d4_e2e_metrics_trt.json` — baseline E2E metrics (latency breakdown + fps + drops)
- `d4_triton_ready.http` — Triton readiness probe output
- `d4_triton_server_trt.log` — Triton server log (TRT backend)

### D4 variants
- `d4_demo_run_trt_overload.log` — overload scenario run trace
- `d4_e2e_metrics_trt_overload.json` — overload E2E metrics
- `d4_demo_run_trt_roi.log` — ROI scenario run trace
- `d4_e2e_metrics_trt_roi.json` — ROI E2E metrics

---

## D3 — TensorRT build + Triton 1-shot
- `d3_trt_build_fp16.log` — `trtexec` build + benchmark summary (FP16)
- `d3_trt_stamp.log` — quick stamp / summary memo
- `d3_triton_server_env.log` — server/container environment dump
- `d3_triton_nvidia_smi.log` — GPU snapshot during run

### D3 Triton server/client
- `d3_triton_server_onnx.log` — Triton server log (ONNX backend)
- `d3_triton_server_trt.log` — Triton server log (TRT backend)
- `d3_triton_client_once.log` — 1-shot client request log (ONNX)
- `d3_triton_client_once_trt.log` — 1-shot client request log (TRT)

### D3 notes
- `d3_fallback_plan.txt` — fallback notes (if TRT/Triton issues)

---

## Misc / quick smoke (utility logs)
- `onnx_smoke_cuda.txt` — ORT CUDA EP smoke for YOLO (single inference avg)
- `trt_env_and_build.log` — TRT environment/build notes
- `loader_smoke_train.txt` — dataloader smoke (train)
- `loader_smoke_val.txt` — dataloader smoke (val)
- `modeling_smoke_row0.txt` — modeling smoke sample output
