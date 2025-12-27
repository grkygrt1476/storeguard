# scripts/onnx_smoke.py
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _fill_shape(shape):
    """ORT input shape may contain None. Fill with sensible defaults for YOLOv8n-320."""
    # expected: (1, 3, 320, 320)
    defaults = [1, 3, 320, 320]
    out = []
    for i, s in enumerate(shape):
        if isinstance(s, int) and s > 0:
            out.append(s)
        else:
            out.append(defaults[i] if i < len(defaults) else 1)
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    model_path = root / "outputs" / "models" / "yolov8n_320.onnx"
    if not model_path.exists():
        raise SystemExit(f"ONNX not found: {model_path}")

    providers = ort.get_available_providers()
    use = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]

    print("Available providers:", providers)
    print("Using providers:", use)

    sess = ort.InferenceSession(str(model_path), providers=use)

    inp = sess.get_inputs()[0]
    in_shape = _fill_shape(inp.shape)
    print(f"Input name={inp.name} shape={inp.shape} -> run_shape={in_shape} dtype={inp.type}")

    x = np.random.rand(*in_shape).astype(np.float32)

    # warmup
    for _ in range(3):
        _ = sess.run(None, {inp.name: x})

    # timed
    iters = 20
    t0 = time.perf_counter()
    out = None
    for _ in range(iters):
        out = sess.run(None, {inp.name: x})
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000 / iters
    print(f"[OK] avg infer: {avg_ms:.2f} ms over {iters} runs")

    # outputs
    out_names = [o.name for o in sess.get_outputs()]
    for name, arr in zip(out_names, out):
        arr = np.asarray(arr)
        print(f"Output {name}: shape={arr.shape} dtype={arr.dtype} min={arr.min():.4f} max={arr.max():.4f}")


if __name__ == "__main__":
    main()
