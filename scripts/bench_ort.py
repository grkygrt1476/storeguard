#!/usr/bin/env python3
# scripts/bench_ort.py
'''
bench ONNX Runtime performance
'''
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _resolve_shape(model_input_shape, fallback=(1, 3, 320, 320)):
    # ONNX input shape may contain None / dynamic dim strings
    shape = []
    for i, d in enumerate(model_input_shape):
        if isinstance(d, int) and d > 0:
            shape.append(d)
        else:
            # dynamic -> use fallback
            shape.append(fallback[i] if i < len(fallback) else 1)
    return tuple(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .onnx")
    ap.add_argument("--provider", default="cuda", choices=["cuda", "cpu"], help="Execution provider")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shape", default="", help='Override input shape, e.g. "1,3,320,320"')
    ap.add_argument("--log", default="", help="Write log to this file (plain text + JSON summary)")
    ap.add_argument("--json", default="", help="Write JSON summary to this file")
    args = ap.parse_args()

    model = args.model
    np.random.seed(args.seed)

    providers = ["CPUExecutionProvider"]
    if args.provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(model, providers=providers)
    used_providers = sess.get_providers()

    inp = sess.get_inputs()[0]
    inp_name = inp.name
    if args.shape:
        shape = tuple(int(x.strip()) for x in args.shape.split(","))
    else:
        shape = _resolve_shape(inp.shape, fallback=(1, 3, 320, 320))

    x = np.random.rand(*shape).astype(np.float32)

    # warmup
    for _ in range(args.warmup):
        sess.run(None, {inp_name: x})

    # measure
    lat_ms = []
    t_all0 = time.perf_counter()
    for _ in range(args.runs):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: x})
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
    t_all1 = time.perf_counter()

    lat = np.array(lat_ms, dtype=np.float64)
    total_s = t_all1 - t_all0
    qps = args.runs / total_s if total_s > 0 else 0.0

    summary = {
        "model": str(model),
        "provider_requested": args.provider,
        "providers_used": used_providers,
        "input_name": inp_name,
        "input_shape": shape,
        "warmup": args.warmup,
        "runs": args.runs,
        "latency_ms": {
            "mean": float(lat.mean()),
            "p50": float(np.percentile(lat, 50)),
            "p90": float(np.percentile(lat, 90)),
            "p95": float(np.percentile(lat, 95)),
            "p99": float(np.percentile(lat, 99)),
            "min": float(lat.min()),
            "max": float(lat.max()),
        },
        "throughput_qps": float(qps),
        "total_walltime_s": float(total_s),
    }

    text_lines = [
        "=== ORT benchmark ===",
        f"model: {model}",
        f"providers_used: {used_providers}",
        f"input: name={inp_name} shape={shape}",
        f"runs: warmup={args.warmup} measure={args.runs}",
        f"throughput_qps: {summary['throughput_qps']:.3f}",
        "latency_ms: "
        + " ".join(
            [
                f"mean={summary['latency_ms']['mean']:.3f}",
                f"p50={summary['latency_ms']['p50']:.3f}",
                f"p90={summary['latency_ms']['p90']:.3f}",
                f"p95={summary['latency_ms']['p95']:.3f}",
                f"p99={summary['latency_ms']['p99']:.3f}",
                f"min={summary['latency_ms']['min']:.3f}",
                f"max={summary['latency_ms']['max']:.3f}",
            ]
        ),
    ]
    print("\n".join(text_lines))

    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(text_lines) + "\n\n")
            f.write(json.dumps(summary, indent=2) + "\n")

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
