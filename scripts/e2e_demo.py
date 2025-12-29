#!/usr/bin/env python3
"""
scripts/e2e_demo.py

E2E demo runner (portfolio stamp):
video -> preprocess -> Triton infer -> HUD overlay -> mp4 + metrics.json

- stage breakdown (pre/infer/post/encode)
- warmup frames excluded from stats
- input/output tensor names are configurable
- optional real-time style dropping (latest-wins) to keep bounded latency
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

import tritonclient.http as httpclient


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def percentile(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p)) if arr.size else 0.0


def stat_ms(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "p95": percentile(arr, 95),
        "max": float(arr.max()),
    }


def preprocess_bgr_to_nchw_fp32(frame_bgr: np.ndarray, size: int = 320) -> np.ndarray:
    # BGR -> RGB -> resize -> normalize [0,1] -> CHW -> NCHW
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = np.expand_dims(x, axis=0)   # NCHW
    return x


def open_writer_mp4(out_path: Path, fps: float, w: int, h: int) -> cv2.VideoWriter:
    # Try common codecs; environments differ (WSL/Ubuntu often picky)
    codecs = ["mp4v", "avc1", "H264", "XVID"]
    for cc in codecs:
        fourcc = cv2.VideoWriter_fourcc(*cc)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Failed to open VideoWriter for {out_path} (tried codecs: {codecs})")


def infer_triton_http(
    client: httpclient.InferenceServerClient,
    model_name: str,
    input_name: str,
    output_name: str,
    x_nchw: np.ndarray,
) -> np.ndarray:
    inp = httpclient.InferInput(input_name, x_nchw.shape, "FP32")
    inp.set_data_from_numpy(x_nchw, binary_data=True)
    out = httpclient.InferRequestedOutput(output_name, binary_data=True)
    resp = client.infer(model_name=model_name, inputs=[inp], outputs=[out])
    y = resp.as_numpy(output_name)
    if y is None:
        raise RuntimeError(f"Triton returned None for output={output_name} (check config.pbtxt)")
    return y


def main():
    p = argparse.ArgumentParser()

    # IO
    p.add_argument("--video", required=True, help="input video path")
    p.add_argument("--out", required=True, help="output mp4 path")
    p.add_argument("--metrics", required=True, help="output metrics json path")

    # Triton
    p.add_argument("--url", default="localhost:8000", help="triton http url host:port")
    p.add_argument("--model", required=True, help="triton model name")
    p.add_argument("--input-name", default="images", help="triton input tensor name")
    p.add_argument("--output-name", default="output0", help="triton output tensor name")

    # Demo control
    p.add_argument("--img", type=int, default=320, help="model input size (square)")
    p.add_argument("--warmup", type=int, default=5, help="warmup frames excluded from stats")
    p.add_argument("--max-frames", type=int, default=0, help="0 means full video")
    p.add_argument("--max-seconds", type=float, default=0.0, help="0 means full video")
    p.add_argument("--show", action="store_true", help="preview window (q to quit)")

    # Real-time style (bounded latency simulation)
    p.add_argument("--queue-size", type=int, default=0, help="0 disables dropping")
    p.add_argument("--drop-policy", choices=["latest"], default="latest", help="drop policy when overloaded")
    p.add_argument("--target-fps", type=float, default=0.0, help="0 uses input fps; used for schedule")

    # Optional future arg (kept for CLI compatibility; not used here)
    p.add_argument("--roi", default="", help="roi json path (not used in this stub)")

    args = p.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out)
    metrics_path = Path(args.metrics)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    fps_schedule = args.target_fps if args.target_fps > 0 else fps_in
    frame_interval = 1.0 / fps_schedule if fps_schedule > 0 else 1.0 / 30.0

    writer = open_writer_mp4(out_path, fps_in, w, h)

    client = httpclient.InferenceServerClient(url=args.url, verbose=False)
    # Light sanity checks (no hard fail if metadata endpoints blocked)
    try:
        if not client.is_server_live():
            raise RuntimeError("Triton server is not live")
        if not client.is_server_ready():
            raise RuntimeError("Triton server is not ready")
    except Exception:
        # Keep going; errors will surface on infer anyway
        pass

    # Stats (exclude warmup)
    lat_pre: List[float] = []
    lat_inf: List[float] = []
    lat_post: List[float] = []
    lat_enc: List[float] = []
    lat_total: List[float] = []

    frames_in = 0
    frames_processed = 0
    frames_written = 0
    drops = 0

    t0 = time.perf_counter()
    start_wall = t0
    schedule_start = t0  # schedule origin

    def elapsed_s() -> float:
        return time.perf_counter() - start_wall

    while True:
        # Stop by time
        if args.max_seconds > 0 and elapsed_s() >= args.max_seconds:
            break

        ok, frame = cap.read()
        if not ok:
            break
        frames_in += 1
        if args.max_frames and frames_in > args.max_frames:
            break

        # --- bounded latency dropping (simulate real-time) ---
        if args.queue_size > 0:
            # "virtual now" vs expected time of this frame
            expected_t = schedule_start + (frames_in - 1) * frame_interval
            lag_s = time.perf_counter() - expected_t
            # if lag exceeds queue budget, drop until within budget
            budget_s = args.queue_size * frame_interval
            if lag_s > budget_s:
                # latest-wins: skip frames to catch up
                while lag_s > budget_s:
                    ok2, _ = cap.read()
                    if not ok2:
                        break
                    frames_in += 1
                    drops += 1
                    expected_t = schedule_start + (frames_in - 1) * frame_interval
                    lag_s = time.perf_counter() - expected_t

        t_total0 = now_ms()

        # preprocess
        t_pre0 = now_ms()
        x = preprocess_bgr_to_nchw_fp32(frame, size=args.img)
        t_pre1 = now_ms()

        # infer
        t_inf0 = now_ms()
        y = infer_triton_http(
            client=client,
            model_name=args.model,
            input_name=args.input_name,
            output_name=args.output_name,
            x_nchw=x,
        )
        t_inf1 = now_ms()

        # postprocess (stub for now; just touch shape)
        t_post0 = now_ms()
        _ = y.shape
        t_post1 = now_ms()

        # HUD overlay
        run_s = time.perf_counter() - t0
        fps_eff = frames_written / run_s if run_s > 0 else 0.0
        hud = (
            f"{args.model} | fps={fps_eff:.1f} | "
            f"pre={t_pre1-t_pre0:.1f}ms inf={t_inf1-t_inf0:.1f}ms enc=?"
        )
        cv2.putText(frame, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)

        # encode/write
        t_enc0 = now_ms()
        writer.write(frame)
        t_enc1 = now_ms()

        t_total1 = now_ms()

        frames_processed += 1
        frames_written += 1

        # record stats after warmup
        if frames_processed > args.warmup:
            lat_pre.append(t_pre1 - t_pre0)
            lat_inf.append(t_inf1 - t_inf0)
            lat_post.append(t_post1 - t_post0)
            lat_enc.append(t_enc1 - t_enc0)
            lat_total.append(t_total1 - t_total0)

        # update HUD encode number cheaply (optional)
        if frames_processed % 30 == 0:
            # print lightweight progress
            print(f"[{frames_processed}] infer={t_inf1-t_inf0:.2f}ms y={tuple(y.shape)} drops={drops}")

        if args.show:
            cv2.imshow("storeguard-e2e", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    total_s = time.perf_counter() - t0
    fps_effective = frames_written / total_s if total_s > 0 else 0.0

    metrics = {
        "video": str(video_path),
        "model": args.model,
        "triton_url": args.url,
        "io_names": {"input": args.input_name, "output": args.output_name},
        "frames_in": int(frames_in),
        "frames_processed": int(frames_processed),
        "frames_written": int(frames_written),
        "drops": int(drops),
        "fps_effective": float(fps_effective),
        "latency_ms": {
            "total": stat_ms(lat_total),
            "preprocess": stat_ms(lat_pre),
            "infer": stat_ms(lat_inf),
            "postprocess": stat_ms(lat_post),
            "encode": stat_ms(lat_enc),
        },
        "warmup_frames_excluded": int(args.warmup),
        "realtime_sim": {
            "enabled": bool(args.queue_size > 0),
            "queue_size": int(args.queue_size),
            "drop_policy": args.drop_policy,
            "target_fps": float(args.target_fps) if args.target_fps > 0 else 0.0,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] video_out={out_path}")
    print(f"[OK] metrics={metrics_path}")


if __name__ == "__main__":
    main()
