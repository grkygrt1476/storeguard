# scripts/e2e_demo.py
#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import tritonclient.http as httpclient


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def stat_ms(xs):
    if not xs:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.array(xs, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def parse_roi(roi: Optional[str], roi_rel: Optional[str], w: int, h: int) -> Tuple[int, int, int, int]:
    # priority: absolute roi > relative roi > default
    if roi:
        x1, y1, x2, y2 = [int(float(x)) for x in roi.split(",")]
        return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
    if roi_rel:
        rx1, ry1, rx2, ry2 = [float(x) for x in roi_rel.split(",")]
        x1, y1 = int(rx1 * w), int(ry1 * h)
        x2, y2 = int(rx2 * w), int(ry2 * h)
        return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)

    # default: center-ish
    x1, y1 = int(0.2 * w), int(0.2 * h)
    x2, y2 = int(0.85 * w), int(0.9 * h)
    return x1, y1, x2, y2


def preprocess_bgr_to_nchw_fp32(frame_bgr: np.ndarray, in_w: int, in_h: int) -> np.ndarray:
    resized = cv2.resize(frame_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = np.expand_dims(x, axis=0)   # NCHW
    return x


def infer_triton_http(url: str, model: str, input_name: str, output_name: str, x: np.ndarray, timeout_s: float) -> np.ndarray:
    client = httpclient.InferenceServerClient(url=url, verbose=False)
    inp = httpclient.InferInput(input_name, x.shape, "FP32")
    inp.set_data_from_numpy(x, binary_data=True)
    out = httpclient.InferRequestedOutput(output_name, binary_data=True)
    resp = client.infer(model_name=model, inputs=[inp], outputs=[out], timeout=int(timeout_s * 1000))
    y = resp.as_numpy(output_name)
    return y


def yolov8_decode_person(
    y: np.ndarray,
    conf_thres: float,
    nms_iou: float,
    in_w: int,
    in_h: int,
    person_class_id: int = 0,
):
    """
    YOLOv8 ONNX typical output: (1, 84, 2100) = [x,y,w,h] + 80 class scores, no NMS.
    Returns xyxy boxes in input-space (0..in_w/in_h) and scores.
    """
    if y is None:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    if y.ndim == 3:
        p = y[0]  # (84, 2100)
    elif y.ndim == 2:
        p = y
    else:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    if p.shape[0] == 84:
        p = p.transpose(1, 0)  # (2100, 84)
    elif p.shape[1] == 84:
        pass
    else:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    boxes_xywh = p[:, :4]
    cls_scores = p[:, 4:]  # (N,80)

    cls_id = np.argmax(cls_scores, axis=1)
    score = cls_scores[np.arange(cls_scores.shape[0]), cls_id]

    keep = (cls_id == person_class_id) & (score >= conf_thres)
    if not np.any(keep):
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    b = boxes_xywh[keep]
    s = score[keep].astype(np.float32)

    x, y0, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x1 = x - w / 2.0
    y1 = y0 - h / 2.0
    x2 = x + w / 2.0
    y2 = y0 + h / 2.0

    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    # clip
    boxes[:, 0] = np.clip(boxes[:, 0], 0, in_w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, in_h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, in_w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, in_h - 1)

    # NMS via OpenCV
    nms_boxes = [[float(bb[0]), float(bb[1]), float(bb[2] - bb[0]), float(bb[3] - bb[1])] for bb in boxes]
    idxs = cv2.dnn.NMSBoxes(nms_boxes, s.tolist(), score_threshold=float(conf_thres), nms_threshold=float(nms_iou))
    if idxs is None or len(idxs) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    idxs = np.array(idxs).reshape(-1)
    return boxes[idxs], s[idxs]


def roi_hit_ratio(box_xyxy, roi_xyxy) -> float:
    x1, y1, x2, y2 = box_xyxy
    rx1, ry1, rx2, ry2 = roi_xyxy
    ix1, iy1 = max(x1, rx1), max(y1, ry1)
    ix2, iy2 = min(x2, rx2), min(y2, ry2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area = max(1.0, (x2 - x1) * (y2 - y1))
    return float(inter / area)


def open_writer(path: str, w: int, h: int, fps: float):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc_candidates = ["mp4v", "avc1", "H264", "XVID"]
    for cc in fourcc_candidates:
        fourcc = cv2.VideoWriter_fourcc(*cc)
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer
    raise RuntimeError("Failed to open VideoWriter (codec issue). Try installing ffmpeg/gstreamer or change codec.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", required=True)

    ap.add_argument("--url", default="localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-name", default="images")
    ap.add_argument("--output-name", default="output0")
    ap.add_argument("--timeout", type=float, default=10.0)

    ap.add_argument("--in-w", type=int, default=320)
    ap.add_argument("--in-h", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--nms-iou", type=float, default=0.45)

    # realtime knobs
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--target-fps", type=float, default=0.0)
    ap.add_argument("--queue-size", type=int, default=2)
    ap.add_argument("--drop-policy", choices=["latest"], default="latest")

    # ROI/event
    ap.add_argument("--roi", default=None, help='absolute "x1,y1,x2,y2" in pixels (original frame)')
    ap.add_argument("--roi-rel", default="0.2,0.2,0.85,0.9", help='relative "x1,y1,x2,y2" (0..1)')
    ap.add_argument("--roi-hit", type=float, default=0.10, help="hit if (intersection area / box area) >= this")
    ap.add_argument("--loiter-sec", type=float, default=3.0)

    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1e-3:
        src_fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_fps = float(args.target_fps) if args.target_fps and args.target_fps > 0 else float(src_fps)
    frame_interval_ms = 1000.0 / out_fps

    roi = parse_roi(args.roi, args.roi_rel, w, h)

    writer = open_writer(args.out, w, h, min(out_fps, 60.0))

    # stats
    lat_pre, lat_inf, lat_post, lat_enc, lat_total = [], [], [], [], []
    frames_in, frames_proc, frames_written, drops = 0, 0, 0, 0

    # event stats
    intrusion_frames, loitering_frames = 0, 0
    roi_presence_ms = 0.0

    # fps overlay (moving)
    tick0 = time.perf_counter()
    proc_in_window = 0
    fps_overlay = 0.0

    t_start = time.perf_counter()

    while True:
        if (time.perf_counter() - t_start) > args.max_seconds:
            break

        # drop policy simulation (latest-wins): if lag too big, skip frames
        if args.queue_size > 0:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            expected_frames = int(elapsed_ms // frame_interval_ms)
            lag = frames_in - expected_frames
            budget = args.queue_size
            if lag > budget:
                # drop extra frames to catch up
                drop_n = lag - budget
                for _ in range(drop_n):
                    ok, _ = cap.read()
                    if not ok:
                        break
                    frames_in += 1
                    drops += 1

        ok, frame = cap.read()
        if not ok:
            break
        frames_in += 1

        t0 = now_ms()

        # preprocess
        t_pre0 = now_ms()
        x = preprocess_bgr_to_nchw_fp32(frame, args.in_w, args.in_h)
        t_pre1 = now_ms()

        # infer
        t_inf0 = now_ms()
        y = infer_triton_http(args.url, args.model, args.input_name, args.output_name, x, timeout_s=args.timeout)
        t_inf1 = now_ms()

        # postprocess: decode person boxes in input-space
        t_post0 = now_ms()
        boxes_in, scores = yolov8_decode_person(y, args.conf, args.nms_iou, args.in_w, args.in_h)
        # scale to original frame space
        sx = w / float(args.in_w)
        sy = h / float(args.in_h)

        event = "normal"
        hit_any = False

        # draw ROI
        rx1, ry1, rx2, ry2 = roi
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)

        for bb in boxes_in:
            x1, y1, x2, y2 = bb
            X1, Y1, X2, Y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
            X1, Y1 = max(0, X1), max(0, Y1)
            X2, Y2 = min(w - 1, X2), min(h - 1, Y2)

            hr = roi_hit_ratio((float(X1), float(Y1), float(X2), float(Y2)), (float(rx1), float(ry1), float(rx2), float(ry2)))
            is_hit = hr >= args.roi_hit
            if is_hit:
                hit_any = True
                cv2.rectangle(frame, (X1, Y1), (X2, Y2), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (X1, Y1), (X2, Y2), (0, 255, 0), 2)

        # event logic
        if hit_any:
            roi_presence_ms += frame_interval_ms
            intrusion_frames += 1
            event = "intrusion"
            if (roi_presence_ms / 1000.0) >= args.loiter_sec:
                event = "loitering"
                loitering_frames += 1
        else:
            roi_presence_ms = 0.0

        t_post1 = now_ms()

        # overlay HUD
        infer_ms = (t_inf1 - t_inf0)
        proc_in_window += 1
        if (time.perf_counter() - tick0) >= 1.0:
            fps_overlay = proc_in_window / (time.perf_counter() - tick0)
            tick0 = time.perf_counter()
            proc_in_window = 0

        hud1 = f"EVENT: {event} | FPS~{fps_overlay:.1f} | infer={infer_ms:.2f}ms | drops={drops}"
        cv2.putText(frame, hud1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 3, cv2.LINE_AA)
        cv2.putText(frame, hud1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 1, cv2.LINE_AA)

        # encode
        t_enc0 = now_ms()
        writer.write(frame)
        t_enc1 = now_ms()

        t1 = now_ms()

        frames_proc += 1
        frames_written += 1

        if frames_proc > args.warmup:
            lat_pre.append(t_pre1 - t_pre0)
            lat_inf.append(t_inf1 - t_inf0)
            lat_post.append(t_post1 - t_post0)
            lat_enc.append(t_enc1 - t_enc0)
            lat_total.append(t1 - t0)

        if frames_proc % 30 == 0:
            print(f"[{frames_proc}] infer={infer_ms:.2f}ms y={tuple(y.shape)} drops={drops} event={event}")

    cap.release()
    writer.release()

    elapsed = time.perf_counter() - t_start
    fps_effective = frames_written / max(1e-6, elapsed)

    metrics = {
        "video": args.video,
        "model": args.model,
        "url": args.url,
        "input": {"name": args.input_name, "shape": [1, 3, args.in_h, args.in_w], "dtype": "FP32"},
        "output": {"name": args.output_name},
        "realtime": {
            "target_fps": out_fps,
            "queue_size": args.queue_size,
            "drop_policy": args.drop_policy,
            "warmup": args.warmup,
        },
        "roi": {"xyxy": list(roi), "roi_hit": args.roi_hit, "loiter_sec": args.loiter_sec},
        "drops": int(drops),
        "fps_effective": float(fps_effective),
        "latency_ms": {
            "preprocess": stat_ms(lat_pre),
            "infer": stat_ms(lat_inf),
            "postprocess": stat_ms(lat_post),
            "encode": stat_ms(lat_enc),
            "total": stat_ms(lat_total),
        },
        "events": {
            "intrusion_frames": int(intrusion_frames),
            "loitering_frames": int(loitering_frames),
        },
        "frames": {
            "in": int(frames_in),
            "processed": int(frames_proc),
            "written": int(frames_written),
        },
    }

    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] video_out={args.out}")
    print(f"[OK] metrics={args.metrics}")


if __name__ == "__main__":
    main()
