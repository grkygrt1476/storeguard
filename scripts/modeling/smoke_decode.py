# scripts/modeling/smoke_decode.py
import argparse
import csv
import os
from dataclasses import dataclass
import cv2


@dataclass
class ClipSpec:
    path: str
    label: str
    start_sec: float
    clip_fps: float
    T: int


def load_row(csv_path: str, row_idx: int) -> ClipSpec:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if row_idx < 0 or row_idx >= len(rows):
        raise IndexError(f"row_idx out of range: 0..{len(rows)-1}")

    r = rows[row_idx]
    return ClipSpec(
        path=r["path"].strip(),
        label=r["label"].strip(),
        start_sec=float(r["start_sec"]),
        clip_fps=float(r["clip_fps"]),
        T=int(r["T"]),
    )


def ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


def sample_clip_frames(spec: ClipSpec):
    cap = cv2.VideoCapture(spec.path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {spec.path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if native_fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Bad metadata: fps={native_fps}, total_frames={total_frames}")

    # Convert start_sec -> start_frame in native fps timeline
    start_frame = int(round(spec.start_sec * native_fps))

    # Downsample step: how many native frames per 1 sampled frame
    step = native_fps / spec.clip_fps  # e.g., 30/8=3.75

    # Target frame indices in native timeline
    idxs = []
    for i in range(spec.T):
        idx = start_frame + int(round(i * step))
        idx = max(0, min(total_frames - 1, idx))
        idxs.append(idx)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)

    cap.release()
    return frames, idxs, native_fps, total_frames, start_frame, step


def write_video(frames, out_mp4: str, fps: float):
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))
    for fr in frames:
        vw.write(fr)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/clips/val_clips.csv")
    ap.add_argument("--row", type=int, default=0, help="0-based row index in csv (excluding header)")
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    spec = load_row(args.csv, args.row)
    ensure_exists(spec.path)

    frames, idxs, native_fps, total_frames, start_frame, step = sample_clip_frames(spec)

    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)
    log_path = os.path.join(args.out_dir, "logs", f"modeling_smoke_row{args.row}.txt")

    with open(log_path, "w") as f:
        f.write("=== storeguard modeling smoke decode ===\n")
        f.write(f"csv={args.csv}, row={args.row}\n")
        f.write(f"path={spec.path}\n")
        f.write(f"label={spec.label}\n")
        f.write(f"start_sec={spec.start_sec}\n")
        f.write(f"native_fps={native_fps:.3f}, clip_fps={spec.clip_fps}, step(native/clip)={step:.3f}\n")
        f.write(f"total_frames={total_frames}, start_frame={start_frame}\n")
        f.write(f"target_idxs(first..last)={idxs[0]}..{idxs[-1]}\n")
        f.write(f"decoded_frames={len(frames)} (expected T={spec.T})\n")
        if frames:
            h, w = frames[0].shape[:2]
            f.write(f"frame_shape(HxW)={h}x{w}\n")

    print(f"[OK] wrote log: {log_path}")
    print(f"decoded_frames={len(frames)}/{spec.T}, native_fps={native_fps:.2f}, step={step:.2f}")
    print(f"idxs[:5]={idxs[:5]} ... idxs[-5:]={idxs[-5:]}")

    if args.save_video and len(frames) == spec.T:
        out_mp4 = os.path.join(args.out_dir, "videos", f"smoke_clip_row{args.row}.mp4")
        write_video(frames, out_mp4, fps=spec.clip_fps)
        print(f"[OK] wrote video: {out_mp4}")


if __name__ == "__main__":
    main()
