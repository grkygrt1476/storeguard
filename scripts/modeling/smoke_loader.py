# scripts/modeling/smoke_loader.py
import argparse
import csv
import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class ClipSpec:
    path: str
    label: str
    start_sec: float
    clip_fps: float
    T: int


def load_specs(csv_path: str, limit: int | None = None):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]

    specs = []
    for r in rows:
        specs.append(
            ClipSpec(
                path=r["path"].strip(),
                label=r["label"].strip(),
                start_sec=float(r["start_sec"]),
                clip_fps=float(r["clip_fps"]),
                T=int(r["T"]),
            )
        )
    return specs


def sample_frames(spec: ClipSpec):
    cap = cv2.VideoCapture(spec.path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {spec.path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if native_fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Bad metadata: fps={native_fps}, total_frames={total_frames}")

    start_frame = int(round(spec.start_sec * native_fps))
    step = native_fps / spec.clip_fps

    idxs = []
    for i in range(spec.T):
        idx = start_frame + int(round(i * step))
        idx = max(0, min(total_frames - 1, idx))
        idxs.append(idx)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    if len(frames) != spec.T:
        raise RuntimeError(f"Decoded {len(frames)}/{spec.T} frames from {spec.path}")

    return frames


def resize_shorter_side(img: np.ndarray, shorter: int = 256):
    h, w = img.shape[:2]
    if h < w:
        new_h = shorter
        new_w = int(round(w * (shorter / h)))
    else:
        new_w = shorter
        new_h = int(round(h * (shorter / w)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop(img: np.ndarray, size: int = 224, mode: str = "center"):
    h, w = img.shape[:2]
    if h < size or w < size:
        # safety: pad if needed
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        h, w = img.shape[:2]

    if mode == "center":
        y0 = (h - size) // 2
        x0 = (w - size) // 2
    elif mode == "random":
        y0 = random.randint(0, h - size)
        x0 = random.randint(0, w - size)
    else:
        raise ValueError("mode must be 'center' or 'random'")

    return img[y0:y0 + size, x0:x0 + size]


def to_tensor_normalized(img_rgb_224: np.ndarray):
    x = img_rgb_224.astype(np.float32) / 255.0  # [0,1]
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))  # CHW
    return torch.from_numpy(x)  # float32


def clip_to_tensor(frames_rgb: list[np.ndarray], crop_mode: str):
    # Apply SAME crop window for all frames (important: temporal consistency)
    resized = [resize_shorter_side(fr, 256) for fr in frames_rgb]

    # decide crop window based on first frame, then apply consistently
    h, w = resized[0].shape[:2]
    size = 224
    if crop_mode == "center":
        y0 = (h - size) // 2
        x0 = (w - size) // 2
    else:
        y0 = random.randint(0, h - size)
        x0 = random.randint(0, w - size)

    cropped = [fr[y0:y0+size, x0:x0+size] for fr in resized]
    tensors = [to_tensor_normalized(fr) for fr in cropped]  # list of [C,H,W]
    x = torch.stack(tensors, dim=0)  # [T,C,H,W]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/clips/train_clips.csv")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--mode", choices=["train", "val"], default="train")
    ap.add_argument("--limit", type=int, default=64, help="limit specs loaded (for speed)")
    args = ap.parse_args()

    crop_mode = "random" if args.mode == "train" else "center"
    specs = load_specs(args.csv, limit=args.limit)

    # sample a small batch
    batch_specs = random.sample(specs, k=min(args.batch, len(specs)))
    xs, ys = [], []
    for spec in batch_specs:
        frames = sample_frames(spec)
        x = clip_to_tensor(frames, crop_mode=crop_mode)  # [T,C,224,224]
        y = 0 if spec.label == "normal" else 1
        xs.append(x)
        ys.append(y)

    X = torch.stack(xs, dim=0)  # [B,T,C,H,W]
    y = torch.tensor(ys, dtype=torch.long)

    os.makedirs("outputs/logs", exist_ok=True)
    log_path = f"outputs/logs/loader_smoke_{args.mode}.txt"
    with open(log_path, "w") as f:
        f.write(f"csv={args.csv}\n")
        f.write(f"mode={args.mode} crop={crop_mode}\n")
        f.write(f"X.shape={tuple(X.shape)} dtype={X.dtype}\n")
        f.write(f"y.shape={tuple(y.shape)} labels={y.tolist()}\n")
        f.write(f"X.min={X.min().item():.3f} X.max={X.max().item():.3f}\n")

    print(f"[OK] wrote {log_path}")
    print(f"X.shape={tuple(X.shape)} (expect [B,T,C,224,224])")
    print(f"X.min/max={X.min().item():.2f}/{X.max().item():.2f}  y={y.tolist()}")


if __name__ == "__main__":
    main()
