# scripts/modeling/make_clips.py
import csv, random, os
import numpy as np

SEED = 42
CLIP_FPS = 8
T = 16
VIDEO_SEC = 10.0

K_VAL = 8      # per video (uniform)
M_TRAIN = 6    # per video (random)

TRAIN_IN = "data/splits/train.csv"
VAL_IN = "data/splits/val.csv"
TRAIN_OUT = "data/clips/train_clips.csv"
VAL_OUT = "data/clips/val_clips.csv"

random.seed(SEED)

clip_len = T / CLIP_FPS
max_start = max(0.0, VIDEO_SEC - clip_len)

def load_split(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def write_clips(out_path, rows):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["path", "label", "start_sec", "clip_fps", "T"]
        )
        w.writeheader()
        w.writerows(rows)

train_rows = []
for r in load_split(TRAIN_IN):
    for _ in range(M_TRAIN):
        s = random.uniform(0.0, max_start) if max_start > 0 else 0.0
        train_rows.append({"path": r["path"], "label": r["label"], "start_sec": f"{s:.3f}", "clip_fps": CLIP_FPS, "T": T})

val_rows = []
for r in load_split(VAL_IN):
    starts = np.linspace(0.0, max_start, num=K_VAL) if K_VAL > 1 else [0.0]
    for s in starts:
        val_rows.append({"path": r["path"], "label": r["label"], "start_sec": f"{float(s):.3f}", "clip_fps": CLIP_FPS, "T": T})

write_clips(TRAIN_OUT, train_rows)
write_clips(VAL_OUT, val_rows)

print(f"train_clips={len(train_rows)} val_clips={len(val_rows)} (clip_fps={CLIP_FPS}, T={T}, clip_len={clip_len:.2f}s)")
