# scripts/modeling/make_split.py
import csv, random, os
from collections import defaultdict

SEED = 42
VAL_RATIO = 0.2
IN_CSV = "data/labels/videos.csv"
OUT_TRAIN = "data/splits/train.csv"
OUT_VAL = "data/splits/val.csv"

random.seed(SEED)

rows = []
with open(IN_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({"path": r["path"].strip(), "label": r["label"].strip()})

by_label = defaultdict(list)
for r in rows:
    by_label[r["label"]].append(r)

train, val = [], []
for label, items in by_label.items():
    random.shuffle(items)
    n_val = max(1, int(len(items) * VAL_RATIO))
    val += items[:n_val]
    train += items[n_val:]

os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
with open(OUT_TRAIN, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["path", "label"])
    w.writeheader(); w.writerows(train)

with open(OUT_VAL, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["path", "label"])
    w.writeheader(); w.writerows(val)

print(f"train={len(train)} val={len(val)} (seed={SEED}, val_ratio={VAL_RATIO})")
