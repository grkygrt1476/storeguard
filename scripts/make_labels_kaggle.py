#!/usr/bin/env python3
# scripts/make_labels_kaggle.py
import csv
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

def infer_label(p: Path, root: Path) -> str:
    rel = p.relative_to(root)
    first = rel.parts[0].lower() if rel.parts else ""
    if first in ["normal", "nonshoplifting", "non_shoplifting", "nonshoplift"]:
        return "normal"
    if first in ["shoplifting", "shoplift", "theft"]:
        return "shoplifting"
    return "unknown"

def main():
    root = Path("data/raw/shoplifting")
    out = Path("data/labels/videos.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    videos = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    videos.sort()

    counts = {"normal": 0, "shoplifting": 0, "unknown": 0}

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        for p in videos:
            label = infer_label(p, root)
            counts[label] += 1
            w.writerow([p.as_posix(), label])

    print(f"[OK] wrote {out} (n={len(videos)})")
    print("[COUNT]", counts)
    if counts["unknown"] > 0:
        print("[WARN] unknown label exists -> check folder names under data/raw/shoplifting/")

if __name__ == "__main__":
    main()
