# scripts/modeling/cache_video_npz.py
import argparse, csv, os, time
import cv2
import numpy as np

def resize_shorter_side(img, shorter=256):
    h, w = img.shape[:2]
    if h < w:
        new_h = shorter
        new_w = int(round(w * (shorter / h)))
    else:
        new_w = shorter
        new_h = int(round(h * (shorter / w)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def to_square(img, size=256):
    h, w = img.shape[:2]

    # 1) pad if smaller
    if h < size or w < size:
        top = max(0, (size - h) // 2)
        bottom = max(0, size - h - top)
        left = max(0, (size - w) // 2)
        right = max(0, size - w - left)
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
        h, w = img.shape[:2]

    # 2) center-crop if larger
    if h > size or w > size:
        y0 = (h - size) // 2
        x0 = (w - size) // 2
        img = img[y0:y0+size, x0:x0+size]

    return img

def out_npz_path(video_path, raw_root, out_dir):
    rel = os.path.relpath(video_path, raw_root)  # shoplifting/normal/xxx.mp4
    rel = os.path.splitext(rel)[0] + ".npz"
    return os.path.join(out_dir, rel)

def cache_one(video_path, out_path, cache_fps=8, size=256):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if native_fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError(f"bad metadata: fps={native_fps}, frames={total_frames}, path={video_path}")

    duration = total_frames / native_fps
    n_out = max(1, int(round(duration * cache_fps)))  # e.g., 10s * 8fps = 80

    # target native frame indices (monotonic)
    step = native_fps / cache_fps
    targets = [min(total_frames - 1, int(round(i * step))) for i in range(n_out)]
    target_set = set(targets)

    frames = []
    cur = 0
    last_rgb = None
    need = set(targets)

    while True:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        if cur in target_set:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = resize_shorter_side(rgb, size)
            rgb = to_square(rgb, size)
            frames.append(rgb.astype(np.uint8))
            last_rgb = frames[-1]
            need.discard(cur)
            if not need:
                break
        cur += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"no frames cached: {video_path}")

    # pad if short (rare)
    while len(frames) < n_out:
        frames.append(last_rgb)

    arr = np.stack(frames, axis=0)  # [N,256,256,3] uint8
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, frames=arr, cache_fps=np.float32(cache_fps), native_fps=np.float32(native_fps))
    return arr.shape[0], native_fps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", default="data/labels/videos.csv")
    ap.add_argument("--raw_root", default="data/raw")
    ap.add_argument("--out_dir", default="data/cache_npz")
    ap.add_argument("--cache_fps", type=float, default=8.0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rows = []
    with open(args.labels_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    done = 0
    skipped = 0
    for r in rows:
        vp = r["path"].strip()
        outp = out_npz_path(vp, args.raw_root, args.out_dir)
        if (not args.force) and os.path.exists(outp):
            skipped += 1
            continue
        n, nfps = cache_one(vp, outp, cache_fps=args.cache_fps, size=args.size)
        done += 1
        if done % 20 == 0:
            print(f"[{done}/{len(rows)}] cached. last N={n}, native_fps={nfps:.2f}")

    dt = time.time() - t0
    print(f"[DONE] cached={done} skipped={skipped} out_dir={args.out_dir} ({dt:.1f}s)")

if __name__ == "__main__":
    main()
