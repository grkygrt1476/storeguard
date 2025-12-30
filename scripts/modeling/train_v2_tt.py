# scripts/modeling/train_v2_tt.py
import argparse
import csv
import json
import os
import random
import time
import glob
from dataclasses import dataclass
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class ClipSpec:
    path: str
    label: int  # 0 normal, 1 shoplifting
    start_sec: float
    clip_fps: float
    T: int


def read_csv_specs(csv_path: str):
    specs = []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        label_str = r["label"].strip()
        y = 0 if label_str == "normal" else 1
        specs.append(
            ClipSpec(
                path=r["path"].strip(),
                label=y,
                start_sec=float(r["start_sec"]),
                clip_fps=float(r["clip_fps"]),
                T=int(r["T"]),
            )
        )
    return specs


def resize_shorter_side(img: np.ndarray, shorter: int = 256):
    h, w = img.shape[:2]
    if h < w:
        new_h = shorter
        new_w = int(round(w * (shorter / h)))
    else:
        new_w = shorter
        new_h = int(round(h * (shorter / w)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop_consistent(frames: list[np.ndarray], size: int, mode: str):
    h, w = frames[0].shape[:2]
    if h < size or w < size:
        padded = []
        for fr in frames:
            ph = max(0, size - fr.shape[0])
            pw = max(0, size - fr.shape[1])
            fr2 = cv2.copyMakeBorder(fr, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            padded.append(fr2)
        frames = padded
        h, w = frames[0].shape[:2]

    if mode == "center":
        y0 = (h - size) // 2
        x0 = (w - size) // 2
    elif mode == "random":
        y0 = random.randint(0, h - size)
        x0 = random.randint(0, w - size)
    else:
        raise ValueError("mode must be 'center' or 'random'")

    return [fr[y0:y0+size, x0:x0+size] for fr in frames]


def to_tensor_normalized(img_rgb_224: np.ndarray) -> torch.Tensor:
    x = img_rgb_224.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))  # CHW
    return torch.from_numpy(x)  # float32


def decode_clip_frames(spec: ClipSpec) -> list[np.ndarray]:
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


def path_to_cache_npz(video_path: str, raw_root: str, cache_root: str) -> str:
    # expects cache layout like: cache_root/<relpath-without-ext>.npz
    rel = os.path.relpath(video_path, raw_root)
    rel = os.path.splitext(rel)[0] + ".npz"
    return os.path.join(cache_root, rel)


class ClipDataset(Dataset):
    def __init__(self, csv_path: str, mode: str, use_cache: bool, raw_root: str, cache_root: str):
        self.specs = read_csv_specs(csv_path)
        self.mode = mode  # "train" or "val"
        self.crop_mode = "random" if mode == "train" else "center"
        self.use_cache = use_cache
        self.raw_root = raw_root
        self.cache_root = cache_root

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx: int):
        spec = self.specs[idx]

        # 1) load frames (cache preferred)
        if self.use_cache:
            npz_path = path_to_cache_npz(spec.path, self.raw_root, self.cache_root)
            if os.path.exists(npz_path):
                obj = np.load(npz_path, mmap_mode="r")
                frames_np = obj["frames"]  # [N,H,W,3] uint8

                # cache fps: prefer new key, keep backward compat, fallback
                if "cache_fps" in obj.files:
                    cache_fps = float(obj["cache_fps"])
                elif "fps" in obj.files:
                    cache_fps = float(obj["fps"])
                else:
                    cache_fps = 8.0

                # ✅ 여기! start_idx 반드시 정의
                start_idx = int(round(spec.start_sec * cache_fps))
                step = cache_fps / spec.clip_fps

                idxs = []
                for i in range(spec.T):
                    j = start_idx + int(round(i * step))
                    j = max(0, min(frames_np.shape[0] - 1, j))
                    idxs.append(j)

                frames = [frames_np[j] for j in idxs]  # list of RGB uint8 [H,W,3]
            else:
                # fallback to mp4 decode (cache not complete)
                frames = decode_clip_frames(spec)
        else:
            frames = decode_clip_frames(spec)

        # 2) resize -> crop -> normalize -> stack
        frames = [resize_shorter_side(fr, 256) for fr in frames]
        frames = crop_consistent(frames, size=224, mode=self.crop_mode)

        xs = [to_tensor_normalized(fr) for fr in frames]      # list of [C,H,W]
        x = torch.stack(xs, dim=0)                            # [T,C,H,W]
        y = torch.tensor(spec.label, dtype=torch.float32)
        video_id = spec.path
        return x, y, video_id


# -------------------- Temporal heads --------------------

class TemporalAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, h):  # [B,T,D]
        s = self.score(h).squeeze(-1)        # [B,T]
        a = torch.softmax(s, dim=1)          # [B,T]
        z = (h * a.unsqueeze(-1)).sum(dim=1) # [B,D]
        return z, a


class TinyTemporalTransformer(nn.Module):
    """
    Very small Transformer encoder over time axis.
    - project frame embedding D -> d_model
    - add learned positional embedding
    - (optional) CLS token
    - TransformerEncoder (L layers)
    - pool: CLS or mean
    """
    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.1,
        max_T: int = 64,
        use_cls: bool = True,
    ):
        super().__init__()
        assert d_model % nhead == 0, "tt_d_model must be divisible by tt_heads"
        self.use_cls = use_cls
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_T + (1 if use_cls else 0), d_model))
        if use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # init
        nn.init.trunc_normal_(self.pos, std=0.02)
        if use_cls:
            nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, h):  # [B,T,in_dim]
        B, T, _ = h.shape
        x = self.proj(h)  # [B,T,d_model]

        if self.use_cls:
            cls = self.cls.expand(B, -1, -1)     # [B,1,d_model]
            x = torch.cat([cls, x], dim=1)       # [B,1+T,d_model]
            x = x + self.pos[:, : (T + 1), :]
        else:
            x = x + self.pos[:, :T, :]

        x = self.enc(x)
        x = self.norm(x)

        if self.use_cls:
            z = x[:, 0, :]               # CLS
        else:
            z = x.mean(dim=1)            # mean pool
        return z, None


class MobileNetV3_Temporal(nn.Module):
    def __init__(
        self,
        backbone_name="mobilenet_v3_small",
        pretrained=True,
        temporal="tap",
        dropout=0.2,
        tt_d_model=128,
        tt_heads=4,
        tt_layers=2,
        tt_dropout=0.1,
        tt_use_cls=True,
    ):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

        if backbone_name == "mobilenet_v3_large":
            m = mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
        else:
            m = mobilenet_v3_small(weights="DEFAULT" if pretrained else None)

        self.features = m.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.avgpool(self.features(dummy)).flatten(1)
            dim = out.shape[1]

        self.temporal = temporal
        if temporal == "tap":
            self.temporal_pool = TemporalAttentionPool(dim)
            head_in = dim
        elif temporal == "tt":
            self.temporal_pool = TinyTemporalTransformer(
                in_dim=dim,
                d_model=tt_d_model,
                nhead=tt_heads,
                num_layers=tt_layers,
                dropout=tt_dropout,
                use_cls=tt_use_cls,
            )
            head_in = tt_d_model
        else:
            raise ValueError("temporal must be 'tap' or 'tt'")

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, 1),
        )

    def forward(self, x):  # [B,T,C,H,W]
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        f = self.features(x)
        f = self.avgpool(f).flatten(1)          # [B*T, D]
        h = f.reshape(B, T, -1)                 # [B,T,D]
        z, attn = self.temporal_pool(h)         # [B, head_in]
        logit = self.head(z).squeeze(-1)        # [B]
        return logit, attn


# -------------------- eval --------------------

def evaluate_clip_level(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y, _vid in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logit, _ = model(x)
            p = torch.sigmoid(logit)
            ys.append(y.cpu())
            ps.append(p.cpu())
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    pred = (p >= 0.5).astype(np.int32)

    metrics = {"clip_acc": float((pred == y.astype(np.int32)).mean())}
    if SKLEARN_OK:
        metrics["clip_f1"] = float(f1_score(y, pred))
        metrics["clip_auc"] = float(roc_auc_score(y, p))
    return metrics


def evaluate_video_level(model, loader, device, topk: int = 3):
    model.eval()
    by_video_probs = defaultdict(list)
    by_video_label = {}

    with torch.no_grad():
        for x, y, vid in loader:
            x = x.to(device, non_blocking=True)
            logit, _ = model(x)
            p = torch.sigmoid(logit).detach().cpu().numpy()

            y_np = y.numpy()
            for i in range(len(vid)):
                v = vid[i]
                by_video_probs[v].append(float(p[i]))
                by_video_label[v] = int(y_np[i])

    yv, pv = [], []
    for v, probs in by_video_probs.items():
        probs_sorted = sorted(probs, reverse=True)
        k = min(topk, len(probs_sorted))
        score = sum(probs_sorted[:k]) / k
        yv.append(by_video_label[v])
        pv.append(score)

    yv = np.array(yv, dtype=np.int32)
    pv = np.array(pv, dtype=np.float32)
    pred = (pv >= 0.5).astype(np.int32)

    metrics = {
        "video_topk": int(topk),
        "video_acc_topkmean": float((pred == yv).mean()),
    }
    if SKLEARN_OK:
        metrics["video_f1_topkmean"] = float(f1_score(yv, pred))
        metrics["video_auc_topkmean"] = float(roc_auc_score(yv, pv))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/clips/train_clips.csv")
    ap.add_argument("--val_csv", default="data/clips/val_clips.csv")
    ap.add_argument("--out_dir", default="outputs/modeling/run_tt_ablation")

    ap.add_argument("--backbone", default="mobilenet_v3_small",
                    choices=["mobilenet_v3_small", "mobilenet_v3_large"])
    ap.add_argument("--temporal", default="tap", choices=["tap", "tt"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--wd", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--freeze_backbone_epochs", type=int, default=2)
    ap.add_argument("--video_topk", type=int, default=3)

    # cache
    ap.add_argument("--use_cache", action="store_true")
    ap.add_argument("--raw_root", default="data/raw")
    ap.add_argument("--cache_root", default="data/cache_npz")
    ap.add_argument("--cache_fps_default", type=float, default=8.0)

    # early stopping
    ap.add_argument("--early_patience", type=int, default=5)
    ap.add_argument("--min_delta", type=float, default=1e-4)

    # tiny transformer knobs (only used when --temporal tt)
    ap.add_argument("--tt_d_model", type=int, default=128)
    ap.add_argument("--tt_heads", type=int, default=4)
    ap.add_argument("--tt_layers", type=int, default=2)
    ap.add_argument("--tt_dropout", type=float, default=0.1)
    ap.add_argument("--tt_use_cls", action="store_true")

    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    run_t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MobileNetV3_Temporal(
        backbone_name=args.backbone,
        pretrained=True,
        temporal=args.temporal,
        dropout=args.dropout,
        tt_d_model=args.tt_d_model,
        tt_heads=args.tt_heads,
        tt_layers=args.tt_layers,
        tt_dropout=args.tt_dropout,
        tt_use_cls=args.tt_use_cls,
    ).to(device)

    # Freeze backbone initially
    for p in model.features.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp) if torch.cuda.is_available() else None
    criterion = nn.BCEWithLogitsLoss()

    train_ds = ClipDataset(args.train_csv, mode="train",
                           use_cache=args.use_cache, raw_root=args.raw_root, cache_root=args.cache_root)
    val_ds   = ClipDataset(args.val_csv, mode="val",
                           use_cache=args.use_cache, raw_root=args.raw_root, cache_root=args.cache_root)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    best_key = "video_auc_topkmean" if SKLEARN_OK else "video_acc_topkmean"
    best_score = -1.0
    history = []

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()

        # unfreeze backbone after N epochs
        if epoch == args.freeze_backbone_epochs + 1:
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.3, weight_decay=args.wd)

        running_loss = 0.0
        for x, y, _vid in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if torch.cuda.is_available() and args.amp:
                with torch.amp.autocast("cuda", enabled=True):
                    logit, _ = model(x)
                    loss = criterion(logit, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logit, _ = model(x)
                loss = criterion(logit, y)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))

        clip_metrics = evaluate_clip_level(model, val_loader, device)
        video_metrics = evaluate_video_level(model, val_loader, device, topk=args.video_topk)

        metrics = {"epoch": epoch, "train_loss": train_loss, **clip_metrics, **video_metrics}
        metrics["epoch_sec"] = round(time.perf_counter() - ep_t0, 3)
        metrics["total_sec"] = round(time.perf_counter() - run_t0, 3)
        history.append(metrics)

        score = metrics.get(best_key, -1.0)
        tsec = metrics["epoch_sec"]
        print(f"[E{epoch}] loss={train_loss:.4f} clip_acc={clip_metrics.get('clip_acc',0):.3f} "
              f"video_topkmean_acc={video_metrics.get('video_acc_topkmean',0):.3f} best({best_key})={best_score:.3f} t={tsec:.3f}s")

        improved = score > (best_score + args.min_delta)
        if improved:
            best_score = score
            bad_epochs = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))
        else:
            bad_epochs += 1

        with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
            json.dump({"best_key": best_key, "best_score": best_score, "history": history}, f, indent=2)

        if bad_epochs >= args.early_patience:
            print(f"[EARLY STOP] epoch={epoch} bad_epochs={bad_epochs} best({best_key})={best_score:.4f}")
            break

    print(f"[DONE] out_dir={args.out_dir} best_score={best_score:.4f} (key={best_key})")


if __name__ == "__main__":
    main()
