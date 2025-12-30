# scripts/d5_cls_onnx.py
'''
python scripts/d5_cls_onnx.py export --run-dir outputs/modeling/run_006_seed_sweep_seed43 --temporal tap --out outputs/models/cls_tap_seed43.onnx
python scripts/d5_cls_onnx.py export --run-dir outputs/modeling/run_010_tt_tiny_seed_sweep_seed43 --temporal tt --out outputs/models/cls_tt_tiny_seed43.onnx
'''
import argparse, json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

from scripts.modeling.train_v2_tt import MobileNetV3_Temporal


def load_cfg(run_dir: Path) -> dict:
    p = run_dir / "config.json"
    return json.load(open(p)) if p.exists() else {}


def detect_temporal_from_state_dict(sd: dict) -> str:
    ks = sd.keys()
    if any("temporal_pool.score" in k for k in ks):
        return "tap"
    if any(("temporal_pool.enc" in k or "temporal_pool.proj" in k or "temporal_pool.pos" in k) for k in ks):
        return "tt"
    raise RuntimeError("Cannot detect temporal head from state_dict keys")


def build_model(run_dir: Path, temporal_mode: str) -> tuple[nn.Module, dict]:
    cfg = load_cfg(run_dir)
    weights = run_dir / "best.pth"
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    sd = torch.load(weights, map_location="cpu")

    detected = detect_temporal_from_state_dict(sd)
    if temporal_mode == "auto":
        temporal = detected
    else:
        temporal = temporal_mode
        # 안전장치: 강제 지정이 state_dict 구조와 완전히 다르면 경고/에러
        if temporal != detected:
            raise RuntimeError(f"temporal mismatch: forced={temporal} but state_dict looks like {detected}")

    backbone = cfg.get("backbone", "mobilenet_v3_small")
    dropout = float(cfg.get("dropout", 0.3))

    kwargs = {}
    if temporal == "tt":
        need = ["tt_d_model", "tt_heads", "tt_layers", "tt_dropout", "tt_use_cls"]
        miss = [k for k in need if k not in cfg]
        if miss:
            raise RuntimeError(f"TT config missing keys in {run_dir}/config.json: {miss}")
        kwargs = {k: cfg[k] for k in need}

    m = MobileNetV3_Temporal(
        backbone_name=backbone,
        pretrained=False,
        temporal=temporal,
        dropout=dropout,
        **kwargs,
    )
    m.load_state_dict(sd, strict=True)
    m.eval()

    stamp = {
        "run_dir": str(run_dir),
        "temporal": temporal,
        "temporal_mode": temporal_mode,
        "detected": detected,
        "backbone": backbone,
        "dropout": dropout,
        "seed": cfg.get("seed", None),
        **({k: cfg[k] for k in kwargs.keys()} if temporal == "tt" else {}),
    }
    return m, stamp


class Wrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        logit, _attn = self.model(x)
        return logit


def cmd_export(args):
    run_dir = Path(args.run_dir)
    m, stamp = build_model(run_dir, args.temporal)
    w = Wrapper(m).eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros((args.batch, args.T, 3, 224, 224), dtype=torch.float32)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"input": {0: "B"}, "logit": {0: "B"}}

    torch.onnx.export(
        w,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logit"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[OK] exported: {out_path} size_mb={size_mb:.2f} opset={args.opset} dynamic_batch={args.dynamic_batch} T={args.T}")
    print(f"[STAMP] {stamp}")


def cmd_io(args):
    m = onnx.load(args.onnx)
    print("opset_import:", [(o.domain, o.version) for o in m.opset_import])

    avail = ort.get_available_providers()
    print("ORT providers:", avail)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    for i in sess.get_inputs():
        print("input :", i.name, i.shape, i.type)
    for o in sess.get_outputs():
        print("output:", o.name, o.shape, o.type)


def cmd_smoke(args):
    avail = ort.get_available_providers()
    if args.device == "cuda" and "CUDAExecutionProvider" in avail:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    print("providers_avail:", avail)
    print("providers_used :", providers)

    sess = ort.InferenceSession(args.onnx, providers=providers)
    inp = sess.get_inputs()[0].name
    x = np.zeros((1, args.T, 3, 224, 224), dtype=np.float32)

    for _ in range(5):
        sess.run(None, {inp: x})

    ts = []
    y = None
    for _ in range(args.iters):
        t0 = time.perf_counter()
        y = sess.run(None, {inp: x})[0]
        ts.append((time.perf_counter() - t0) * 1000)

    ts_sorted = sorted(ts)
    p50 = ts_sorted[int(len(ts_sorted) * 0.50)]
    p95 = ts_sorted[int(len(ts_sorted) * 0.95) - 1]
    print("out_shape:", y.shape, "sample:", float(y.reshape(-1)[0]))
    print("lat_ms: mean=%.3f p50=%.3f p95=%.3f" % (sum(ts) / len(ts), p50, p95))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_e = sub.add_parser("export")
    ap_e.add_argument("--run-dir", required=True)
    ap_e.add_argument("--temporal", choices=["auto", "tap", "tt"], default="auto")
    ap_e.add_argument("--out", required=True)
    ap_e.add_argument("--T", type=int, default=16)
    ap_e.add_argument("--batch", type=int, default=1)
    ap_e.add_argument("--opset", type=int, default=18)
    ap_e.add_argument("--dynamic-batch", action="store_true")
    ap_e.set_defaults(func=cmd_export)

    ap_i = sub.add_parser("io")
    ap_i.add_argument("--onnx", required=True)
    ap_i.set_defaults(func=cmd_io)

    ap_s = sub.add_parser("smoke")
    ap_s.add_argument("--onnx", required=True)
    ap_s.add_argument("--T", type=int, default=16)
    ap_s.add_argument("--iters", type=int, default=50)
    ap_s.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap_s.set_defaults(func=cmd_smoke)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
