#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def load_json(path: Path) -> dict:
    if not path:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        warn(f"metrics json not found: {path}")
    except json.JSONDecodeError as exc:
        warn(f"failed to parse json: {path} ({exc})")
    return {}


def get_metric(data: dict, keys, default=None):
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def to_str(val, precision=2) -> str:
    if val is None:
        return "na"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)


def escape_drawtext(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def build_overlay_text(kind: str, metrics: dict) -> str:
    cls_calls = get_metric(metrics, ["cls_calls"])
    fps_effective = get_metric(metrics, ["fps_effective"])
    drops = get_metric(metrics, ["drops"])
    parts = [
        f"cls_calls={to_str(cls_calls, 0)}",
        f"fps={to_str(fps_effective)}",
        f"drops={to_str(drops, 0)}",
    ]
    if kind == "loiter":
        loiter_frames = get_metric(metrics, ["events", "loitering_frames"])
        parts.append(f"loiter={to_str(loiter_frames, 0)}")
    return " | ".join(parts)


def validate_metrics(kind: str, metrics: dict) -> None:
    cls_calls = get_metric(metrics, ["cls_calls"])
    if cls_calls in (None, 0):
        warn(f"{kind}: cls_calls missing or zero; proceeding with defaults")

    event_key = "intrusion_frames" if kind == "intrusion" else "loitering_frames"
    event_frames = get_metric(metrics, ["events", event_key])
    if event_frames in (None, 0):
        warn(f"{kind}: events.{event_key} missing or zero; proceeding with defaults")


def run_ffmpeg(video: Path, ts: float, out_path: Path, overlay_text: str | None) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(ts),
        "-i",
        str(video),
        "-frames:v",
        "1",
    ]

    if overlay_text:
        text = escape_drawtext(overlay_text)
        drawtext = (
            "drawtext="
            f"text='{text}'"
            ":x=w-tw-12:y=h-th-12"
            ":fontsize=18"
            ":fontcolor=white"
            ":box=1"
            ":boxcolor=black@0.5"
        )
        cmd += ["-vf", drawtext]

    cmd += ["-y", str(out_path)]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed")


def extract_frame(video: Path, ts: float, out_path: Path, overlay_text: str | None) -> None:
    try:
        run_ffmpeg(video, ts, out_path, overlay_text)
    except RuntimeError as exc:
        if overlay_text:
            warn(f"drawtext failed; retrying without overlay ({exc})")
            run_ffmpeg(video, ts, out_path, None)
        else:
            raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="outputs/videos/d4_e2e_with_cls.mp4")
    ap.add_argument("--intrusion-json", default="outputs/logs/d4_e2e_metrics_onnx_cls_PROOF_intrusion.json")
    ap.add_argument("--loiter-json", default="outputs/logs/d4_e2e_metrics_onnx_cls_PROOF_loiter.json")
    ap.add_argument("--outdir", default="outputs/portfolio_assets")
    ap.add_argument("--t-intrusion", type=float, default=5.0)
    ap.add_argument("--t-loiter", type=float, default=8.0)
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found in PATH", file=sys.stderr)
        return 2

    video = Path(args.video)
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2

    intrusion_json = Path(args.intrusion_json)
    loiter_json = Path(args.loiter_json)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    intrusion_metrics = load_json(intrusion_json)
    loiter_metrics = load_json(loiter_json)

    validate_metrics("intrusion", intrusion_metrics)
    validate_metrics("loiter", loiter_metrics)

    intrusion_text = build_overlay_text("intrusion", intrusion_metrics)
    loiter_text = build_overlay_text("loiter", loiter_metrics)

    intrusion_out = outdir / "storeguard_intrusion.png"
    loiter_out = outdir / "storeguard_loitering.png"

    extract_frame(video, args.t_intrusion, intrusion_out, intrusion_text)
    extract_frame(video, args.t_loiter, loiter_out, loiter_text)

    print(f"[OK] {intrusion_out}")
    print(f"[OK] {loiter_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
