# scripts/modeling/plot_metrics.py
import argparse, json, os
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="outputs/modeling/run_001",
                    help="run directory that contains metrics.json (e.g., outputs/modeling/run_005_...)")
    ap.add_argument("--metrics", default=None,
                    help="(optional) explicit metrics.json path. If set, overrides --run_dir.")
    ap.add_argument("--out_dir", default=None,
                    help="(optional) output directory for plots. Default: <run_dir>/plots")
    ap.add_argument("--prefix", default="history",
                    help="file prefix for saved plots (default: history)")
    args = ap.parse_args()

    metrics_path = args.metrics or os.path.join(args.run_dir, "metrics.json")
    out_dir = args.out_dir or os.path.join(args.run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    with open(metrics_path, "r") as f:
        obj = json.load(f)
    hist = obj.get("history", [])

    if not hist:
        raise RuntimeError(f"No history found in: {metrics_path}")

    epochs = [h["epoch"] for h in hist]
    train_loss = [h.get("train_loss") for h in hist]
    clip_acc = [h.get("clip_acc") for h in hist]
    video_acc = [h.get("video_acc_topkmean") for h in hist]
    epoch_sec = [h.get("epoch_sec") for h in hist]

    loss_png   = os.path.join(out_dir, f"{args.prefix}_loss.png")
    metric_png = os.path.join(out_dir, f"{args.prefix}_metrics.png")
    time_png   = os.path.join(out_dir, f"{args.prefix}_time.png")

    plt.figure()
    plt.plot(epochs, train_loss)
    plt.xlabel("epoch"); plt.ylabel("train_loss"); plt.title("Train Loss")
    plt.savefig(loss_png, dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, clip_acc, label="clip_acc")
    plt.plot(epochs, video_acc, label="video_acc_topkmean")
    plt.xlabel("epoch"); plt.ylabel("score"); plt.title("Val Metrics"); plt.legend()
    plt.savefig(metric_png, dpi=150)
    plt.close()

    if any(v is not None for v in epoch_sec):
        plt.figure()
        plt.plot(epochs, epoch_sec)
        plt.xlabel("epoch"); plt.ylabel("sec"); plt.title("Epoch Time (sec)")
        plt.savefig(time_png, dpi=150)
        plt.close()

    print("saved:", loss_png, metric_png, time_png if os.path.exists(time_png) else "(no time plot)")

if __name__ == "__main__":
    main()
