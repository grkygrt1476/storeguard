import argparse, json, os
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="outputs/modeling/run_001/metrics.json")
    ap.add_argument("--out", default="outputs/modeling/run_001/history.png")
    args = ap.parse_args()

    with open(args.metrics, "r") as f:
        obj = json.load(f)
    hist = obj["history"]

    epochs = [h["epoch"] for h in hist]
    train_loss = [h.get("train_loss") for h in hist]
    clip_acc = [h.get("clip_acc") for h in hist]
    video_acc = [h.get("video_acc_topkmean") for h in hist]
    epoch_sec = [h.get("epoch_sec") for h in hist]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    plt.figure()
    plt.plot(epochs, train_loss)
    plt.xlabel("epoch"); plt.ylabel("train_loss"); plt.title("Train Loss")
    plt.savefig(args.out.replace(".png", "_loss.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, clip_acc, label="clip_acc")
    plt.plot(epochs, video_acc, label="video_acc_topkmean")
    plt.xlabel("epoch"); plt.ylabel("score"); plt.title("Val Metrics"); plt.legend()
    plt.savefig(args.out.replace(".png", "_metrics.png"), dpi=150)
    plt.close()

    if any(v is not None for v in epoch_sec):
        plt.figure()
        plt.plot(epochs, epoch_sec)
        plt.xlabel("epoch"); plt.ylabel("sec"); plt.title("Epoch Time (sec)")
        plt.savefig(args.out.replace(".png", "_time.png"), dpi=150)
        plt.close()

    print("saved:", args.out.replace(".png", "_loss.png"),
          args.out.replace(".png", "_metrics.png"),
          args.out.replace(".png", "_time.png"))

if __name__ == "__main__":
    main()
