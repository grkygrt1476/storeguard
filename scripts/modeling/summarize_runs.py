# scripts/modeling/summarize_runs.py
'''
python scripts/modeling/summarize_runs.py \
  --glob "outputs/modeling/run_006_seed_sweep_seed*/metrics.json" \
  --out outputs/modeling/run_006_seed_sweep_summary.csv
'''
import argparse, csv, glob, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/modeling/run_*_seed*/metrics.json",
                    help="glob pattern for metrics.json")
    ap.add_argument("--out", default="outputs/modeling/seed_sweep_summary.csv")
    args = ap.parse_args()

    rows = []
    for mp in sorted(glob.glob(args.glob)):
        run_dir = os.path.dirname(mp)
        with open(mp, "r") as f:
            obj = json.load(f)

        best_key = obj.get("best_key")
        best_score = obj.get("best_score")
        hist = obj.get("history", [])

        # find epoch of best_score (first match)
        best_epoch = None
        for h in hist:
            if abs(h.get(best_key, -999) - best_score) < 1e-12:
                best_epoch = h.get("epoch")
                break

        # pull some useful context
        last = hist[-1] if hist else {}
        rows.append({
            "run_dir": run_dir,
            "best_key": best_key,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "video_topk": last.get("video_topk"),
            "seed": _infer_seed(run_dir),
            "total_sec": (hist[-1].get("total_sec") if hist else None),
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["run_dir","best_key","best_score","best_epoch","video_topk","seed","total_sec"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("saved:", args.out)
    # also print a compact view
    for r in rows:
        print(f"{r['seed']}\tbest={r['best_score']:.4f}\tepoch={r['best_epoch']}\tsec={r['total_sec']}  {r['run_dir']}")

def _infer_seed(run_dir: str):
    # try to parse "...seed42" from dir name
    import re
    m = re.search(r"seed(\d+)", run_dir)
    return int(m.group(1)) if m else None

if __name__ == "__main__":
    main()
