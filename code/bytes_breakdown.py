"""
bytes_breakdown.py - where the misclassified traffic of two strategies differs (Nov 12).

Usage (after bootstrap_bytes.py):
    python bytes_breakdown.py 1                       # LIGHT vs FULL_W120 at 250k
    python bytes_breakdown.py 1 --budget 50000
    python bytes_breakdown.py 1 --a LIGHT --b REFIT --budget 10000

Uses the results/wrong/ files saved by bootstrap_bytes.py, so it only takes a
minute. For KEEP and the two strategies (averaged over seeds and test weeks)
it gives misclassified GB per week:
  - per class
  - per flow size group (top 0.1% of flows by bytes, next 0.9%, next 9%,
    bottom 90%), to see if the difference comes from a few large flows
and the difference B - A (positive = A misclassifies less).
Saves results/bytes_breakdown_c{chain}_{A}_vs_{B}_B{budget}.csv
"""
import argparse
import json

import numpy as np
import pandas as pd

import common as C
import run_grid as R

SIZE_GROUPS = [("top 0.1%", 0.999, 1.0), ("next 0.9%", 0.99, 0.999),
               ("next 9%", 0.90, 0.99), ("bottom 90%", 0.0, 0.90)]


def load_wrong(run_id, week, n):
    path = C.RESULTS_DIR / "wrong" / f"{run_id}_w{week:02d}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} missing: run bootstrap_bytes.py {run_id[1]} first")
    return np.unpackbits(np.load(path), count=n).astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chain", type=int, choices=[1, 2])
    ap.add_argument("--a", default="LIGHT")
    ap.add_argument("--b", default="FULL_W120")
    ap.add_argument("--budget", type=int, default=250_000)
    a = ap.parse_args()
    chain, cfg = a.chain, R.CHAINS[a.chain]
    seeds = json.loads((C.SAMPLES_DIR / f"c{chain}_samples_meta.json").read_text())["seeds"]
    names = C.class_names()

    def ids(st):
        if st == "KEEP":
            return [C.make_run_id(chain, "main", "KEEP", 0, "none", 0)]
        tier = R.TIER_OF.get(st, "main")
        return [C.make_run_id(chain, tier, st, a.budget, "random", s) for s in seeds]

    strategies = {"KEEP": ids("KEEP"), a.a: ids(a.a), a.b: ids(a.b)}
    weeks = cfg["tests"]
    by_class = {st: np.zeros(C.N_CLASSES) for st in strategies}
    by_size = {st: np.zeros(len(SIZE_GROUPS)) for st in strategies}
    size_share = np.zeros(len(SIZE_GROUPS))
    for w in weeks:
        df = C.load_week(w)                                  # only labels and bytes are used
        idx = C.load_indices(f"c{chain}_test_w{w:02d}")
        y = df["y"].to_numpy()[idx]
        b = df["bytes_total"].to_numpy()[idx]
        del df
        scale = R.week_rows(w) / len(idx) / C.GB / len(weeks)   # to GB per week, averaged over weeks
        rank = b.argsort().argsort() / (len(b) - 1)              # 0 = smallest, 1 = largest
        groups = [(rank >= lo) & (rank <= hi if hi == 1.0 else rank < hi)
                  for _, lo, hi in SIZE_GROUPS]
        size_share += np.array([b[g].sum() for g in groups]) / b.sum() / len(weeks)
        for st, run_ids in strategies.items():
            for rid in run_ids:
                wb = b * load_wrong(rid, w, len(y)) * scale / len(run_ids)
                by_class[st] += np.bincount(y, weights=wb, minlength=C.N_CLASSES)
                by_size[st] += np.array([wb[g].sum() for g in groups])
        print(f"  week {w}: done")

    rows = []
    for i, name in enumerate(names):
        rows.append({"breakdown": "class", "group": name,
                     **{f"{st}_gb_week": by_class[st][i] for st in strategies},
                     f"{a.a}_saves_vs_{a.b}": by_class[a.b][i] - by_class[a.a][i]})
    for j, (label, _, _) in enumerate(SIZE_GROUPS):
        rows.append({"breakdown": "flow size", "group": f"{label} ({size_share[j]:.0%} of bytes)",
                     **{f"{st}_gb_week": by_size[st][j] for st in strategies},
                     f"{a.a}_saves_vs_{a.b}": by_size[a.b][j] - by_size[a.a][j]})
    out = pd.DataFrame(rows)
    path = C.RESULTS_DIR / f"bytes_breakdown_c{chain}_{a.a}_vs_{a.b}_B{a.budget}.csv"
    out.to_csv(path, index=False)
    C.sync_to_drive(path)
    tot = {st: by_class[st].sum() for st in strategies}
    print(f"\nchain {chain}, B = {a.budget}: misclassified GB/week (mean over seeds and weeks)")
    print("  totals: " + ", ".join(f"{st} {v:.1f}" for st, v in tot.items()))
    print(out.round(2).to_string(index=False))
    print(f"\nSaved {path.name}")


if __name__ == "__main__":
    main()
