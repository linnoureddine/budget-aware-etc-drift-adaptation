"""
bootstrap_bytes.py - confidence intervals for the byte error and GB/week (Nov 10).

Usage (after run_grid main + capacity):
    python bootstrap_bytes.py 1            # chain 1, takes 3-4 hours the first time
    python bootstrap_bytes.py 1 --boots 2000
    python bootstrap_bytes.py 1 --budgets 2000 250000   # quick test

A few very large flows carry most of the bytes, so the byte error depends a
lot on which test flows were sampled, not just on the label seed. So we
bootstrap over both.

Steps:
  1. predicts the 500k test flows of every week with each saved model and
     saves which flows are wrong (results/wrong/, reused on re-runs)
  2. checks the byte error matches results_c{chain}.csv
  3. bootstrap (1000 by default): resample the seeds with replacement and
     reweight the test flows with Poisson(1) weights (same weights for all
     models in one replicate, so comparisons stay paired)
  4. saves bootstrap_bytes_c{chain}.csv: byte error, GB/week and GB/week
     saved vs KEEP, FULL, FULL_W, FULL_W120 with 95% intervals
  5. saves bytes_by_class_c{chain}.csv: share of bytes and of misclassified
     bytes per class for KEEP

GB per week = misclassified GB in the sample x (10-class flows in the week / 500k).
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import common as C
import run_grid as R

STRATEGIES = ["LIGHT", "REFIT", "FULL", "FULL_W", "FULL_W120"]
COMPARE_TO = ["KEEP", "FULL", "FULL_W", "FULL_W120"]


def wrong_mask(run_id, week, X, y):
    path = C.RESULTS_DIR / "wrong" / f"{run_id}_w{week:02d}.npy"
    if path.exists():
        return np.unpackbits(np.load(path), count=len(y)).astype(bool)
    booster = lgb.Booster(model_file=str(C.MODELS_DIR / f"{run_id}.txt"))
    wrong = C.predict_classes(booster, X) != y
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.packbits(wrong))
    return wrong


def model_list(chain, seeds, budgets):
    """list of (strategy, budget, seed, run_id), skips groups with a missing model"""
    out = [("KEEP", 0, 0, C.make_run_id(chain, "main", "KEEP", 0, "none", 0)),
           ("FULL_ALL", 0, 0, C.make_run_id(chain, "main", "FULL_ALL", 0, "all", 0))]
    for st in STRATEGIES:
        tier = R.TIER_OF.get(st, "main")
        for B in budgets:
            group = [(st, B, s, C.make_run_id(chain, tier, st, B, "random", s)) for s in seeds]
            missing = [g[3] for g in group if not (C.MODELS_DIR / f"{g[3]}.txt").exists()]
            if missing:
                print(f"  skipped {st} B{B}: {len(missing)} model(s) missing, e.g. {missing[0]}")
                continue
            out += group
    for g in out[:2]:
        if not (C.MODELS_DIR / f"{g[3]}.txt").exists():
            raise FileNotFoundError(f"{g[3]}.txt not found in {C.MODELS_DIR}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chain", type=int, choices=[1, 2])
    ap.add_argument("--boots", type=int, default=1000)
    ap.add_argument("--budgets", type=int, nargs="*")
    a = ap.parse_args()
    chain, cfg = a.chain, R.CHAINS[a.chain]
    meta = json.loads((C.SAMPLES_DIR / f"c{chain}_samples_meta.json").read_text())
    seeds, budgets = meta["seeds"], a.budgets or meta["budgets"]
    models = model_list(chain, seeds, budgets)
    print(f"chain {chain}: {len(models)} models x {len(cfg['tests'])} test weeks, B = {a.boots}")

    # ---- 1. wrong flows per model ----
    weeks = cfg["tests"]
    tests, wrong = {}, {}
    for w in weeks:
        idx = C.load_indices(f"c{chain}_test_w{w:02d}")
        X, y, b = R.take(w, idx, ten_classes_only=True)
        tests[w] = (y, b.astype(np.float64), R.week_rows(w) / len(idx))
        wrong[w] = np.stack([wrong_mask(m[3], w, X, y) for m in models])
        print(f"  week {w}: done")
        del X

    # ---- 2. compare with the results file ----
    res = pd.read_csv(C.results_path(chain)).set_index(["run_id", "test_week"])["byte_err"]
    diff = max(abs(float(wrong[w][i] @ tests[w][1] / tests[w][1].sum()) - res[(m[3], w)])
               for w in weeks for i, m in enumerate(models) if (m[3], w) in res.index)
    print(f"  check: max |recomputed - saved byte_err| = {diff:.2e}")
    if diff > 1e-6:
        raise ValueError("recomputed byte error does not match results file; models or samples changed?")

    # ---- 3. bootstrap ----
    rng = np.random.default_rng(12345)
    n_s = len(seeds)
    seed_draws = rng.choice(n_s, size=(a.boots, n_s), replace=True)
    M = len(models)
    err = np.zeros((a.boots + 1, M))        # row 0 = point estimate
    gbw = np.zeros((a.boots + 1, M))
    for w in weeks:
        y, b, scale = tests[w]
        Wf = wrong[w].astype(np.float32)
        for r in range(a.boots + 1):
            v = b if r == 0 else b * rng.poisson(1.0, len(b))
            mis = Wf @ v.astype(np.float32)
            err[r] += mis / v.sum() / len(weeks)
            gbw[r] += mis / C.GB * scale / len(weeks)
        print(f"  bootstrap week {w}: done")

    # one value per (strategy, budget) and replicate, averaged over the seeds
    def series(st, B, arr):
        cols = [i for i, m in enumerate(models) if m[0] == st and m[1] == B]
        if len(cols) == 1:                                   # KEEP, FULL_ALL
            return arr[:, cols[0]]
        cols = np.array(cols)
        point = arr[0, cols].mean()
        boot = arr[1:, cols][np.arange(a.boots)[:, None], seed_draws].mean(axis=1)
        return np.concatenate([[point], boot])

    groups = sorted({(m[0], m[1]) for m in models}, key=lambda g: (g[0], g[1]))
    rows = []

    def add(st, B, quantity, comparison, s):
        lo, hi = np.percentile(s[1:], [2.5, 97.5])
        rows.append({"chain": chain, "strategy": st, "budget": B, "quantity": quantity,
                     "comparison": comparison, "point": s[0], "ci_low": lo, "ci_high": hi,
                     "share_le_0": float((s[1:] <= 0).mean()) if comparison != "level" else np.nan})

    for st, B in groups:
        e, g = series(st, B, err) * 100, series(st, B, gbw)
        add(st, B, "byte_err_pct", "level", e)
        add(st, B, "misclassified_gb_week", "level", g)
        for other in COMPARE_TO:
            if other == st:
                continue
            oB = 0 if other == "KEEP" else B
            if (other, oB) not in groups or st in ("KEEP",):
                continue
            add(st, B, "byte_err_pts_saved", f"vs_{other}", series(other, oB, err) * 100 - e)
            add(st, B, "gb_week_saved", f"vs_{other}", series(other, oB, gbw) - g)
    out = pd.DataFrame(rows)
    path = C.RESULTS_DIR / f"bootstrap_bytes_c{chain}.csv"
    out.to_csv(path, index=False)
    C.sync_to_drive(path)

    # ---- 5. KEEP byte error per class ----
    names = C.class_names()
    k = [i for i, m in enumerate(models) if m[0] == "KEEP"][0]
    tot = np.zeros(C.N_CLASSES)
    mis = np.zeros(C.N_CLASSES)
    for w in weeks:
        y, b, _ = tests[w]
        tot += np.bincount(y, weights=b, minlength=C.N_CLASSES)
        mis += np.bincount(y, weights=b * wrong[w][k], minlength=C.N_CLASSES)
    by_class = pd.DataFrame({"class_id": range(C.N_CLASSES), "service": names,
                             "share_of_test_bytes": tot / tot.sum(),
                             "share_of_misclassified_bytes": mis / mis.sum(),
                             "class_byte_err": mis / np.maximum(tot, 1)})
    by_class = by_class.sort_values("share_of_misclassified_bytes", ascending=False)
    cpath = C.RESULTS_DIR / f"bytes_by_class_c{chain}.csv"
    by_class.to_csv(cpath, index=False)
    C.sync_to_drive(cpath)

    show = out[(out.comparison.isin(["vs_KEEP", "vs_FULL_W", "vs_FULL_W120"]))
               & (out.quantity == "gb_week_saved")]
    print("\nGB/week saved (point [95% CI]):")
    for _, r in show.iterrows():
        print(f"  {r.strategy:9s} B{r.budget:<7d} {r.comparison:13s} "
              f"{r.point:7.1f} [{r.ci_low:6.1f}, {r.ci_high:6.1f}]")
    print(f"\nKEEP byte error by class:\n{by_class.round(3).to_string(index=False)}")
    print(f"\nSaved {path.name} and {cpath.name}")
    C.log_event(f"bootstrap_bytes chain {chain}: B={a.boots}, {len(models)} models")


if __name__ == "__main__":
    main()