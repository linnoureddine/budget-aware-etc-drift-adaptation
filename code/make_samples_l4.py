"""
make_samples_l4.py - label budget samples, control samples and label yield.

Usage:
    python make_samples_l4.py             # both chains
    python make_samples_l4.py 1           # only chain 1
    python make_samples_l4.py 1 --force   # overwrite existing samples

For each seed (42, 7, 123, 11, 2026) and budget B (2k, 10k, 50k, 250k):

  budget: c{chain}_budget_B{B}_s{seed}.npy
    B flows drawn uniformly at random from all flows of the label week
    (22 / 40), any service. Only flows in our 10 classes can be used (n_used),
    the rest are wasted labels -> label yield = n_used / B.
    Budgets are nested: for a seed, 2k is inside 10k, inside 50k, inside 250k.

  older (recency control): c{chain}_older_B{B}_s{seed}.npy
    same n_used as the budget sample, from the 10-class flows of week 17 / 35

  strat (stratified control): c{chain}_strat_B{B}_s{seed}.npy
    same n_used, from the label week, equal number per class

  c{chain}_label_yield.csv: n_queried, n_used, yield and counts per class
    for every sample

Indices are row positions in week_XX.parquet (all services kept), saved
sorted as int64. Budget files include the other services too; run_grid.py
keeps only the 10-class rows.

Each (seed, chain, sample type) has its own random generator, so the files
are reproducible. Both laptops use the same saved files.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

# 5 seeds (was 3 before Nov 5) so we can get confidence intervals over seeds
SEEDS = [42, 7, 123, 11, 2026]
BUDGETS = [2_000, 10_000, 50_000, 250_000]       # smallest first
TYPE_CODE = {"budget": 0, "older": 1, "strat": 2}


# ---------------- helpers ----------------
def read_apps(week: int) -> np.ndarray:
    """service name of every row in the processed week"""
    path = C.week_path(week)
    if not path.exists():
        path = C.week_path(week, shared=True)
    if not path.exists():
        raise FileNotFoundError(f"week_{week:02d}.parquet not found (preprocess it first)")
    return pd.read_parquet(path, columns=[C.LABEL_COL])[C.LABEL_COL].to_numpy()


def rng_for(seed: int, chain: int, kind: str) -> np.random.Generator:
    return np.random.default_rng([seed, chain, TYPE_CODE[kind]])


def random_order(rng: np.random.Generator, n_rows: int, k: int) -> np.ndarray:
    """k different rows out of n_rows in random order. taking the first B
    rows gives a random sample of size B, so the budgets are nested"""
    if k > n_rows:
        raise ValueError(f"asked for {k} rows but only {n_rows} exist")
    return rng.choice(n_rows, size=k, replace=False, shuffle=True)


def equal_quotas(total: int, available: dict[str, int]) -> dict[str, int]:
    """splits total evenly over the classes. if a class doesn't have enough
    rows, it gives what it has and the rest goes to the other classes"""
    names = sorted(available)
    quota = {c: 0 for c in names}
    left = total
    open_ = [c for c in names if available[c] > 0]
    while left > 0 and open_:
        share, extra = divmod(left, len(open_))
        for i, c in enumerate(open_):
            want = share + (1 if i < extra else 0)
            take = min(want, available[c] - quota[c])
            quota[c] += take
            left -= take
        open_ = [c for c in open_ if quota[c] < available[c]]
    if left > 0:
        raise ValueError(f"not enough 10-class rows for {total} stratified labels")
    return quota


def class_counts(apps: np.ndarray, idx: np.ndarray, classes: list[str]) -> dict[str, int]:
    vc = pd.Series(apps[idx]).value_counts()
    return {c: int(vc.get(c, 0)) for c in classes}


def save_idx(name: str, idx: np.ndarray, force: bool) -> Path:
    idx = np.sort(np.asarray(idx, dtype=np.int64))
    assert len(np.unique(idx)) == len(idx), f"{name}: duplicate rows"
    path = C.SAMPLES_DIR / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        old = np.load(path)
        if not np.array_equal(old, idx):
            raise FileExistsError(f"{path} exists with DIFFERENT content. "
                                  "Samples are frozen; use --force only if you mean it.")
        return path                                   # already saved
    np.save(path, idx)
    return path


# ---------------- one chain ----------------
def make_chain(chain: int, classes: list[str], force: bool) -> pd.DataFrame:
    pool_w = C.LABEL_WEEKS[chain]["pool"]
    old_w = C.LABEL_WEEKS[chain]["older"]
    print(f"\nchain {chain}: pool week {pool_w}, older week {old_w}")

    pool_apps = read_apps(pool_w)
    old_apps = read_apps(old_w)
    keep = set(classes)
    pool10 = np.flatnonzero(pd.Series(pool_apps).isin(keep).to_numpy())
    old10 = np.flatnonzero(pd.Series(old_apps).isin(keep).to_numpy())
    pool_by_class = {c: np.flatnonzero(pool_apps == c) for c in classes}
    print(f"  pool week: {len(pool_apps):,} flows, {len(pool10):,} in our 10 "
          f"({len(pool10) / len(pool_apps):.1%})")
    print(f"  older week: {len(old_apps):,} flows, {len(old10):,} in our 10")

    rows = []
    for seed in SEEDS:
        # budget: one random order of the label week, each budget takes the first B
        order = random_order(rng_for(seed, chain, "budget"), len(pool_apps), max(BUDGETS))
        n_used = {}
        for B in BUDGETS:
            idx = order[:B]
            used = idx[np.isin(pool_apps[idx], classes)]
            n_used[B] = len(used)
            save_idx(f"c{chain}_budget_B{B}_s{seed}", idx, force)
            rows.append(("budget", pool_w, B, seed, B, used))

        # older: n_used labels from the older week's 10-class flows
        o_order = old10[random_order(rng_for(seed, chain, "older"), len(old10), n_used[max(BUDGETS)])]
        for B in BUDGETS:
            idx = o_order[:n_used[B]]
            save_idx(f"c{chain}_older_B{B}_s{seed}", idx, force)
            rows.append(("older", old_w, B, seed, n_used[B], idx))

        # strat: n_used labels from the label week, equal number per class
        rng = rng_for(seed, chain, "strat")
        avail = {c: len(pool_by_class[c]) for c in classes}
        q_max = equal_quotas(n_used[max(BUDGETS)], avail)
        c_order = {c: pool_by_class[c][random_order(rng, avail[c], q_max[c])] for c in sorted(classes)}
        for B in BUDGETS:
            q = equal_quotas(n_used[B], avail)
            idx = np.concatenate([c_order[c][:q[c]] for c in sorted(classes)])
            save_idx(f"c{chain}_strat_B{B}_s{seed}", idx, force)
            rows.append(("strat", pool_w, B, seed, n_used[B], idx))
        print(f"  seed {seed}: n_used = " + ", ".join(f"B{B}: {n_used[B]:,}" for B in BUDGETS))

    # ---------------- label yield table ----------------
    table = []
    for kind, week, B, seed, n_queried, idx in rows:
        apps = pool_apps if week == pool_w else old_apps
        if kind == "budget":
            idx = idx[np.isin(apps[idx], classes)]           # usable rows only
        counts = class_counts(apps, idx, classes)
        table.append({
            "chain": chain, "sample": kind, "label_week": week, "budget": B, "seed": seed,
            "n_queried": n_queried, "n_used": len(idx),
            "yield": round(len(idx) / n_queried, 4) if kind == "budget" else np.nan,
            "smallest_class_n": min(counts.values()),
            "smallest_class": min(counts, key=counts.get),
            "classes_present": sum(v > 0 for v in counts.values()),
            **{f"n_{c}": counts[c] for c in classes},
            "file": f"c{chain}_{kind}_B{B}_s{seed}.npy",
        })
    df = pd.DataFrame(table)
    check_chain(chain, df, pool_apps, old_apps, classes)

    out = C.SAMPLES_DIR / f"c{chain}_label_yield.csv"
    df.to_csv(out, index=False)
    meta = {"chain": chain, "pool_week": pool_w, "older_week": old_w, "seeds": SEEDS,
            "budgets": BUDGETS, "pool_rows": int(len(pool_apps)), "older_rows": int(len(old_apps)),
            "rng": "np.random.default_rng([seed, chain, code]), code budget=0 older=1 strat=2",
            "index_convention": "row positions in week_XX.parquet "
                                "(= load_week(week, ten_classes_only=False) order), sorted int64"}
    (C.SAMPLES_DIR / f"c{chain}_samples_meta.json").write_text(json.dumps(meta, indent=2))

    summary = (df[df["sample"] == "budget"].groupby("budget")
               .agg(n_used_mean=("n_used", "mean"), yield_mean=("yield", "mean"),
                    smallest_class_min=("smallest_class_n", "min"),
                    classes_present_min=("classes_present", "min")))
    print(f"\n  label yield, chain {chain} (mean over seeds):\n{summary.round(3).to_string()}")
    C.log_event(f"made chain-{chain} budget/older/strat samples "
                f"(seeds {SEEDS}, budgets {BUDGETS}); yield table {out.name}")
    return df


# ---------------- checks ----------------
def check_chain(chain, df, pool_apps, old_apps, classes):
    """checks the samples are sorted, in range, nested and matched on n_used"""
    keep = set(classes)
    for seed in SEEDS:
        for kind in ("budget", "older", "strat"):
            prev = None
            for B in BUDGETS:
                idx = np.load(C.SAMPLES_DIR / f"c{chain}_{kind}_B{B}_s{seed}.npy")
                apps = old_apps if kind == "older" else pool_apps
                assert idx.dtype == np.int64 and np.all(np.diff(idx) > 0), f"{kind} B{B} s{seed}: not sorted/unique"
                assert idx.min() >= 0 and idx.max() < len(apps), f"{kind} B{B} s{seed}: index out of range"
                if kind != "budget":
                    assert set(apps[idx]) <= keep, f"{kind} B{B} s{seed}: has non-10-class rows"
                if prev is not None:
                    assert np.isin(prev, idx).all(), f"{kind} B{B} s{seed}: not nested"
                prev = idx
            # controls must have the same n_used as the budget sample
            used = df[(df["sample"] == "budget") & (df["seed"] == seed)].set_index("budget")["n_used"]
            for other in ("older", "strat"):
                o = df[(df["sample"] == other) & (df["seed"] == seed)].set_index("budget")["n_used"]
                assert (o == used).all(), f"{other} s{seed}: n_used does not match budget"
    print("  checks passed: sorted, in range, nested, 10-class only, matched n_used")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    chains = [int(a) for a in args] or [1, 2]
    classes = C.load_classes()
    for chain in chains:
        try:
            make_chain(chain, classes, force)
        except FileNotFoundError as e:
            print(f"\nchain {chain}: SKIPPED ({e})")
    print(f"\nSamples are in {C.SAMPLES_DIR}. Copy them to the shared samples folder.")


if __name__ == "__main__":
    main()