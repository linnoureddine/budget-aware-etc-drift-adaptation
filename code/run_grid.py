"""
run_grid.py - trains every strategy and tests it on every test week.

Usage:
    python run_grid.py 1                 # chain 1, main grid
    python run_grid.py 1 --plan rounds   # LIGHT with 10 / 50 rounds
    python run_grid.py 1 --plan capacity # FULL_W120 (added Nov 10)
    python run_grid.py 1 --plan timing   # update time + memory, 3 runs, median
    python run_grid.py 1 --plan predict  # prediction speed of saved models
    python run_grid.py 1 --plan timing --strategies FULL_W120   # only these
    python run_grid.py 1 --seeds 42 --budgets 2000   # quick test

Needs in samples/:
    c{chain}_train_S.npy, c{chain}_test_w{WW}.npy   (Aya)
    budget / older / strat samples + c{chain}_samples_meta.json  (make_samples_l4.py)

Main grid, for every seed and budget:
    LIGHT, REFIT, FULL, FULL_W, FULL120 on the budget sample
    LIGHT_OLD (recency control), LIGHT and FULL on the strat sample
    and once per chain: KEEP and FULL_ALL
Rounds plan: LIGHT_R10, LIGHT_R50 at 2k and 50k.
Capacity plan: FULL_W120 at every budget and seed (added after the first results).

All models are tested on the same 500k flows of each test week. Results are
appended to results/results_c{chain}.csv as they finish, so a stopped run
can be restarted and skips what's done.

Timing plan: each update 3 times on seed 42, each in a new process, median
saved to timing_c{chain}.csv. These are the cost numbers we report; the
times in the main grid are single runs, only for checking.
Predict plan: times prediction on the first test week (500k flows) for the
models of one budget and seed (default 50k, seed 42) + KEEP and FULL_ALL,
5 times each in a new process. Saves predict_c{chain}.csv.
Close other programs for timing and predict.
"""
import argparse
import gc
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import common as C

CHAINS = {1: {"train": 18, "tests": list(range(23, 31))},
          2: {"train": 36, "tests": list(range(41, 49))}}
ROUNDS_PLAN_BUDGETS = [2_000, 50_000]
TIMING_REPEATS = 3
PREDICT_REPEATS = 5
PREDICT_STRATEGIES = ["KEEP", "REFIT", "FULL", "FULL_W", "FULL120", "FULL_W120",
                      "FULL_ALL", "LIGHT_R10", "LIGHT", "LIGHT_R50"]
TIMING_STRATEGIES = ["LIGHT", "REFIT", "FULL", "FULL_W", "FULL120", "FULL_W120"]
TIER_OF = {"LIGHT_R10": "rounds", "LIGHT_R50": "rounds", "FULL_W120": "capacity"}


# ---------------- data ----------------
def week_rows(week: int) -> int:
    path = C.week_path(week)
    if not path.exists():
        path = C.week_path(week, shared=True)
    return pq.ParquetFile(path).metadata.num_rows


def take(week: int, idx: np.ndarray, ten_classes_only: bool):
    """X, y and bytes for the given rows of a week"""
    df = C.load_week(week, ten_classes_only=ten_classes_only)
    if idx.max() >= len(df):
        raise ValueError(f"week {week}: index {idx.max()} >= {len(df)} rows; "
                         "samples and parquet do not match")
    X, y = C.to_xy(df.iloc[idx])
    b = df["bytes_total"].to_numpy()[idx]
    del df
    gc.collect()
    return X, y, b


class Pool:
    """10-class flows of the label week, looked up by row position"""

    def __init__(self, week: int, expected_rows: int):
        df = C.load_week(week, ten_classes_only=False)
        if len(df) != expected_rows:
            raise ValueError(f"pool week {week}: {len(df)} rows, samples were made on "
                             f"{expected_rows}. Re-check the parquet / samples.")
        y = df["y"].to_numpy()
        self.pos = np.flatnonzero(y >= 0)                     # rows in our 10 classes
        self.X, self.y = C.to_xy(df.iloc[self.pos])
        self.n_rows = len(df)
        del df
        gc.collect()

    def usable(self, idx: np.ndarray):
        """X, y of the sampled rows that are in our 10 classes"""
        k = np.searchsorted(self.pos, idx)
        k = np.minimum(k, len(self.pos) - 1)
        hit = self.pos[k] == idx
        return self.X[k[hit]], self.y[k[hit]]


class Older:
    """loads only the rows of the older week used by the older samples"""

    def __init__(self, chain, week, seeds, budgets):
        files = [f"c{chain}_older_B{B}_s{s}" for s in seeds for B in budgets]
        need = np.unique(np.concatenate([C.load_indices(f) for f in files]))
        self.pos = need
        self.X, self.y, _ = take(week, need, ten_classes_only=False)
        assert (self.y >= 0).all(), "older samples must be 10-class only"

    def get(self, idx):
        k = np.searchsorted(self.pos, idx)
        assert np.array_equal(self.pos[k], idx)
        return self.X[k], self.y[k]


# ---------------- training ----------------
def train_new(X, y, n_rounds, weight=None):
    ds = lgb.Dataset(X, label=y, weight=weight, free_raw_data=True)
    return lgb.train(C.lgb_params(), ds, num_boost_round=n_rounds)


def update(strategy, base, S, new):
    """runs one strategy, returns (model, train_rows). S and new are (X, y)"""
    R = C.rounds()
    Xn, yn = new
    if strategy.startswith("LIGHT"):                       # LIGHT, LIGHT_OLD, LIGHT_R10/50
        extra = {"LIGHT_R10": 10, "LIGHT_R50": 50}.get(strategy, R["light_extra"])
        ds = lgb.Dataset(Xn, label=yn, free_raw_data=True)
        return lgb.train(C.lgb_params(), ds, num_boost_round=extra, init_model=base), len(yn)
    if strategy == "REFIT":
        return base.refit(Xn, yn, decay_rate=C.REFIT_DECAY), len(yn)
    Xs, ys = S
    X = np.concatenate([Xs, Xn])
    y = np.concatenate([ys, yn])
    if strategy in ("FULL_W", "FULL_W120"):
        share = C.NEW_LABEL_WEIGHT_SHARE
        w_new = share / (1 - share) * len(ys) / len(yn)
        w = np.concatenate([np.ones(len(ys)), np.full(len(yn), w_new)])
        n = R["full120"] if strategy == "FULL_W120" else R["full"]
        return train_new(X, y, n, weight=w), len(y)
    n = R["full120"] if strategy == "FULL120" else R["full"]  # FULL, FULL120, FULL_ALL
    return train_new(X, y, n), len(y)


def measured(fn):
    gc.collect()
    with C.Measure() as m:
        out = fn()
    return out, m


# ---------------- evaluation ----------------
def score_and_save(model, info, tests, m, done):
    path = C.MODELS_DIR / f"{info['run_id']}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    kb = C.model_kb(path)
    for w, (X, y, b, scale) in tests.items():
        if (info["run_id"], w) in done:
            continue
        pred, secs, fps = C.predict_timed(model, X)
        summary, per_class, cm = C.evaluate(y, pred, b, week_scale=scale)
        C.append_result({**info, "test_week": w, **summary,
                         "n_trees": model.current_iteration(),
                         "update_time_s": round(m.seconds, 3),
                         "peak_mem_mb": round(m.peak_mb, 1),
                         "peak_mem_delta_mb": round(m.delta_mb, 1),
                         "model_kb": round(kb, 1),
                         "predict_time_s": round(secs, 3), "flows_per_s": round(fps)},
                        per_class, cm)
    print(f"  {info['run_id']}: n_used={info['n_used']:,} "
          f"time={m.seconds:.1f}s mem+={m.delta_mb:.0f}MB")


def jobs(chain, plan, seeds, budgets):
    """list of (strategy, sampling, budget, seed) to run for a plan"""
    out = []
    if plan == "main":
        out += [("KEEP", "none", 0, 0), ("FULL_ALL", "all", 0, 0)]
        for s in seeds:
            for B in budgets:
                out += [(st, "random", B, s) for st in
                        ("LIGHT", "REFIT", "FULL", "FULL_W", "FULL120", "LIGHT_OLD")]
                out += [("LIGHT", "strat", B, s), ("FULL", "strat", B, s)]
    elif plan == "rounds":
        for s in seeds:
            for B in [b for b in budgets if b in ROUNDS_PLAN_BUDGETS]:
                out += [("LIGHT_R10", "random", B, s), ("LIGHT_R50", "random", B, s)]
    elif plan == "capacity":
        for s in seeds:
            for B in budgets:
                out += [("FULL_W120", "random", B, s)]
    return out


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chain", type=int, choices=[1, 2])
    ap.add_argument("--plan", default="main",
                    choices=["main", "rounds", "capacity", "timing", "predict"])
    ap.add_argument("--strategies", nargs="*",
                    help="timing / predict only: measure just these and merge into the existing file")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--budgets", type=int, nargs="*")
    a = ap.parse_args()
    chain = a.chain
    cfg = CHAINS[chain]
    pool_w, old_w = C.LABEL_WEEKS[chain]["pool"], C.LABEL_WEEKS[chain]["older"]
    meta = json.loads((C.SAMPLES_DIR / f"c{chain}_samples_meta.json").read_text())
    seeds = a.seeds or meta["seeds"]
    budgets = a.budgets or meta["budgets"]
    C.save_machine_info()
    print(f"chain {chain}, plan {a.plan}, seeds {seeds}, budgets {budgets}")
    print(f"  {C.lib_versions()}")
    if a.plan == "predict":
        return predict_speed(chain, cfg, budget=(a.budgets or [50_000])[0],
                             seed=(a.seeds or [42])[0], only=a.strategies)

    # training sample S and test samples (same for all strategies)
    S_idx = C.load_indices(f"c{chain}_train_S")
    Xs, ys, _ = take(cfg["train"], S_idx, ten_classes_only=True)
    S = (Xs, ys)
    print(f"  S: {len(ys):,} flows of week {cfg['train']}")
    tests = {}
    if a.plan != "timing":
        for w in cfg["tests"]:
            idx = C.load_indices(f"c{chain}_test_w{w:02d}")
            X, y, b = take(w, idx, ten_classes_only=True)
            tests[w] = (X, y, b, week_rows(w) / len(idx))   # scale for GB per week
        print(f"  test weeks {cfg['tests']}: {len(idx):,} flows each")
    pool = Pool(pool_w, meta["pool_rows"])
    older = Older(chain, old_w, seeds, budgets)
    done = C.done_keys(chain)

    # base model (KEEP), trained once and reused by LIGHT and REFIT
    base_path = C.MODELS_DIR / f"c{chain}_base.txt"
    keep_id = C.make_run_id(chain, "main", "KEEP", 0, "none", 0)
    if base_path.exists():
        base = lgb.Booster(model_file=str(base_path))
        m_base = None
    else:
        base, m_base = measured(lambda: train_new(Xs, ys, C.rounds()["base"]))
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base.save_model(str(base_path))
        C.sync_to_drive(base_path)

    if a.plan == "timing":
        return timing(chain, base, S, pool, budgets, only=a.strategies)

    for strategy, sampling, B, seed in jobs(chain, a.plan, seeds, budgets):
        tier = a.plan
        run_id = C.make_run_id(chain, tier, strategy, B, sampling, seed)
        if all((run_id, w) in done for w in tests):
            continue
        info = {"run_id": run_id, "chain": chain, "tier": tier, "strategy": strategy,
                "budget": B, "sampling": sampling, "seed": seed}
        if strategy == "KEEP":
            if m_base is None:                      # base was loaded, retrain to time it
                _, m_base = measured(lambda: train_new(Xs, ys, C.rounds()["base"]))
            score_and_save(base, {**info, "label_week": cfg["train"], "n_queried": 0,
                                  "n_used": 0, "train_rows": len(ys)}, tests, m_base, done)
            continue
        if strategy == "FULL_ALL":
            new = (pool.X, pool.y)
            info.update(label_week=pool_w, n_queried=pool.n_rows, n_used=len(pool.y))
        elif strategy == "LIGHT_OLD":
            new = older.get(C.load_indices(f"c{chain}_older_B{B}_s{seed}"))
            info.update(label_week=old_w, n_queried=len(new[1]), n_used=len(new[1]))
        elif sampling == "strat":
            new = pool.usable(C.load_indices(f"c{chain}_strat_B{B}_s{seed}"))
            info.update(label_week=pool_w, n_queried=len(new[1]), n_used=len(new[1]))
        else:
            new = pool.usable(C.load_indices(f"c{chain}_budget_B{B}_s{seed}"))
            info.update(label_week=pool_w, n_queried=B, n_used=len(new[1]))
        (model, train_rows), m = measured(lambda: update(strategy, base, S, new))
        info["train_rows"] = train_rows
        score_and_save(model, info, tests, m, done)
        del model
    C.log_event(f"run_grid chain {chain} plan {a.plan} finished")
    print("\nALL DONE.")


def _timed_in_child(strategy, base_path, S, new):
    """runs in a new process, only the update itself is timed"""
    base = lgb.Booster(model_file=base_path) if base_path else None
    if strategy == "BASE_TRAIN":
        _, m = measured(lambda: train_new(*S, C.rounds()["base"]))
    else:
        _, m = measured(lambda: update(strategy, base, S, new))
    return m.seconds, m.delta_mb


def fresh_process_measure(strategy, base_path, S, new):
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(_timed_in_child, strategy, base_path, S, new).result()


def merge_save(rows, path, only, key="strategy"):
    """saves rows. with --strategies only those rows are replaced in the file"""
    new = pd.DataFrame(rows)
    if only and path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old[key].isin(only)], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    C.sync_to_drive(path)


def timing(chain, base, S, pool, budgets, only=None):
    """each update 3 times on seed 42 in a new process, keeps the median"""
    base_path = str(C.MODELS_DIR / f"c{chain}_base.txt")
    rows = []
    strategies = [s for s in TIMING_STRATEGIES if not only or s in only]
    for B in budgets:
        new = pool.usable(C.load_indices(f"c{chain}_budget_B{B}_s42"))
        for strategy in strategies:
            times, mems = [], []
            for r in range(TIMING_REPEATS):
                secs, mem = fresh_process_measure(strategy, base_path, S, new)
                times.append(secs)
                mems.append(mem)
            rows.append({"chain": chain, "strategy": strategy, "budget": B,
                         "n_used": len(new[1]), "median_s": float(np.median(times)),
                         "min_s": min(times), "max_s": max(times),
                         "median_mem_delta_mb": float(np.median(mems)),
                         "machine": C.machine_info()["machine"], "versions": C.lib_versions()})
            print(f"  B{B} {strategy}: {np.median(times):.1f}s")
    if not only or "BASE_TRAIN" in only:
        res = [fresh_process_measure("BASE_TRAIN", None, S, None) for _ in range(TIMING_REPEATS)]
        times = [r[0] for r in res]
        rows.append({"chain": chain, "strategy": "BASE_TRAIN", "budget": 0, "n_used": 0,
                     "median_s": float(np.median(times)), "min_s": min(times), "max_s": max(times),
                     "median_mem_delta_mb": float(np.median([r[1] for r in res])),
                     "machine": C.machine_info()["machine"], "versions": C.lib_versions()})
    path = C.RESULTS_DIR / f"timing_c{chain}.csv"
    merge_save(rows, path, only)
    print(f"\nTiming saved to {path}")


# ---------------- prediction speed ----------------
def _predict_in_child(model_path, X):
    """new process: load the model, warm up on 1000 rows, then time the prediction"""
    booster = lgb.Booster(model_file=model_path)
    C.predict_timed(booster, X[:1000])
    _, secs, fps = C.predict_timed(booster, X)
    return secs, fps, booster.current_iteration()


def predict_speed(chain, cfg, budget, seed, only=None):
    w = cfg["tests"][0]
    X, _, _ = take(w, C.load_indices(f"c{chain}_test_w{w:02d}"), ten_classes_only=True)
    print(f"  prediction speed on week {w}: {len(X):,} flows, models B{budget} seed {seed}")
    rows = []
    for strategy in [s for s in PREDICT_STRATEGIES if not only or s in only]:
        if strategy == "KEEP":
            run_id = C.make_run_id(chain, "main", "KEEP", 0, "none", 0)
        elif strategy == "FULL_ALL":
            run_id = C.make_run_id(chain, "main", "FULL_ALL", 0, "all", 0)
        else:
            tier = TIER_OF.get(strategy, "main")
            run_id = C.make_run_id(chain, tier, strategy, budget, "random", seed)
        path = C.MODELS_DIR / f"{run_id}.txt"
        if not path.exists():
            print(f"  {strategy}: model {path.name} not found, skipped")
            continue
        res = []
        for _ in range(PREDICT_REPEATS):
            with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
                res.append(ex.submit(_predict_in_child, str(path), X).result())
        secs = float(np.median([r[0] for r in res]))
        rows.append({"chain": chain, "strategy": strategy, "run_id": run_id,
                     "n_trees": res[0][2], "model_kb": round(C.model_kb(path), 1),
                     "test_flows": len(X), "median_s": round(secs, 3),
                     "min_s": round(min(r[0] for r in res), 3),
                     "max_s": round(max(r[0] for r in res), 3),
                     "flows_per_s": round(len(X) / secs),
                     "us_per_flow": round(secs / len(X) * 1e6, 2),
                     "threads": C.lgb_params()["num_threads"],
                     "machine": C.machine_info()["machine"], "versions": C.lib_versions()})
        print(f"  {strategy}: {rows[-1]['n_trees']} rounds, {rows[-1]['flows_per_s']:,} flows/s")
    path = C.RESULTS_DIR / f"predict_c{chain}.csv"
    merge_save(rows, path, only)
    print(f"\nPrediction speed saved to {path}")


if __name__ == "__main__":
    main()