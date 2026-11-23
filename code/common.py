"""
common.py - shared code for the CMPS 342 project (CESNET-TLS-Year22).
Features, labels, LightGBM params, metrics and the results format all live
here so both of us use the same setup. If you change something, note it in
logs/run_log.md.

Changes:
  Nov 2  - num_threads fixed in params.json so both laptops train identical models
         - bytes_total read from BYTES_TOTAL (exact int64 from preprocess.py)
         - finished result files are copied to the shared Drive folder
  Nov 5  - added FULL_W, REFIT and LIGHT_R10/R50
         - Measure also gives the memory increase during training (delta_mb)
         - library versions saved with every result row
  Nov 10 - added FULL_W120 (FULL_W with 120 rounds). Added after the first
           results, since LIGHT has 120 trees and FULL_W only 100.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

# ---------------- paths ----------------
# PROJECT_ROOT: shared Drive folder
# OUT_ROOT: local folder we write to while running (Drive locks files when
#   syncing), finished files get copied to PROJECT_ROOT
# RAW_ROOT: downloaded raw weeks
# set once per laptop:
#   conda env config vars set CESNET_RAW=D:\cesnet_raw CESNET_OUT=D:\cesnet_work
PROJECT_ROOT = Path(os.environ.get("CESNET_PROJECT", Path(__file__).resolve().parents[1]))
OUT_ROOT = Path(os.environ.get("CESNET_OUT", PROJECT_ROOT))
RAW_ROOT = Path(os.environ.get("CESNET_RAW", r"C:\cesnet_raw"))

CONFIG_DIR = PROJECT_ROOT / "config"
SAMPLES_DIR = PROJECT_ROOT / "samples"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"        # shared copy
PROCESSED_LOCAL = OUT_ROOT / "data" / "processed"          # written here first
MODELS_DIR = OUT_ROOT / "models"
RESULTS_DIR = OUT_ROOT / "results"
LOGS_DIR = OUT_ROOT / "logs"

LABEL_COL = "APP"                      # service name (from the SNI in CESNET)
BYTES_COLS = ("BYTES", "BYTES_REV")    # added together for the byte-weighted error
N_CLASSES = 10

# ---------------- features ----------------
# we can't use these as features, they leak the label or identify the server
EXCLUDED = {
    "TLS_SNI", "TLS_JA3", "SRC_IP", "DST_IP", "DST_ASN", "SRC_PORT", "DST_PORT",
    "TIME_FIRST", "TIME_LAST", "APP", "CATEGORY", "ID",
}

# the 28 features from the dataset docs: 8 flow stats + 16 TCP flags +
# 4 flow end reasons. only used by write_features_json(), after checking
# it against the pilot notebook
_FLAGS = ["CWR", "ECE", "URG", "ACK", "PSH", "RST", "SYN", "FIN"]
CANDIDATE_FEATURES = (
    ["DURATION", "BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV",
     "PPI_LEN", "PPI_DURATION", "PPI_ROUNDTRIPS"]
    + [f"FLAG_{f}" for f in _FLAGS] + [f"FLAG_{f}_REV" for f in _FLAGS]
    + ["FLOW_ENDREASON_IDLE", "FLOW_ENDREASON_ACTIVE",
       "FLOW_ENDREASON_END", "FLOW_ENDREASON_OTHER"])


def write_features_json(features: list[str] = CANDIDATE_FEATURES) -> None:
    """writes config/features.json (run once)"""
    path = CONFIG_DIR / "features.json"
    if path.exists():
        raise FileExistsError(f"{path} already exists; it is frozen")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"features": list(features)}, indent=2))


def _load_features() -> list[str]:
    # reads the 28 features from config/features.json (same list on both laptops)
    path = CONFIG_DIR / "features.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create it with the pilot's 28 feature names, "
            'e.g. {"features": ["DURATION", "BYTES", ...]}, or, after checking '
            "the candidate list against the pilot: "
            'python -c "import common; common.write_features_json()"')
    feats = json.loads(path.read_text())["features"]
    if len(feats) != 28 or len(set(feats)) != 28:
        raise ValueError(f"features.json must list 28 unique names, got {len(feats)}")
    # make sure no excluded column is in there
    bad = [f for f in feats if f in EXCLUDED]
    if bad:
        raise ValueError(f"Excluded columns found in features.json: {bad}")
    return feats


try:
    FEATURES: list[str] = _load_features()
except FileNotFoundError:
    FEATURES = []          # so write_features_json() can still run


def _need_features():
    if not FEATURES:
        _load_features()   # raises FileNotFoundError with instructions


# ---------------- config + label encoder ----------------
def load_json(name: str) -> dict | list:
    return json.loads((CONFIG_DIR / name).read_text())


def save_json(name: str, obj) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / name).write_text(json.dumps(obj, indent=2))


def load_classes() -> list[str]:
    # top 10 services of week 18 by flow count
    # label numbers come from config/label_encoder.json, not from this list
    data = load_json("classes.json")
    if isinstance(data, dict):
        key = next((k for k in ("classes_ranked", "classes") if k in data), None)
        if key is None:
            raise KeyError(f"classes.json has no 'classes_ranked' list; keys: {list(data)}")
        classes = data[key]
    else:
        classes = data
    assert len(classes) == N_CLASSES, f"expected {N_CLASSES} classes, got {len(classes)}"
    assert len(set(classes)) == N_CLASSES, "classes.json has duplicates"
    return list(classes)


def build_label_encoder(classes: list[str]) -> dict[str, int]:
    # uses Aya's config/label_encoder.json if it exists (only checks it).
    # otherwise makes one from the sorted names
    path = CONFIG_DIR / "label_encoder.json"
    if path.exists():
        return load_label_encoder()
    names = sorted(classes)
    save_json("label_encoder.json",
              {"classes_sorted": names, "index": {c: i for i, c in enumerate(names)}})
    return load_label_encoder()


def load_label_encoder() -> dict[str, int]:
    # reads the label encoder ("index" or the older "mapping" key) and checks
    # it has the same 10 names as classes.json and labels 0..9
    enc = load_json("label_encoder.json")
    mapping = enc.get("index") or enc.get("mapping")
    if mapping is None:
        raise KeyError(f"label_encoder.json has no 'index' mapping; keys: {list(enc)}")
    mapping = {c: int(i) for c, i in mapping.items()}
    if set(mapping) != set(load_classes()):
        raise ValueError("label_encoder.json names do not match classes.json")
    if sorted(mapping.values()) != LABELS:
        raise ValueError("label_encoder.json numbers must be 0..9, each used once")
    return mapping


def class_names() -> list[str]:
    # class_names()[i] = service with label i
    m = load_label_encoder()
    return sorted(m, key=m.get)


def encode_labels(labels: pd.Series, mapping: dict[str, int]) -> np.ndarray:
    # service name -> 0..9, other services -> -1
    return labels.map(mapping).fillna(-1).astype(np.int16).to_numpy()


# ---------------- raw files (used by preprocess.py) ----------------
def raw_week_files(week: int) -> list[Path]:
    # csv day files of one week (skips .part files)
    folder = RAW_ROOT / f"WEEK-2022-{week:02d}"
    files = sorted(p for p in folder.glob("*.csv*") if not p.name.endswith(".part"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")
    return files


def day_from_name(path: Path) -> int:
    m = re.search(r"(20\d{6})", path.name)
    return int(m.group(1)) if m else -1


def peek_raw_columns(week: int) -> list[str]:
    # prints the raw columns and which of our columns are missing
    cols = list(pd.read_csv(raw_week_files(week)[0], nrows=5).columns)
    want = [LABEL_COL, *BYTES_COLS, *(FEATURES or CANDIDATE_FEATURES)]
    missing = [c for c in want if c not in cols]
    print(f"{len(cols)} raw columns: {cols}")
    print(f"missing expected columns: {missing or 'none'}")
    return cols


# ---------------- loading processed data ----------------
def week_path(week: int, shared: bool = False) -> Path:
    base = PROCESSED_DIR if shared else PROCESSED_LOCAL
    return base / f"week_{week:02d}.parquet"


def load_week(week: int, ten_classes_only: bool = True, extra_cols: tuple = ()) -> pd.DataFrame:
    """
    Loads one processed week (local copy first, then Drive).
    Returns the 28 features (float32), y (0-9, -1 = other service) and bytes_total.
    Row order is kept so the saved .npy indices still match.
    Use ten_classes_only=False for the pool week, since other services
    still cost a label.
    """
    _need_features()
    path = week_path(week)
    if not path.exists():
        path = week_path(week, shared=True)
    cols = list(dict.fromkeys([*FEATURES, LABEL_COL, "BYTES_TOTAL", *extra_cols]))
    df = pd.read_parquet(path, columns=cols)
    out = df[FEATURES].astype(np.float32)
    out["y"] = encode_labels(df[LABEL_COL], load_label_encoder())
    out["bytes_total"] = df["BYTES_TOTAL"].to_numpy(np.float64)
    for c in extra_cols:
        out[c] = df[c]
    if ten_classes_only:
        out = out[out["y"] >= 0].reset_index(drop=True)
    return out


def to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # X, y as numpy arrays for LightGBM
    return np.ascontiguousarray(df[FEATURES].to_numpy(np.float32)), df["y"].to_numpy(np.int32)


def load_indices(name: str) -> np.ndarray:
    # e.g. load_indices("c1_budget_10000_seed42")
    return np.load(SAMPLES_DIR / f"{name}.npy")


# ---------------- LightGBM params ----------------
# fixed before running, not tuned on test weeks, same for every strategy.
# saved to config/params.json on the first run
DEFAULT_LGB = {
    "objective": "multiclass",
    "num_class": N_CLASSES,
    "learning_rate": 0.1,       # default, same as pilot
    "num_leaves": 64,           # pilot value (default is 31)
    "min_data_in_leaf": 20,     # default
    "max_bin": 255,             # default
    "feature_fraction": 1.0,    # no column subsampling
    "bagging_fraction": 1.0,    # no row subsampling
    "lambda_l2": 0.0,           # default
    "deterministic": True,
    "force_row_wise": True,     # needed for identical re-runs
    "num_threads": 4,           # same on both laptops, results depend on it
    "verbosity": -1,
}
DEFAULT_ROUNDS = {"base": 100, "light_extra": 20, "full": 100, "full120": 120}


def _params_file() -> dict:
    path = CONFIG_DIR / "params.json"
    if not path.exists():
        save_json("params.json", {"lgb": DEFAULT_LGB, "rounds": DEFAULT_ROUNDS, "S": None})
    return load_json("params.json")


def lgb_params(seed: int = 42) -> dict:
    # model seed is always 42 (label sample seeds are separate)
    p = {**DEFAULT_LGB, **_params_file().get("lgb", {})}
    p["seed"] = seed
    return p


def rounds() -> dict:
    return {**DEFAULT_ROUNDS, **_params_file().get("rounds", {})}


def predict_classes(booster, X: np.ndarray) -> np.ndarray:
    # fixed number of rounds, no early stopping
    return booster.predict(X).argmax(axis=1).astype(np.int32)


def predict_timed(booster, X: np.ndarray, num_threads: int | None = None):
    """
    Predicts and times it. Returns (predictions, seconds, flows per second).
    Uses the same num_threads as training.
    """
    threads = num_threads or lgb_params()["num_threads"]
    t0 = time.perf_counter()
    pred = booster.predict(X, num_threads=threads).argmax(axis=1).astype(np.int32)
    secs = time.perf_counter() - t0
    return pred, secs, (len(X) / secs if secs > 0 else float("nan"))


# ---------------- metrics ----------------
LABELS = list(range(N_CLASSES))


GB = 1e9   # decimal GB


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, bytes_total: np.ndarray,
             week_scale: float | None = None):
    """
    Returns (summary dict, per-class table, confusion matrix).
    macro_f1: over all 10 classes (missing class = 0)
    macro_f1_supported: only classes present that week
    byte_err: bytes of misclassified flows / all bytes
    misclassified_gb, test_gb: in the test sample
    week_scale = 10-class flows in the week / test sample size; used to
      estimate misclassified GB for the whole week
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.min() < 0 or y_true.max() >= N_CLASSES:
        raise ValueError("evaluate() needs 10-class labels only (0..9); filter y >= 0 first")
    bytes_total = np.asarray(bytes_total, dtype=np.float64)
    support = np.bincount(y_true, minlength=N_CLASSES)
    supported = [c for c in LABELS if support[c] > 0]
    wrong = y_true != y_pred
    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "macro_f1_supported": f1_score(y_true, y_pred, labels=supported, average="macro",
                                       zero_division=0) if supported else 0.0,
        "zero_support": ";".join(str(c) for c in LABELS if support[c] == 0),
        "byte_err": float(bytes_total[wrong].sum() / bytes_total.sum()) if bytes_total.sum() else 0.0,
        "misclassified_gb": float(bytes_total[wrong].sum() / GB),
        "test_gb": float(bytes_total.sum() / GB),
        "misclassified_gb_week_est": (float(bytes_total[wrong].sum() / GB * week_scale)
                                      if week_scale else float("nan")),
    }
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    per_class = pd.DataFrame({"class_id": LABELS, "precision": prec, "recall": rec,
                              "f1": f1, "support": sup})
    try:                                   # add service names if config exists
        per_class.insert(1, "service", class_names())
    except (FileNotFoundError, KeyError, ValueError):
        pass
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return summary, per_class, cm


# ---------------- time + memory ----------------
class Measure:
    """
    Times a block and tracks memory:
        with Measure() as m:
            booster = lgb.train(...)
    A thread reads memory every 20 ms and keeps the max.
    peak_mb: highest memory of the process
    delta_mb: peak minus memory before the block (what we report)
    Run gc.collect() before using it.
    """

    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self.peak_mb = 0.0
        self.base_mb = 0.0
        self.seconds = 0.0

    def _watch(self):
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._proc.memory_info().rss / 2**20)
            self._stop.wait(self.interval)

    @property
    def delta_mb(self) -> float:
        return max(0.0, self.peak_mb - self.base_mb)

    def __enter__(self):
        self.base_mb = self.peak_mb = self._proc.memory_info().rss / 2**20
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, self._proc.memory_info().rss / 2**20)
        return False


def model_kb(path: Path) -> float:
    return Path(path).stat().st_size / 1024


def machine_info() -> dict:
    # cpu, cores, ram of this laptop (for the cost table)
    return {
        "machine": platform.node(),
        "cpu": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": platform.platform(),
        "versions": lib_versions(),
    }


def lib_versions() -> str:
    # versions are pinned on both laptops, saved with every result
    import lightgbm, sklearn
    return f"lightgbm={lightgbm.__version__};numpy={np.__version__};sklearn={sklearn.__version__}"


def save_machine_info() -> Path:
    info = machine_info()
    path = RESULTS_DIR / f"machine_{info['machine']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2))
    sync_to_drive(path)
    return path


# ---------------- copying to Drive ----------------
def sync_to_drive(local: Path) -> None:
    """copies a finished file from OUT_ROOT to the same path under PROJECT_ROOT"""
    local = Path(local)
    if OUT_ROOT.resolve() == PROJECT_ROOT.resolve():
        return                                  # already writing to Drive
    try:
        target = PROJECT_ROOT / local.resolve().relative_to(OUT_ROOT.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, target)
    except (OSError, ValueError) as e:          # Drive offline, keep the local copy
        print(f"  (Drive copy skipped for {local.name}: {e})")


# ---------------- saving results ----------------
RESULT_COLUMNS = [
    "run_id", "chain", "tier", "strategy", "budget", "sampling", "seed", "label_week",
    "n_queried", "n_used", "train_rows", "n_trees", "test_week",
    "accuracy", "macro_f1", "macro_f1_supported", "zero_support", "byte_err",
    "misclassified_gb", "test_gb", "misclassified_gb_week_est",
    "update_time_s", "peak_mem_mb", "peak_mem_delta_mb", "model_kb",
    "predict_time_s", "flows_per_s", "machine", "versions", "timestamp",
]
_AUTO_COLUMNS = {"machine", "versions", "timestamp"}


# strategies:
#   KEEP       base model (100 rounds on S), no update
#   LIGHT      base + 20 rounds on the new labels (init_model)
#   LIGHT_R10/R50  same with 10 / 50 rounds
#   REFIT      same trees, leaf values refit on the new labels (Booster.refit)
#   FULL       new model, 100 rounds on S + new labels
#   FULL_W     FULL but new labels get half of the total weight
#   FULL120    FULL with 120 rounds
#   FULL_W120  FULL_W with 120 rounds, same size as LIGHT (added Nov 10, after results)
#   LIGHT_OLD  recency control, LIGHT with labels from week 17/35 (same n_used)
#   FULL_ALL   FULL with all 10-class flows of the pool week
# stratified control = LIGHT / FULL with sampling="strat" (oracle, needs the
# classes before labeling)
STRATEGIES = {"KEEP", "LIGHT", "LIGHT_R10", "LIGHT_R50", "REFIT", "FULL", "FULL_W",
              "FULL120", "FULL_W120", "LIGHT_OLD", "FULL_ALL"}
NEW_LABEL_WEIGHT_SHARE = 0.5   # FULL_W: old and new data get equal total weight
REFIT_DECAY = 0.9              # LightGBM default for refit
LABEL_WEEKS = {1: {"pool": 22, "older": 17}, 2: {"pool": 40, "older": 35}}


def make_run_id(chain: int, tier: str, strategy: str, budget: int = 0,
                sampling: str = "random", seed: int = 0) -> str:
    # e.g. c1_main_LIGHT_B10000_random_s42, KEEP is c1_main_KEEP_B0_none_s0
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy}; use one of {sorted(STRATEGIES)}")
    return f"c{chain}_{tier}_{strategy}_B{budget}_{sampling}_s{seed}"


def results_path(chain: int) -> Path:
    # one file per chain, so the two laptops don't write to the same file
    return RESULTS_DIR / f"results_c{chain}.csv"


def done_keys(chain: int) -> set[tuple[str, int]]:
    # (run_id, test_week) pairs already done, so a restarted run skips them
    path = results_path(chain)
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["run_id", "test_week"])
    return set(zip(df["run_id"], df["test_week"].astype(int)))


def append_result(row: dict, per_class: pd.DataFrame | None = None,
                  cm: np.ndarray | None = None) -> None:
    # saves one row (one model, one test week) + per-class and confusion
    # matrix files, then copies them to Drive
    row = {"machine": platform.node(), "versions": lib_versions(),
           "timestamp": datetime.now().isoformat(timespec="seconds"), **row}
    missing = [c for c in RESULT_COLUMNS if c not in row and c not in _AUTO_COLUMNS]
    if missing:
        raise KeyError(f"result row missing columns: {missing}")
    if row["strategy"] not in STRATEGIES:
        raise ValueError(f"unknown strategy {row['strategy']}")
    path = results_path(row["chain"])
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())   # write to disk now in case the laptop shuts down
    sync_to_drive(path)
    tag = f"{row['run_id']}_w{int(row['test_week']):02d}"
    if per_class is not None:
        p = RESULTS_DIR / "per_class" / f"{tag}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        per_class.to_csv(p, index=False)
        sync_to_drive(p)
    if cm is not None:
        p = RESULTS_DIR / "confusion" / f"{tag}.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, cm)
        sync_to_drive(p)


def log_event(message: str, who: str = "Lin") -> None:
    # adds a dated line to logs/run_log.md
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / "run_log.md"
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"- {datetime.now():%Y-%m-%d %H:%M} ({who}): {message}\n")
    sync_to_drive(path)
