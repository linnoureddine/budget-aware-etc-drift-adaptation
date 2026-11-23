"""
preprocess.py - converts each raw CESNET week into one parquet file (L2).

Usage:
    python preprocess.py            # all chain 1 weeks, then chain 2
    python preprocess.py 18 22      # only these weeks
    python preprocess.py 44 --allow-partial   # allow a week with < 7 day files

For each week:
  - reads the 7 day files in 1M-row chunks to save RAM
  - keeps APP, DAY, the 28 features (float32, booleans as 0/1) and
    BYTES_TOTAL = BYTES + BYTES_REV (int64)
  - train/test weeks: only our 10 classes. label weeks (17, 22, 35, 40):
    all services, since the budget samples from every service
  - keeps the original row order (the sample indices depend on it)
  - writes data/processed/week_XX.parquet and copies it to Drive
  - saves flow counts per service to results/support/
  - skips weeks that are already done or not fully downloaded
"""
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import common as C

CHAIN1 = [18, 22, 23, 24, 25, 26, 27, 28, 29, 30, 17]
CHAIN2 = [36, 40, 41, 42, 43, 44, 45, 46, 47, 48, 35]
KEEP_ALL_SERVICES = {17, 22, 35, 40}          # older-label and pool weeks
CHUNK = 1_000_000
DAYS_PER_WEEK = 7


def _to01(s: pd.Series) -> pd.Series:
    # True/False -> 1/0
    if s.dtype == bool:
        return s.astype("float32")
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "t"]).astype("float32")


def clean_chunk(chunk: pd.DataFrame, day: int) -> pd.DataFrame:
    out = pd.DataFrame(index=chunk.index)
    out["DAY"] = day
    out[C.LABEL_COL] = chunk[C.LABEL_COL].astype(str)
    total = 0
    for c in C.BYTES_COLS:
        total = total + pd.to_numeric(chunk[c], errors="coerce").fillna(0).astype("int64")
    out["BYTES_TOTAL"] = total
    for f in C.FEATURES:
        s = chunk[f]
        if pd.api.types.is_numeric_dtype(s) and s.dtype != bool:
            out[f] = pd.to_numeric(s, errors="coerce").astype("float32")
        else:
            out[f] = _to01(s)
    return out


def build_week(week: int, classes: list[str]) -> dict:
    keep_all = week in KEEP_ALL_SERVICES
    schema = pa.schema([("DAY", pa.int32()), (C.LABEL_COL, pa.string()),
                        ("BYTES_TOTAL", pa.int64())]
                       + [(f, pa.float32()) for f in C.FEATURES])
    cols = ["DAY", C.LABEL_COL, "BYTES_TOTAL", *C.FEATURES]
    usecols = list(dict.fromkeys([C.LABEL_COL, *C.BYTES_COLS, *C.FEATURES]))

    out = C.week_path(week)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    counts = pd.Series(dtype="int64")
    rows_in = rows_out = 0
    keep = set(classes)
    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
    try:
        for f in C.raw_week_files(week):
            t0 = time.perf_counter()
            for raw in pd.read_csv(f, usecols=usecols, chunksize=CHUNK, low_memory=False):
                chunk = clean_chunk(raw, C.day_from_name(f))
                rows_in += len(chunk)
                counts = counts.add(chunk[C.LABEL_COL].value_counts(), fill_value=0)
                if not keep_all:
                    chunk = chunk[chunk[C.LABEL_COL].isin(keep)]
                if len(chunk):
                    rows_out += len(chunk)
                    writer.write_table(pa.Table.from_pandas(chunk[cols], schema=schema,
                                                            preserve_index=False))
            print(f"    {f.name}: {time.perf_counter() - t0:.0f} s")
    finally:
        writer.close()
    tmp.replace(out)                       # rename only when complete
    C.sync_to_drive(out)

    counts_df = (counts.astype("int64").sort_values(ascending=False)
                 .rename_axis("app").reset_index(name="flows"))
    counts_df["in_our_10"] = counts_df["app"].isin(keep)
    cpath = C.RESULTS_DIR / "support" / f"week_{week:02d}_counts.csv"
    cpath.parent.mkdir(parents=True, exist_ok=True)
    counts_df.to_csv(cpath, index=False)
    C.sync_to_drive(cpath)

    ours = counts_df.loc[counts_df["in_our_10"], "flows"]
    return {"week": week, "rows_raw": rows_in, "rows_saved": rows_out,
            "our10_flows": int(ours.sum()),
            "our10_smallest_class": int(ours.min()) if len(ours) else 0,
            "our10_present": int(len(ours)),
            "mb": round(out.stat().st_size / 2**20, 1)}


def main():
    C._need_features()
    allow_partial = "--allow-partial" in sys.argv
    weeks = [int(a) for a in sys.argv[1:] if not a.startswith("--")] or CHAIN1 + CHAIN2
    classes = C.load_classes()
    C.build_label_encoder(classes)            # creates or checks the encoder
    print(f"classes: {classes}")

    log = C.LOGS_DIR / "preprocess_log.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    failed = []
    for week in weeks:
        if C.week_path(week).exists():
            print(f"week {week:02d}: already done, skipped")
            continue
        try:
            files = C.raw_week_files(week)
        except FileNotFoundError as e:
            print(f"week {week:02d}: SKIPPED ({e}) - download it first")
            continue
        folder = files[0].parent
        parts = list(folder.glob("*.part"))
        if (len(files) != DAYS_PER_WEEK or parts) and not allow_partial:
            print(f"week {week:02d}: SKIPPED - {len(files)} of {DAYS_PER_WEEK} day files"
                  f"{', unfinished: ' + ', '.join(p.name for p in parts) if parts else ''}"
                  " - finish the download (or use --allow-partial if the dataset really lacks a day)")
            continue
        print(f"week {week:02d}: {'all services' if week in KEEP_ALL_SERVICES else 'our 10 classes'}")
        t0 = time.perf_counter()
        try:
            stats = build_week(week, classes)
        except Exception as e:
            # don't stop the other weeks if one fails
            msg = f"week {week:02d}: FAILED ({type(e).__name__}: {e}) - fix and re-run this week"
            print("  " + msg)
            C.log_event(msg)
            failed.append(week)
            continue
        stats["minutes"] = round((time.perf_counter() - t0) / 60, 1)
        print(f"  -> {stats}")
        pd.DataFrame([stats]).to_csv(log, mode="a", header=not log.exists(), index=False)
        C.sync_to_drive(log)
        C.log_event(f"preprocessed week {week:02d}: {stats}")
    print("\nALL DONE." if not failed else f"\nDONE, but these weeks FAILED: {failed}")


if __name__ == "__main__":
    main()
