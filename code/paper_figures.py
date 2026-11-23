"""
paper_figures.py - all results figures and tables for the paper, in one style
(updated Nov 19). Only reads the result files, so it runs in seconds.

Usage:
    python paper_figures.py --dirs D:\\cesnet_work\\results "G:\\...\\CESNET_Project\\results"
    python paper_figures.py --dirs ... --wide      # 7.16 in wide (both columns)

Each chain's files are taken from the first --dirs folder that has them:
    results_c{1,2}.csv  bootstrap_bytes_c{1,2}.csv  timing_c{1,2}.csv  predict_c{1,2}.csv
    per_class/c{1,2}_main_KEEP_B0_none_s0_wWW.csv   (only for fig_class_drift)
Missing files -> that chain or figure is skipped.

Outputs go to <first dir>/../figures/ (PDF + PNG). One column wide (3.5 in),
chain 1 and chain 2 stacked as panels (a) and (b).
    fig_drift              macro-F1 per test week: KEEP, LIGHT and FULL_W120 at 250k
    fig_class_drift        per-class F1 change, first to last test week (KEEP)
    fig_budget_f1          macro-F1 vs budget per strategy (95% CI over seeds)
    fig_light_vs_fullw120  LIGHT - FULL_W120 per budget (95% CI over seeds)
    fig_traffic_saved      GB/week saved vs KEEP per budget (95% bootstrap CI)
    fig_cost_benefit       macro-F1 change vs KEEP against update time (log)
Tables: table_cost (B = 250k) and table_rounds (LIGHT with 10/20/50 rounds).
Each figure also gets a *_data.csv with the plotted numbers.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

# ---------------- style ----------------
# IEEE-style figures: serif font, boxed axes, ticks inside, dashed grid.
# each strategy has a fixed color, marker and line style (readable in grayscale)
STYLE = {
    "KEEP":      dict(color="black",   marker="",  ls="--", label="KEEP"),
    "LIGHT":     dict(color="#1f77b4", marker="o", ls="-",  label="LIGHT"),
    "REFIT":     dict(color="#ff7f0e", marker="s", ls="--", label="REFIT"),
    "FULL":      dict(color="#d62728", marker="v", ls=":",  label="FULL"),
    "FULL_W120": dict(color="#2ca02c", marker="^", ls="-.", label="FULL_W120"),
    "FULL_ALL":  dict(color="0.45",    marker="",  ls=":",  label="FULL_ALL"),
}
CHAIN_STYLE = {1: dict(color="black", marker="o", mfc="black", label="Chain 1"),
               2: dict(color="0.4", marker="s", mfc="white", label="Chain 2")}
MUTED, GRID = "0.35", "0.82"
COL_W = 3.5                     # one IEEE column in inches (--wide uses 7.16)
BUDGETS = [2_000, 10_000, 50_000, 250_000]
BUDGET_LABEL = {2_000: "2k", 10_000: "10k", 50_000: "50k", 250_000: "250k"}
T95 = 2.776                     # t value for a 95% CI with 5 seeds (df = 4)
WIDE = False

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "0.6",
    "legend.fancybox": False, "legend.borderpad": 0.3, "legend.handlelength": 2.2,
    "lines.linewidth": 1.3, "lines.markersize": 4.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 600,
})


# ---------------- loading ----------------
def find(name, dirs):
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None


def load_chain(chain, dirs):
    files = {k: find(f"{k}_c{chain}.csv", dirs)
             for k in ("results", "bootstrap_bytes", "timing", "predict")}
    missing = [k for k, v in files.items() if v is None]
    if missing:
        print(f"chain {chain}: skipped, missing {missing}")
        return None
    d = {k: pd.read_csv(v) for k, v in files.items()}
    d["per_class_dirs"] = [Path(x) / "per_class" for x in dirs if (Path(x) / "per_class").exists()]
    return d


def seed_level(res, strategy, sampling="random"):
    """macro-F1 (%) per (budget, seed), averaged over the test weeks"""
    r = res[(res.strategy == strategy) & (res.sampling == sampling)]
    return r.groupby(["budget", "seed"]).macro_f1.mean() * 100


def keep_f1(res):
    return res[res.strategy == "KEEP"].macro_f1.mean() * 100


def ci_stats(v):
    v = np.asarray(v, dtype=float)
    m = v.mean()
    h = T95 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    return m, h


# ---------------- layout helpers ----------------
def new_fig(n, height_per=2.05, sharex=True, sharey=True):
    """n panels stacked vertically"""
    w = 7.16 if WIDE else COL_W
    fig, axes = plt.subplots(n, 1, figsize=(w, height_per * n + 0.45), sharex=sharex,
                             sharey=sharey, squeeze=False)
    return fig, axes[:, 0]


def style_axis(ax, zero=False):
    ax.grid(True, color=GRID, lw=0.4, ls="--")
    ax.set_axisbelow(True)
    if zero:
        ax.axhline(0, color="black", lw=0.6)


def panel_label(ax, i, ch):
    ax.text(0.02, 0.96, f"({'ab'[i]}) Chain {ch}", transform=ax.transAxes, ha="left",
            va="top", fontsize=8, bbox=dict(fc="white", ec="none", pad=1.0))


def top_legend(fig, axes, ncol=None):
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=ncol or len(l), bbox_to_anchor=(0.55, 1.0),
               columnspacing=1.0, handletextpad=0.5)


def finish(fig, axes, xlabel, ylabel, ncol=None, top=0.94):
    axes[-1].set_xlabel(xlabel)
    for ax in axes:
        ax.set_ylabel(ylabel)
    top_legend(fig, axes, ncol)
    fig.tight_layout(h_pad=0.4, rect=(0, 0, 1, top))


def save(fig, out, stem, rows=None):
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    if rows is not None:
        pd.DataFrame(rows).to_csv(out / f"{stem}_data.csv", index=False)
    print(f"saved {stem}.pdf / .png")


def budget_axis(ax):
    ax.set_xticks(range(len(BUDGETS)), [BUDGET_LABEL[B] for B in BUDGETS])
    ax.set_xlim(-0.4, len(BUDGETS) - 0.6)


# ---------------- fig_drift ----------------
def fig_drift(data, out):
    chains = sorted(data)
    fig, axes = new_fig(len(chains), sharex=False, sharey=False)
    rows = []
    for i, (ax, ch) in enumerate(zip(axes, chains)):
        res = data[ch]["results"]
        series = [("KEEP", res[res.strategy == "KEEP"]),
                  ("LIGHT", res[(res.strategy == "LIGHT") & (res.sampling == "random")
                                & (res.budget == 250_000)]),
                  ("FULL_W120", res[(res.strategy == "FULL_W120") & (res.budget == 250_000)])]
        for st, r in series:
            if r.empty:
                continue
            w = r.groupby("test_week").macro_f1.mean() * 100
            sty = STYLE[st]
            lab = st if st == "KEEP" else f"{st} (250k)"
            ax.plot(range(1, len(w) + 1), w.values, color=sty["color"], ls=sty["ls"],
                    marker=sty["marker"] or "o", mfc=sty["color"], mec=sty["color"], label=lab)
            rows += [{"chain": ch, "series": lab, "test_week": wk, "macro_f1": v}
                     for wk, v in w.items()]
        weeks = sorted(res.test_week.unique())
        ax.set_xticks(range(1, len(weeks) + 1), [str(int(x)) for x in weeks])
        ax.set_xlabel(f"Test week (chain {ch})")
        style_axis(ax)
        ax.margins(y=0.15)
        panel_label(ax, i, ch)
    for ax in axes:
        ax.set_ylabel("Macro-F1 (%)")
    top_legend(fig, axes)
    fig.tight_layout(h_pad=0.6, rect=(0, 0, 1, 0.94))
    save(fig, out, "fig_drift", rows)


# ---------------- fig_class_drift ----------------
def fig_class_drift(data, out):
    rows = []
    for ch in sorted(data):
        files = {}
        for d in data[ch]["per_class_dirs"]:
            for p in d.glob(f"c{ch}_main_KEEP_B0_none_s0_w*.csv"):
                files.setdefault(int(re.search(r"_w(\d+)\.csv$", p.name).group(1)), p)
        if len(files) < 2:
            print(f"fig_class_drift: chain {ch} skipped (KEEP per-class files not found)")
            continue
        first, last = min(files), max(files)
        a = pd.read_csv(files[first]).set_index("service").f1 * 100
        b = pd.read_csv(files[last]).set_index("service").f1 * 100
        for svc in a.index:
            rows.append({"chain": ch, "service": svc, "f1_first": a[svc], "f1_last": b[svc],
                         "change_pts": b[svc] - a[svc], "weeks": f"{first}->{last}"})
    if not rows:
        return
    df = pd.DataFrame(rows)
    order = df.groupby("service").change_pts.mean().sort_values().index.tolist()
    chains = sorted(df.chain.unique())
    w = 7.16 if WIDE else COL_W
    fig, ax = plt.subplots(figsize=(w, 0.22 * len(order) + 0.9))
    y = np.arange(len(order))
    hgt = 0.38 if len(chains) == 2 else 0.6
    for k, ch in enumerate(chains):
        d = df[df.chain == ch].set_index("service").reindex(order)
        off = (k - (len(chains) - 1) / 2) * hgt
        ax.barh(y + off, d.change_pts, height=hgt, color="0.25" if ch == 1 else "0.75",
                edgecolor="black", lw=0.4, label=CHAIN_STYLE[ch]["label"])
    ax.set_yticks(y, order)
    ax.axvline(0, color="black", lw=0.6)
    ax.grid(True, axis="x", color=GRID, lw=0.4, ls="--")
    ax.set_axisbelow(True)
    ax.set_xlabel("F1 change, first to last test week (pts), KEEP")
    ax.legend(loc="best")
    fig.tight_layout()
    save(fig, out, "fig_class_drift", rows)


# ---------------- fig_budget_f1 ----------------
def fig_budget_f1(data, out):
    chains = sorted(data)
    fig, axes = new_fig(len(chains), sharey=False)
    rows = []
    offsets = {"LIGHT": -0.15, "REFIT": -0.05, "FULL": 0.05, "FULL_W120": 0.15}
    for i, (ax, ch) in enumerate(zip(axes, chains)):
        res = data[ch]["results"]
        k = keep_f1(res)
        ax.axhline(k, color=STYLE["KEEP"]["color"], ls="--", lw=1.0, label="KEEP")
        fa = res[res.strategy == "FULL_ALL"].macro_f1.mean() * 100
        ax.axhline(fa, color=STYLE["FULL_ALL"]["color"], ls=":", lw=1.0, label="FULL_ALL")
        rows += [{"chain": ch, "strategy": "KEEP", "budget": 0, "macro_f1": k},
                 {"chain": ch, "strategy": "FULL_ALL", "budget": 0, "macro_f1": fa}]
        for st in ("LIGHT", "REFIT", "FULL", "FULL_W120"):
            s = seed_level(res, st)
            if s.empty:
                continue
            ms, hs = [], []
            for B in BUDGETS:
                if B in s.index.get_level_values(0):
                    m, h = ci_stats(s.loc[B].values)
                else:
                    m, h = np.nan, np.nan
                ms.append(m); hs.append(h)
                rows.append({"chain": ch, "strategy": st, "budget": B, "macro_f1": m, "ci95": h})
            sty = STYLE[st]
            ax.errorbar(np.arange(len(BUDGETS)) + offsets[st], ms, yerr=hs, color=sty["color"],
                        marker=sty["marker"], ls=sty["ls"], capsize=2.2, elinewidth=0.8,
                        capthick=0.8, label=sty["label"])
        budget_axis(ax)
        style_axis(ax)
        ax.margins(y=0.1)
        panel_label(ax, i, ch)
    finish(fig, axes, "Label budget (flows queried)", "Macro-F1 (%)", ncol=3, top=0.90)
    save(fig, out, "fig_budget_f1", rows)


# ---------------- fig_light_vs_fullw120 ----------------
def fig_light_vs_fullw120(data, out):
    w = 7.16 if WIDE else COL_W
    fig, ax = plt.subplots(figsize=(w, 2.3))
    rows = []
    for k, ch in enumerate(sorted(data)):
        res = data[ch]["results"]
        a, b = seed_level(res, "LIGHT"), seed_level(res, "FULL_W120")
        if b.empty:
            print(f"fig_light_vs_fullw120: chain {ch} has no FULL_W120 rows, skipped")
            continue
        diff = (a - b).dropna()
        ms, hs = [], []
        for B in BUDGETS:
            m, h = ci_stats(diff.loc[B].values)
            ms.append(m); hs.append(h)
            rows.append({"chain": ch, "budget": B, "light_minus_fullw120": m, "ci95": h})
        sty = CHAIN_STYLE[ch]
        ax.errorbar(np.arange(len(BUDGETS)) + (k - 0.5) * 0.14, ms, yerr=hs, color=sty["color"],
                    marker=sty["marker"], mfc=sty["mfc"], mec=sty["color"], ls="-", capsize=2.2,
                    elinewidth=0.8, capthick=0.8, label=sty["label"])
    budget_axis(ax)
    style_axis(ax, zero=True)
    ax.set_xlabel("Label budget (flows queried)")
    ax.set_ylabel("LIGHT $-$ FULL_W120 (F1 pts)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, out, "fig_light_vs_fullw120", rows)


# ---------------- fig_traffic_saved ----------------
def fig_traffic_saved(data, out):
    chains = sorted(data)
    fig, axes = new_fig(len(chains))
    rows = []
    offsets = {"LIGHT": -0.12, "REFIT": 0.0, "FULL_W120": 0.12}
    for i, (ax, ch) in enumerate(zip(axes, chains)):
        b = data[ch]["bootstrap_bytes"]
        s = b[(b.quantity == "gb_week_saved") & (b.comparison == "vs_KEEP")]
        x = np.arange(len(BUDGETS))
        for st in ("LIGHT", "REFIT", "FULL_W120"):
            d = s[s.strategy == st].set_index("budget").reindex(BUDGETS)
            if d.point.isna().all():
                continue
            sty = STYLE[st]
            yerr = np.vstack([d.point - d.ci_low, d.ci_high - d.point])
            ax.errorbar(x + offsets[st], d.point, yerr=yerr, color=sty["color"],
                        marker=sty["marker"], ls=sty["ls"], capsize=2.2, elinewidth=0.8,
                        capthick=0.8, label=sty["label"])
            rows += [{"chain": ch, "strategy": st, "budget": B, "gb_week_saved": r.point,
                      "ci_low": r.ci_low, "ci_high": r.ci_high} for B, r in d.iterrows()]
        fa = s[s.strategy == "FULL_ALL"]
        if len(fa):
            v = float(fa.point.iloc[0])
            ax.axhline(v, color=STYLE["FULL_ALL"]["color"], lw=1.0, ls=":")
            ax.text(len(BUDGETS) - 0.65, v, "all labels (FULL_ALL)", ha="right", va="bottom",
                    fontsize=7, color=MUTED)
            rows.append({"chain": ch, "strategy": "FULL_ALL", "budget": 0, "gb_week_saved": v,
                         "ci_low": fa.ci_low.iloc[0], "ci_high": fa.ci_high.iloc[0]})
        budget_axis(ax)
        style_axis(ax, zero=True)
        ax.margins(y=0.08)
        panel_label(ax, i, ch)
    finish(fig, axes, "Label budget (flows queried)", "Saved vs. KEEP (GB/week)")
    save(fig, out, "fig_traffic_saved", rows)


# ---------------- fig_cost_benefit ----------------
def label_budgets(ax, st, Bs, xs, ys):
    """budget labels next to the points (close points share one label)"""
    if st not in ("LIGHT", "FULL_W120"):
        return
    groups = []
    for i in np.argsort(ys):
        if groups and abs(ys[i] - ys[groups[-1][-1]]) < 0.35:
            groups[-1].append(i)
        else:
            groups.append([i])
    for g in groups:
        txt = ", ".join(BUDGET_LABEL[Bs[i]] for i in sorted(g, key=lambda i: Bs[i]))
        x, y = max(xs[i] for i in g), float(np.mean([ys[i] for i in g]))
        off, ha = ((-4, 7), "right") if (st == "LIGHT" and txt == "50k") else ((4, 0), "left")
        ax.annotate(txt, (x, y), textcoords="offset points", xytext=off, ha=ha,
                    va="center", fontsize=7, color=MUTED)


def fig_cost_benefit(data, out):
    chains = sorted(data)
    fig, axes = new_fig(len(chains))
    rows = []
    for i, (ax, ch) in enumerate(zip(axes, chains)):
        res, t = data[ch]["results"], data[ch]["timing"]
        k = keep_f1(res)
        for st in ("LIGHT", "REFIT", "FULL_W120"):
            s = seed_level(res, st)
            tt = t[t.strategy == st].set_index("budget").median_s
            Bs = [B for B in BUDGETS if B in tt.index and B in s.index.get_level_values(0)]
            if not Bs:
                continue
            xs = tt.loc[Bs].values
            ys = np.array([s.loc[B].mean() - k for B in Bs])
            sty = STYLE[st]
            # FULL_W120 takes ~40 s at every budget, so no line, just markers
            ax.plot(xs, ys, color=sty["color"], marker=sty["marker"],
                    ls="none" if st == "FULL_W120" else sty["ls"], label=sty["label"])
            rows += [{"chain": ch, "strategy": st, "budget": B, "update_time_s": xv,
                      "f1_change_pts": yv} for B, xv, yv in zip(Bs, xs, ys)]
            label_budgets(ax, st, Bs, xs, ys)
        style_axis(ax, zero=True)
        ax.margins(y=0.12)
        ax.set_xscale("log")
        ax.set_xlim(0.04, 300)
        ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
        panel_label(ax, i, ch)
    finish(fig, axes, "Update time (s, log scale)", "$\\Delta$ Macro-F1 vs. KEEP (pts)")
    save(fig, out, "fig_cost_benefit", rows)


# ---------------- tables ----------------
def table_cost(data, out):
    rows = []
    for ch in sorted(data):
        t, p = data[ch]["timing"], data[ch]["predict"]
        pr = p.set_index("strategy")
        base = t[t.strategy == "BASE_TRAIN"]
        for st in ["KEEP", "LIGHT", "REFIT", "FULL_W120"]:
            row = {"chain": ch, "strategy": st}
            if st in pr.index:
                row.update(trees=int(pr.loc[st, "n_trees"]), model_kb=float(pr.loc[st, "model_kb"]),
                           flows_per_s=int(pr.loc[st, "flows_per_s"]))
            if st == "KEEP" and len(base):
                row.update(update_s_250k=float(base.median_s.iloc[0]), note="initial training")
            elif st != "KEEP":
                tt = t[t.strategy == st].set_index("budget")
                if 250_000 in tt.index:
                    row.update(update_s_2k=float(tt.median_s.get(2_000, np.nan)),
                               update_s_250k=float(tt.loc[250_000, "median_s"]),
                               mem_mb_250k=float(tt.loc[250_000, "median_mem_delta_mb"]))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "table_cost.csv", index=False)
    fmt = lambda v, f: "--" if v is None or pd.isna(v) else format(v, f)
    lines = [r"\begin{tabular}{llrrrrr}", r"\hline",
             r"Chain & Strategy & Trees & Update 2k (s) & Update 250k (s) & Memory (MB) & Flows/s \\",
             r"\hline"]
    for _, r in df.iterrows():
        lines.append(f"{r.chain} & {r.strategy.replace('_', chr(92) + '_')} & "
                     f"{fmt(r.get('trees'), '.0f')} & {fmt(r.get('update_s_2k'), '.1f')} & "
                     f"{fmt(r.get('update_s_250k'), '.1f')} & {fmt(r.get('mem_mb_250k'), '.0f')} & "
                     f"{fmt(r.get('flows_per_s'), ',.0f')} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (out / "table_cost.tex").write_text("\n".join(lines))
    print("\nTable: cost\n" + df.round(1).to_string(index=False))


def table_rounds(data, out):
    rows = []
    for ch in sorted(data):
        res = data[ch]["results"]
        k = keep_f1(res)
        for st, label in [("LIGHT_R10", "LIGHT, 10 rounds"), ("LIGHT", "LIGHT, 20 rounds"),
                          ("LIGHT_R50", "LIGHT, 50 rounds"), ("FULL_W120", "FULL_W120")]:
            s = seed_level(res, st)
            for B in (2_000, 50_000):
                if B in s.index.get_level_values(0):
                    m, h = ci_stats(s.loc[B].values)
                    rows.append({"chain": ch, "strategy": label, "budget": B, "macro_f1": m,
                                 "change_vs_keep_pts": m - k, "ci95": h})
    df = pd.DataFrame(rows)
    df.to_csv(out / "table_rounds.csv", index=False)
    piv = df.pivot_table(index=["chain", "strategy"], columns="budget",
                         values="change_vs_keep_pts", sort=False)
    lines = [r"\begin{tabular}{llrr}", r"\hline",
             r"Chain & Update & $\Delta$F1 at 2k & $\Delta$F1 at 50k \\", r"\hline"]
    for (ch, st), r in piv.iterrows():
        lines.append(f"{ch} & {st.replace('_', chr(92) + '_')} & {r.get(2000, np.nan):+.1f} & "
                     f"{r.get(50000, np.nan):+.1f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (out / "table_rounds.tex").write_text("\n".join(lines))
    print("\nTable: rounds (macro-F1 change vs KEEP, points)\n" + piv.round(1).to_string())


# ---------------- main ----------------
def main():
    global WIDE
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=[str(C.RESULTS_DIR)])
    ap.add_argument("--wide", action="store_true",
                    help="figures span both columns (7.16 in) instead of one (3.5 in)")
    a = ap.parse_args()
    WIDE = a.wide
    data = {ch: d for ch in (1, 2) if (d := load_chain(ch, a.dirs)) is not None}
    if not data:
        raise SystemExit("no chain has all four result files; check --dirs")
    out = Path(a.dirs[0]).parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_drift, fig_class_drift, fig_budget_f1, fig_light_vs_fullw120,
              fig_traffic_saved, fig_cost_benefit, table_cost, table_rounds):
        try:
            f(data, out)
        except Exception as e:                      # keep going if one figure fails
            print(f"{f.__name__}: FAILED ({type(e).__name__}: {e})")
    print(f"\nAll outputs in {out}")


if __name__ == "__main__":
    main()
