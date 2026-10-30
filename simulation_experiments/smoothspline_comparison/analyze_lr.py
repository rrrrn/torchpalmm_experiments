"""
Learning-rate diagnostic for the "mixed" (sn-scaling) experiment: estimation
accuracy and wall time of SmoothingSplineMixed's LBFGS fit at each swept lr
(run_python.py's LBFGS_LR_SWEEP = [1e-3, 1e-1, 1.0]), faceted over the full
(N, D_type, sn) design grid -- 16 cells, the largest lr-swept grid of any
experiment in this repo.

Reads the *_sweep.csv files under results/ (every lr's fit for every
replicate) rather than the main mixed_*.csv, which keeps only the
min-REML-selected lr per replicate. This is a separate script rather than
folded into analyze_results.py's own _plot_lr_diagnostic (which pools sn
together into one box per (N, D_type) and has no time panel or table) so the
much larger analyze_results.py pipeline stays untouched; shared formatting
helpers are imported from it instead of duplicated.

One summary plot (plot_lr_summary): rows = D_type, columns = the four
accuracy metrics plus wall time as the last column. Each panel is a grouped
boxplot vs sn (m), colored by lr. Restricted to N=_N_FOCUS (see analyze_results.py's own
_save_param_table, which makes the same restriction).

Run from: experiments/smoothspline_comparison/
    python analyze_lr.py
"""
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType in PDFs, as journals require
matplotlib.rcParams.update({"font.family": "Arial", "mathtext.fontset": "custom",
                            "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                            "mathtext.bf": "Arial:bold"})

from analyze_results import DATA_DIR, _SN_KEEP

# N=500 is excluded throughout: this diagnostic focuses on the N=1000
# production point, mirroring analyze_results._save_param_table's own
# restriction to N=1000.
_N_FOCUS = 1000

# sn (m) values drawn in the summary figure (the table keeps all of _SN_KEEP).
_SN_PLOT = (5, 10, 15)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)

# Output folder shared with analyze_results.py and plot_curve_library.py
_DIAG_DIR = "plots"

# mode "paired_diff": boxplot of |value at lr - value at a reference lr|
# (same dataset refit at each lr, so the pairing is exact) rather than the
# raw values -- the natural view for "does this lr cost you accuracy vs the
# reference", log-scaled like the raw panels since the gap itself can span
# several decades. Wall time stays "raw": a grouped boxplot of the values.
_ACCURACY_METRICS = [
    ("beta_mae",       r"MAE($\hat{\beta}$)",        "paired_diff"),
    ("sigma2_abs_err", r"error($\hat{\sigma}^2$)",   "paired_diff"),
    ("D_mae",          r"MAE($\hat{D}$)",            "paired_diff"),
    ("ise_compare",    "ISE",                         "paired_diff"),
]
_TIME_METRIC = [("time_s", "Run time (s)", "raw")]


def _log_exponent_yaxis(ax):
    """Bare-exponent tick labels on a log y-axis (ported from
    scaling_py/analyze_results.py's _log_exponent_yaxis) instead of
    matplotlib's default 10^n formatting, which crowds at these figure
    sizes."""
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _pos: f"{round(np.log10(y))}" if y > 0 else ""))


def _load_sweep():
    """Every lr's fit for every (pz, sn, D_type, repeat_id) replicate at
    N=_N_FOCUS, restricted to the sn values this experiment actively tracks
    (analyze_results._SN_KEEP)."""
    files = glob.glob(f"{DATA_DIR}/mixed_*_sweep.csv")
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, on_bad_lines="skip") for f in files], ignore_index=True)
    if df.empty or "lr" not in df.columns:
        return None
    for col in ("N", "sn", "lr", "beta_mae", "sigma2_abs_err", "D_mae", "time_s"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["sn"].isin(_SN_KEEP) & (df["N"] == _N_FOCUS)]
    if df.empty:
        return None
    ise_comp_cols = sorted(c for c in df.columns if c.startswith("ise_") and c[4:].isdigit())
    df["ise_compare"] = df[ise_comp_cols].median(axis=1) if ise_comp_cols else np.nan
    return df


# Reference (smallest) lr in grey; the compared lrs in Okabe-Ito orange and blue,
# which stay distinct under color-vision deficiency and in grayscale.
_LR_PALETTE = ["#b3b3b3", "#e69f00", "#0072b2"]
_EDGE = "#3d3d3d"


def _lr_colors(lrs):
    if len(lrs) <= len(_LR_PALETTE):
        return {lr: _LR_PALETTE[i] for i, lr in enumerate(lrs)}
    return {lr: plt.cm.tab10(i) for i, lr in enumerate(lrs)}


def _median_color(fill):
    """White median line on dark fills, black on light ones."""
    r, g, b = matplotlib.colors.to_rgb(fill)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.4 else "black"


def _draw_boxes(ax, x_vals, series, colors):
    """Grouped boxes: series maps lr -> one array per x value. Outliers are kept:
    in the |difference| panels they are the replicates where lr mattered."""
    lrs = list(series)
    bar_w = 0.8 / len(lrs)
    for i, lr in enumerate(lrs):
        offset = (i - (len(lrs) - 1) / 2) * bar_w
        bp = ax.boxplot(series[lr], positions=[j + 1 + offset for j in range(len(x_vals))],
                        widths=bar_w * 0.8, patch_artist=True, manage_ticks=False,
                        medianprops=dict(color=_median_color(colors[lr]), linewidth=1.0),
                        boxprops=dict(linewidth=0.6, edgecolor=_EDGE),
                        whiskerprops=dict(linewidth=0.6, color=_EDGE),
                        capprops=dict(linewidth=0.6, color=_EDGE),
                        flierprops=dict(marker="o", markersize=1.6, markerfacecolor=colors[lr],
                                        markeredgecolor="none", alpha=0.8))
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[lr])
    ax.set_xticks(range(1, len(x_vals) + 1))
    ax.set_xticklabels([f"{v:g}" for v in x_vals])


def _paired_diffs(fdf, x_col, x_vals, metric, other_lrs, ref_lr):
    """|value at lr - value at ref_lr| per x value, paired on the same dataset
    (repeat_id), so the sign is dropped and the gap can be log-scaled. None if
    ref_lr has no data."""
    wide = fdf.pivot_table(index=["repeat_id", x_col], columns="lr", values=metric)
    if ref_lr not in wide.columns:
        return None
    series = {}
    for lr in other_lrs:
        if lr in wide.columns:
            diff = (wide[lr] - wide[ref_lr]).abs().rename("diff").reset_index()
            series[lr] = [diff.loc[diff[x_col] == xv, "diff"].dropna().values for xv in x_vals]
    return series or None


def _plot_lr_grid(conv_df, metric_specs, out_path):
    """One row per D_type, one column per metric (n is fixed at _N_FOCUS), sized
    for print (about 7 in wide). "paired_diff" columns show |value at lr - value
    at the smallest lr| on the same dataset; "raw" columns show the values. Both
    are log10-scaled with bare-exponent ticks and colored by lr."""
    lrs = sorted(conv_df["lr"].unique())
    colors = _lr_colors(lrs)
    ref_lr, other_lrs = lrs[0], lrs[1:]
    D_types = sorted(conv_df["D_type"].dropna().unique())
    sn_vals = sorted(conv_df["sn"].unique())

    nrows, ncols = len(D_types), len(metric_specs)
    fig_h = 2.8
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.16, fig_h), squeeze=False)
    for r_idx, D_type in enumerate(D_types):
        fdf = conv_df[conv_df["D_type"] == D_type]
        for c_idx, (metric_col, metric_title, mode) in enumerate(metric_specs):
            ax = axes[r_idx][c_idx]
            if fdf.empty or metric_col not in fdf.columns or fdf[metric_col].isna().all():
                ax.set_visible(False)
                continue
            if mode == "paired_diff":
                series = _paired_diffs(fdf, "sn", sn_vals, metric_col, other_lrs, ref_lr)
                if series is None:
                    ax.set_visible(False)
                    continue
            else:
                series = {lr: [fdf.loc[(fdf["lr"] == lr) & (fdf["sn"] == v), metric_col].dropna().values
                               for v in sn_vals] for lr in lrs}
            _draw_boxes(ax, sn_vals, series, colors)

            ax.set_yscale("log")
            _log_exponent_yaxis(ax)
            ax.yaxis.set_minor_locator(mticker.NullLocator())
            ax.tick_params(labelsize=8, length=2.5, width=0.6, pad=2)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_linewidth(0.6)
            if r_idx == nrows - 1:
                ax.set_xlabel("$m$", fontsize=9, labelpad=1)
            if c_idx == 0:
                ax.set_ylabel(rf"{D_type.capitalize()} $D$", fontsize=9)
            if r_idx == 0:
                ax.set_title((r"$|\Delta|$ " if mode == "paired_diff" else "") + metric_title
                             , fontsize=8.5, pad=4)

    def _lr_label(lr):
        e = int(round(np.log10(lr)))
        return rf"$\eta = 10^{{{e}}}$" if e else r"$\eta = 1$"
    handles = [Patch(facecolor=colors[lr], edgecolor=_EDGE, linewidth=0.6, label=_lr_label(lr))
               for lr in lrs]
    fig.legend(handles=handles, loc="upper center", ncol=len(lrs), fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=2.0)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.27 / fig_h], w_pad=0.6, h_pad=0.8)
    os.makedirs(_DIAG_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_lr_summary(label="mixed"):
    """Estimation accuracy (beta MAE, |sigma2-hat|, D MAE, ISE) and wall time
    (last column) vs lr at n=_N_FOCUS subjects, converged runs only. Rows are D_type."""
    df = _load_sweep()
    if df is None:
        print(f"[{label}] no sweep CSVs found -- skipping lr summary plot")
        return
    conv_df = df[(df["converged"] == True) & df["sn"].isin(_SN_PLOT)]
    if conv_df.empty:
        return
    _plot_lr_grid(conv_df, _ACCURACY_METRICS + _TIME_METRIC,
                  os.path.join(_DIAG_DIR, f"{label}_lr_summary.png"))


def main():
    plot_lr_summary()


if __name__ == "__main__":
    main()
