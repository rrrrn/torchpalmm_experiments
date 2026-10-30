"""
Task-level relative-error boxplots for the scaling_py pilots (pz_scaling_ss, ss_scaling):
rows = pilot, columns = metric, one box per D_type at each swept value.

Run from: experiments/scaling_py/
    python analyze_results.py
"""
import importlib.util
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams.update({"font.family": "Arial", "mathtext.fontset": "custom",
                            "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                            "mathtext.bf": "Arial:bold"})
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)

PLOT_DIR = "plots"

# (subfolder, swept column, axis symbol, display label); rows top to bottom
TASK_PILOTS = [
    ("pz_scaling_ss", "pz", r"$q$", "Random-effect scaling"),
    ("ss_scaling",    "sn", r"$m$", "Smooth-term scaling"),
]

# (column, title, log-y, zero_ref)
TASK_RELERR_METRICS = [
    ("time_s",       "Runtime (s)",                     True, False),
    ("beta_mape",    r"$\beta$ MAPE",                   True, True),
    ("D_relerr",     "Relative $D$ error\n(Frobenius)", True, True),
    ("ise_relerr",   "Relative ISE\nof $f$",            True, True),
]
FORCE_HLINE = ("D_relerr", "ise_relerr")

TIME_CAP_SECONDS = 1800
LEGEND_TITLE = r"Structure of $D$"
BOX_WHISKERS = (5, 95)   # percentiles; points beyond them are not drawn
D_TYPE_COLORS = {"diagonal": "#00558A", "unstructured": "#C77700"}


def _load_pilot(name):
    """Import <name>/analyze_results.py and return its _load()."""
    path = os.path.join(_SCRIPT_DIR, name, "analyze_results.py")
    spec = importlib.util.spec_from_file_location(f"_scaling_py_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    df = mod._load()
    os.chdir(_SCRIPT_DIR)
    return df


def _log_exponent_yaxis(ax):
    """Bare-exponent tick labels (n for 10^n) on a log y-axis."""
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=30))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _pos: f"{round(np.log10(y))}" if y > 0 else ""))


def _zero_ref_line(ax):
    """Dashed y=0 line, only if 0 falls within the panel's ylim."""
    ylo, yhi = ax.get_ylim()
    if ylo <= 0 <= yhi:
        ax.axhline(0, color="k", lw=0.8, ls="--")


def _panel_box(ax, df, value_col, swept_col, logy, axhline_y=None, zero_ref=False):
    """Per-rep boxes (one per D_type) at each swept value that has a converged fit."""
    if df is None or df.empty or value_col not in df.columns:
        ax.text(0.5, 0.5, "no results", ha="center", va="center",
                transform=ax.transAxes, color="#8a8a86", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
        return
    conv = df[df["converged"]] if "converged" in df.columns else df
    fitted = set(conv[swept_col].dropna())
    values = [v for v in sorted(df[swept_col].dropna().unique()) if v in fitted]
    pos = {v: i for i, v in enumerate(values)}
    offsets = {"diagonal": -0.2, "unstructured": 0.2}
    for D_type in ("diagonal", "unstructured"):
        grp = conv[conv["D_type"] == D_type]
        boxes, xs = [], []
        for v in values:
            vals = grp.loc[grp[swept_col] == v, value_col].dropna().values
            if len(vals) == 0:
                continue
            boxes.append(vals)
            xs.append(pos[v] + offsets[D_type])
        if not boxes:
            continue
        color = D_TYPE_COLORS[D_type]
        bp = ax.boxplot(boxes, positions=xs, widths=0.34, patch_artist=True,
                        whis=BOX_WHISKERS, showfliers=False, manage_ticks=False)
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.5, edgecolor=color)
        for key in ("whiskers", "caps"):
            for artist in bp[key]:
                artist.set(color=color, alpha=0.8)
        for med in bp["medians"]:
            med.set(color=color, lw=1.4)
    if axhline_y is not None:
        ax.axhline(axhline_y, color="k", lw=0.8, ls="--")
    if logy:
        ax.set_yscale("log")
        _log_exponent_yaxis(ax)
    if zero_ref:
        _zero_ref_line(ax)
    ax.set_xlim(-0.5, len(values) - 0.5)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([int(v) if float(v).is_integer() else v for v in values], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.25, ls=":", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def _metric_hlines(data, metrics, pilots):
    """Reference line per metric: the time cap for time_s, else y=1 when the
    95th-percentile box top reaches it or the metric is in FORCE_HLINE."""
    hlines = {}
    for value_col, *_ in metrics:
        if value_col == "time_s":
            hlines[value_col] = TIME_CAP_SECONDS
            continue
        top_max = float("-inf")
        for name, swept_col, _sym, _label in pilots:
            df = data[name]
            if df is None or df.empty or value_col not in df.columns:
                continue
            conv = df[df["converged"]] if "converged" in df.columns else df
            if conv[value_col].notna().any():
                top = (conv.groupby([swept_col, "D_type"])[value_col]
                           .quantile(BOX_WHISKERS[1] / 100).max())
                top_max = max(top_max, top)
        hlines[value_col] = 1.0 if (top_max >= 1.0 or value_col in FORCE_HLINE) else None
    return hlines


def _task_grid(data, metrics, pilots, out_name, col_width=1.75, row_height=1.55,
               font_scale=0.9, legend_fs=8, legend_height=0.34):
    fs = lambda size: size * font_scale
    hlines = _metric_hlines(data, metrics, pilots)
    fig_h = row_height * len(pilots) + legend_height
    fig, axes = plt.subplots(len(pilots), len(metrics),
                             figsize=(col_width * len(metrics), fig_h),
                             squeeze=False, sharey="col")
    for r, (name, swept_col, sym, label) in enumerate(pilots):
        for c, (value_col, title, logy, zero_ref) in enumerate(metrics):
            ax = axes[r][c]
            _panel_box(ax, data[name], value_col, swept_col, logy,
                       axhline_y=hlines[value_col], zero_ref=zero_ref)
            ax.tick_params(labelsize=fs(8))
            ax.set_xlabel(sym, fontsize=fs(10), labelpad=1)
            if r == 0:
                ax.set_title(title, fontsize=fs(9))
            if c == 0:
                ax.set_ylabel(label.replace(" ", "\n", 1), fontsize=fs(9))
    handles = [Patch(facecolor=D_TYPE_COLORS[d], edgecolor=D_TYPE_COLORS[d], alpha=0.5, label=d.capitalize())
               for d in ("diagonal", "unstructured")]
    fig.tight_layout(rect=[0, 0, 1, 1 - legend_height / fig_h], w_pad=0.8, h_pad=0.6)
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=legend_fs,
               title=LEGEND_TITLE, title_fontsize=legend_fs)
    out = f"{PLOT_DIR}/{out_name}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    data = {}
    for name, *_ in TASK_PILOTS:
        data[name] = _load_pilot(name)
        n = 0 if data[name] is None else len(data[name])
        print(f"[scaling_py] {name}: {n} rows loaded")
    _task_grid(data, TASK_RELERR_METRICS, TASK_PILOTS, "scaling_task_boxplot_relerr")


if __name__ == "__main__":
    main()
