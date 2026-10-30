"""
Pair the GPU rows from run_gpu.py against CPU rows borrowed from
experiments/scaling_py/, at the largest cell of each pilot.

CPU rows come from each pilot's own _load() (imported from its analyze_results.py,
so its local/fallback logic is exercised unchanged) filtered to that pilot's extreme
axis value -- the same reuse pattern scaling_py/analyze_results.py already uses.
Pairing is on (D_type, repeat_id, optimizer, lr): same dataset, same hyperparameters,
device the only difference.

Only reps where BOTH devices converged are compared. Writes plots/gpu_cpu_combined.png.

Run from: experiments/gpu_cpu_comparison/
    python analyze_results.py
"""
import glob
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerBase

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)

_PILOT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "../scaling_py"))
PLOT_DIR = "plots"

# (cell, pilot subfolder, axis column, extreme value, display label)
CELLS = [
    ("sn10",  "ss_scaling",    "sn", 10,     r"$m$=10"),
    ("sn30",  "ss_scaling",    "sn", 30,     r"$m$=30"),
]


# Cells in the combined time + agreement figure (the m comparison).
COMBINED_CELLS = ("sn10", "sn30")

# Error metrics both devices compute against the same truth.
AGREE_COLS = ["beta_mae", "D_mae", "sigma2_abs_err", "ise_macro"]

DEVICE_COLORS = {"cpu": "#6BAED6", "gpu": "#eb6834"}
COMBINED_METRICS = [c for c in AGREE_COLS if c != "beta_mae"]

METRIC_COLORS = {"beta_mae": "#0072B2", "D_mae": "#CC79A7",
                 "sigma2_abs_err": "#009E73", "ise_macro": "#E69F00"}
DIFF_LABELS = {"D_mae": r"MAE($\hat{D}$)", "sigma2_abs_err": r"$\hat{\sigma}^2$ error",
               "ise_macro": "ISE"}


def _log_exponent_yaxis(ax):
    """Bare-exponent log-y tick labels (1, 2, 3 for 10^1..; -1, -2 for 10^-1..)
    instead of matplotlib's default "10^n" -- at this figure size the default
    stacked notation crowds and can visually overlap adjacent labels. Always
    apply this on a log y-axis rather than matplotlib's default (mirrors
    scaling_py/analyze_results.py's identical helper -- dataviz convention for
    this project, not specific to this plot)."""
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _pos: f"{round(np.log10(y))}".replace("-", "\u2212") if y > 0 else ""))



def _load_pilot_cpu(pilot):
    path = os.path.join(_PILOT_DIR, pilot, "analyze_results.py")
    spec = importlib.util.spec_from_file_location(f"_gpu_cmp_an_{pilot}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)      # chdir's into the pilot folder
    df = mod._load()
    os.chdir(_SCRIPT_DIR)
    return df




def _load_gpu(cell, device):
    csvs = [c for c in glob.glob(f"results/{cell}_{device}*.csv") if "_summary" not in c]
    if not csvs:
        return None
    df = pd.concat([pd.read_csv(c, on_bad_lines="skip") for c in csvs], ignore_index=True)
    # parallel jobs (--out-suffix) can refit the same (D_type, rep, lr); keep the first
    df = df.drop_duplicates(subset=["D_type", "repeat_id", "optimizer", "lr"], keep="first")
    return df if not df.empty else None


def _pair(cpu, gpu, axis, value):
    """Inner-join CPU and GPU on the shared fit key, restricted to the extreme cell."""
    if cpu is None or gpu is None:
        return None
    cpu = cpu[pd.to_numeric(cpu[axis], errors="coerce") == value].copy()
    if cpu.empty:
        return None
    keys = ["D_type", "repeat_id", "optimizer", "lr"]
    for k in ("repeat_id", "lr"):
        for d in (cpu, gpu):
            d[k] = pd.to_numeric(d[k], errors="coerce")
    keep = keys + ["converged", "time_s"] + [c for c in AGREE_COLS if c in cpu.columns]
    cpu_k = cpu[[c for c in keep if c in cpu.columns]]
    gpu_time = "time_s_outer" if "time_s_outer" in gpu.columns else "time_s"
    # AGREE_COLS must be in gpu_keep too, else the two frames don't overlap on those
    # names and merge() never applies the _cpu/_gpu suffixes to them -- they'd survive
    # unsuffixed from the CPU side only, and _summarise's "<c>_gpu" lookups silently
    # find nothing (this is why max_abs_metric_diff was empty in every row on the
    # first run, independent of the CUDA failure below).
    gpu_keep = keys + ["converged", gpu_time, "gpu_mem_mb", "gpu_name"] + \
        [c for c in AGREE_COLS if c in gpu.columns]
    gpu_k = gpu[[c for c in gpu_keep if c in gpu.columns]].rename(
        columns={gpu_time: "time_s"})
    m = cpu_k.merge(gpu_k, on=keys, suffixes=("_cpu", "_gpu"))
    return m if not m.empty else None










def _box(ax, vals, pos, color, width):
    """One box at x=pos: box = IQR, whiskers = 2.5th/97.5th percentiles over reps
    (a 95% interval of the replicate distribution), line = median."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    bp = ax.boxplot([vals], positions=[pos], widths=width, whis=(2.5, 97.5),
                    patch_artist=True, showfliers=False, manage_ticks=False)
    for box in bp["boxes"]:
        box.set(facecolor=color, alpha=0.6, edgecolor=color)
    for k in ("whiskers", "caps"):
        for a in bp[k]:
            a.set(color=color, alpha=0.9)
    dark = tuple(0.55 * c for c in matplotlib.colors.to_rgb(color))
    for med in bp["medians"]:
        med.set(color=dark, lw=0.9)


def _plot_combined(pairs):
    """Fit time (left) and CPU/GPU agreement (right), each over the same x-axis of
    settings (COMBINED_CELLS x D_type), as boxplots over replications: box = IQR,
    whiskers = 2.5-97.5th percentiles, so the spread reads as a 95% interval across
    reps. Left: fit time per device, each over the reps on which that device
    converged. Right: |CPU - GPU| per metric (absolute units, log scale) over reps
    where both converged. A device or setting with no converged rep is annotated
    rather than left blank."""
    settings, keys = [], []
    for cell, _pilot, _axis, _value, label in CELLS:
        if cell not in COMBINED_CELLS:
            continue
        for D_type in ("diagonal", "unstructured"):
            g = pairs[(pairs["cell"] == cell) & (pairs["D_type"] == D_type)]
            if g.empty:
                continue
            settings.append(f"{label}\n{D_type[:5]}")
            keys.append(g)
    if not settings:
        return
    x = np.arange(len(settings))

    fig, (ax_t, ax_a) = plt.subplots(1, 2, figsize=(7.16, 2.4))
    ax_t.set_yscale("log")
    ax_a.set_yscale("log")

    for i, g in enumerate(keys):
        cpu_ok, gpu_ok = g["converged_cpu"].astype(bool), g["converged_gpu"].astype(bool)
        n_cpu, n_gpu = int(cpu_ok.sum()), int(gpu_ok.sum())
        _box(ax_t, g.loc[cpu_ok, "time_s_cpu"], i - 0.2, DEVICE_COLORS["cpu"], 0.32)
        _box(ax_t, g.loc[gpu_ok, "time_s_gpu"], i + 0.2, DEVICE_COLORS["gpu"], 0.32)
        for xpos, k in ((i - 0.2, n_cpu), (i + 0.2, n_gpu)):
            if k == len(g):
                continue
            if k:
                ax_t.text(xpos, 1.02, f"{k}/{len(g)}",
                          transform=ax_t.get_xaxis_transform(), ha="center", va="bottom",
                          fontsize=6.5, color="#333")

        both = g[cpu_ok & gpu_ok]
        if both.empty:
            ax_a.text(i, 0.5, "no paired converged reps", transform=ax_a.get_xaxis_transform(),
                      ha="center", va="center", fontsize=8, color="#555")
            continue
        for off, metric in zip(np.linspace(-0.3, 0.3, len(COMBINED_METRICS)), COMBINED_METRICS):
            if f"{metric}_cpu" not in both.columns or f"{metric}_gpu" not in both.columns:
                continue
            diff = (both[f"{metric}_cpu"] - both[f"{metric}_gpu"]).abs().clip(lower=1e-16)
            _box(ax_a, diff, i + off, METRIC_COLORS[metric], 0.15)

    _log_exponent_yaxis(ax_t)
    _log_exponent_yaxis(ax_a)
    ax_t.set_ylabel("Fit time (s)", fontsize=8)
    ax_a.set_ylabel(r"|CPU $-$ GPU|", fontsize=8)
    for ax in (ax_t, ax_a):
        ax.set_xticks(x); ax.set_xticklabels(settings, fontsize=8)
        ax.set_xlim(-0.6, len(settings) - 0.4)
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(labelsize=7, length=2.5, width=0.6, pad=2)
        ax.grid(alpha=0.4, ls=":", lw=0.5, axis="y")
        ax.spines[["top", "right"]].set_visible(False)

    dev_handles = [_RowTitle("Fit time:"), Patch(facecolor=DEVICE_COLORS["cpu"], alpha=0.6, label="CPU"),
                   Patch(facecolor=DEVICE_COLORS["gpu"], alpha=0.6, label="GPU")]
    met_handles = [_RowTitle("Error metric:")] + [Patch(facecolor=METRIC_COLORS[m], alpha=0.6, label=DIFF_LABELS[m]) for m in COMBINED_METRICS]
    fig.tight_layout(rect=[0, 0.09, 1, 0.98], w_pad=2.0)
    legs = [fig.legend(handles=h, loc="lower center", ncol=5, handler_map={_RowTitle: _NoHandle()},
                       frameon=False, fontsize=7.5, columnspacing=1.0, handlelength=1.6,
                       bbox_to_anchor=((ax.get_position().x0 + ax.get_position().x1) / 2, 0.0))
            for h, ax in ((dev_handles, ax_t), (met_handles, ax_a))]
    for leg in legs:
        leg.get_texts()[0].set_fontweight("bold")
    out = f"{PLOT_DIR}/gpu_cpu_combined.png"
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  Saved {out}")






class _RowTitle(Patch):
    def __init__(self, label):
        super().__init__(label=label)


class _NoHandle(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        return [Rectangle((0, 0), 0, 0, alpha=0, transform=trans)]








def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    pairs_all = []
    for cell, pilot, axis, value, label in CELLS:
        if cell not in COMBINED_CELLS:
            continue
        gpu = _load_gpu(cell, "gpu")
        if gpu is None:
            print(f"[gpu_cpu] {cell}: no GPU results yet, skipping.")
            continue
        m = _pair(_load_pilot_cpu(pilot), gpu, axis, value)
        if m is None:
            print(f"[gpu_cpu] {cell}: no CPU rows pair with the GPU rows.")
            continue
        pairs_all.append(m.assign(cell=cell, label=label, cpu_source="borrowed"))
    if not pairs_all:
        print("[gpu_cpu] Nothing to report.")
        return
    _plot_combined(pd.concat(pairs_all, ignore_index=True))


if __name__ == "__main__":
    main()
