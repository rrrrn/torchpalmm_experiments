"""Plot the standardized smooth-function library and its default-cell recovery.

Each library member is fit as the *single* smooth term in the default simulation
cell (n=1000, n_i=15, p=q=2), with 100 diagonal-D replicates. The dashed curve
in every panel is the fitted replicate whose overall ISE is closest to the
median ISE across those 100 fits.

Run from the project root; reads the default-cell fits in ``results/curve_library/``:
    python simulation_experiments/smoothspline_comparison/plot_curve_library.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType in PDFs, as journals require
matplotlib.rcParams.update({
    "font.family": "Arial",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})
# Same blue/orange pair as the scaling_py figures.
TRUE_COLOR, FIT_COLOR = "#00558A", "#C77700"
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_SCRIPT_DIR, "../..")))
from SimScripts.LMMSS import MIXED_SS_CURVE_LABELS  # noqa: E402

PLOT_DIR = os.path.join(_SCRIPT_DIR, "plots")
DEFAULT_CELL_DIR = os.path.join(_SCRIPT_DIR, "results", "curve_library")
LABELS = {"√·sin": "Doppler",   # clearer names for the paper, in the covariate s
          "2·sin(πt)": "2·sin(πs)", "4t(1−t)": "4s(1−s)",
          "sin(4πt)": "sin(4πs)", "|t−0.5|": "|s−0.5|"}








def _plain(x, sig=2):
    """Plain-decimal rounding (no scientific notation) to `sig` significant figures."""
    d = sig - 1 - int(np.floor(np.log10(abs(x))))
    return f"{x:.{max(d, 0)}f}"




def plot_default_cell():
    """The manuscript figure: truth and pointwise-median default-cell fit."""
    fit_files = sorted(glob.glob(os.path.join(DEFAULT_CELL_DIR, "fits_curve*.csv")))
    curve_files = sorted(glob.glob(os.path.join(DEFAULT_CELL_DIR, "curves_curve*.csv")))
    if not fit_files or not curve_files:
        raise FileNotFoundError(
            f"No default-cell library results in {DEFAULT_CELL_DIR}. "
            "Expected fits_curve*.csv and curves_curve*.csv for curve ids 0--9.")
    fits = pd.concat([pd.read_csv(path) for path in fit_files], ignore_index=True)
    curves = pd.concat([pd.read_csv(path) for path in curve_files], ignore_index=True)
    # Some previous workers may have written unstructured-D rows before that
    # branch was cancelled. They are deliberately excluded from this figure.
    fits = fits[fits["D_type"] == "diagonal"].copy()
    curves = curves[curves["D_type"] == "diagonal"].copy()
    if "curve" not in curves:
        raise ValueError("Default-cell curve CSVs must contain a `curve` column.")
    fits["converged"] = fits["converged"].astype(str).str.lower().isin(["true", "1"])
    expected = set(range(len(MIXED_SS_CURVE_LABELS)))
    found = set(curves["curve"].unique())
    missing = sorted(expected - found)
    if missing:
        raise ValueError(f"Missing fitted default-cell curves: {missing}")

    fig, axes = plt.subplots(2, 5, figsize=(6.8, 3.8), squeeze=False, layout="constrained")
    fig.patch.set_facecolor("white")
    for cid, ax in enumerate(axes.flat):
        cd = curves[curves["curve"] == cid]
        wide = cd.pivot_table(index="t", columns=["D_type", "repeat_id"], values="fitted",
                              aggfunc="first").sort_index()
        n_fit = wide.shape[1]
        if n_fit != 100:
            raise ValueError(f"Curve {cid} has {n_fit} fitted curves; expected 100.")
        t = wide.index.to_numpy(float)
        truth = cd.drop_duplicates("t").set_index("t")["true"].reindex(t).to_numpy(float)
        ise = fits.loc[(fits.get("curve", cid) == cid) & fits["converged"], "ise_macro"]
        # Per-file fit tables carry the curve id in their filename rather than a
        # column; attach it here when reading old intermediate outputs.
        if ise.empty:
            fpath = fit_files[cid]
            local = pd.read_csv(fpath)
            local["converged"] = local["converged"].astype(str).str.lower().isin(["true", "1"])
            ise = local.loc[local["converged"], "ise_macro"]
        if len(ise) != 100:
            raise ValueError(f"Curve {cid} has {len(ise)} converged ISE values; expected 100.")
        fit_rows = fits.loc[(fits["curve"] == cid) & fits["converged"],
                            ["D_type", "repeat_id", "ise_macro"]]
        median_ise = float(ise.median())
        representative = fit_rows.loc[(fit_rows["ise_macro"] - median_ise).abs().idxmin()]
        median_fit = (cd[cd["repeat_id"] == representative["repeat_id"]]
                      .drop_duplicates("t").set_index("t")["fitted"].reindex(t).to_numpy(float))
        label = LABELS.get(MIXED_SS_CURVE_LABELS[cid], MIXED_SS_CURVE_LABELS[cid])
        ax.plot(t, truth, color=TRUE_COLOR, lw=1.1, label="True curve")
        ax.plot(t, median_fit, color=FIT_COLOR, lw=1.1, ls=(0, (3.2, 1.8)),
                label="Fitted curve at median ISE")
        ax.set_title(f"{label}\nMed. ISE = {_plain(median_ise)}",
                     fontsize=10, pad=3, linespacing=1.5)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.25, ls=":", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside upper center", ncol=2,
               fontsize=10, frameon=False, handlelength=2.2, columnspacing=2.0,
               borderaxespad=0.1)
    fig.supxlabel("Nonparametric covariate, $s$", fontsize=11)
    fig.supylabel("Standardized smooth function, $f(s)$", fontsize=11)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.05)
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, "mixed_curve_library.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {out}; 100 diagonal-D fitted curves per library member.")


def main():
    plot_default_cell()



if __name__ == "__main__":
    main()
