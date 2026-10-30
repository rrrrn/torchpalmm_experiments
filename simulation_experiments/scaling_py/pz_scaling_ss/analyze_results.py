"""
Analyze the pz_scaling_ss pilot (N=1000, px=2, sn=1 fixed, pz swept).
Python-only -- no baseline comparison, so the table is plain mean(SD) per
(D_type, pz), not the bold-on-significance form used where a baseline is
present (compare randeff_comparison/analyze_results.py).

Local results/ only -- no longer merges in ../../randeff_comparison as a
fallback for pz values this pilot hasn't fit itself: that sibling experiment
predates metrics like ise_relerr and used a different D_relerr definition, so
values it "covered" were silently missing columns or on a different footing
than a genuinely local fit. A pz with no local rows now just shows "no
results" until it's actually fit here.

Run from: experiments/scaling_py/pz_scaling_ss/
    python analyze_results.py
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)

PLOT_DIR = "plots"
CORE_METRICS = ["beta_mse", "D_mse", "sigma2_abs_err", "ise_macro"]

# The pz values actively tracked: matches this pilot's own pz_list / the
# slurm submit list. Trimmed here so plots/tables only ever show the values
# we're actually tracking.
_PZ_KEEP = (5, 10, 20, 30)


def _select_best_lr(group_df):
    """Pick the row with the largest lr for the rep. Reps where that run
    diverged keep converged=False so the convergence rate stays accurate."""
    if "lr" not in group_df.columns or len(group_df) == 1:
        return group_df.iloc[0]
    return group_df.loc[group_df["lr"].idxmax()]


def _load_local():
    csvs = [c for c in glob.glob("results/pz*.csv") if "_summary" not in c]
    if not csvs:
        return None
    return pd.concat([pd.read_csv(c, on_bad_lines="skip") for c in csvs], ignore_index=True)


def _load():
    df = _load_local()
    if df is None or df.empty:
        return None

    for c in ("pz", "repeat_id", "lr", "time_s", "beta_mse", "beta_mae", "D_mse", "D_mae",
              "sigma2_abs_err", "ise_macro", "beta_mape", "sigma2_mape", "D_relerr", "ise_relerr"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "converged" in df.columns:
        df["converged"] = df["converged"].astype(str).str.lower().isin(["true", "1", "1.0"])
    df = df[df["pz"].isin(_PZ_KEEP)]
    if df.empty:
        return None
    key = ["pz", "D_type", "repeat_id"]
    best = pd.DataFrame([_select_best_lr(g) for _, g in df.groupby(key)]).reset_index(drop=True)
    return best


def _summary(df):
    agg_cols = {"conv_rate": ("converged", "mean"), "mean_time_s": ("time_s", "mean"),
                "beta_mse": ("beta_mse", "mean"), "D_mse": ("D_mse", "mean"),
                "sigma2_abs_err": ("sigma2_abs_err", "mean"), "ise_macro": ("ise_macro", "mean"),
                "n": ("repeat_id", "count")}
    g = (df.groupby(["D_type", "pz"]).agg(**agg_cols).round(4).reset_index())
    os.makedirs("results", exist_ok=True)
    g.to_csv("results/summary.csv", index=False)
    print(g.to_string(index=False))
    print("  Saved results/summary.csv")
    return g


def _save_table(df):
    table_dir = "tables"
    os.makedirs(table_dir, exist_ok=True)
    metrics = [("beta_mae", "beta MAE"), ("sigma2_abs_err", "|sigma2-hat|"),
               ("D_mae", "D MAE"), ("ise_macro", "MISE")]
    rows = []
    for (D_type, pz), g in df.groupby(["D_type", "pz"]):
        conv = g[g["converged"]]
        row = {"D_type": D_type, "pz": int(pz), "n": len(conv),
               "conv_pct": 100 * len(conv) / len(g) if len(g) else float("nan"),
               "time_s": conv["time_s"].mean() if len(conv) else float("nan")}
        for col, _lbl in metrics:
            row[f"{col}_mean"] = conv[col].mean() if col in conv and len(conv) else float("nan")
            row[f"{col}_sd"]   = conv[col].std() if col in conv and len(conv) else float("nan")
        rows.append(row)
    tbl = pd.DataFrame(rows).sort_values(["D_type", "pz"])
    out_csv = os.path.join(table_dir, "pz_scaling_ss_table.csv")
    tbl.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}")


def _scaling_plot(df, value, ylabel, fname_base, logy):
    pzs = sorted(df["pz"].dropna().unique())
    pz_pos = {p: i for i, p in enumerate(pzs)}
    D_types = sorted(df["D_type"].dropna().unique())
    fig, axes = plt.subplots(1, len(D_types), figsize=(4.6 * len(D_types), 4.0), squeeze=False)
    fig.suptitle(f"[pz_scaling_ss]  {ylabel} vs pz    (N=1000, px=2, sn=1)")
    conv = df[df["converged"]] if "converged" in df.columns else df
    for c, dt in enumerate(D_types):
        ax = axes[0][c]
        s = (conv[conv["D_type"] == dt].groupby("pz")[value].median()
             .reset_index().sort_values("pz").dropna(subset=[value]))
        if not s.empty:
            xs = [pz_pos[v] for v in s["pz"]]
            ax.plot(xs, s[value], marker="o", lw=1.4, ms=6, color="#1F77B4")
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(range(len(pzs))); ax.set_xticklabels([int(p) for p in pzs])
        ax.set(title=f"D={dt}", xlabel="pz", ylabel=ylabel)
        ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = f"{PLOT_DIR}/{fname_base}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  Saved {out}")


def _convergence_plot(df):
    pzs = sorted(df["pz"].dropna().unique())
    pz_pos = {p: i for i, p in enumerate(pzs)}
    D_types = sorted(df["D_type"].dropna().unique())
    fig, axes = plt.subplots(1, len(D_types), figsize=(4.6 * len(D_types), 4.0),
                             squeeze=False, sharey=True)
    fig.suptitle("[pz_scaling_ss]  convergence rate vs pz    N=1000, px=2, sn=1")
    for c, dt in enumerate(D_types):
        ax = axes[0][c]
        s = (df[df["D_type"] == dt].groupby("pz")["converged"].mean().mul(100)
             .reset_index().sort_values("pz"))
        xs = [pz_pos[v] for v in s["pz"]]
        ax.plot(xs, s["converged"], marker="o", lw=1.4, ms=6, color="#1F77B4")
        ax.set_ylim(0, 105)
        ax.set_xticks(range(len(pzs))); ax.set_xticklabels([int(p) for p in pzs])
        ax.set(title=f"D={dt}", xlabel="pz")
        if c == 0:
            ax.set_ylabel("converged (%)")
        ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    out = f"{PLOT_DIR}/convergence_vs_pz.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  Saved {out}")


def _progress_plot(df, n_reps):
    piv = (df.pivot_table(index="D_type", columns="pz", values="converged",
                          aggfunc="sum", fill_value=0).sort_index())
    fig, ax = plt.subplots(figsize=(2.2 + 1.1 * piv.shape[1], 1.4 + 0.5 * piv.shape[0]))
    fig.suptitle(f"[pz_scaling_ss] converged reps per cell  (target n_reps={n_reps})")
    ax.imshow(piv.values, vmin=0, vmax=max(n_reps, 1), cmap="Greens", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([int(c) for c in piv.columns])
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    ax.set(xlabel="pz")
    for (i, j), v in np.ndenumerate(piv.values):
        ax.text(j, i, int(v), ha="center", va="center",
                color="white" if v > n_reps / 2 else "black")
    fig.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = f"{PLOT_DIR}/progress.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  Saved {out}")


def main():
    df = _load()
    if df is None:
        print("[pz_scaling_ss] No results in results/."); return
    os.makedirs(PLOT_DIR, exist_ok=True)
    _summary(df)

    n_reps = int(df.groupby(["pz", "D_type"])["repeat_id"].nunique().max())
    _progress_plot(df, n_reps)

    _scaling_plot(df, "time_s", "median runtime (s)", "runtime_vs_pz", logy=True)
    _scaling_plot(df, "beta_mse", "beta MSE", "beta_mse_vs_pz", logy=True)
    _scaling_plot(df, "D_mse", "D MSE", "D_mse_vs_pz", logy=True)
    _scaling_plot(df, "sigma2_abs_err", "sigma^2 abs err", "sigma2_vs_pz", logy=True)
    _scaling_plot(df, "ise_macro", "MISE (spline recovery)", "ise_vs_pz", logy=True)
    _convergence_plot(df)
    _save_table(df)


if __name__ == "__main__":
    main()
