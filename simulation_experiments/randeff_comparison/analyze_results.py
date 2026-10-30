"""
Random-effects scaling experiment (sn=1, pz swept): writes the R-failure
diagnostic plot and the parameter-recovery LaTeX table from the lr = 1.0
Python fits merged with the R baselines.

Run from: experiments/randeff_comparison/
    python analyze_results.py
"""

import glob
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)

PLOT_DIR = "plots"

DATA_DIR = os.environ.get("RE_DATA_DIR", "data")

# Keep in sync with run_r_baselines.R's TIME_CAP. A row can be recorded
# converged=True with time_s > this: run_r_baselines.R's per-fit timer resets
# on internal retries (and, for gamm's k-halving loop in the ss experiment,
# per attempt), so a fit that only succeeds after several near-cap attempts
# can log a cumulative time_s well past TIME_CAP without ever tripping the
# R-side timed_out flag. Treated as a timeout here regardless of what R
# recorded, so it can't quietly dominate the mean/SD wall-time columns.
R_TIME_CAP = 1800
_PZ_KEEP = (5, 8, 20)

# compound_symmetry dropped from the comparison: bam runs iid only, so CS cells
# could never put all three methods on comparable footing. Existing CS rows stay
# on disk and are filtered out at load. Keep in sync with CORR_LIST in
# slurm/run_array_randeff.sh.
_CORR_KEEP = ("iid",)

_NUMERIC_COLS = {
    "N", "pz", "sn", "repeat_id", "converged", "time_s", "ise_macro", "ise_compare",
    "n_iter", "mem_mb", "D_mse", "D_mae", "sigma2_abs_err", "beta_mse", "beta_mae",
    "cond_xtvinvx", "tau_sq_max", "tau_sq_min", "rho_abs_err", "D_full_mse",
    "rho_est", "D_frob", "D_stein", "ise_total", "final_loss", "n_knots_min",
    "n_knots_max", "curve_ids", "lr",
}

# Internal method key stays "python" everywhere (data loading, filtering,
# column names); this only renames it for display.
METHOD_LABELS = {"python": "torchpalmm"}







LR = 1.0   # every reported torchpalmm fit uses L-BFGS with lr = 1.0


def _lr_fit(group_df):
    """A rep's lr = LR fit (the last one if a resumed run wrote it twice). Callers
    keep failed fits, so a rep that diverged still counts as not converged."""
    return group_df.iloc[-1]


def _load_python():
    """Load Python results, keeping the lr = 1.0 fit per
    (N, pz, D_type, corr, repeat_id)."""
    csvs = [c for c in sorted(glob.glob("results/randeff_*_rho_moment.csv"))
            if "_summary" not in c]
    if not csvs:
        return None
    df = pd.concat([pd.read_csv(c, on_bad_lines="skip") for c in csvs], ignore_index=True)
    for _col in df.columns:
        if _col in _NUMERIC_COLS or _col.startswith("ise_"):
            df[_col] = pd.to_numeric(df[_col], errors="coerce")
    if "ise_macro" not in df.columns or df.empty:
        return None

    # "precision" is a legacy column kept for CSV schema stability; all rows are
    # fp64 now that the mixed-precision arm has been removed.
    if "precision" in df.columns:
        df = df[df["precision"].fillna("fp64") == "fp64"].copy()

    # Keep the lr = 1.0 row of every rep, failed runs included, so reps whose fit
    # diverged still appear with converged=False, keeping the convergence rate
    # accurate. Metric plots drop NaN via .dropna() later.
    if "corr" in df.columns:
        df = df[df["corr"].isin(_CORR_KEEP)]

    key = ["N", "pz", "D_type", "corr", "repeat_id"]
    best_rows = [_lr_fit(grp) for _, grp in df[pd.to_numeric(df["lr"], errors="coerce") == LR].groupby(key)]
    best = pd.DataFrame(best_rows).reset_index(drop=True)
    best["method"] = "python"
    return best


def _load_r():
    csvs = [c for c in sorted(glob.glob("results/r_randeff_*.csv")) if "_summary" not in c]
    if not csvs:
        return None
    df = pd.concat([pd.read_csv(c, on_bad_lines="skip") for c in csvs], ignore_index=True)
    # lmesp dropped (lmeSplines/nlme doesn't scale, see the SS runner); filter any
    # legacy lmesp rows so they never enter the comparison.
    if "method" in df.columns:
        df = df[df["method"] != "lmesp"].copy()
    if "corr" in df.columns:
        df = df[df["corr"].isin(_CORR_KEEP)]
    # A rep can be fit twice and land in two files (e.g. an old combined csv
    # and a newer --only-split one, or two overlapping resumed runs) -- each
    # run's own skip-set only sees what was on disk when IT started, so
    # neither catches the other's row. Left undeduplicated, _save_param_table
    # builds a repeat_id-indexed Series with a duplicate label, which crashes
    # pd.concat's alignment in _bold_mask_sig's paired t-test.
    key = [c for c in ["N", "pz", "D_type", "corr", "repeat_id", "method"] if c in df.columns]
    if key:
        n_before = len(df)
        df = df.drop_duplicates(subset=key, keep="last")
        if len(df) < n_before:
            print(f"  [_load_r] dropped {n_before - len(df)} duplicate-key rows "
                  f"(same {key} fit more than once)")
    return df if "ise_macro" in df.columns else None




def _combined():
    py = _load_python()
    r  = _load_r()
    frames = [f for f in (py, r) if f is not None]
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for _col in combined.columns:
        if _col in _NUMERIC_COLS or _col.startswith("ise_"):
            combined[_col] = pd.to_numeric(combined[_col], errors="coerce")
    combined = combined[combined["pz"].isin(_PZ_KEEP)]     # corr filtered in the loaders

    # R-only: python has no with_time_cap() analog, so its time_s is never a
    # timeout artifact.
    if {"method", "time_s", "converged"}.issubset(combined.columns):
        over_cap = (combined["method"] != "python") & (combined["time_s"] > R_TIME_CAP)
        combined.loc[over_cap, "converged"] = False
    return combined


# ---------------------------------------------------------------------------
# Single-method metric-vs-pz plot (cols = D_type, lines = N × method)
#
# Used for the python-only diagnostics, which are deliberately NOT restricted to
# the jointly-converged reps: they show python's full coverage (N=5000, pz=16),
# where the R baselines never produced a run to join against.
# ---------------------------------------------------------------------------





def _plot_r_failure_diagnostic(r_df):
    """
    Stacked bar chart of R convergence failure types by (method, D_type, pz),
    sized for a single IEEE column (3.5 in). One panel per (method, D_type) cell
    with data, x-axis = pz.

    Failure taxonomy (mirrors run_r_baselines.R):
      converged  — fit succeeded and converged flag is TRUE
      fit_null   — with_time_cap() returned NULL: timeout or R-level error
                   (converged=FALSE, time_s is NA)
      vcov_nonPD — gamm only: fit succeeded (ise_macro present, time_s not NA)
                   but converged=FALSE — LME optimizer produced non-PD apVar
      other      — converged=FALSE, time_s not NA, ise_macro is NA

    A (N, pz, D_type, corr, method) cell that run_r_baselines.R gave up on after
    DEFER_THRESHOLD consecutive timeouts (logged to results/r_deferred.csv) stops
    accumulating reps early, so its raw completed-rep counts alone would understate
    its failure rate. Each such cell's shortfall against the experiment's target
    rep count (_expected_reps()) is padded in here as additional fit_null (timeout)
    reps, since that shortfall exists precisely because the runner kept timing out.
    """
    if r_df is None or r_df.empty:
        return
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = r_df[r_df["pz"].isin(_PZ_KEEP)].copy()
    if df.empty:
        return
    df["fail_type"] = "converged"
    df.loc[~df["converged"] & df["time_s"].isna(),                            "fail_type"] = "fit_null"
    df.loc[~df["converged"] & df["time_s"].notna() & df["ise_macro"].notna(), "fail_type"] = "vcov_nonPD"
    df.loc[~df["converged"] & df["time_s"].notna() & df["ise_macro"].isna(),  "fail_type"] = "other"

    deferred_path = "results/r_deferred.csv"
    if os.path.exists(deferred_path):
        try:
            dd = pd.read_csv(deferred_path, on_bad_lines="skip")
        except Exception:
            dd = pd.DataFrame()
        if {"N", "pz", "D_type", "corr", "method"}.issubset(dd.columns):
            expected_reps = _expected_reps()
            dd = dd.drop_duplicates(subset=["N", "pz", "D_type", "corr", "method"])
            pad_rows = []
            for r in dd.itertuples(index=False):
                if r.pz not in _PZ_KEEP:
                    continue
                cell = df[(df["N"] == r.N) & (df["pz"] == r.pz) & (df["D_type"] == r.D_type)
                         & (df["corr"] == r.corr) & (df["method"] == r.method)]
                n_missing = expected_reps - len(cell)
                if n_missing > 0:
                    pad_rows.extend([{"N": r.N, "pz": r.pz, "D_type": r.D_type,
                                      "corr": r.corr, "method": r.method,
                                      "fail_type": "fit_null"}] * n_missing)
            if pad_rows:
                df = pd.concat([df, pd.DataFrame(pad_rows)], ignore_index=True)

    methods  = sorted(df["method"].unique())
    D_types  = sorted(df["D_type"].unique())
    pz_vals  = sorted(df["pz"].unique())
    types    = [t for t in ["converged", "vcov_nonPD", "fit_null", "other"]
                if (df["fail_type"] == t).any()]
    # CVD- and grayscale-separable (green/red is not); edges carry the light amber.
    colors   = {"converged": "#00558A", "vcov_nonPD": "#E3A21A",
                "fit_null": "#C0392B", "other": "#9e9e9e"}
    labels   = {"converged": "Converged", "vcov_nonPD": "Non-PD covariance",
                "fit_null": "Timeout/error", "other": "Other"}
    edge     = "#3d3d3d"
    panels   = [(m, d) for d in D_types for m in methods
                if ((df["method"] == m) & (df["D_type"] == d)).any()]

    rc = {"pdf.fonttype": 42, "font.family": "Arial", "font.size": 7,
          "mathtext.fontset": "custom", "mathtext.rm": "Arial",
          "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(panels), figsize=(3.5, 2.0), sharey=True,
                                 squeeze=False, layout="constrained")
        x = np.arange(len(pz_vals))
        for ax, (method, D_type) in zip(axes[0], panels):
            mdf = df[(df["method"] == method) & (df["D_type"] == D_type)]
            counts = (mdf.groupby(["pz", "fail_type"]).size().unstack(fill_value=0)
                         .reindex(index=pz_vals, columns=types, fill_value=0))
            totals = counts.sum(axis=1).values.astype(float)
            totals = np.where(totals == 0, np.nan, totals)

            bottom = np.zeros(len(pz_vals))
            for ft in types:
                prop = np.nan_to_num(counts[ft].values / totals)
                ax.bar(x, prop, bottom=bottom, width=0.7, color=colors[ft],
                       edgecolor=edge, linewidth=0.4)
                bottom += prop

            ax.set_xticks(x, [str(p) for p in pz_vals])
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
            # Monospace method name marks it as software, not a scenario.
            ax.set_title(f"$\\mathtt{{{method}}}$\n{D_type.capitalize()} $D$", fontsize=7,
                         pad=3, linespacing=1.2)
            ax.tick_params(labelsize=6, length=2.5, width=0.6, pad=2)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_linewidth(0.6)
        axes[0][0].set_ylabel("Proportion")
        fig.supxlabel("$q$", fontsize=7)

        # Figure-level legend: a panel may be 100% converged and show no failure types.
        handles = [Patch(facecolor=colors[t], edgecolor=edge, linewidth=0.4, label=labels[t])
                   for t in types]
        fig.legend(handles=handles, loc="outside upper center", ncol=len(types), frameon=False,
                   fontsize=7, handlelength=1.0, handletextpad=0.4, columnspacing=1.0, borderpad=0.9)
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04)
        out = f"{PLOT_DIR}/randeff_r_failure_diagnostic.png"
        fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print(f"  Saved {out}")


def _expected_reps():
    """Target repeat count per config, from data/config_randeff.json's repeat_ids
    list (the same list generate_data.py used to seed every N/pz/D_type/corr
    cell). Falls back to 25 if the config file is unavailable."""
    config_file = f"{DATA_DIR}/config_randeff.json"
    if os.path.exists(config_file):
        try:
            with open(config_file) as f:
                cfg = json.load(f)
            reps_from_config = len(cfg.get("repeat_ids", []))
            if reps_from_config > 0:
                return reps_from_config
        except Exception as e:
            print(f"  [progress] Error reading config: {e}")
    return 25






# ---------------------------------------------------------------------------
# Jointly-converged comparison grids
#
# Every python-vs-R figure has one of two shapes, so each shape is drawn by a
# single function instead of one function per metric:
#     _plot_metric_grid   median metric vs pz, python and R lines
#     _plot_ratio_grid    R/python ratio of those same medians
# Rows = R method, cols = corr x D_type, one figure per metric. All of them read
# from the jointly-converged subset (reps matched on repeat_id via _joint_subset)
# so python and the baseline are always summarised over identical datasets.
# ---------------------------------------------------------------------------


# Parameter recovery (β, σ², D) is a sameness result: python and the baselines
# agree to several significant figures, so it goes to _save_param_table as one
# table in absolute units rather than six near-flat figures. The grids below keep
# the metrics that have a real pz trend.



# (column, filename/CSV slug, LaTeX header)
_PARAM_METRICS = [
    ("beta_mae",       "beta",   r"$\beta$ MAE"),
    ("sigma2_abs_err", "sigma2", r"$|\sigma^2-\hat{\sigma}^2|$"),
    ("D_mae",          "D",      r"$D$ MAE"),
    ("ise_macro",      "ise",    r"MISE"),
]
















_HIGHER_BETTER = {"conv_pct"}          # every other reported column is an error or a cost
_BOLD_MARGIN   = 0.10                  # margin rule, used for time_s/conv_pct only
_BOLD_ALPHA    = 0.05                  # paired-t-test rule, used for the accuracy columns


def _bold_mask(tbl, cell_cols, cols):
    """True where a value beats every other method in the same setting by more
    than _BOLD_MARGIN. Compared against the runner-up rather than the mean, so a
    method only bolds when it leads the whole field; ties never bold. Columns
    where only one method reports a value are skipped, since D MAE with bam
    absent is not a win over anything."""
    mask = pd.DataFrame(False, index=tbl.index, columns=cols)
    for _, grp in tbl.groupby(cell_cols):
        for col in cols:
            s = grp[col].dropna()
            if len(s) < 2:
                continue
            lower_better = col not in _HIGHER_BETTER
            s = s.sort_values(ascending=lower_better)
            best, runner = s.iloc[0], s.iloc[1]
            if best == runner:
                continue
            if lower_better:
                wins = runner > 0 and best < (1 - _BOLD_MARGIN) * runner
            else:
                wins = best > (1 + _BOLD_MARGIN) * runner if runner > 0 else best > 0
            if wins:
                mask.at[s.index[0], col] = True
    return mask


def _bold_mask_sig(tbl, cell_cols, metric_slugs, raw_vals, alpha=_BOLD_ALPHA, margin=_BOLD_MARGIN):
    """True where a method's mean is significantly lower than every other
    method's in the same cell AND ahead by more than `margin` (relative): a
    paired t-test on repeat_id-matched per-replicate errors (two-sided,
    p < alpha) against each other present method, plus the same margin rule
    _bold_mask uses. The margin check matters because a paired t-test alone has
    enough power at n~100 replicates to flag differences at the floating-point-
    noise level -- e.g. two methods that solve for a parameter via essentially
    the same closed-form step can differ by ~1e-8, yet that still tests
    "significant" since the sign is consistent across replicates. A method only
    bolds when it beats the whole field; an untestable pair (fewer than two
    matched reps, a non-lower mean, or a sub-margin gap) blocks the bold rather
    than defaulting to significant."""
    mask = pd.DataFrame(False, index=tbl.index, columns=metric_slugs)
    for cell_key, grp in tbl.groupby(cell_cols):
        key = cell_key if isinstance(cell_key, tuple) else (cell_key,)
        for slug in metric_slugs:
            means = grp.set_index("method")[f"{slug}_mean"].dropna()
            if slug == "D":
                means = means.drop(index="bam", errors="ignore")   # D not reported for bam
            if len(means) < 2:
                continue
            best_m = means.idxmin()
            best_vals = raw_vals.get((*key, best_m, slug))
            if best_vals is None or best_vals.empty:
                continue
            sig_all = True
            for m in means.index:
                if m == best_m:
                    continue
                other_vals = raw_vals.get((*key, m, slug))
                if other_vals is None or other_vals.empty or means[best_m] >= means[m]:
                    sig_all = False
                    break
                if not (means[m] > 0 and means[best_m] < (1 - margin) * means[m]):
                    sig_all = False
                    break
                paired = pd.concat([best_vals, other_vals], axis=1, join="inner").dropna()
                if len(paired) < 2:
                    sig_all = False
                    break
                try:
                    _, p = stats.ttest_rel(paired.iloc[:, 0], paired.iloc[:, 1])
                except Exception:
                    sig_all = False
                    break
                if not (p < alpha):
                    sig_all = False
                    break
            if sig_all:
                row_idx = grp.index[grp["method"] == best_m][0]
                mask.at[row_idx, slug] = True
    return mask


_TABLE_SIG_FIGS = 3


def _fmt_sig(x, n=_TABLE_SIG_FIGS):
    """Round x to n significant figures, always in plain decimal notation --
    never scientific ("1.68e+03"), which reads badly in a LaTeX table.
    Consistent across columns spanning very different magnitudes (small MAE
    values vs. wall time in the hundreds/thousands)."""
    if pd.isna(x) or x == 0:
        return "0" if x == 0 else "nan"
    e = math.floor(math.log10(abs(x)))
    decimals = n - 1 - e
    if decimals <= 0:
        return f"{round(x, decimals):.0f}"
    return f"{x:.{decimals}f}"


# Fixed, cross-table display scale per metric slug -- NOT computed from this
# table's own data. Keeps the same metric in the same units across every
# param table in the paper (randeff_comparison and
# smoothspline_comparison share this convention); a data-adaptive
# exponent would otherwise vary run to run and table to table for the same
# metric name, which reads as inconsistent (or wrong) side by side.
_FIXED_COL_EXPONENT = {"beta": 3, "sigma2": 3, "D": 3, "ise": 3, "mise": 3}


def _column_scale(slug):
    """Fixed x10^p display multiplier for a metric slug (see
    _FIXED_COL_EXPONENT). Returns (multiplier, exponent); exponent=0 means
    "don't scale". mult is applied to the raw value before printing; the
    header shows 10^{-exponent} such that displayed*10^{-exponent} recovers
    the true value."""
    exponent = _FIXED_COL_EXPONENT.get(slug, 0)
    return 10.0 ** exponent, exponent


def _save_param_table(combined, min_n=5):
    """Parameter recovery, wall time and convergence as one table.

    N=1000 only. Grouped by D_type only, with rows (pz, method) and python as
    its own row. Accuracy and time are means (SD in parentheses) over the reps
    on which python and the participating baselines all converged, so the
    rows of a cell describe identical datasets. Errors stay in their own
    units, since a ratio of two near-zero, or two equally poor, errors is not
    interpretable.

    Only iid reaches here, the loaders having filtered on _CORR_KEEP.

    The convergence column is deliberately NOT computed on the matched subset,
    where it would be 100% by construction. It is each method's
    converged/attempted rate over all reps of the cell.
    """
    if combined is None or combined.empty or "converged" not in combined.columns:
        return
    combined = combined[combined["N"] == 1000]
    if combined.empty:
        return
    table_dir = os.path.join(os.path.dirname(PLOT_DIR), "tables")
    os.makedirs(table_dir, exist_ok=True)

    cell_cols = ["D_type", "pz"]
    no_conv   = []

    # Pass 1: reps to compare on, chosen by how many methods produced any
    # converged results for this (corr, D_type, N, pz) cell:
    #   - only python converged      -> report python's own full set (baseline
    #     rows read n=0/0% from their own attempted rows in pass 2)
    #   - python + one baseline      -> jointly-converged reps of those two
    #   - python + both baselines    -> jointly-converged reps of all three
    # A baseline that has only converged a handful of times (e.g. still
    # mid-sweep, or right at a wall-time cap) still counts as "has results"
    # and enters the intersection -- pass 2 reports whatever n that leaves,
    # it no longer drops the cell for having a small joint n.
    shared = {}
    for (corr, D_type, pz), g in combined.groupby(["corr"] + cell_cols):
        if "python" not in g["method"].values:
            continue
        methods = ["python"] + sorted(m for m in g["method"].unique() if m != "python")
        conv   = {m: set(g[(g["method"] == m) & (g["converged"] == True)]["repeat_id"])
                  for m in methods}
        if not conv["python"]:
            continue
        usable = [m for m in methods if conv[m]]
        if len(usable) < 2:
            no_conv.append(f"{corr}/{D_type}/pz={int(pz)}")
            shared[(corr, D_type, pz)] = conv["python"]
            continue
        shared[(corr, D_type, pz)] = set.intersection(*(conv[m] for m in usable))

    # Pass 2: pool those rep sets across corr.
    rows, thin, gaps = [], [], {}
    raw_vals = {}   # (D_type, pz, method, slug) -> Series of per-rep values, keyed by repeat_id
    for (D_type, pz), g in combined.groupby(cell_cols):
        parts = [g[(g["corr"] == corr) & (g["repeat_id"].isin(reps))]
                 for corr in sorted(g["corr"].unique())
                 for reps in [shared.get((corr, D_type, pz))] if reps]
        if not parts:
            continue
        gj = pd.concat(parts)

        n_py = len(gj[(gj["method"] == "python") & (gj["converged"] == True)])
        if n_py == 0:
            continue
        if n_py < min_n:
            thin.append(f"{D_type}/pz={int(pz)} (n={n_py})")

        cell_cov = {}
        for m in ["python"] + sorted(x for x in g["method"].unique() if x != "python"):
            # Converged rows only: a failed run's wall time is not comparable.
            gm  = gj[(gj["method"] == m) & (gj["converged"] == True)]
            ga  = g[g["method"] == m]
            att = len(ga.groupby(["corr", "repeat_id"]))
            ok  = len(ga[ga["converged"] == True].groupby(["corr", "repeat_id"]))
            row = {"D_type": D_type, "pz": int(pz), "method": m,
                   "n": len(gm),
                   "time_s": gm["time_s"].mean() if "time_s" in gm else float("nan"),
                   "time_s_sd": gm["time_s"].std() if "time_s" in gm else float("nan"),
                   "conv_pct": 100 * ok / att if att else float("nan")}
            for col, slug, _ in _PARAM_METRICS:
                vals = gm.set_index("repeat_id")[col] if col in gm else pd.Series(dtype=float)
                row[f"{slug}_mean"] = vals.mean()
                row[f"{slug}_sd"]   = vals.std()
                raw_vals[(D_type, int(pz), m, slug)] = vals
            cell_cov[m] = (set(gm["corr"].unique()), row)
            rows.append(row)

        # Coverage-matched gaps: score each baseline against python restricted to
        # the corr settings that baseline actually covers. Differencing bam (iid
        # only) against python's pooled mean would report a regime difference
        # as an accuracy difference.
        gpy = gj[(gj["method"] == "python") & (gj["converged"] == True)]
        for m, (corrs_m, mrow) in cell_cov.items():
            if m == "python" or not corrs_m:
                continue
            ref = gpy[gpy["corr"].isin(corrs_m)]
            for col, slug, _ in _PARAM_METRICS:
                v = mrow[f"{slug}_mean"]
                r = ref[col].mean() if col in ref else float("nan")
                if not pd.isna(v) and not pd.isna(r):
                    gaps.setdefault(slug, []).append(abs(v - r))

    if not rows:
        print("[randeff] no cell has any converged python reps — skipping param table")
        return

    tbl = pd.DataFrame(rows)
    tbl["_mo"] = (tbl["method"] != "python").astype(int)      # python first in each cell
    tbl = (tbl.sort_values(cell_cols + ["_mo", "method"])
              .drop(columns="_mo").reset_index(drop=True))

    metric_slugs = [slug for _, slug, _h in _PARAM_METRICS]
    mean_sd_cols = [c for slug in metric_slugs for c in (f"{slug}_mean", f"{slug}_sd")]
    col_order    = (["D_type", "pz", "method", "n"]
                    + mean_sd_cols + ["time_s", "time_s_sd", "conv_pct"])

    # Headline claim, in each metric's own units: worst coverage-matched gap
    # between a baseline's mean and python's, collected in pass 2.
    devs = {slug: max(v) for slug, v in gaps.items() if v}
    if devs:
        print(f"  max |R - python| over {tbl.groupby(cell_cols).ngroups} cells:  "
              + ", ".join(f"{s} {v:.2g}" for s, v in devs.items()))
    if thin:
        print(f"  small joint n (< {min_n}), reported anyway: " + "; ".join(thin))
    if no_conv:
        print(f"  no baseline converged ({len(no_conv)} corr-cells): " + "; ".join(no_conv))

    # --- LaTeX: one block per D_type, which is now the only grouping factor ---
    def _tex(s):
        return str(s).replace("_", r"\_")

    # Per-column x10^k display scale, decided once over the whole table (not
    # per D_type block) so the header annotation applies uniformly.
    col_scale = {slug: _column_scale(slug) for slug in metric_slugs}

    ncol  = 2 + len(_PARAM_METRICS) + 3          # pz,method,n,metrics,time,conv
    # Header prints -p (not the +p used to compute the displayed numbers):
    # a reader recovers the true value as shown x 10^{header}, e.g. a cell of
    # 58.3 under "(x10^{-4})" means the real value is 58.3 x 10^-4 = 0.00583,
    # directly, without inverting the exponent themselves.
    metric_hdrs = [
        hdr + (rf" ($\times 10^{{{-col_scale[slug][1]}}}$)" if col_scale[slug][1] else "")
        for _col, slug, hdr in _PARAM_METRICS
    ]
    lines = [
        r"% \cdashline between cells requires \usepackage{arydshln} in the preamble.",
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{rl" + "r" * (ncol - 2) + "}",
        r"\toprule",
        r"$p_z$ & method & $n$ & "
        + " & ".join(metric_hdrs)
        + r" & time (s) & conv. \\",
    ]
    bold_sig    = _bold_mask_sig(tbl, cell_cols, metric_slugs, raw_vals)
    bold_margin = _bold_mask(tbl, cell_cols, ["time_s", "conv_pct"])

    def _val(r, col, txt, sig=False):
        b = bold_sig.at[r.name, col] if sig else bold_margin.at[r.name, col]
        return rf"\textbf{{{txt}}}" if b else txt

    for D_type, block in tbl.groupby("D_type", sort=False):
        lines += [
            r"\midrule",
            rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{$D$ = {_tex(D_type)}}}}} \\",
        ]
        # Dashed rule between cells, so the method rows being compared read as a
        # group. Not after the last cell, where \midrule or \bottomrule follows.
        groups = list(block.groupby("pz", sort=False))
        for gi, (_key, grp) in enumerate(groups):
            for _, r in grp.iterrows():
                cells = [str(r["pz"]),
                         rf"\texttt{{{_tex(METHOD_LABELS.get(r['method'], r['method']))}}}",
                         str(r["n"])]
                for slug in metric_slugs:
                    mult, _exp = col_scale[slug]
                    mean_v, sd_v = r[f"{slug}_mean"], r[f"{slug}_sd"]
                    if slug == "D" and r["method"] == "bam":
                        # bam's fitted random-effect representation isn't a
                        # like-for-like D comparison against ours -- not reported.
                        txt = "--"
                    elif pd.isna(mean_v):
                        txt = "--"
                    elif pd.isna(sd_v):
                        txt = _fmt_sig(mean_v * mult)
                    else:
                        txt = f"{_fmt_sig(mean_v * mult)} ({_fmt_sig(sd_v * mult)})"
                    cells.append(_val(r, slug, txt, sig=True))
                mean_t, sd_t = r["time_s"], r["time_s_sd"]
                if pd.isna(mean_t):
                    time_txt = "--"
                elif pd.isna(sd_t):
                    time_txt = _fmt_sig(mean_t)
                else:
                    time_txt = f"{_fmt_sig(mean_t)} ({_fmt_sig(sd_t)})"
                cells.append(_val(r, "time_s", time_txt))
                cells.append(_val(r, "conv_pct",
                                  "--" if pd.isna(r["conv_pct"]) else rf"{r['conv_pct']:.0f}\%"))
                lines.append(" & ".join(cells) + r" \\")
            if gi < len(groups) - 1:
                lines.append(rf"\cdashline{{1-{ncol}}}")

    dev_str = ", ".join(f"{s} {v:.2g}" for s, v in devs.items()) if devs else "n/a"
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Parameter recovery and wall time (mean, SD in parentheses, over "
        r"the $n$ replicates), and convergence under iid residual "
        r"correlation, at $N=1000$. Within each ($D$, $p_z$) cell, accuracy and time are "
        r"computed over the $n$ replicates on which python and the participating "
        r"baselines all converged, so those rows describe identical datasets; a "
        r"baseline that never converged in a cell is listed at $n=0$ and does not "
        r"restrict the others. Errors are in their own units rather than as ratios to "
        r"python, since a ratio of two near-zero or two equally poor errors is not "
        r"interpretable. The convergence column is the exception to the matching, "
        r"giving each method's converged/attempted rate over \emph{all} replicates of "
        r"the cell. Largest gap between a baseline mean and python's: " + dev_str
        + r". Bold on an accuracy column marks the method whose mean is both "
          r"significantly lower than every other method's in that cell (paired "
          r"$t$-test on replicate-matched errors, $p<" + f"{_BOLD_ALPHA:g}" + r"$) and "
          r"ahead of it by more than " + f"{100 * _BOLD_MARGIN:.0f}"
          + r"\%, so a floating-point-level but consistently-signed gap cannot bold "
            r"on significance alone; bold on time or convergence marks a value "
            r"beating the runner-up by more than "
        + f"{100 * _BOLD_MARGIN:.0f}" + r"\%. \texttt{bam} reports no $D$ estimate.}",
        r"\label{tab:randeff-param}", r"\end{table}",
    ]
    out_tex = os.path.join(table_dir, "randeff_param_table.tex")
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Saved {out_tex}")




def main():
    combined = _combined()
    if combined is None:
        print("[randeff] No results found in results/.")
        return

    _plot_r_failure_diagnostic(_load_r())
    _save_param_table(combined)


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--rank_label", type=str, default="",
                         help="Output subdirectory label, e.g. 'rank30'. "
                              "Plots go to results/{rank_label}/plots, "
                              "tables to results/{rank_label}/tables.")
    _args = _parser.parse_args()
    if _args.rank_label:
        PLOT_DIR = f"results/{_args.rank_label}/plots"
    main()
