"""
Mixed (sn-scaling) experiment: writes tables/mixed_param_table.tex and
plots/mixed_r_failure_diagnostic.png from the lr = 1.0 Python fits and the
R baselines. Also provides DATA_DIR and _SN_KEEP to analyze_lr.py.

Run from: experiments/smoothspline_comparison/
    python analyze_results.py
"""

import glob
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
DATA_DIR = "results"



# Keep in sync with run_r_baselines.R's TIME_CAP (mixed chain). A row can be
# recorded converged=True with time_s > this: gamm's k-halving retry loop
# resets the per-attempt timer on each retry, so a fit that only succeeds
# after several near-cap attempts can log a cumulative time_s well past
# TIME_CAP without ever tripping the R-side timed_out flag. Treated as a
# timeout here regardless of what R recorded, so it can't quietly dominate
# the mean/SD wall-time columns.
R_TIME_CAP = 3600
_SN_KEEP = (1, 5, 10, 15)
_N_KEEP_MIXED = (500, 1000)

_NUMERIC_COLS = {
    "N", "pz", "sn", "repeat_id", "converged", "time_s", "ise_macro", "ise_compare",
    "n_iter", "mem_mb", "D_mse", "D_mae", "sigma2_abs_err", "beta_mse", "beta_mae",
    "cond_xtvinvx", "tau_sq_max", "tau_sq_min", "rho_abs_err", "D_full_mse",
    "rho_est", "D_frob", "D_stein", "ise_total", "final_loss", "n_knots_min",
    "n_knots_max", "lr",
    # NB: curve_ids is intentionally NOT here. It is a comma-joined list of ids
    # ("0,1,...,9"), and coercing it to a number turns every multi-component row
    # into NaN, which silently breaks the per-curve pooling in _plot_curve_library.
}

# Internal method key stays "python" everywhere (data loading, filtering,
# column names); this only renames it for display, mirroring randeff_comparison's
# METHOD_LABELS (which maps its internal "python" to "torchpalmm").
METHOD_LABELS = {"python": "torchpalmm"}







# ---------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------

class _Timeout(Exception):
    pass






# ---------------------------------------------------------------------------
# Build combined DataFrame (Python lr = 1.0 fits + R baselines)
# ---------------------------------------------------------------------------

def _verify_checksums(py_df, r_df, data_cols, label):
    """Confirm Python and R fit identical data (they read from separate dirs).
    Compares the per-dataset y checksum emitted by both runners."""
    if (r_df is None or "data_checksum" not in py_df.columns
            or "data_checksum" not in r_df.columns):
        return
    keys = [c for c in data_cols + ["repeat_id"]
            if c in py_df.columns and c in r_df.columns]
    pj = py_df[keys + ["data_checksum"]].dropna().drop_duplicates(keys).copy()
    rj = (r_df[keys + ["data_checksum"]].dropna().drop_duplicates(keys)
          .rename(columns={"data_checksum": "data_checksum_r"})).copy()
    for k in keys:
        pj[k] = pj[k].astype(str)
        rj[k] = rj[k].astype(str)
    m = pj.merge(rj, on=keys, how="inner")
    m = m[(m["data_checksum"].astype(str) != "")
          & (m["data_checksum_r"].astype(str) != "")]
    if m.empty:
        return
    bad = m[m["data_checksum"].astype(str) != m["data_checksum_r"].astype(str)]
    if len(bad):
        print(f"[{label}] WARNING: {len(bad)}/{len(m)} datasets have MISMATCHED "
              f"Python/R checksums — methods did NOT fit identical data.")
        print(bad[keys + ["data_checksum", "data_checksum_r"]].to_string(index=False))
    else:
        print(f"[{label}] checksum OK: {len(m)} datasets fit identical data "
              f"in Python and R.")


def _read_r_csv(path):
    """Read one R baseline CSV, tolerating header/row width drift from the R
    runner's append-resume: later rows carry extra columns the stale header
    lacks (ise_median inserted after ise_macro, plus error_msg/k_final/k_retries
    appended), so pd.read_csv(on_bad_lines='skip') silently DROPS them. Columns
    through 'ise_macro' are stable across schema versions; the trailing fields
    are [data_checksum] (old) or [ise_median, data_checksum, ...] (new),
    disambiguated by count. Returns a DataFrame or None."""
    import csv as _csv, io
    with open(path, newline="") as fh:
        rows = list(_csv.reader(fh))
    if len(rows) < 2:
        return None
    header, data = rows[0], rows[1:]
    if all(len(r) == len(header) for r in data):
        df = pd.read_csv(path)                       # consistent file: normal read
    elif "ise_macro" not in header:
        df = pd.read_csv(path, on_bad_lines="skip")  # can't anchor: fall back
    else:
        k      = header.index("ise_macro")           # last stable column
        prefix = header[:k + 1]
        recs   = []
        for r in data:
            if len(r) < k + 2:
                continue
            d    = dict(zip(prefix, r[:k + 1]))
            rest = r[k + 1:]
            if len(rest) == 1:                        # old tail: data_checksum only
                d["data_checksum"] = rest[0]
            else:                                     # new tail: ise_median, checksum, …
                d["ise_median"], d["data_checksum"] = rest[0], rest[1]
            recs.append(d)
        buf = io.StringIO()
        pd.DataFrame(recs).to_csv(buf, index=False)   # round-trip for dtype inference
        buf.seek(0)
        df = pd.read_csv(buf)
    if "converged" in df.columns:                     # normalise R's TRUE/FALSE/NA
        df["converged"] = df["converged"].map(
            lambda v: str(v).strip().lower() in ("true", "1", "1.0"))
    return df


def _load_r_baselines(r_csv_pattern):
    csvs = [c for c in sorted(glob.glob(r_csv_pattern)) if "_summary" not in c]
    if not csvs:
        return None
    frames = [_read_r_csv(c) for c in csvs]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    # lmesp dropped: lmeSplines puts one RE column per knot at a single group, so
    # nlme::lme is ~cubic in sn*K_c and times out on every sn>=5 cell (all-NA
    # rows). Filter any legacy lmesp rows so they never enter the comparison.
    if "method" in df.columns:
        df = df[df["method"] != "lmesp"].copy()
    return df if "ise_macro" in df.columns else None


def _build_combined(py_best, r_df, data_cols, ise_comp_cols):
    """Merge Python lr = 1.0 rows and R rows into one DataFrame.

    ise_compare is the *median* per-component ISE for both methods (falling back
    to the mean, ise_macro, for older result files without an ise_median column).
    The median is robust to a single spatially-inhomogeneous curve (the Doppler,
    curve 5) whose ISE is ~20-40x the others and dominates the mean at large sn.
      Python: ise_median — median over components of ISE(f̂_d, f_d)
      R:      ise_median — median over components of ISE(s(t_d), f_d)
    Both are on the same shared integration grid (see run_python.py / the R
    baseline), so the comparison is on an equal footing.

    Parameter-recovery columns (D_mse, sigma2_abs_err, beta_mse, …) are
    pre-computed in run_python.py and passed through unchanged.
    """
    PARAM_COLS = ["D_mse", "D_mae", "sigma2_abs_err",
                  "beta_mse", "beta_mae", "n_iter"]
    keep_py = (data_cols
               + ["repeat_id", "converged", "time_s", "ise_macro", "ise_median"]
               + ise_comp_cols
               + [c for c in PARAM_COLS if c not in data_cols])
    py = py_best[[c for c in keep_py if c in py_best.columns]].copy()

    # Compare on the Doppler-robust median per component. Older Python CSVs lack
    # an ise_median column; recompute it from the per-component ise_d columns
    # rather than falling back to ise_macro (the mean), which is Doppler-inflated
    # at large sn and would unfairly plot Python's mean against R's median.
    if "ise_median" in py_best.columns and py_best["ise_median"].notna().any():
        py["ise_compare"] = py_best["ise_median"].values
    elif ise_comp_cols:
        py["ise_compare"] = py_best[ise_comp_cols].median(axis=1).values
    else:
        py["ise_compare"] = py_best["ise_macro"].values
    py["method"] = "python"

    if r_df is None:
        return py

    keep_r = (data_cols
              + ["repeat_id", "method", "converged", "time_s",
                 "ise_macro", "ise_median"]
              + [c for c in ["D_mse", "D_mae", "sigma2_abs_err",
                             "beta_mse", "beta_mae", "n_iter"]
                 if c in r_df.columns])
    r = r_df[[c for c in keep_r if c in r_df.columns]].copy()
    _r_cmp = "ise_median" if "ise_median" in r.columns else "ise_macro"
    r["ise_compare"] = r[_r_cmp]   # median (robust) per-component ISE, mean fallback

    # Fill Python-only columns with NaN in R rows
    py_only_cols = list(ise_comp_cols)
    if "ise_macro" not in r.columns:
        py_only_cols.append("ise_macro")
    for col in py_only_cols:
        if col not in r.columns:
            r[col] = float("nan")

    combined = pd.concat([py, r], ignore_index=True, sort=False)
    for _col in combined.columns:
        if _col in _NUMERIC_COLS or _col.startswith("ise_"):
            combined[_col] = pd.to_numeric(combined[_col], errors="coerce")
    if "sn" in combined.columns:
        combined = combined[combined["sn"].isin(_SN_KEEP)]
    if "pz" in combined.columns:
        is_mixed = combined["pz"].fillna(0) > 0
        combined = combined[~is_mixed | combined["N"].isin(_N_KEEP_MIXED)]

    # R-only: python has no with_time_cap() analog, so its time_s is never a
    # timeout artifact.
    if {"method", "time_s", "converged"}.issubset(combined.columns):
        over_cap = (combined["method"] != "python") & (combined["time_s"] > R_TIME_CAP)
        combined.loc[over_cap, "converged"] = False
    return combined


# ---------------------------------------------------------------------------
# Parameter recovery table
#
# Accuracy is a sameness result here, so it goes to one table rather than a grid
# of near-flat figures. Errors are reported in their own units: a ratio of two
# near-zero, or two equally poor, errors is not interpretable.
# ---------------------------------------------------------------------------

# (column, CSV/filename slug, LaTeX header)
_PARAM_METRICS = [
    ("beta_mae",       "beta",   r"$\beta$ MAE"),
    ("sigma2_abs_err", "sigma2", r"$|\sigma^2-\hat{\sigma}^2|$"),
    ("D_mae",          "D",      r"$D$ MAE"),
    ("ise_compare",    "mise",   r"MISE"),
]

_HIGHER_BETTER = {"conv_pct"}          # every other reported column is an error or a cost
_BOLD_MARGIN   = 0.10                  # margin rule, used for time_s/conv_pct only
_BOLD_ALPHA    = 0.05                  # paired-t-test rule, used for the accuracy columns
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


def _bold_mask_sig(tbl, cell_cols, metric_slugs, raw_vals, alpha=_BOLD_ALPHA, margin=_BOLD_MARGIN):
    """True where a method's mean is significantly lower than every other
    method's in the same cell AND ahead by more than `margin` (relative): a
    paired t-test on repeat_id-matched per-replicate errors (two-sided,
    p < alpha) against each other present method, plus the same margin rule
    _bold_mask uses. See randeff_comparison/analyze_results.py's version of
    this function for the fuller rationale (the margin guards against a
    floating-point-noise-level but consistently-signed gap testing
    "significant" at n~100 replicates). A method only bolds when it beats the
    whole field; an untestable pair (fewer than two matched reps, a
    non-lower mean, or a sub-margin gap) blocks the bold rather than
    defaulting to significant."""
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


def _bold_mask(tbl, cell_cols, cols):
    """True where a value beats every other method in the same setting by more
    than _BOLD_MARGIN. Compared against the runner-up rather than the mean, so a
    method only bolds when it leads the whole field; ties never bold. Columns
    where only one method reports a value are skipped, since a metric no
    baseline reports is not a win over anything."""
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


def _save_param_table(combined, label, min_n=5):
    """Parameter recovery, wall time and convergence as one table.

    N=1000 only. Grouped by D_type only, with rows (sn, method) and python as
    its own row. Accuracy and time are means (SD in parentheses) over the reps
    on which python and the participating baselines all converged, so the
    rows of a cell describe identical datasets. Errors stay in their own
    units, since a ratio of two near-zero, or two equally poor, errors is not
    interpretable.

    Mirrors randeff_comparison/analyze_results.py's _save_param_table (see
    that file for the fuller rationale); this experiment has no corr axis, so
    the pass-1 rep selection keys directly on (D_type, sn) instead of pooling
    across corr subsets.

    The convergence column is deliberately NOT computed on the matched subset,
    where it would be 100% by construction. It is each method's
    converged/attempted rate over all reps of the cell.
    """
    if combined is None or combined.empty or "converged" not in combined.columns:
        return
    if not {"N", "sn", "D_type"}.issubset(combined.columns):
        return
    combined = combined[combined["N"] == 1000]
    if combined.empty:
        return
    table_dir = os.path.join(os.path.dirname(PLOT_DIR), "tables")
    os.makedirs(table_dir, exist_ok=True)

    cell_cols = ["D_type", "sn"]

    # Pass 1: reps to compare on, chosen by how many methods produced any
    # converged results for this (D_type, sn) cell:
    #   - only python converged   -> report python's own full set (baseline
    #     rows read n=0/0% from their own attempted rows in pass 2)
    #   - python + >=1 baseline   -> jointly-converged reps of those methods
    # A baseline that has only converged a handful of times still counts as
    # "has results" and enters the intersection -- pass 2 reports whatever n
    # that leaves, it no longer drops the cell for having a small joint n.
    shared, no_conv = {}, []
    for key, g in combined.groupby(cell_cols):
        if "python" not in g["method"].values:
            continue
        methods = ["python"] + sorted(m for m in g["method"].unique() if m != "python")
        conv   = {m: set(g[(g["method"] == m) & (g["converged"] == True)]["repeat_id"])
                  for m in methods}
        if not conv["python"]:
            continue
        usable = [m for m in methods if conv[m]]
        if len(usable) < 2:
            no_conv.append("/".join(str(x) for x in key))
            shared[key] = conv["python"]
            continue
        shared[key] = set.intersection(*(conv[m] for m in usable))

    # Pass 2.
    rows, thin, gaps = [], [], {}
    raw_vals = {}   # (D_type, sn, method, slug) -> Series of per-rep values, keyed by repeat_id
    for key, g in combined.groupby(cell_cols):
        reps = shared.get(key)
        if not reps:
            continue
        D_type, sn = key
        gj = g[g["repeat_id"].isin(reps)]

        n_py = len(gj[(gj["method"] == "python") & (gj["converged"] == True)])
        if n_py == 0:
            continue
        if n_py < min_n:
            thin.append(f"{D_type}/sn={int(sn)} (n={n_py})")

        cell_cov = {}
        for m in ["python"] + sorted(x for x in g["method"].unique() if x != "python"):
            # Converged rows only: a failed run's wall time is not comparable.
            gm  = gj[(gj["method"] == m) & (gj["converged"] == True)]
            ga  = g[g["method"] == m]
            att = ga["repeat_id"].nunique()
            ok  = ga[ga["converged"] == True]["repeat_id"].nunique()
            row = {"D_type": D_type, "sn": int(sn), "method": m,
                   "n": len(gm),
                   "time_s": gm["time_s"].mean() if "time_s" in gm else float("nan"),
                   "time_s_sd": gm["time_s"].std() if "time_s" in gm else float("nan"),
                   "conv_pct": 100 * ok / att if att else float("nan")}
            for col, slug, _ in _PARAM_METRICS:
                if m == "bam" and slug == "D":
                    # Not comparable, not just noisy: bam's D_c = sigma2/lambda_c
                    # (mgcv's per-term smoothing parameters) systematically
                    # under-penalizes every random-slope column after the first
                    # when fitting >1 jointly under the same grouping factor,
                    # inflating that column's estimate ~10-30x in every replicate
                    # tested (verified by direct sp inspection), independent of
                    # the underlying data. Left NaN so it renders as "--" like
                    # any other unreported cell, rather than averaging a
                    # meaningless number.
                    vals = pd.Series(dtype=float)
                else:
                    vals = gm.set_index("repeat_id")[col] if col in gm else pd.Series(dtype=float)
                row[f"{slug}_mean"] = vals.mean()
                row[f"{slug}_sd"]   = vals.std()
                raw_vals[(D_type, int(sn), m, slug)] = vals
            cell_cov[m] = row
            rows.append(row)

        gpy = gj[(gj["method"] == "python") & (gj["converged"] == True)]
        for m, mrow in cell_cov.items():
            if m == "python":
                continue
            for col, slug, _ in _PARAM_METRICS:
                v = mrow[f"{slug}_mean"]
                r = gpy[col].mean() if col in gpy else float("nan")
                if not pd.isna(v) and not pd.isna(r):
                    gaps.setdefault(slug, []).append(abs(v - r))

    if not rows:
        print(f"[{label}] no cell has any converged python reps — skipping param table")
        return

    tbl = pd.DataFrame(rows)
    tbl["_mo"] = (tbl["method"] != "python").astype(int)      # python first in each cell
    tbl = (tbl.sort_values(cell_cols + ["_mo", "method"])
              .drop(columns="_mo").reset_index(drop=True))

    metric_slugs = [slug for _, slug, _h in _PARAM_METRICS]
    mean_sd_cols = [c for slug in metric_slugs for c in (f"{slug}_mean", f"{slug}_sd")]
    col_order    = (["D_type", "sn", "method", "n"]
                    + mean_sd_cols + ["time_s", "time_s_sd", "conv_pct"])

    devs = {slug: max(v) for slug, v in gaps.items() if v}
    if devs:
        print(f"  max |R - python| over {tbl.groupby(cell_cols).ngroups} cells:  "
              + ", ".join(f"{s} {v:.2g}" for s, v in devs.items()))
    if thin:
        print(f"  small joint n (< {min_n}), reported anyway: " + "; ".join(thin))
    if no_conv:
        print(f"  no baseline converged ({len(no_conv)} cells): " + "; ".join(no_conv))

    # --- LaTeX: one block per D_type, which is the only grouping factor ---
    def _tex(s):
        return str(s).replace("_", r"\_")

    # Per-column x10^k display scale, decided once over the whole table.
    col_scale = {slug: _column_scale(slug) for slug in metric_slugs}

    ncol  = 2 + len(_PARAM_METRICS) + 3          # sn,method,n,metrics,time,conv
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
        r"$s_n$ & method & $n$ & "
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
        groups = list(block.groupby("sn", sort=False))
        for gi, (_key, grp) in enumerate(groups):
            for _, r in grp.iterrows():
                cells = [str(r["sn"]),
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
        r"\caption{Smoothing accuracy, parameter recovery and wall time (mean, SD "
        r"in parentheses, over the $n$ replicates), and convergence, at $N=1000$. "
        r"Within each ($D$, $s_n$) cell, accuracy and time are computed over the $n$ "
        r"replicates on which python and the participating baselines all converged, "
        r"so those rows describe identical datasets; a baseline that never converged "
        r"in a cell is listed at $n=0$ and does not restrict the others. Errors are "
        r"in their own units rather than as ratios to python, since a ratio of two "
        r"near-zero or two equally poor errors is not interpretable. MISE is the "
        r"median per-component ISE. The convergence column is the exception to the "
        r"matching, giving each method's converged/attempted rate over \emph{all} "
        r"replicates of the cell. Largest gap between a baseline mean and python's: "
        + dev_str + r". Bold on an accuracy column marks the method whose mean is "
        r"both significantly lower than every other method's in that cell (paired "
        r"$t$-test on replicate-matched errors, $p<" + f"{_BOLD_ALPHA:g}" + r"$) and "
        r"ahead of it by more than " + f"{100 * _BOLD_MARGIN:.0f}"
        + r"\%, so a floating-point-level but consistently-signed gap cannot bold "
          r"on significance alone; bold on time or convergence marks a value "
          r"beating the runner-up by more than "
        + f"{100 * _BOLD_MARGIN:.0f}" + r"\%. \texttt{bam}'s $D$ MAE is marked "
          r"-- : its per-column variance estimate ($D_c=\hat\sigma^2/\lambda_c$, "
          r"read off \texttt{mgcv}'s per-term smoothing parameters) is not "
          r"comparable, since jointly fitting more than one random-slope term "
          r"under the same grouping factor systematically under-penalizes every "
          r"column after the first, inflating its estimate by roughly an order "
          r"of magnitude in every replicate tested regardless of the "
          r"underlying data.}",
        rf"\label{{tab:{label}-param}}", r"\end{table}",
    ]
    out_tex = os.path.join(table_dir, f"{label}_param_table.tex")
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Saved {out_tex}")






# ---------------------------------------------------------------------------
# R failure diagnostic
# ---------------------------------------------------------------------------

def _plot_r_failure_diagnostic(r_df, label):
    """
    Stacked bar chart of R convergence failure types by (method, D_type, sn),
    sized for a single IEEE column (3.5 in). N=1000 only, matching
    _save_param_table's scope. One panel per (method, D_type) cell with data,
    x-axis = sn.

    Failure taxonomy:
      converged  — fit succeeded
      fit_null   — with_time_cap() returned NULL (timeout / R error)
      vcov_nonPD — gamm only: fit returned but apVar non-PD
      other      — fit returned but post-fit extraction failed
    """
    if r_df is None or r_df.empty:
        return
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = r_df[r_df["N"] == 1000].copy()
    if "sn" in df.columns:
        df = df[df["sn"].isin(_SN_KEEP)]
    if df.empty:
        return
    df["fail_type"] = "converged"
    df.loc[~df["converged"] & df["time_s"].isna(),                            "fail_type"] = "fit_null"
    df.loc[~df["converged"] & df["time_s"].notna() & df["ise_macro"].notna(), "fail_type"] = "vcov_nonPD"
    df.loc[~df["converged"] & df["time_s"].notna() & df["ise_macro"].isna(),  "fail_type"] = "other"

    methods = sorted(df["method"].unique())
    D_types = sorted(df["D_type"].unique())
    sn_vals = sorted(df["sn"].unique())
    types   = [t for t in ["converged", "vcov_nonPD", "fit_null", "other"]
               if (df["fail_type"] == t).any()]
    # CVD- and grayscale-separable (green/red is not); edges carry the light amber.
    colors  = {"converged": "#00558A", "vcov_nonPD": "#E3A21A",
               "fit_null": "#C0392B", "other": "#9e9e9e"}
    labels  = {"converged": "Converged", "vcov_nonPD": "Non-PD covariance",
               "fit_null": "Timeout/error", "other": "Other"}
    edge    = "#3d3d3d"
    panels  = [(m, d) for d in D_types for m in methods
               if ((df["method"] == m) & (df["D_type"] == d)).any()]

    rc = {"pdf.fonttype": 42, "font.family": "Arial", "font.size": 7,
          "mathtext.fontset": "custom", "mathtext.rm": "Arial",
          "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(panels), figsize=(3.5, 2.0), sharey=True,
                                 squeeze=False, layout="constrained")
        x = np.arange(len(sn_vals))
        for ax, (method, D_type) in zip(axes[0], panels):
            mdf = df[(df["method"] == method) & (df["D_type"] == D_type)]
            counts = (mdf.groupby(["sn", "fail_type"]).size().unstack(fill_value=0)
                         .reindex(index=sn_vals, columns=types, fill_value=0))
            totals = counts.sum(axis=1).values.astype(float)
            totals = np.where(totals == 0, np.nan, totals)

            bottom = np.zeros(len(sn_vals))
            for ft in types:
                prop = np.nan_to_num(counts[ft].values / totals)
                ax.bar(x, prop, bottom=bottom, width=0.7, color=colors[ft],
                       edgecolor=edge, linewidth=0.4)
                bottom += prop

            ax.set_xticks(x, [str(s) for s in sn_vals])
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
        fig.supxlabel("$m$", fontsize=7)

        # Figure-level legend: a panel may be 100% converged and show no failure types.
        handles = [Patch(facecolor=colors[t], edgecolor=edge, linewidth=0.4, label=labels[t])
                   for t in types]
        fig.legend(handles=handles, loc="outside upper center", ncol=len(types), frameon=False,
                   fontsize=7, handlelength=1.0, handletextpad=0.4, columnspacing=1.0, borderpad=0.9)
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04)
        out = f"{PLOT_DIR}/{label}_r_failure_diagnostic.png"
        fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------





LR = 1.0   # every reported torchpalmm fit uses L-BFGS with lr = 1.0


def _read_python(path):
    """The lr = LR fits behind one Python results file: its *_sweep.csv sibling holds
    every lr's fit, while the file itself keeps the lowest-REML one per dataset."""
    sweep = path.replace(".csv", "_sweep.csv")
    df = pd.read_csv(sweep if os.path.exists(sweep) else path, on_bad_lines="skip")
    return df[pd.to_numeric(df["lr"], errors="coerce") == LR] if "lr" in df.columns else df


def _analyse(label, csv_pattern, config_cols, r_csv_pattern=None):
    csvs = [c for c in sorted(glob.glob(csv_pattern))
            if "_summary" not in c and "_sweep" not in c and "_by_curve" not in c]
    if not csvs:
        print(f"[{label}] No Python result CSVs found ({csv_pattern}).")
        return

    df = pd.concat([_read_python(c) for c in csvs], ignore_index=True)
    for _col in df.columns:
        if _col in _NUMERIC_COLS or _col.startswith("ise_"):
            df[_col] = pd.to_numeric(df[_col], errors="coerce")
    if "ise_macro" not in df.columns:
        print(f"[{label}] 'ise_macro' column missing — results may be pre-update.")
        return

    # "precision" is a legacy column kept for CSV schema stability; all rows are
    # fp64 now that the mixed-precision arm has been removed.
    if "precision" in df.columns:
        df = df[df["precision"].fillna("fp64") == "fp64"].copy()

    data_cols = [c for c in config_cols if c in df.columns and c not in ("optimizer", "lr")]

    ise_comp_cols = sorted(
        c for c in df.columns
        if c.startswith("ise_") and c[4:].isdigit()
    )

    has_lr_sweep = "optimizer" in df.columns and "lr" in df.columns

    if has_lr_sweep:
        group_keys = data_cols + ["repeat_id"]

        valid_mask = df.groupby(group_keys)["ise_macro"].transform(lambda s: s.notna().any())
        df_valid   = df[ valid_mask.fillna(False)].copy()
        df_invalid = df[~valid_mask.fillna(False)].copy()

        # One row per rep (all rows are lr = LR; duplicates come from resumed
        # runs): prefer a converged fit, then the lower ISE
        best_rows = []
        for group_key, group_df in df_valid.groupby(group_keys):
            converged = group_df[group_df["converged"] == True]
            if not converged.empty:
                row = converged.loc[converged["ise_macro"].idxmin()]
            else:
                row = group_df.loc[group_df["ise_macro"].idxmin()]
            best_rows.append(row)

        # Reps where NO LR produced a finite ISE: include one row per rep so that
        # convergence-rate plots count them as failed (converged=False) rather than
        # omitting them entirely, which would inflate the apparent success rate.
        for group_key, group_df in df_invalid.groupby(group_keys):
            row = group_df.iloc[0].copy()
            row["converged"] = False
            row["ise_macro"] = float("nan")
            best_rows.append(row)

        best = pd.DataFrame(best_rows).drop_duplicates(subset=group_keys).reset_index(drop=True)
    else:
        # New files like pure_N100.csv already contain final Python results.
        best = df[df["ise_macro"].notna()].copy().reset_index(drop=True)

    # --- load R baselines and build combined DataFrame ----------------------
    r_df     = _load_r_baselines(r_csv_pattern) if r_csv_pattern else None
    _verify_checksums(best, r_df, data_cols, label)
    combined = _build_combined(best, r_df, data_cols, ise_comp_cols)

    if r_df is None and r_csv_pattern:
        print(f"\n[{label}] No R baseline CSVs found ({r_csv_pattern}).")

    _save_param_table(combined, label)
    return r_df


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def analyse_mixed():
    r_df = _analyse(
        label         = "mixed",
        csv_pattern   = f"{DATA_DIR}/mixed_*.csv",
        config_cols   = ["N", "pz", "sn", "D_type", "optimizer", "lr"],
        r_csv_pattern = "results/r_mixed_*.csv",
    )
    _plot_r_failure_diagnostic(r_df, "mixed")


def main():
    analyse_mixed()


if __name__ == "__main__":
    main()
