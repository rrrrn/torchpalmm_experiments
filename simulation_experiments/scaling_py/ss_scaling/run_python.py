"""
Fit the smoothing-spline mixed model on the ss_scaling data (N=1000, px=2,
pz=2 fixed, sn swept). Driver: resume, chronic-timeout skip, LR sweep with
every row written; computes one ISE per smooth component, since sn > 1 here.

Run from: simulation_experiments/scaling_py/ss_scaling/
    python run_python.py --sn 10 --D_type diagonal
    python run_python.py                 # all configs in data/config.json
"""

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(_SCRIPT_DIR, "../../..")))

import argparse
import json
import resource
import signal
import time
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from SimScripts.LMMSS import MIXED_SS_CURVES
from torchpalmm import SmoothingSplineMixed

try:
    from scipy.interpolate import CubicSpline as _CubicSpline
except Exception:
    _CubicSpline = None

ISE_GRID_N = 500
TIME_CAP_SECONDS = 1800
# Skip the remaining reps of an (sn, D_type) config once this many of its reps
# have timed out at every lr (chronic-timeout heuristic).
TIMEOUT_SKIP_MIN_REPS = 2


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return ru.ru_maxrss / (1024.0**2 if sys.platform == "darwin" else 1024.0)
    except Exception:
        return float("nan")


def load_data(sn, D_type, rid):
    with open(f"data/sn{sn}_D{D_type}_repeat{rid}.json") as f:
        d = json.load(f)
    cids = [int(c) for c in d["curve_ids"]]
    return {
        "y":            np.array(d["y"]),
        "X":            np.array(d["X"]),
        "Z":            np.array(d["Z"]),
        "splines_vars": np.array(d["splines_vars"]),
        "subject_id":   np.array(d["subject_id"]),
        "beta":         np.array(d["beta"]),
        "D":            np.array(d["D"]),
        "sigma2":       float(d["sigma2"]),
        "curve_ids":    cids,
        "curve_means":  [float(m) for m in d["curve_means"]],
        "curve_stds":   [float(s) for s in d["curve_stds"]],
        "data_checksum": str(d.get("data_checksum", "")),
        "sn": sn, "D_type": D_type,
    }


class _FitTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _FitTimeout()


def _true_curve_fn(data, d):
    cid    = data["curve_ids"][d]
    mean_d = data["curve_means"][d]
    std_d  = data["curve_stds"][d]
    raw    = MIXED_SS_CURVES[cid]
    return lambda t: (raw(t) - mean_d) / std_d


def _eval_on_grid(x_vals, theta_fitted, grid):
    if _CubicSpline is not None and len(x_vals) >= 2:
        return _CubicSpline(x_vals, theta_fitted, bc_type="natural")(grid)
    return np.interp(grid, x_vals, theta_fitted)


def compute_ise(ss_model, data):
    """Per-component out-of-sample ISE, integrated on a shared dense grid over
    each component's observed support (matches the R baselines elsewhere in
    the repo, though this pilot has none of its own). Also returns each
    component's relative ISE (ISE / the true curve's own integrated variance
    over that grid), macro-averaged the same way as the absolute ISE."""
    sn = data["sn"]
    try:
        fitteddf, _ = ss_model.get_blup()
    except Exception as exc:
        print(f"[compute_ise] get_blup failed: {type(exc).__name__}: {exc}", flush=True)
        return [float("nan")] * sn, [float("nan")] * sn

    ise_list, ise_rel_list = [], []
    for d in range(sn):
        sub = (fitteddf[fitteddf["spline_id"] == d]
               .sort_values("spline_var").reset_index(drop=True))
        if len(sub) < 2:
            ise_list.append(float("nan"))
            ise_rel_list.append(float("nan"))
            continue
        x_vals       = sub["spline_var"].values.astype(np.float64)
        theta_fitted = sub["theta_fitted"].values.astype(np.float64)
        curve_fn     = _true_curve_fn(data, d)

        grid     = np.linspace(x_vals.min(), x_vals.max(), ISE_GRID_N)
        fitted_g = _eval_on_grid(x_vals, theta_fitted, grid)
        true_g   = curve_fn(torch.tensor(grid, dtype=torch.float32)).numpy().astype(np.float64)
        true_gc  = true_g - true_g.mean()
        ise         = float(np.trapezoid((fitted_g - fitted_g.mean() - true_gc) ** 2, grid))
        true_energy = float(np.trapezoid(true_gc ** 2, grid))
        ise_list.append(ise)
        ise_rel_list.append((ise / true_energy) if true_energy > 0 else float("nan"))
    return ise_list, ise_rel_list


def _diag_cond_xtvinvx(ss_model):
    try:
        lmm = ss_model.lmm
        mask_vec = lmm.mask.unsqueeze(-1)
        mask_mat = mask_vec @ mask_vec.transpose(1, 2)
        X_eff    = lmm.masked_X if hasattr(lmm, "masked_X") else (lmm.X_grouped * mask_vec)
        ZDZ = lmm.masked_Z @ ss_model.D @ lmm.masked_Z.transpose(-1, -2)
        NB_spline = (lmm.NB_NB * ss_model.tau_sq_list.view(-1, 1, 1, 1)).sum(0)
        V = ss_model.sigma2 * lmm.eye_nmax.unsqueeze(0) + ZDZ + NB_spline
        V = V * mask_mat + (1.0 - mask_mat) * lmm.eye_nmax * 1e6
        V = 0.5 * (V + V.transpose(-1, -2))
        L = torch.linalg.cholesky(V)
        Vinv_X  = torch.cholesky_solve(X_eff, L)
        XtVinvX = (X_eff.transpose(1, 2) @ Vinv_X).sum(0)
        if XtVinvX.numel() == 0:
            return float("nan")
        return float(torch.linalg.cond(XtVinvX))
    except Exception:
        return float("nan")


def run_one(data, optimizer_cls, lr, max_iter=1000, tol=1e-6, n_restarts=1):
    y            = torch.tensor(data["y"],            dtype=torch.float64)
    X            = torch.tensor(data["X"],            dtype=torch.float64)
    Z            = torch.tensor(data["Z"],            dtype=torch.float64)
    splines_vars = torch.tensor(data["splines_vars"], dtype=torch.float64)
    subject_id   = torch.tensor(data["subject_id"],  dtype=torch.long)

    mem_before = _rss_mb()
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TIME_CAP_SECONDS)
    t0 = time.time()

    try:
        ss_model = SmoothingSplineMixed(
            y=y, X=X, Z=Z, spline_inputs=splines_vars, obs_id=subject_id,
            corr_structure="iid", rand_D_type=data["D_type"],
        )
        ss_model.fit(
            lr=lr, max_iter=max_iter, verbose=False, optimizer=optimizer_cls,
            warm_start=True, convg_check="loss", tol=tol, fit_type="reml",
            grad_control_tol=1e-2, n_restarts=n_restarts,
        )
        signal.alarm(0)
        elapsed   = time.time() - t0
        mem_after = _rss_mb()

        lmm       = getattr(ss_model, "lmm", ss_model)
        converged = bool(getattr(lmm, "convg", False))
        n_iter    = int(getattr(lmm, "n_iter", 0))

        px          = getattr(ss_model, "px", data["X"].shape[1])
        beta_est    = ss_model.beta[:px].detach().cpu().numpy().flatten()
        sigma2_est  = float(ss_model.sigma2)
        D_est       = ss_model.D.detach().cpu().numpy()

        true_beta   = data["beta"].flatten()
        true_D_diag = np.diag(data["D"]) if data["D"].ndim == 2 else data["D"]
        D_diag_est  = np.diag(D_est) if D_est.ndim == 2 else D_est

        d_beta   = beta_est - true_beta
        beta_mse = float(np.mean(d_beta ** 2)) if len(d_beta) > 0 else float("nan")
        beta_mae = float(np.mean(np.abs(d_beta))) if len(d_beta) > 0 else float("nan")
        sigma2_abs_err = abs(sigma2_est - data["sigma2"])

        if len(D_diag_est) > 0 and len(true_D_diag) > 0:
            d_D   = D_diag_est - true_D_diag
            D_mse = float(np.mean(d_D ** 2))
            D_mae = float(np.mean(np.abs(d_D)))
        else:
            D_mse = D_mae = float("nan")

        true_D = data["D"]
        if true_D.ndim == 2 and D_est.ndim == 2 and true_D.shape == D_est.shape:
            D_full_mse = float(np.mean((D_est - true_D) ** 2))
        else:
            D_full_mse = float("nan")

        # Relative errors, in addition to the absolute ones above: beta entries
        # are away from 0 ([1,...,px]), so entrywise MAPE is safe. D's
        # off-diagonals can be exactly 0 (rand_D_type="diagonal"), so the
        # relative Frobenius-norm error is restricted to true D's non-zero
        # entries.
        beta_mape = float(np.mean(np.abs(d_beta) / np.abs(true_beta))) if len(d_beta) > 0 else float("nan")
        sigma2_mape = sigma2_abs_err / data["sigma2"]
        if true_D.ndim == 2 and D_est.ndim == 2 and true_D.shape == D_est.shape:
            nz_mask     = true_D != 0
            true_D_norm = float(np.linalg.norm(true_D[nz_mask]))
            D_relerr    = (float(np.linalg.norm((D_est - true_D)[nz_mask])) / true_D_norm) if true_D_norm > 0 else float("nan")
        else:
            D_relerr = float("nan")

        ise_list, ise_rel_list = compute_ise(ss_model, data)
        ise_macro  = float(np.nanmean(ise_list))
        ise_relerr = float(np.nanmean(ise_rel_list))
        tau_sq = ss_model.tau_sq_list.detach().cpu().numpy().flatten()

        return {
            "converged":      converged,
            "n_iter":         n_iter,
            "time_s":         elapsed,
            "mem_mb":         mem_after - mem_before,
            "beta_mse":       beta_mse,
            "beta_mae":       beta_mae,
            "sigma2_abs_err": sigma2_abs_err,
            "D_mse":          D_mse,
            "D_mae":          D_mae,
            "D_full_mse":     D_full_mse,
            "beta_mape":      beta_mape,
            "sigma2_mape":    sigma2_mape,
            "D_relerr":       D_relerr,
            "tau_sq_min":     float(np.min(tau_sq)) if tau_sq.size else float("nan"),
            "tau_sq_max":     float(np.max(tau_sq)) if tau_sq.size else float("nan"),
            "cond_xtvinvx":   _diag_cond_xtvinvx(ss_model),
            "ise_macro":      ise_macro,
            "ise_relerr":     ise_relerr,
            "error":          None,
        }

    except _FitTimeout:
        return _fail_row(_rss_mb() - mem_before, float(TIME_CAP_SECONDS),
                         f"timeout>{TIME_CAP_SECONDS}s")
    except Exception as exc:
        signal.alarm(0)
        return _fail_row(_rss_mb() - mem_before, time.time() - t0, str(exc))
    finally:
        signal.alarm(0)


def _fail_row(mem, t, err):
    return {
        "converged": False, "n_iter": 0, "time_s": t, "mem_mb": mem,
        "beta_mse": float("nan"), "beta_mae": float("nan"),
        "sigma2_abs_err": float("nan"), "D_mse": float("nan"),
        "D_mae": float("nan"), "D_full_mse": float("nan"),
        "beta_mape": float("nan"), "sigma2_mape": float("nan"), "D_relerr": float("nan"),
        "tau_sq_min": float("nan"), "tau_sq_max": float("nan"),
        "cond_xtvinvx": float("nan"), "ise_macro": float("nan"), "ise_relerr": float("nan"),
        "error": err,
    }


# Every reported fit uses L-BFGS with lr = 1.0 (smaller lrs were slower and
# sometimes failed to converge).
OPTIMIZER_CONFIGS = [
    ("LBFGS", torch.optim.LBFGS, 1.0),
]


def _resume(out_csv, key_cols, full_cols=None):
    if not os.path.exists(out_csv):
        return set(), True
    try:
        existing = pd.read_csv(out_csv, on_bad_lines="skip")
    except Exception:
        existing = pd.DataFrame()
    if not set(key_cols).issubset(existing.columns):
        print(f"[_resume] {out_csv} corrupted — removing.")
        os.remove(out_csv)
        return set(), True
    # A file written by an older code version (fewer columns, e.g. before a
    # metric like ise_relerr existed) would otherwise silently misalign every
    # appended row against this file's stale header (header=False on append
    # only writes values, not names). Upgrade the file to the current schema
    # first -- old rows get NaN in any newly-added column -- so appends stay
    # column-aligned regardless of which code version wrote what's here.
    if full_cols is not None and list(existing.columns) != list(full_cols):
        existing = existing.reindex(columns=full_cols)
        existing.to_csv(out_csv, index=False)
        print(f"[_resume] {out_csv}: migrated schema to {len(full_cols)} columns.")
    done = set(zip(*(existing[c] for c in key_cols)))
    print(f"Resuming: {len(done)} rows done in {out_csv}")
    return done, False


def _rep_timed_out(errs):
    return len(errs) > 0 and all(str(e).startswith("timeout") for e in errs)


def _seed_timeout_history(csv_paths):
    rep_errors = defaultdict(list)
    need = {"sn", "D_type", "repeat_id", "error"}
    for p in csv_paths:
        if not os.path.exists(p):
            continue
        try:
            dfh = pd.read_csv(p, on_bad_lines="skip")
        except Exception:
            continue
        if not need.issubset(dfh.columns):
            continue
        for rec in dfh[list(need)].itertuples(index=False):
            rep_errors[(rec.sn, rec.D_type, rec.repeat_id)].append(rec.error)
    return rep_errors


def _flat(res):
    return {k: (v.replace("\n", " ").replace("\r", " ") if isinstance(v, str) else v)
            for k, v in res.items()}


def _summarise(out_csv, group_cols):
    df = pd.read_csv(out_csv, on_bad_lines="skip")
    summary = (df.groupby(group_cols)
                 .agg(conv_rate=("converged", "mean"),
                      mean_time_s=("time_s", "mean"),
                      mean_mem_mb=("mem_mb", "mean"),
                      beta_mse=("beta_mse", "mean"),
                      D_mse=("D_mse", "mean"),
                      D_full_mse=("D_full_mse", "mean"),
                      sigma2_abs_err=("sigma2_abs_err", "mean"),
                      ise_macro=("ise_macro", "mean"))
                 .round(4))
    summary_path = out_csv.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path)
    print(f"Saved {out_csv}\nSaved {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sn",     type=int, default=None)
    parser.add_argument("--D_type", type=str, default=None, choices=["diagonal", "unstructured"])
    parser.add_argument("--reps",   type=int, default=None,
                        help="pilot: fit only the first --reps repeat_ids")
    parser.add_argument("--n_restarts", type=int, default=1,
                        help="random restarts per fit; best REML kept")
    return parser.parse_args()


def run(args, cfg):
    sn_list     = [args.sn]      if args.sn      is not None else cfg["sn_list"]
    D_type_list = [args.D_type]  if args.D_type  is not None else cfg["D_type_list"]
    repeat_ids  = cfg["repeat_ids"][:args.reps] if args.reps is not None else cfg["repeat_ids"]

    slice_mode = all(v is not None for v in [args.sn, args.D_type])
    out_csv = (f"results/sn{args.sn}_D{args.D_type}.csv" if slice_mode
               else "results/results.csv")

    key_cols    = ["sn", "D_type", "repeat_id", "optimizer", "lr"]
    summary_key = ["sn", "D_type", "optimizer", "lr"]
    full_cols   = key_cols + list(_fail_row(0.0, 0.0, "").keys())

    os.makedirs("results", exist_ok=True)
    done, write_header = _resume(out_csv, key_cols, full_cols)
    csv_paths = [out_csv]
    if not slice_mode:
        import glob as _glob
        for _p in _glob.glob("results/sn*.csv"):
            if "_summary" not in _p:
                done |= _resume(_p, key_cols)[0]
                csv_paths.append(_p)

    rep_errors   = _seed_timeout_history(csv_paths)
    timeout_rids = defaultdict(set)
    for (sn_, dt_, rid_), errs in rep_errors.items():
        if _rep_timed_out(errs):
            timeout_rids[(sn_, dt_)].add(rid_)

    def _emit(sn, D_type, rid, opt_name, lr, res):
        nonlocal write_header
        row = pd.DataFrame([{"sn": sn, "D_type": D_type, "repeat_id": rid,
                             "optimizer": opt_name, "lr": lr, **_flat(res)}])
        row.to_csv(out_csv, mode="a", header=write_header, index=False)
        write_header = False

    for sn, D_type in tqdm(list(product(sn_list, D_type_list)), desc="sn×D"):
        cfg_key = (sn, D_type)
        for rid in tqdm(repeat_ids, desc="repeats", leave=False):
            if len(timeout_rids[cfg_key]) >= TIMEOUT_SKIP_MIN_REPS:
                reason = (f"skipped: chronic timeout "
                          f"({len(timeout_rids[cfg_key])} reps timed out at all lr)")
                for opt_name, opt_cls, lr in OPTIMIZER_CONFIGS:
                    if (sn, D_type, rid, opt_name, lr) in done:
                        continue
                    _emit(sn, D_type, rid, opt_name, lr, _fail_row(0.0, 0.0, reason))
                continue

            data = load_data(sn, D_type, rid)
            rep_key = (sn, D_type, rid)
            for opt_name, opt_cls, lr in OPTIMIZER_CONFIGS:
                if (sn, D_type, rid, opt_name, lr) in done:
                    continue
                res = run_one(data, opt_cls, lr, n_restarts=args.n_restarts)
                rep_errors[rep_key].append(res["error"])
                _emit(sn, D_type, rid, opt_name, lr, res)
            if _rep_timed_out(rep_errors[rep_key]):
                timeout_rids[cfg_key].add(rid)

    _summarise(out_csv, summary_key)


def main():
    args = parse_args()
    with open("data/config.json") as f:
        cfg = json.load(f)
    run(args, cfg)


if __name__ == "__main__":
    main()
