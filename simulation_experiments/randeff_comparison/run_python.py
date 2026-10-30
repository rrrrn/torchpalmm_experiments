"""
Fit the smoothing-spline mixed model on the random-effects scaling data
(sn=1, pz swept). Mirrors smoothspline_comparison/run_python.py but sweeps pz
instead of sn.

Run from: experiments/randeff_comparison/
    python run_python.py --N 500 --pz 4 --D_type diagonal
    python run_python.py                 # all configs in config_randeff.json
"""

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(_SCRIPT_DIR, "../..")))

DATA_DIR = os.environ.get("RE_DATA_DIR", "data")

import argparse
import json
import signal
import time
import resource
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from SimScripts.LMMSS import MIXED_SS_CURVES, generate_smoothing_spline_lmm_data
from torchpalmm import SmoothingSplineMixed

# ISE is integrated on a fixed dense grid over [0, 1] (matches the R gamm/bam
# baselines' t_grid) rather than the training design points, so the smooth is
# scored out-of-sample. The fit is a natural cubic spline, so cubic
# interpolation through its fitted knot values evaluates it exactly.
try:
    from scipy.interpolate import CubicSpline as _CubicSpline
except Exception:
    _CubicSpline = None

ISE_GRID_N = 500
OOS_T_GRID = np.linspace(0.0, 1.0, ISE_GRID_N)


def _eval_on_grid(x_vals, theta_fitted, grid):
    if _CubicSpline is not None and len(x_vals) >= 2:
        return _CubicSpline(x_vals, theta_fitted, bc_type="natural")(grid)
    return np.interp(grid, x_vals, theta_fitted)


_CFG = dict(px=2, sigma_scale=1.0, lmin=20, lmax=30)
TIME_CAP_SECONDS = 1800
# Skip the remaining reps of an (N, pz, D_type, corr) config once this many of
# its reps have timed out at every lr (chronic-timeout heuristic).
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


def load_randeff(N, pz, D_type, corr, rid):
    from SimScripts.RandeffLMM import generate_randeff_data

    path = f"{DATA_DIR}/randeff_N{N}_pz{pz}_D{D_type}_C{corr}_repeat{rid}.json"
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return {
            "y":            np.array(d["y"]),
            "X":            np.array(d["X"]),
            "Z":            np.array(d["Z"]),
            "splines_vars": np.array(d["splines_vars"]),
            "subject_id":   np.array(d["subject_id"]),
            "beta":         np.array(d["beta"]),
            "D":            np.array(d["D"]),
            "sigma2":       float(d["sigma2"]),
            # Older datasets predate the persisted rho; fall back to the generator value.
            "rho":          float(d.get("rho", 0.4 if corr == "compound_symmetry" else 0.0)),
            "pz": pz, "sn": 1, "D_type": D_type, "corr": corr,
        }
    # File absent (cleanup ran) — regenerate from same seed.
    x, z, y, sv, sid, beta, D, sigma2, rho = generate_randeff_data(
        N=N, px=_CFG["px"], pz=pz, rand_D=D_type, corr_structure=corr,
        sigma_scale=_CFG["sigma_scale"], lmin=_CFG["lmin"], lmax=_CFG["lmax"],
        repeat_id=rid)
    y = y.view(-1, 1); y = y - y.mean()
    return {
        "y":            y.numpy(),
        "X":            np.asarray(x),
        "Z":            np.asarray(z),
        "splines_vars": np.asarray(sv),
        "subject_id":   np.asarray(sid, dtype=np.int32),
        "beta":         np.asarray(beta),
        "D":            np.asarray(D),
        "sigma2":       float(sigma2),
        "rho":          float(rho),
        "pz": pz, "sn": 1, "D_type": D_type, "corr": corr,
    }


class _FitTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _FitTimeout()


def compute_ise(ss_model):
    """Out-of-sample ISE of the fitted smooth vs the true curve f0 = 2 sin(pi t),
    evaluated on OOS_T_GRID rather than the training design points."""
    try:
        fitteddf, _ = ss_model.get_blup()
    except Exception as exc:
        print(f"[compute_ise] get_blup failed: {type(exc).__name__}: {exc}", flush=True)
        return float("nan")
    sub = (fitteddf[fitteddf["spline_id"] == 0]
           .sort_values("spline_var").reset_index(drop=True))
    if len(sub) < 2:
        return float("nan")
    x_vals       = sub["spline_var"].values.astype(np.float64)
    theta_fitted = sub["theta_fitted"].values.astype(np.float64)
    th     = _eval_on_grid(x_vals, theta_fitted, OOS_T_GRID)
    true_v = MIXED_SS_CURVES[0](torch.tensor(OOS_T_GRID, dtype=torch.float32)).numpy()
    thc = th - th.mean()
    tvc = true_v - true_v.mean()
    return float(np.trapezoid((thc - tvc) ** 2, OOS_T_GRID))


def _diag_cond_xtvinvx(ss_model):
    """Condition number of X_fitᵀ V⁻¹ X_fit at the fitted (σ², D, τ²)."""
    try:
        lmm    = ss_model.lmm
        device = ss_model.device
        mask_vec = lmm.mask.unsqueeze(-1)
        mask_mat = mask_vec @ mask_vec.transpose(1, 2)
        X_eff    = lmm.masked_X if hasattr(lmm, "masked_X") else (lmm.X_grouped * mask_vec)
        ZDZ = lmm.masked_Z @ ss_model.D @ lmm.masked_Z.transpose(-1, -2)
        NB_spline = (lmm.NB_NB * ss_model.tau_sq_list.view(-1, 1, 1, 1)).sum(0)
        R = lmm.resid_corr() if hasattr(lmm, "resid_corr") else lmm.eye_nmax
        V = ss_model.sigma2 * R.unsqueeze(0) + ZDZ + NB_spline
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


def run_one(data, optimizer_cls, lr, max_iter=10000, tol=1e-6, n_restarts=1, rho_init="fixed"):
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
            corr_structure=data["corr"], rand_D_type=data["D_type"],
        )
        ss_model.fit(
            lr=lr, max_iter=max_iter, verbose=False, optimizer=optimizer_cls,
            warm_start=True, convg_check="loss", tol=tol, fit_type="reml",
            grad_control_tol=1e-2,
            n_restarts=n_restarts, rho_init=rho_init,
        )
        signal.alarm(0)
        elapsed   = time.time() - t0
        mem_after = _rss_mb()

        lmm       = getattr(ss_model, "lmm", ss_model)
        converged = bool(getattr(lmm, "convg", False))
        n_iter    = int(getattr(lmm, "n_iter", 0))

        px         = getattr(ss_model, "px", data["X"].shape[1])
        beta_est   = ss_model.beta[:px].detach().cpu().numpy().flatten()
        sigma2_est = float(ss_model.sigma2)
        D_est      = ss_model.D.detach().cpu().numpy()

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

        # Full-matrix D error (captures off-diagonal recovery for unstructured)
        true_D = data["D"]
        if true_D.ndim == 2 and D_est.ndim == 2 and true_D.shape == D_est.shape:
            D_full_mse = float(np.mean((D_est - true_D) ** 2))
        else:
            D_full_mse = float("nan")

        ise = compute_ise(ss_model)
        tau_sq = ss_model.tau_sq_list.detach().cpu().numpy().flatten()

        rho_est = float(getattr(ss_model, "rho", 0.0))
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
            "rho_est":        rho_est,
            "rho_abs_err":    abs(rho_est - data["rho"]),
            "tau_sq_min":     float(np.min(tau_sq)) if tau_sq.size else float("nan"),
            "tau_sq_max":     float(np.max(tau_sq)) if tau_sq.size else float("nan"),
            "cond_xtvinvx":   _diag_cond_xtvinvx(ss_model),
            "ise_macro":      ise,
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
        "rho_est": float("nan"), "rho_abs_err": float("nan"),
        "tau_sq_min": float("nan"), "tau_sq_max": float("nan"),
        "cond_xtvinvx": float("nan"), "ise_macro": float("nan"),
        "error": err,
    }


# Every reported fit uses L-BFGS with lr = 1.0 (smaller lrs were slower and
# sometimes failed to converge).
OPTIMIZER_CONFIGS = [
    ("LBFGS", torch.optim.LBFGS, 1.0),
]


def _resume(out_csv, key_cols):
    if not os.path.exists(out_csv):
        return set(), True
    try:
        existing = pd.read_csv(out_csv, on_bad_lines="skip")
    except Exception:
        existing = pd.DataFrame()
    # Backfill the precision key for CSVs written before the fp64/mixed comparison
    # existed, so old exact-float64 rows are preserved as the "fp64" arm.
    if ("precision" in key_cols and "precision" not in existing.columns
            and not existing.empty):
        existing["precision"] = "fp64"
    if not set(key_cols).issubset(existing.columns):
        print(f"[_resume] {out_csv} corrupted — removing.")
        os.remove(out_csv)
        return set(), True
    done = set(zip(*(existing[c] for c in key_cols)))
    print(f"Resuming: {len(done)} rows done in {out_csv}")
    return done, False


def _rep_timed_out(errs):
    """A rep is 'fully timed out' when it has recorded attempts and every one
    is a timeout (skipped rows and other errors disqualify it)."""
    return len(errs) > 0 and all(str(e).startswith("timeout") for e in errs)


def _seed_timeout_history(csv_paths):
    """Tally recorded errors per (N, pz, D_type, corr, repeat_id) from history."""
    rep_errors = defaultdict(list)
    need = {"N", "pz", "D_type", "corr", "repeat_id", "error"}
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
            key = (rec.N, rec.pz, rec.D_type, rec.corr, rec.repeat_id)
            rep_errors[key].append(rec.error)
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
    parser.add_argument("--N",      type=int, default=None)
    parser.add_argument("--pz",     type=int, default=None)
    parser.add_argument("--D_type", type=str, default=None,
                        choices=["diagonal", "unstructured"])
    parser.add_argument("--corr",   type=str, default=None,
                        choices=["iid", "compound_symmetry"])
    parser.add_argument("--reps",     type=int,  default=None,
                        help="pilot: fit only the first --reps repeat_ids")
    parser.add_argument("--n_restarts", type=int, default=1,
                        help="random restarts per fit; best REML kept (default 1 = no restarts)")
    parser.add_argument("--rho_init", type=str, default="moment",
                        choices=["fixed", "moment"],
                        help="rho initialisation: fixed (rho_raw=0) or moment (OLS-residual estimate)")
    return parser.parse_args()


def run(args, cfg):
    N_list      = [args.N]      if args.N      is not None else cfg["N_list"]
    pz_list     = [args.pz]     if args.pz     is not None else cfg["pz_list"]
    D_type_list = [args.D_type] if args.D_type is not None else cfg["D_type_list"]
    corr_list   = [args.corr]   if args.corr   is not None else cfg["corr_structure_list"]
    repeat_ids  = cfg["repeat_ids"][:args.reps] if args.reps is not None else cfg["repeat_ids"]

    rho_tag  = "" if args.rho_init == "fixed" else f"_rho_{args.rho_init}"
    slice_mode = all(v is not None for v in [args.N, args.pz, args.D_type, args.corr])
    if slice_mode:
        out_csv = f"results/randeff_N{args.N}_pz{args.pz}_D{args.D_type}_C{args.corr}{rho_tag}.csv"
    else:
        out_csv = f"results/randeff_results{rho_tag}.csv"

    key_cols    = ["N", "pz", "D_type", "corr", "repeat_id", "optimizer", "lr", "precision"]
    summary_key = ["N", "pz", "D_type", "corr", "optimizer", "lr", "precision"]

    os.makedirs("results", exist_ok=True)
    done, write_header = _resume(out_csv, key_cols)
    csv_paths = [out_csv]
    if not slice_mode:
        import glob as _glob
        for _p in _glob.glob("results/randeff_N*.csv"):
            if "_summary" not in _p:
                done |= _resume(_p, key_cols)[0]
                csv_paths.append(_p)

    # Seed chronic-timeout state from history: per config, the set of reps that
    # timed out at every lr. Once it reaches TIMEOUT_SKIP_MIN_REPS, remaining
    # reps of that config are skipped instead of re-run.
    rep_errors   = _seed_timeout_history(csv_paths)
    timeout_rids = defaultdict(set)
    for (N_, pz_, dt_, cr_, rid_), errs in rep_errors.items():
        if _rep_timed_out(errs):
            timeout_rids[(N_, pz_, dt_, cr_)].add(rid_)

    def _emit(N, pz, D_type, corr, rid, opt_name, lr, res):
        nonlocal write_header
        row = pd.DataFrame([{"N": N, "pz": pz, "D_type": D_type, "corr": corr,
                             "repeat_id": rid, "optimizer": opt_name,
                             "lr": lr, "precision": "fp64", **_flat(res)}])
        row.to_csv(out_csv, mode="a", header=write_header, index=False)
        write_header = False

    for N, pz, D_type, corr in tqdm(list(product(N_list, pz_list, D_type_list, corr_list)),
                                     desc="N×pz×D×corr"):
        cfg = (N, pz, D_type, corr)
        for rid in tqdm(repeat_ids, desc="repeats", leave=False):
            if len(timeout_rids[cfg]) >= TIMEOUT_SKIP_MIN_REPS:
                reason = (f"skipped: chronic timeout "
                          f"({len(timeout_rids[cfg])} reps timed out at all lr)")
                for opt_name, opt_cls, lr in OPTIMIZER_CONFIGS:
                    if (N, pz, D_type, corr, rid, opt_name, lr, "fp64") in done:
                        continue
                    _emit(N, pz, D_type, corr, rid, opt_name, lr,
                          _fail_row(0.0, 0.0, reason))
                continue

            data = load_randeff(N, pz, D_type, corr, rid)
            rep_key = (N, pz, D_type, corr, rid)
            for opt_name, opt_cls, lr in OPTIMIZER_CONFIGS:
                if (N, pz, D_type, corr, rid, opt_name, lr, "fp64") in done:
                    continue
                res = run_one(data, opt_cls, lr, n_restarts=args.n_restarts,
                              rho_init=args.rho_init)
                rep_errors[rep_key].append(res["error"])
                _emit(N, pz, D_type, corr, rid, opt_name, lr, res)
            if _rep_timed_out(rep_errors[rep_key]):
                timeout_rids[cfg].add(rid)

    _summarise(out_csv, summary_key)


def main():
    args = parse_args()
    with open(f"{DATA_DIR}/config_randeff.json") as f:
        cfg = json.load(f)
    run(args, cfg)


if __name__ == "__main__":
    main()
