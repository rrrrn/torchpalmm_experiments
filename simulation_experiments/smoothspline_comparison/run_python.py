"""
Run from: simulation_experiments/smoothspline_comparison/
    python run_python.py --type pure  [--N N]
    python run_python.py --type mixed [--N N] [--pz PZ] [--sn SN] [--D_type diagonal|unstructured]
"""

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)

# Data directory — overridable so Python and R chains use separate dirs and
# never race on the same JSON files.
DATA_DIR = os.environ.get("SS_DATA_DIR", "data")

import argparse
import json
import signal
import time
from itertools import product

import sys
import resource

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def _rss_mb():
    """Current process RSS in MB. Uses /proc on Linux, resource module elsewhere."""
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

from simscripts.LMMSS import (global_curve, local_curve, MIXED_SS_CURVES,
                               generate_pure_ss_data,
                               generate_smoothing_spline_lmm_data,
                               ss_curve_ids, MIXED_SS_CURVE_STATS,
                               MIXED_SS_CURVE_LABELS, y_checksum)
from torchpalmm import SmoothingSplineMixed

# ISE is integrated on a shared dense grid (below) so Python and the R baselines
# use the same integration support; the Python fit is a natural cubic spline, so
# cubic interpolation through its fitted knot values evaluates it exactly.
try:
    from scipy.interpolate import CubicSpline as _CubicSpline
except Exception:
    _CubicSpline = None

# Points used to integrate ISE on the common support of each component's grid.
ISE_GRID_N = 500

# Must match generate_data.py configs so regenerated data is identical.
_PURE_CFG  = dict(sigma_scale=1.0, lmin=15, lmax=30)
_MIXED_CFG = dict(px=2, sigma_scale=1.0, lmin=15, lmax=30)

# Maximum expected unique knots per spline dimension (grid_size = 20 + 1.5*lmax = 65).
# Files generated before the discrete-grid commit have continuous t ~ total_obs >> this.
_MAX_UNIQUE_KNOTS = 200


def load_pure(N, scenario, rid):
    path = f"{DATA_DIR}/pure_N{N}_scenario{scenario}_repeat{rid}.json"
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
            "scenario":     str(d["scenario"]),
            "data_checksum": str(d.get("data_checksum", "")),
            "pz": 0, "sn": 1, "D_type": "diagonal",
        }
    # Data file deleted by cleanup step — regenerate from same seed.
    x, z, y, sv, sid, beta, D, sigma2 = generate_pure_ss_data(
        N=N, scenario=scenario, repeat_id=rid, **_PURE_CFG)
    import torch
    y = y.view(-1, 1); y = y - y.mean()
    return {
        "y":            y.numpy(),
        "X":            x.numpy(),
        "Z":            z.numpy(),
        "splines_vars": sv.numpy(),
        "subject_id":   sid.numpy(),
        "beta":         beta.numpy(),
        "D":            D.numpy() if hasattr(D, "numpy") else np.array(D),
        "sigma2":       float(sigma2),
        "scenario":     scenario,
        "data_checksum": y_checksum(y.numpy()),
        "pz": 0, "sn": 1, "D_type": "diagonal",
    }


def load_mixed(N, pz, sn, D_type, rid):
    path = f"{DATA_DIR}/mixed_N{N}_pz{pz}_sn{sn}_D{D_type}_repeat{rid}.json"
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        cids = [int(c) for c in d.get("curve_ids", ss_curve_ids(rid, sn))]
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
            "curve_means":  [float(m) for m in d.get("curve_means",
                             [MIXED_SS_CURVE_STATS[c][0] for c in cids])],
            "curve_stds":   [float(s) for s in d.get("curve_stds",
                             [MIXED_SS_CURVE_STATS[c][1] for c in cids])],
            "data_checksum": str(d.get("data_checksum", "")),
            "pz": pz, "sn": sn, "D_type": D_type,
        }
    # Data file deleted by cleanup step — regenerate from same seed.
    x, z, y, sv, sid, beta, D, sigma2 = generate_smoothing_spline_lmm_data(
        N=N, pz=pz, spline_dimension=sn, repeat_id=rid, rand_D=D_type,
        **_MIXED_CFG)
    import torch
    y = y.view(-1, 1); y = y - y.mean()
    cids = ss_curve_ids(rid, sn)
    return {
        "y":            y.numpy(),
        "X":            x.numpy(),
        "Z":            z.numpy(),
        "splines_vars": sv.numpy(),
        "subject_id":   sid.numpy(),
        "beta":         beta.numpy(),
        "D":            D.numpy() if hasattr(D, "numpy") else np.array(D),
        "sigma2":       float(sigma2),
        "curve_ids":    cids,
        "curve_means":  [MIXED_SS_CURVE_STATS[c][0] for c in cids],
        "curve_stds":   [MIXED_SS_CURVE_STATS[c][1] for c in cids],
        "data_checksum": y_checksum(y.numpy()),
        "pz": pz, "sn": sn, "D_type": D_type,
    }


TIME_CAP_SECONDS = 3600


class _FitTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _FitTimeout()


def _true_curve_fn(data, d):
    """True (standardized) curve for component d of this dataset."""
    scenario = data.get("scenario", None)
    if scenario is not None:
        return global_curve if scenario == "global" else local_curve
    cid    = data["curve_ids"][d]
    mean_d = data["curve_means"][d]
    std_d  = data["curve_stds"][d]
    raw    = MIXED_SS_CURVES[cid]
    return lambda t: (raw(t) - mean_d) / std_d


def _eval_on_grid(x_vals, theta_fitted, grid):
    """Evaluate the fitted natural cubic spline at `grid` (exact via cubic
    interpolation through the fitted knot values; linear fallback if no scipy)."""
    if _CubicSpline is not None and len(x_vals) >= 2:
        return _CubicSpline(x_vals, theta_fitted, bc_type="natural")(grid)
    return np.interp(grid, x_vals, theta_fitted)


def compute_ise(ss_model, data):
    """Return (ise_list, ise_total), both out-of-sample.

    ise_list  — per-component ISE, each integrated on a shared dense grid over
                its observed support (ISE_GRID_N points), matching the R
                baselines so the numbers are comparable.
    ise_total — recovery of the *total* additive smooth g(t)=Σ_d f_d(t_d), out
                of sample: component d contributes its own OOS-grid evaluation
                (the same fitted_g/true_g used for ise_list[d]), summed
                pointwise across components by grid index. Each component has
                its own marginal domain, so there is no single joint grid to
                evaluate on; pairing by index (position i in every component's
                own ISE_GRID_N-point grid) is a synthetic out-of-sample
                combination rather than an actual joint draw, but it keeps the
                metric off the training design without needing a combinatorial
                joint grid. If per-component ISE steps up with sn while
                ise_total stays flat, the issue is per-component *attribution*,
                not total fit quality.
    """
    sn       = data["sn"]
    nan_list = [float("nan")] * sn

    try:
        fitteddf, _ = ss_model.get_blup()
    except Exception as _blup_exc:
        print(f"[compute_ise] get_blup failed: {type(_blup_exc).__name__}: {_blup_exc}", flush=True)
        return nan_list, float("nan")

    ise_list = []
    fit_tot  = np.zeros(ISE_GRID_N)
    true_tot = np.zeros(ISE_GRID_N)
    n_valid  = 0
    for d in range(sn):
        sub = (fitteddf[fitteddf["spline_id"] == d]
               .sort_values("spline_var").reset_index(drop=True))
        if len(sub) < 2:
            ise_list.append(float("nan"))
            continue

        x_vals       = sub["spline_var"].values.astype(np.float64)
        theta_fitted = sub["theta_fitted"].values.astype(np.float64)
        curve_fn     = _true_curve_fn(data, d)

        grid     = np.linspace(x_vals.min(), x_vals.max(), ISE_GRID_N)
        fitted_g = _eval_on_grid(x_vals, theta_fitted, grid)
        true_g   = curve_fn(torch.tensor(grid, dtype=torch.float32)).numpy().astype(np.float64)
        ise_list.append(float(np.trapezoid(
            (fitted_g - fitted_g.mean() - (true_g - true_g.mean())) ** 2, grid)))

        # accumulate the total additive predictor out of sample (grid, not t_obs)
        fit_tot  += fitted_g
        true_tot += true_g
        n_valid  += 1

    if n_valid > 0:
        fit_tot  -= fit_tot.mean()
        true_tot -= true_tot.mean()
        ise_total = float(np.mean((fit_tot - true_tot) ** 2))
    else:
        ise_total = float("nan")

    return ise_list, ise_total


def _diag_cond_xtvinvx(ss_model):
    """Condition number of X_fitᵀ V⁻¹ X_fit at the fitted (σ², D, τ²).

    Diagnoses fixed-effect collinearity (e.g. the redundant linear columns
    each spline contributes). Returns nan on any numerical failure.
    """
    try:
        lmm    = ss_model.lmm
        device = ss_model.device
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


def _d_full_losses(D_est, true_D):
    """Full-matrix D losses: Frobenius ||D̂−D||_F and Stein/entropy loss
    tr(D̂D⁻¹) − log|D̂D⁻¹| − p. These capture off-diagonal (covariance) error
    that the diagonal-only D_mse misses in the unstructured case."""
    try:
        De = np.asarray(D_est, dtype=np.float64)
        Dt = np.asarray(true_D, dtype=np.float64)
        if De.ndim != 2 or De.shape != Dt.shape or De.shape[0] == 0:
            return float("nan"), float("nan")
        frob = float(np.linalg.norm(De - Dt, "fro"))
        p    = Dt.shape[0]
        M    = De @ np.linalg.inv(Dt)
        sign, logdet = np.linalg.slogdet(M)
        stein = (float(np.trace(M) - logdet - p)
                 if sign > 0 and np.isfinite(logdet) else float("nan"))
        return frob, stein
    except Exception:
        return float("nan"), float("nan")


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
            y=y, X=X, Z=Z,
            spline_inputs=splines_vars,
            obs_id=subject_id,
            corr_structure="iid",
            rand_D_type=data["D_type"],
        )
        ss_model.fit(
            lr=lr,
            max_iter=max_iter,
            verbose=False,
            optimizer=optimizer_cls,
            warm_start=True,
            convg_check="loss",
            tol=tol,
            fit_type="reml",
            grad_control_tol=1e-2,
            n_restarts=n_restarts,
        )
        signal.alarm(0)
        elapsed  = time.time() - t0
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

        D_frob, D_stein = _d_full_losses(D_est, data["D"])
        # beta entries are away from 0, so entrywise MAPE is safe. D's off-diagonals
        # can approach 0 (rand_D_type="unstructured" AR(1)-like decay), so entrywise
        # % error is unstable there; report the relative Frobenius-norm error instead.
        beta_mape   = float(np.mean(np.abs(d_beta) / np.abs(true_beta))) if len(d_beta) > 0 else float("nan")
        sigma2_mape = sigma2_abs_err / data["sigma2"]
        true_D_norm = float(np.linalg.norm(np.asarray(data["D"], dtype=np.float64), "fro"))
        D_relerr    = (D_frob / true_D_norm) if (np.isfinite(D_frob) and true_D_norm > 0) else float("nan")

        ise_list, ise_total = compute_ise(ss_model, data)
        ise_macro  = float(np.nanmean(ise_list))
        # Robust across-component summary: the median is unaffected by a single
        # spatially-inhomogeneous curve (e.g. the Doppler, curve 5) whose ISE is
        # ~20-40x the others and otherwise dominates the mean.
        ise_median = float(np.nanmedian(ise_list))

        tau_sq = ss_model.tau_sq_list.detach().cpu().numpy().flatten()
        tau_sq_min = float(np.min(tau_sq)) if tau_sq.size else float("nan")
        tau_sq_max = float(np.max(tau_sq)) if tau_sq.size else float("nan")
        cond_xtvinvx = _diag_cond_xtvinvx(ss_model)

        # Realized interior-knot count per component (data-driven #unique − 2).
        n_knots = [int(v.numel()) - 2 for v in ss_model.unique_spline_vec]
        final_loss = float(getattr(lmm, "final_loss", float("nan")))
        return {
            "converged":      converged,
            "n_iter":         n_iter,
            "final_loss":     final_loss,
            "time_s":         elapsed,
            "mem_mb":         mem_after - mem_before,
            "beta_mse":       beta_mse,
            "beta_mae":       beta_mae,
            "sigma2_abs_err": sigma2_abs_err,
            "D_mse":          D_mse,
            "D_mae":          D_mae,
            "D_frob":         D_frob,
            "D_stein":        D_stein,
            "beta_mape":      beta_mape,
            "sigma2_mape":    sigma2_mape,
            "D_relerr":       D_relerr,
            "tau_sq_min":     tau_sq_min,
            "tau_sq_max":     tau_sq_max,
            "cond_xtvinvx":   cond_xtvinvx,
            "n_knots_min":    int(min(n_knots)) if n_knots else 0,
            "n_knots_max":    int(max(n_knots)) if n_knots else 0,
            "data_checksum":  data.get("data_checksum", ""),
            "curve_ids":      ",".join(str(c) for c in data.get("curve_ids", [])),
            **{f"ise_{d}": ise_list[d] for d in range(len(ise_list))},
            "ise_macro":      ise_macro,
            "ise_median":     ise_median,
            "ise_total":      ise_total,
            "error":          None,
        }

    except _FitTimeout:
        sn = data["sn"]
        return {
            "converged":      False,
            "n_iter":         0,
            "final_loss":     float("nan"),
            "time_s":         float(TIME_CAP_SECONDS),
            "mem_mb":         _rss_mb() - mem_before,
            "beta_mse":       float("nan"), "beta_mae":       float("nan"),
            "sigma2_abs_err": float("nan"), "D_mse":          float("nan"),
            "D_mae":          float("nan"),
            "D_frob":         float("nan"), "D_stein":        float("nan"),
            "beta_mape":      float("nan"), "sigma2_mape":    float("nan"), "D_relerr": float("nan"),
            "tau_sq_min":     float("nan"), "tau_sq_max":     float("nan"),
            "cond_xtvinvx":   float("nan"),
            "n_knots_min":    0, "n_knots_max":    0,
            "data_checksum":  data.get("data_checksum", ""),
            "curve_ids":      ",".join(str(c) for c in data.get("curve_ids", [])),
            **{f"ise_{d}": float("nan") for d in range(sn)},
            "ise_macro":      float("nan"),
            "ise_median":     float("nan"),
            "ise_total":      float("nan"),
            "error":          f"timeout>{TIME_CAP_SECONDS}s",
        }

    except Exception as exc:
        signal.alarm(0)
        sn = data["sn"]
        return {
            "converged":      False,
            "n_iter":         0,
            "final_loss":     float("nan"),
            "time_s":         time.time() - t0,
            "mem_mb":         _rss_mb() - mem_before,
            "beta_mse":       float("nan"), "beta_mae":       float("nan"),
            "sigma2_abs_err": float("nan"), "D_mse":          float("nan"),
            "D_mae":          float("nan"),
            "D_frob":         float("nan"), "D_stein":        float("nan"),
            "beta_mape":      float("nan"), "sigma2_mape":    float("nan"), "D_relerr": float("nan"),
            "tau_sq_min":     float("nan"), "tau_sq_max":     float("nan"),
            "cond_xtvinvx":   float("nan"),
            "n_knots_min":    0, "n_knots_max":    0,
            "data_checksum":  data.get("data_checksum", ""),
            "curve_ids":      ",".join(str(c) for c in data.get("curve_ids", [])),
            **{f"ise_{d}": float("nan") for d in range(sn)},
            "ise_macro":      float("nan"),
            "ise_median":     float("nan"),
            "ise_total":      float("nan"),
            "error":          str(exc),
        }

    finally:
        signal.alarm(0)


# Every reported fit uses L-BFGS with lr = 1.0 (smaller lrs were slower and
# sometimes failed to converge). Kept as a list so a sensitivity sweep can pass
# more; with several, the lowest-final-REML fit is kept and every fit goes to
# *_sweep.csv.
LBFGS_LR_SWEEP = [1.0]


def _select_min_reml(sweep):
    """sweep: list of (lr, res). Pick the fit with lowest final REML among the
    converged ones with a finite loss; fall back to lowest finite loss; else
    the first candidate."""
    def finite(r):
        fl = r.get("final_loss", float("nan"))
        return fl is not None and np.isfinite(fl)

    conv = [(lr, r) for lr, r in sweep if r.get("converged") and finite(r)]
    pool = conv if conv else [(lr, r) for lr, r in sweep if finite(r)]
    if not pool:
        return sweep[0]
    return min(pool, key=lambda kr: kr[1]["final_loss"])


def run_dataset(data, n_restarts=1, lr_sweep=None):
    """Fit every LBFGS lr; return (sweep, selected_lr, selected_res)."""
    if lr_sweep is None:
        lr_sweep = LBFGS_LR_SWEEP
    sweep = [(lr, run_one(data, torch.optim.LBFGS, lr, n_restarts=n_restarts))
             for lr in lr_sweep]
    sel_lr, sel_res = _select_min_reml(sweep)
    return sweep, sel_lr, sel_res


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",     type=str, required=True, choices=["pure", "mixed"])
    parser.add_argument("--N",        type=int, default=None)
    parser.add_argument("--scenario", type=str, default=None, choices=["global", "local"])
    parser.add_argument("--pz",       type=int, default=None)
    parser.add_argument("--sn",       type=int, default=None)
    parser.add_argument("--D_type",   type=str, default=None, choices=["diagonal", "unstructured"])
    parser.add_argument("--reps",     type=int, default=None,
                        help="pilot: fit only the first --reps repeat_ids")
    parser.add_argument("--n_restarts", type=int, default=1,
                        help="random log-τ² restarts per fit; keep lowest REML")
    parser.add_argument("--rank_label", type=str, default="",
                        help="Output subdirectory label for CSV files")
    return parser.parse_args()


def run_pure(args, cfg):
    N_list        = [args.N]        if args.N        is not None else cfg["N_list"]
    scenario_list = [args.scenario] if args.scenario is not None else cfg["scenario_list"]
    repeat_ids    = cfg["repeat_ids"][:args.reps] if args.reps is not None else cfg["repeat_ids"]

    # Build results directory path with optional rank_label
    results_dir = "results"
    if args.rank_label:
        results_dir = f"results/{args.rank_label}"

    if args.N is not None and args.scenario is not None:
        out_csv = f"{results_dir}/pure_N{args.N}_scenario{args.scenario}.csv"
    elif args.N is not None:
        out_csv = f"{results_dir}/pure_N{args.N}.csv"
    else:
        out_csv = f"{results_dir}/pure_results.csv"

    # One row per dataset (REML-selected); resume on the dataset key only.
    key_cols    = ["N", "scenario", "repeat_id"]
    summary_key = ["N", "scenario", "optimizer"]
    sweep_csv   = out_csv.replace(".csv", "_sweep.csv")

    os.makedirs(results_dir, exist_ok=True)
    done, write_header = _resume(out_csv, key_cols)
    # In aggregate mode, also absorb previously computed slice CSVs
    if out_csv.endswith("pure_results.csv"):
        import glob as _glob
        for _p in _glob.glob(f"{results_dir}/pure_N*.csv"):
            if "_summary" not in _p and "_sweep" not in _p:
                done |= _resume(_p, key_cols)[0]
    sweep_header = not os.path.exists(sweep_csv)

    for N in tqdm(N_list, desc="N"):
        for scenario in tqdm(scenario_list, desc="scenario", leave=False):
            for rid in tqdm(repeat_ids, desc="repeats", leave=False):
                if (N, scenario, rid) in done:
                    continue
                data = load_pure(N, scenario, rid)
                sweep, sel_lr, sel_res = run_dataset(data, n_restarts=args.n_restarts)

                for lr, res in sweep:
                    sweep_row = pd.DataFrame([{"N": N, "scenario": scenario, "repeat_id": rid,
                                               "optimizer": "LBFGS", "lr": lr, **_flat(res)}])
                    sweep_row.to_csv(sweep_csv, mode="a", header=sweep_header, index=False)
                    sweep_header = False

                row = pd.DataFrame([{"N": N, "scenario": scenario, "repeat_id": rid,
                                     "optimizer": "LBFGS", "lr": sel_lr,
                                     "selection": "min_reml", **_flat(sel_res)}])
                row.to_csv(out_csv, mode="a", header=write_header, index=False)
                write_header = False

    _summarise(out_csv, summary_key)


def run_mixed(args, cfg):
    N_list      = [args.N]      if args.N      is not None else cfg["N_list"]
    pz_list     = [args.pz]     if args.pz     is not None else cfg["pz_list"]
    sn_list     = [args.sn]     if args.sn     is not None else cfg["sn_list"]
    D_type_list = [args.D_type] if args.D_type is not None else cfg["D_type_list"]
    repeat_ids  = cfg["repeat_ids"][:args.reps] if args.reps is not None else cfg["repeat_ids"]

    # Build results directory path with optional rank_label
    results_dir = "results"
    if args.rank_label:
        results_dir = f"results/{args.rank_label}"

    slice_mode = all(v is not None for v in [args.N, args.pz, args.sn, args.D_type])
    if slice_mode:
        out_csv = f"{results_dir}/mixed_N{args.N}_pz{args.pz}_sn{args.sn}_D{args.D_type}.csv"
    else:
        out_csv = f"{results_dir}/mixed_results.csv"

    # One row per (dataset, precision); resume on that combined key.
    key_cols    = ["N", "pz", "sn", "D_type", "repeat_id", "precision"]
    summary_key = ["N", "pz", "sn", "D_type", "precision", "optimizer"]
    sweep_csv   = out_csv.replace(".csv", "_sweep.csv")

    os.makedirs(results_dir, exist_ok=True)
    done, write_header = _resume(out_csv, key_cols)
    # In aggregate mode, also absorb previously computed slice CSVs
    if not slice_mode:
        import glob as _glob
        for _p in _glob.glob(f"{results_dir}/mixed_N*.csv"):
            if "_summary" not in _p and "_sweep" not in _p:
                done |= _resume(_p, key_cols)[0]
    sweep_header = not os.path.exists(sweep_csv)

    for N, pz, sn, D_type in tqdm(list(product(N_list, pz_list, sn_list, D_type_list)),
                                   desc="N×pz×sn×D"):
        for rid in tqdm(repeat_ids, desc="repeats", leave=False):
            if (N, pz, sn, D_type, rid, "fp64") in done:
                continue
            data = load_mixed(N, pz, sn, D_type, rid)
            sweep, sel_lr, sel_res = run_dataset(data, n_restarts=args.n_restarts)

            for lr, res in sweep:
                sweep_row = pd.DataFrame([{"N": N, "pz": pz, "sn": sn, "D_type": D_type,
                                           "repeat_id": rid, "precision": "fp64",
                                           "optimizer": "LBFGS", "lr": lr,
                                           **_flat(res)}])
                sweep_row.to_csv(sweep_csv, mode="a", header=sweep_header, index=False)
                sweep_header = False

            row = pd.DataFrame([{"N": N, "pz": pz, "sn": sn, "D_type": D_type,
                                 "repeat_id": rid, "precision": "fp64",
                                 "optimizer": "LBFGS", "lr": sel_lr,
                                 "selection": "min_reml", **_flat(sel_res)}])
            row.to_csv(out_csv, mode="a", header=write_header, index=False)
            write_header = False

    _summarise(out_csv, summary_key)


def _resume(out_csv, key_cols):
    """Return (done_set, write_header).

    key_cols is an ordered list; tuples in done_set follow that same order,
    so callers can check membership with a matching tuple literal.
    If the file exists but is corrupted (header missing or misplaced),
    it is deleted so the next write starts clean.

    Only converged=True rows count as "done" -- a failed attempt should be
    retried (e.g. after a convergence-check fix), not skipped forever. Failed
    rows are pruned from the file here so the retry's row doesn't duplicate them.
    """
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
        print(f"[_resume] {out_csv} is corrupted (missing key columns) — removing.")
        os.remove(out_csv)
        return set(), True
    if "converged" in existing.columns:
        converged = existing["converged"].astype(str).str.lower().isin(("true", "1", "1.0"))
        n_failed = int((~converged).sum())
        if n_failed:
            existing[converged].to_csv(out_csv, index=False)
            print(f"[_resume] {out_csv}: pruned {n_failed} unconverged rows for retry.")
        existing = existing[converged]
    done = set(zip(*(existing[c] for c in key_cols)))
    print(f"Resuming: {len(done)} rows done in {out_csv}")
    return done, existing.empty


def _flat(res):
    out = {}
    for k, v in res.items():
        if isinstance(v, str):
            v = v.replace("\n", " ").replace("\r", " ")
        out[k] = v
    return out


def _summarise(out_csv, group_cols):
    df = pd.read_csv(out_csv, on_bad_lines="skip")
    group_cols = [c for c in group_cols if c in df.columns]

    agg = {
        "n_reps":           ("ise_macro",      "count"),
        "conv_rate":        ("converged",      "mean"),
        "mean_iters":       ("n_iter",         "mean"),
        "mean_time_s":      ("time_s",         "mean"),
        "mean_mem_mb":      ("mem_mb",         "mean"),
        "beta_mse":         ("beta_mse",       "mean"),
        "D_mse":            ("D_mse",          "mean"),
        "sigma2_abs_err":   ("sigma2_abs_err", "mean"),
        "ise_macro":        ("ise_macro",      "mean"),
        "ise_macro_std":    ("ise_macro",      "std"),
        "ise_macro_median": ("ise_macro",      "median"),
        "ise_macro_q25":    ("ise_macro",      lambda s: s.quantile(0.25)),
        "ise_macro_q75":    ("ise_macro",      lambda s: s.quantile(0.75)),
    }
    # Robust headline: median over components per replicate (Doppler-insensitive).
    if "ise_median" in df.columns:
        agg.update({
            "ise_median":        ("ise_median", "mean"),
            "ise_median_std":    ("ise_median", "std"),
            "ise_median_median": ("ise_median", "median"),
            "ise_median_q25":    ("ise_median", lambda s: s.quantile(0.25)),
            "ise_median_q75":    ("ise_median", lambda s: s.quantile(0.75)),
        })
    for col in ("D_frob", "D_stein", "ise_total"):
        if col in df.columns:
            agg[col] = (col, "mean")
    for col in sorted(c for c in df.columns
                      if c.startswith("ise_") and c[4:].isdigit()):
        agg[col] = (col, "mean")

    summary = df.groupby(group_cols).agg(**agg)
    # Monte-Carlo standard error of the across-replicate mean ISE.
    n_sqrt = np.sqrt(summary["n_reps"].clip(lower=1))
    summary["ise_macro_se"] = summary["ise_macro_std"] / n_sqrt
    if "ise_median_std" in summary.columns:
        summary["ise_median_se"] = summary["ise_median_std"] / n_sqrt
    summary = summary.round(5)
    summary_path = out_csv.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path)
    print(f"Saved {out_csv}\nSaved {summary_path}")

    _summarise_by_curve(out_csv, group_cols)


def _summarise_by_curve(out_csv, group_cols):
    """Per-curve ISE table. Because the curve subset is randomized per replicate,
    positional ise_<d> mixes curves across replicates; remap each ise_<d> to its
    actual curve via the recorded curve_ids so ISE can be read by curve label."""
    df = pd.read_csv(out_csv, on_bad_lines="skip")
    if "curve_ids" not in df.columns:
        return
    df = df[df["curve_ids"].notna() & (df["curve_ids"].astype(str) != "")]
    if df.empty:
        return

    ise_cols = [c for c in df.columns if c.startswith("ise_") and c[4:].isdigit()]
    keep = [c for c in group_cols if c in df.columns and c != "optimizer"]
    records = []
    for _, row in df.iterrows():
        cids = str(row["curve_ids"]).split(",")
        for c in ise_cols:
            d = int(c.split("_")[1])
            if d >= len(cids) or pd.isna(row[c]):
                continue
            rec = {k: row[k] for k in keep}
            rec["curve_id"] = int(cids[d])
            rec["ise"] = float(row[c])
            records.append(rec)
    if not records:
        return

    long = pd.DataFrame(records)
    g = [c for c in keep if c != "repeat_id"] + ["curve_id"]
    summ = (long.groupby(g)["ise"]
                .agg(n="count", mean="mean", std="std", median="median")
                .reset_index())
    summ["se"] = summ["std"] / np.sqrt(summ["n"].clip(lower=1))
    summ["curve_label"] = summ["curve_id"].map(
        lambda i: MIXED_SS_CURVE_LABELS[int(i) % len(MIXED_SS_CURVE_LABELS)])
    path = out_csv.replace(".csv", "_by_curve.csv")
    summ.round(5).to_csv(path, index=False)
    print(f"Saved {path}")


def main():
    args = parse_args()
    cfg_file = f"{DATA_DIR}/config_pure.json" if args.type == "pure" else f"{DATA_DIR}/config_mixed.json"
    with open(cfg_file) as f:
        cfg = json.load(f)

    if args.type == "pure":
        run_pure(args, cfg)
    else:
        run_mixed(args, cfg)


if __name__ == "__main__":
    main()
