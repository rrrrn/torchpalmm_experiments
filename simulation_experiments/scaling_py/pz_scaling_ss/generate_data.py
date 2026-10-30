"""
pz-scaling pilot (our method only): number of random effects pz swept, with
N=1000, px=2, sn=1 fixed. One of two single-axis scaling pilots in
simulation_experiments/scaling_py/ (ss_scaling varies sn; pz_scaling_ss varies pz)
sharing the same default point (sn=1, px=2, pz=2, N=1000) and no R baselines. Isolates how random-effects
covariance (D) estimation degrades as D grows (unstructured D has
pz(pz+1)/2 free params) -- mirrors randeff_comparison's own pz sweep, minus
the R baselines and the corr axis (iid only here).

Uses simscripts.RandeffLMM.generate_randeff_data (sn=1 hardcoded, single
un-standardized true curve f(t)=2*sin(pi*t)), the same generator
randeff_comparison uses.

Run from: simulation_experiments/scaling_py/pz_scaling_ss/
    python generate_data.py                 # all cells/reps
    python generate_data.py --pz 10          # one pz slice
    python generate_data.py --reps 3         # pilot
"""

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(_SCRIPT_DIR, "../..")))

import json
from itertools import product

import numpy as np
import torch

from simscripts.RandeffLMM import generate_randeff_data


N  = 1000    # subjects, fixed
PX = 2        # fixed-effect dim, fixed
SN = 1        # spline dimension, fixed (hardcoded in generate_randeff_data)
LMIN = LMAX = 15    # observations per subject, fixed (balanced)

CONFIG = {
    "N":           N,
    "px":          PX,
    "sn":          SN,
    "pz_list":     [5, 10, 20, 30],
    "D_type_list": ["diagonal", "unstructured"],
    "corr_structure": "iid",
    "sigma_scale": 1.0,
    "lmin":        LMIN, "lmax": LMAX,
    "repeat_ids":  list(range(1, 101)),
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return super().default(obj)


def data_path(pz, D_type, rid):
    return f"data/pz{pz}_D{D_type}_repeat{rid}.json"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pz", type=int, default=None)
    p.add_argument("--D_type", type=str, default=None, choices=["diagonal", "unstructured"])
    p.add_argument("--reps", type=int, default=None, help="pilot: first --reps repeat_ids")
    args = p.parse_args()

    os.makedirs("data", exist_ok=True)
    cfg = dict(CONFIG)
    pz_list = [args.pz] if args.pz is not None else cfg["pz_list"]
    D_list  = [args.D_type] if args.D_type is not None else cfg["D_type_list"]
    reps    = cfg["repeat_ids"][:args.reps] if args.reps is not None else cfg["repeat_ids"]
    cfg = {**cfg, "repeat_ids": reps}

    combos = list(product(pz_list, D_list, reps))
    for pz, D_type, rid in combos:
        out = data_path(pz, D_type, rid)
        if os.path.exists(out):
            continue
        x, z, y, sv, sid, beta, D, sigma2, rho = generate_randeff_data(
            N=N, px=PX, pz=pz, rand_D=D_type, corr_structure=cfg["corr_structure"],
            sigma_scale=cfg["sigma_scale"], lmin=cfg["lmin"], lmax=cfg["lmax"],
            repeat_id=rid,
        )
        y = y.view(-1, 1) if y.dim() == 1 else y
        y = y - torch.mean(y)
        d_out = {
            "y":            y.numpy(),
            "X":            np.asarray(x),
            "Z":            np.asarray(z),
            "splines_vars": np.asarray(sv),
            "subject_id":   np.asarray(sid, dtype=np.int64),
            "beta":         np.asarray(beta),
            "D":            np.asarray(D),
            "sigma2":       float(sigma2),
            "rho":          float(rho),
            "N": N, "px": PX, "pz": pz, "sn": SN, "D_type": D_type,
        }
        with open(out, "w") as f:
            json.dump(d_out, f, cls=NumpyEncoder)
        print(f"  pz={pz} D={D_type} rep={rid} -> {out}")

    with open("data/config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Done. {len(combos)} datasets.")


if __name__ == "__main__":
    main()
