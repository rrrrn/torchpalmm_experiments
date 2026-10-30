"""
Random-effects scaling experiment: a single smooth (sn=1) with the number of
random effects pz swept upward. Isolates how random-effects covariance (D)
estimation degrades as D grows — unstructured D has pz(pz+1)/2 free params.

Run from: simulation_experiments/randeff_comparison/
    python generate_data.py --N 500
    python generate_data.py            # all N
    python generate_data.py --N 1000 --pz 8
"""

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(_SCRIPT_DIR, "../..")))

# Data directory — overridable so Python and R chains use separate dirs and
# never race on the same JSON files.
DATA_DIR = os.environ.get("RE_DATA_DIR", "data")

import json
from itertools import product

import numpy as np
import torch

from SimScripts.RandeffLMM import generate_randeff_data


RANDEFF_CONFIG = {
    "type":        "randeff",
    "N_list":      [500, 1000],
    "px":          2,
    "pz_list":     [5, 8, 20],
    "sn":          1,                       # single smooth, fixed
    "D_type_list": ["diagonal", "unstructured"],
    "corr_structure_list": ["iid"],         # compound_symmetry dropped (bam is iid-only)
    "sigma_scale": 1.0,
    "lmin":        15,
    "lmax":        15,                      # fixed 15 observations per subject
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


def data_path(N, pz, D_type, corr, rid):
    return f"{DATA_DIR}/randeff_N{N}_pz{pz}_D{D_type}_C{corr}_repeat{rid}.json"


def save_dataset(out_path, x, z, y, splines_vars, subject_id, beta, D, sigma2, rho):
    def to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    y = to_numpy(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    y = y - np.mean(y)
    d_out = {
        "y":            y,
        "X":            to_numpy(x),
        "Z":            to_numpy(z),
        "splines_vars": to_numpy(splines_vars),
        "subject_id":   to_numpy(subject_id),
        "beta":         to_numpy(beta),
        "D":            to_numpy(D),
        "sigma2":       float(sigma2),
        "rho":          float(rho),
    }
    with open(out_path, "w") as f:
        json.dump(d_out, f, cls=NumpyEncoder)


def generate(cfg):
    os.makedirs(DATA_DIR, exist_ok=True)
    combos = list(product(cfg["N_list"], cfg["pz_list"], cfg["D_type_list"],
                          cfg["corr_structure_list"], cfg["repeat_ids"]))
    for N, pz, D_type, corr, rid in combos:
        out = data_path(N, pz, D_type, corr, rid)
        if os.path.exists(out):
            continue
        x, z, y, sv, sid, beta, D, sigma2, rho = generate_randeff_data(
            N=N, px=cfg["px"], pz=pz,
            sigma_scale=cfg["sigma_scale"],
            lmin=cfg["lmin"], lmax=cfg["lmax"],
            repeat_id=rid, rand_D=D_type, corr_structure=corr,
        )
        save_dataset(out, x, z, y, sv, sid, beta, D, sigma2, rho)
        print(f"  randeff N={N}, pz={pz}, D={D_type}, corr={corr}, repeat={rid} -> {out}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--pz", type=int, default=None,
                        help="generate a single pz (override pz_list)")
    parser.add_argument("--reps", type=int, default=None,
                        help="pilot: use only the first --reps repeat_ids")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    rids = RANDEFF_CONFIG["repeat_ids"]
    cfg = {**RANDEFF_CONFIG,
           "N_list": [args.N] if args.N is not None else RANDEFF_CONFIG["N_list"],
           "pz_list": [args.pz] if args.pz is not None else RANDEFF_CONFIG["pz_list"],
           "repeat_ids": rids[:args.reps] if args.reps is not None else rids}
    generate(cfg)
    with open(f"{DATA_DIR}/config_randeff.json", "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()
