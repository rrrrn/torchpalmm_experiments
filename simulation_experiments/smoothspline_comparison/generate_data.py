"""
Run from: experiments/smoothspline_comparison/
    python generate_data.py
"""

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)

# Data directory — overridable so Python and R chains use separate dirs and
# never race on the same JSON files.
DATA_DIR = os.environ.get("SS_DATA_DIR", "data")

import json
from itertools import product

import numpy as np
import torch

from SimScripts.LMMSS import (generate_pure_ss_data, generate_smoothing_spline_lmm_data,
                              ss_curve_ids, ss_knot_count, MIXED_SS_CURVE_STATS,
                              y_checksum)


PURE_SS_CONFIG = {
    "type":          "pure",
    "N_list":        [100, 500, 5000],
    "scenario_list": ["global", "local"],
    "px":            0,
    "pz":            0,
    "sn":            1,
    "sigma_scale":   1.0,
    "lmin":          15,
    "lmax":          16,
    "repeat_ids":    list(range(1, 21)),
}

MIXED_SS_CONFIG = {
    "type":        "mixed",
    "N_list":      [500, 1000],
    "px":          2,
    "pz_list":     [2],
    "sn_list":     [1, 5, 10, 15],
    "D_type_list": ["diagonal", "unstructured"],
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


def pure_data_path(N, scenario, rid):
    return f"{DATA_DIR}/pure_N{N}_scenario{scenario}_repeat{rid}.json"


def mixed_data_path(N, pz, sn, D_type, rid):
    return f"{DATA_DIR}/mixed_N{N}_pz{pz}_sn{sn}_D{D_type}_repeat{rid}.json"


def save_dataset(out_path, x, z, y, splines_vars, subject_id, beta, D, sigma2, cfg_extra):
    y = y.view(-1, 1)
    y = y - torch.mean(y)

    d_out = {
        "y":           y.numpy(),
        "X":           x.numpy(),
        "Z":           z.numpy(),
        "splines_vars": splines_vars.numpy(),
        "subject_id":  subject_id.numpy(),
        "beta":        beta.numpy(),
        "D":           D.numpy() if isinstance(D, torch.Tensor) else np.array(D),
        "sigma2":      float(sigma2),
        "data_checksum": y_checksum(y.numpy()),
        **cfg_extra,
    }
    with open(out_path, "w") as f:
        json.dump(d_out, f, cls=NumpyEncoder)


def generate_pure(cfg):
    os.makedirs(DATA_DIR, exist_ok=True)
    combos = list(product(cfg["N_list"], cfg["scenario_list"], cfg["repeat_ids"]))
    for N, scenario, rid in combos:
        out = pure_data_path(N, scenario, rid)
        if os.path.exists(out):
            continue
        x, z, y, sv, sid, beta, D, sigma2 = generate_pure_ss_data(
            N=N, scenario=scenario,
            sigma_scale=cfg["sigma_scale"],
            lmin=cfg["lmin"], lmax=cfg["lmax"],
            repeat_id=rid,
        )
        save_dataset(out, x, z, y, sv, sid, beta, D, sigma2,
                     {"scenario": scenario,
                      "K_c": ss_knot_count(N, cfg["lmin"], cfg["lmax"])})
        print(f"  pure N={N}, scenario={scenario}, repeat={rid} -> {out}")


def generate_mixed(cfg):
    os.makedirs(DATA_DIR, exist_ok=True)
    combos = list(product(
        cfg["N_list"], cfg["pz_list"], cfg["sn_list"],
        cfg["D_type_list"], cfg["repeat_ids"],
    ))
    for N, pz, sn, D_type, rid in combos:
        out = mixed_data_path(N, pz, sn, D_type, rid)
        if os.path.exists(out):
            continue
        x, z, y, sv, sid, beta, D, sigma2 = generate_smoothing_spline_lmm_data(
            N=N, px=cfg["px"], pz=pz, spline_dimension=sn,
            sigma_scale=cfg["sigma_scale"],
            lmin=cfg["lmin"], lmax=cfg["lmax"],
            repeat_id=rid, rand_D=D_type,
        )
        # Persist the ground-truth curve subset and its standardization
        # constants so both runners use the exact simulated truth for ISE.
        cids = ss_curve_ids(rid, sn)
        save_dataset(out, x, z, y, sv, sid, beta, D, sigma2,
                     {"curve_ids":   cids,
                      "curve_means": [MIXED_SS_CURVE_STATS[c][0] for c in cids],
                      "curve_stds":  [MIXED_SS_CURVE_STATS[c][1] for c in cids],
                      "K_c":         ss_knot_count(N, cfg["lmin"], cfg["lmax"])})
        print(f"  mixed N={N}, pz={pz}, sn={sn}, D={D_type}, repeat={rid} -> {out}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["pure", "mixed"], required=True)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--sn", type=int, default=None,
                        help="mixed only: generate a single sn (override sn_list)")
    parser.add_argument("--reps", type=int, default=None,
                        help="pilot: use only the first --reps repeat_ids")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    def _reps(cfg):
        rids = cfg["repeat_ids"]
        return rids[:args.reps] if args.reps is not None else rids

    if args.type == "pure":
        cfg = {**PURE_SS_CONFIG,
               "N_list": [args.N] if args.N is not None else PURE_SS_CONFIG["N_list"],
               "repeat_ids": _reps(PURE_SS_CONFIG)}
        generate_pure(cfg)
        with open(f"{DATA_DIR}/config_pure.json", "w") as f:
            json.dump(cfg, f, indent=2)
    else:
        cfg = {**MIXED_SS_CONFIG,
               "N_list": [args.N] if args.N is not None else MIXED_SS_CONFIG["N_list"],
               "sn_list": [args.sn] if args.sn is not None else MIXED_SS_CONFIG["sn_list"],
               "repeat_ids": _reps(MIXED_SS_CONFIG)}
        generate_mixed(cfg)
        with open(f"{DATA_DIR}/config_mixed.json", "w") as f:
            json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()
