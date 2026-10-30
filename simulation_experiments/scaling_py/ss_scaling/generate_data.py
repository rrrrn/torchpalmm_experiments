"""
sn-scaling pilot (our method only): number of additive smooth components sn
swept, with N=1000, px=2, pz=2 fixed. One of two single-axis scaling pilots
in experiments/scaling_py/ (ss_scaling varies sn; pz_scaling_ss varies pz)
sharing the same default point (sn=1, px=2, pz=2, N=1000) and no R baselines.

Each of the sn smooth components uses its own independently-sampled t and its
own true curve, drawn per replicate from SimScripts.LMMSS.MIXED_SS_CURVES via
SimScripts.LMMSS.ss_curve_ids(repeat_id, sn) (curves standardized to zero
mean/unit variance, so growing sn changes model dimension without also
changing per-observation SNR) -- this is the one sub-experiment that cannot
reuse SimScripts.RandeffLMM.generate_randeff_data (hardcoded to sn=1).

Run from: experiments/scaling_py/ss_scaling/
    python generate_data.py                 # all cells/reps
    python generate_data.py --sn 10          # one sn slice
    python generate_data.py --reps 3         # pilot
"""

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(_SCRIPT_DIR, "../../..")))

import json
from itertools import product

import numpy as np
import torch

from SimScripts.LMMSS import (generate_smoothing_spline_lmm_data, ss_curve_ids,
                              ss_knot_count, MIXED_SS_CURVE_STATS, y_checksum)


N  = 1000    # subjects, fixed
PX = 2        # fixed-effect dim, fixed
PZ = 2        # random-effect dim, fixed
LMIN = LMAX = 15    # observations per subject, fixed (balanced)

CONFIG = {
    "N":           N,
    "px":          PX,
    "pz":          PZ,
    "sn_list":     [5, 10, 20, 30],
    "D_type_list": ["diagonal", "unstructured"],
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


def data_path(sn, D_type, rid):
    return f"data/sn{sn}_D{D_type}_repeat{rid}.json"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sn", type=int, default=None)
    p.add_argument("--D_type", type=str, default=None, choices=["diagonal", "unstructured"])
    p.add_argument("--reps", type=int, default=None, help="pilot: first --reps repeat_ids")
    args = p.parse_args()

    os.makedirs("data", exist_ok=True)
    cfg = dict(CONFIG)
    sn_list = [args.sn] if args.sn is not None else cfg["sn_list"]
    D_list  = [args.D_type] if args.D_type is not None else cfg["D_type_list"]
    reps    = cfg["repeat_ids"][:args.reps] if args.reps is not None else cfg["repeat_ids"]
    cfg = {**cfg, "repeat_ids": reps}

    combos = list(product(sn_list, D_list, reps))
    for sn, D_type, rid in combos:
        out = data_path(sn, D_type, rid)
        if os.path.exists(out):
            continue
        x, z, y, sv, sid, beta, D, sigma2 = generate_smoothing_spline_lmm_data(
            N=N, px=PX, pz=PZ, spline_dimension=sn,
            sigma_scale=cfg["sigma_scale"],
            lmin=cfg["lmin"], lmax=cfg["lmax"],
            repeat_id=rid, rand_D=D_type,
        )
        y = y.view(-1, 1)
        y = y - torch.mean(y)
        cids = ss_curve_ids(rid, sn)
        d_out = {
            "y":            y.numpy(),
            "X":            x.numpy(),
            "Z":            z.numpy(),
            "splines_vars": sv.numpy(),
            "subject_id":   sid.numpy(),
            "beta":         beta.numpy(),
            "D":            D.numpy() if isinstance(D, torch.Tensor) else np.array(D),
            "sigma2":       float(sigma2),
            "data_checksum": y_checksum(y.numpy()),
            "curve_ids":    cids,
            "curve_means":  [MIXED_SS_CURVE_STATS[c][0] for c in cids],
            "curve_stds":   [MIXED_SS_CURVE_STATS[c][1] for c in cids],
            "K_c":          ss_knot_count(N, LMIN, LMAX),
            "N": N, "px": PX, "pz": PZ, "sn": sn, "D_type": D_type,
        }
        with open(out, "w") as f:
            json.dump(d_out, f, cls=NumpyEncoder)
        print(f"  sn={sn} D={D_type} rep={rid} -> {out}")

    with open("data/config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Done. {len(combos)} datasets.")


if __name__ == "__main__":
    main()
