"""
Refit the largest cells of simulation_experiments/scaling_py/ss_scaling (sn=10, 30) on one device.

Reuses each pilot's own run_python.py (load_data + run_one + OPTIMIZER_CONFIGS)
so the datasets, seeds, optimizer/lr grid, tolerances and metric code are
identical to the CPU rows this is compared against; only the device changes.
SmoothingSplineMixed(device=None) resolves to CUDA when one is visible, so the
device is chosen by what the job can see, not by a separate code path.

Run from: simulation_experiments/gpu_cpu_comparison/
    python run_gpu.py --device gpu --reps 20
    python run_gpu.py --device cpu --reps 5 --cell sn30    # same-node control
"""

import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(_SCRIPT_DIR, "../..")))

import argparse
import importlib.util
import time

import pandas as pd
import torch
from tqdm import tqdm

_PILOT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "../scaling_py"))

# (cell, pilot subfolder, axis column, extreme value)
CELLS = [
    ("sn10",  "ss_scaling",    "sn", 10),
    ("sn30",  "ss_scaling",    "sn", 30),
]
D_TYPES = ["diagonal", "unstructured"]


def _import_pilot(name):
    """Import <pilot>/run_python.py as its own module. It chdir's into its own
    folder at import time, which is what makes its relative data/ paths resolve;
    cwd is restored by the caller."""
    path = os.path.join(_PILOT_DIR, name, "run_python.py")
    spec = importlib.util.spec_from_file_location(f"_gpu_cmp_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _device_info(device):
    if device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device gpu but torch.cuda.is_available() is False. Submit to a "
                "partition with --gres=gpu:1, or use --device cpu."
            )
        return {"device": "gpu",
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda}
    if torch.cuda.is_available():
        raise RuntimeError(
            "--device cpu but a GPU is visible; SmoothingSplineMixed would silently "
            "use it. Re-run with CUDA_VISIBLE_DEVICES='' to hide it."
        )
    return {"device": "cpu", "gpu_name": "", "cuda_version": ""}


def _sync(device):
    if device == "gpu":
        torch.cuda.synchronize()


def _timed_fit(mod, data, opt_cls, lr, device):
    """Pilot's own run_one, wrapped in a synchronized outer timer and GPU peak-memory
    accounting. time_s_outer is authoritative on GPU: the pilot's internal time_s uses
    a bare time.time() around async kernel launches."""
    if device == "gpu":
        torch.cuda.reset_peak_memory_stats()
    _sync(device)
    t0 = time.time()
    res = mod.run_one(data, opt_cls, lr)
    _sync(device)
    res["time_s_outer"] = time.time() - t0
    res["gpu_mem_mb"] = (torch.cuda.max_memory_allocated() / 1024**2
                         if device == "gpu" else float("nan"))
    return res


def _resume(out_csv, key_cols):
    if not os.path.exists(out_csv):
        return set(), True
    try:
        existing = pd.read_csv(out_csv, on_bad_lines="skip")
    except Exception:
        existing = pd.DataFrame()
    if not set(key_cols).issubset(existing.columns):
        print(f"[_resume] {out_csv} corrupted -- removing.")
        os.remove(out_csv)
        return set(), True
    done = set(zip(*(existing[c] for c in key_cols)))
    print(f"Resuming: {len(done)} rows done in {out_csv}")
    return done, False


def _flat(res):
    return {k: (v.replace("\n", " ").replace("\r", " ") if isinstance(v, str) else v)
            for k, v in res.items()}


def run_cell(cell, pilot, axis, value, args, info):
    mod = _import_pilot(pilot)                 # chdir's into the pilot folder
    optim_configs = mod.OPTIMIZER_CONFIGS
    repeat_ids = list(range(1, args.reps + 1))
    d_types = [args.D_type] if args.D_type else D_TYPES

    os.chdir(_SCRIPT_DIR)
    os.makedirs("results", exist_ok=True)
    out_csv = f"results/{cell}_{args.device}{args.out_suffix}.csv"
    key_cols = ["D_type", "repeat_id", "optimizer", "lr"]
    done, write_header = _resume(out_csv, key_cols)

    warmed = not args.warmup
    for D_type in d_types:
        for rid in tqdm(repeat_ids, desc=f"{cell}/{D_type}", leave=False):
            os.chdir(os.path.join(_PILOT_DIR, pilot))
            try:
                data = mod.load_data(value, D_type, rid)
            except FileNotFoundError:
                os.chdir(_SCRIPT_DIR)
                print(f"[{cell}] missing data for {D_type} rep {rid}; run the pilot's "
                      f"generate_data.py for {axis}={value} first.")
                continue

            if not warmed:
                # Absorb CUDA context init / cuBLAS handle setup outside the timed loop.
                _timed_fit(mod, data, optim_configs[0][1], optim_configs[0][2], args.device)
                warmed = True

            for opt_name, opt_cls, lr in optim_configs:
                if (D_type, rid, opt_name, lr) in done:
                    continue
                res = _timed_fit(mod, data, opt_cls, lr, args.device)
                os.chdir(_SCRIPT_DIR)
                row = pd.DataFrame([{
                    "cell": cell, "pilot": pilot, "axis": axis, "axis_value": value,
                    "D_type": D_type, "repeat_id": rid,
                    "optimizer": opt_name, "lr": lr,
                    **info, **_flat(res),
                }])
                row.to_csv(out_csv, mode="a", header=write_header, index=False)
                write_header = False
                os.chdir(os.path.join(_PILOT_DIR, pilot))
            os.chdir(_SCRIPT_DIR)
    print(f"Saved {out_csv}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--cell", choices=[c[0] for c in CELLS], default=None)
    p.add_argument("--D_type", choices=D_TYPES, default=None)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--out-suffix", default="",
                   help="append to the results filename (e.g. _u) so a parallel job never writes the same CSV")
    p.add_argument("--no-warmup", dest="warmup", action="store_false",
                   help="skip the discarded warm-up fit (GPU timings will be inflated)")
    return p.parse_args()


def main():
    args = parse_args()
    info = _device_info(args.device)
    print(f"[gpu_cpu_comparison] device={info['device']} "
          f"gpu={info['gpu_name'] or 'n/a'} torch={torch.__version__}")
    cells = [c for c in CELLS if args.cell is None or c[0] == args.cell]
    for cell, pilot, axis, value in cells:
        run_cell(cell, pilot, axis, value, args, info)


if __name__ == "__main__":
    main()
