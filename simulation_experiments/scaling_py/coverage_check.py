"""
Coverage check used by slurm/py_scaling_all.sh: for one scaling_py pilot's
swept column, reports which candidate values already have >= --target
converged reps for EVERY requested D_type.

Reuses that pilot's own analyze_results.py _load() (via this directory's
combined analyze_results.py _load_pilot helper) -- the same local-only load,
best-lr dedup, and _KEEP filtering that produces scaling_overview.png /
scaling_progress.png. So "covered" here means exactly what "covered" means
in those figures, unlike the old file-existence check (>=1 CSV file, however
few reps it holds, counted as "done") that let e.g. G=100000 sit at ~25/100
converged reps indefinitely without ever being resubmitted. Each pilot's
_load() only reads its own results/ (no sibling-experiment fallback merge
anymore), so a value with zero local reps always reports missing.

Usage:
    python coverage_check.py <pilot> <swept_col> <target> \
        --d-types D1 [D2 ...] --values V1 [V2 ...]

Prints exactly two lines:
    COVERED: v1 v2 ...
    MISSING: v3 v4 ...
"""
import argparse
import importlib.util
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_combined():
    """Import this directory's own analyze_results.py (the task-boxplot
    generator) purely to reuse its _load_pilot -- avoids
    re-implementing the local/fallback merge + dedup a second time."""
    path = os.path.join(_SCRIPT_DIR, "analyze_results.py")
    spec = importlib.util.spec_from_file_location("_scaling_py_combined_for_coverage", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot")
    ap.add_argument("swept_col")
    ap.add_argument("target", type=int)
    ap.add_argument("--d-types", nargs="+", required=True)
    ap.add_argument("--values", nargs="+", required=True)
    args = ap.parse_args()

    combined = _load_combined()
    df = combined._load_pilot(args.pilot)
    os.chdir(_SCRIPT_DIR)

    covered, missing = [], []
    for v_str in args.values:
        v = float(v_str)
        ok = df is not None and not df.empty and "converged" in df.columns
        if ok:
            for d in args.d_types:
                n = int(((df[args.swept_col] == v) & (df["D_type"] == d) & df["converged"]).sum())
                if n < args.target:
                    ok = False
                    break
        (covered if ok else missing).append(v_str)

    print("COVERED: " + " ".join(covered))
    print("MISSING: " + " ".join(missing))


if __name__ == "__main__":
    main()
