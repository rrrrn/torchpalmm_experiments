"""
Remove failed bam rows (D_type=diagonal, pz=1, fail_type='other') from R result CSVs
so the R runner will attempt those reps again on the next submission.

fail_type='other': converged=FALSE AND time_s not NA AND ise_macro is NA
  (i.e. bam ran but crashed without producing estimates -- not a timeout)

Run from: experiments/randeff_comparison/
    uv run python clean_bam_failures.py [--dry-run]
"""
import argparse
import glob
import os

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)


def _is_other_failure(df):
    conv = df["converged"].astype(str).str.upper().isin(["TRUE", "1"])
    return (
        ~conv
        & df["time_s"].notna()
        & df["ise_macro"].isna()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be removed without writing")
    args = parser.parse_args()

    csvs = sorted(glob.glob("results/r_randeff_*.csv"))
    if not csvs:
        print("No R result CSVs found in results/.")
        return

    total_removed = 0
    for path in csvs:
        df = pd.read_csv(path, on_bad_lines="skip")
        if df.empty:
            continue

        df["pz"] = pd.to_numeric(df["pz"], errors="coerce")

        mask = (
            (df["method"] == "bam")
            & (df["D_type"] == "diagonal")
            & (df["pz"] == 1)
            & _is_other_failure(df)
        )
        n = int(mask.sum())
        if n == 0:
            continue

        reps = sorted(df.loc[mask, "repeat_id"].unique())
        print(f"  {os.path.basename(path)}: {n} row(s) to remove  repeat_ids={reps}")
        total_removed += n

        if not args.dry_run:
            df[~mask].to_csv(path, index=False)

    if total_removed == 0:
        print("Nothing matched. No changes made.")
    elif args.dry_run:
        print(f"\n[dry-run] Would remove {total_removed} row(s). Re-run without --dry-run to apply.")
    else:
        print(f"\nRemoved {total_removed} row(s). Resubmit the R chain to fill them in.")


if __name__ == "__main__":
    main()
