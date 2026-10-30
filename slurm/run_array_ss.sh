#!/bin/bash
# Smoothing-spline mixed-model comparison (ours vs mgcv::gamm/bam), pure + mixed.
#
# One script for the whole comparison, with a --lang py/r/both switch like
# run_array_randeff.sh.
# Python and R chains use separate data dirs (data_py/data_r) so they never
# race on the same JSON files.
#
# The mixed-chain R fit submits gamm and bam as SEPARATE, parallel array jobs
# (--only_bam / --only_gamm, each writing its own suffixed csv) rather than
# one job fitting both sequentially -- bam is markedly slower than gamm at
# larger sn, so sharing one job meant bam's slow reps could
# starve gamm's later cells of wall-clock, mirroring run_array_randeff.sh.
# Each job's skip-set is seeded from every results/r_mixed*.csv on disk, so
# already-generated results (including pre-split combined files) are never
# re-fit. --only restricts this pair down to a single job.
#
# Usage (from project root):
#   bash slurm/run_array_ss.sh                          # full run, both languages
#   bash slurm/run_array_ss.sh --type mixed             # mixed chain only
#   bash slurm/run_array_ss.sh --lang py                # Python chain only
#   bash slurm/run_array_ss.sh --lang r                 # R chain: gamm + bam as SEPARATE array jobs
#   bash slurm/run_array_ss.sh --lang r --only bam      # R chain, bam only
#   bash slurm/run_array_ss.sh --lang r --only gamm     # R chain, gamm only
#   bash slurm/run_array_ss.sh --reps 3                 # pilot: 3 repeats per param set
#   bash slurm/run_array_ss.sh --N 500                  # restrict to one N
#   bash slurm/run_array_ss.sh --rank_label rank30      # one-off ablation: disposable data, labeled output
#   bash slurm/run_array_ss.sh --rerun_bam --N 500      # recovery: strip old bam rows, refit bam only (diagonal)
#
# Data is kept after a normal run (generate_data.py skips datasets that already
# exist). --rank_label and --rerun_bam runs use disposable data instead,
# deleted once the (labeled) analyze step finishes.

TYPE="both"; LANG="both"; ONLY=""; REPS=""; RANK_LABEL=""; RERUN_BAM=""; N_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --type)       TYPE="$2";       shift 2 ;;
    --lang)       LANG="$2";       shift 2 ;;
    --only)       ONLY="$2";       shift 2 ;;   # bam | gamm -- mixed R chain only
    --reps)       REPS="$2";       shift 2 ;;
    --rank_label) RANK_LABEL="$2"; shift 2 ;;
    --rerun_bam)  RERUN_BAM="1";   shift 1 ;;
    --N)          N_OVERRIDE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

REPS_ARG=""; [[ -n "$REPS" ]] && REPS_ARG="--reps $REPS"
RANK_LABEL_ARG=""; [[ -n "$RANK_LABEL" ]] && RANK_LABEL_ARG="--rank_label $RANK_LABEL"

# R methods for the mixed chain: both, unless --only restricts to one.
R_METHODS=(gamm bam)
[[ "$ONLY" == "bam"  ]] && R_METHODS=(bam)
[[ "$ONLY" == "gamm" ]] && R_METHODS=(gamm)

# Factor grid -- keep in sync with experiments/smoothspline_comparison/generate_data.py CONFIG.
N_LIST_PURE=(100 500 5000)
N_LIST_MIXED=(500 1000)
if [[ -n "$N_OVERRIDE" ]]; then N_LIST_PURE=("$N_OVERRIDE"); N_LIST_MIXED=("$N_OVERRIDE"); fi
PZ_LIST=(2)
SN_LIST=(1 5 10 15)
D_TYPE_LIST=(diagonal unstructured)

PARTITION="standard"; R_MODULE="R"
ACCOUNT="${SLURM_ACCOUNT:?export SLURM_ACCOUNT=<your SLURM account>}"

PROJECT_ROOT="$(pwd)"
bash "$PROJECT_ROOT/slurm/check_torchpalmm.sh" || exit 1
EXP_DIR="$PROJECT_ROOT/experiments/smoothspline_comparison"
LOG_DIR="$PROJECT_ROOT/slurm/logs_ss"
mkdir -p "$LOG_DIR"

# rank-label ablations and bam recovery use disposable data (fresh-generated,
# deleted after); the main comparison keeps its data across runs.
DISPOSABLE=0
[[ -n "$RANK_LABEL" || -n "$RERUN_BAM" ]] && DISPOSABLE=1

ALL_FIT_IDS=()

# ---------------------------------------------------------------------------
# --rerun_bam recovery mode: strip stale bam rows from existing R results,
# regenerate diagonal-only mixed data (bam never runs unstructured), and
# refit bam only. Short-circuits everything below.
# ---------------------------------------------------------------------------
if [[ -n "$RERUN_BAM" ]]; then
  DATA_DIR="data_r_bam"

  python3 - <<PYEOF
import pandas as pd, glob, os
results_dir = "$EXP_DIR/results"
n_stripped = 0
for f in sorted(glob.glob(f"{results_dir}/r_mixed_*.csv")):
    df = pd.read_csv(f)
    if "method" in df.columns and (df["method"] == "bam").any():
        cleaned = df[df["method"] != "bam"].reset_index(drop=True)
        cleaned.to_csv(f, index=False)
        removed = len(df) - len(cleaned)
        print(f"  stripped {removed} bam rows from {os.path.basename(f)}")
        n_stripped += removed
print(f"Total bam rows removed: {n_stripped}")
PYEOF

  for N in "${N_LIST_MIXED[@]}"; do
    MAPPING="$LOG_DIR/rerun_bam_mapping_N${N}.txt"; : > "$MAPPING"; IDX=0
    for SN in "${SN_LIST[@]}"; do echo "$IDX $SN" >> "$MAPPING"; IDX=$((IDX + 1)); done
    MAX_IDX=$((IDX - 1))

    JID_GEN=$(sbatch --parsable --job-name=ss_bam_gen_N${N} \
      --output="$LOG_DIR/%j_bam_gen_N${N}.out" --error="$LOG_DIR/%j_bam_gen_N${N}.err" \
      --time=01:00:00 --mem=20G --cpus-per-task=1 --account="$ACCOUNT" --partition="$PARTITION" \
      --wrap="cd $PROJECT_ROOT; export SS_DATA_DIR=$DATA_DIR
        rm -f $EXP_DIR/$DATA_DIR/mixed_N${N}_*_Ddiagonal_*.json
        uv run python $EXP_DIR/generate_data.py --type mixed --N $N $REPS_ARG")

    JID_FIT=$(sbatch --parsable --dependency=afterok:${JID_GEN} --job-name=ss_bam_fit_N${N} \
      --output="$LOG_DIR/%A_%a_bam_fit_N${N}.out" --error="$LOG_DIR/%A_%a_bam_fit_N${N}.err" \
      --array="0-${MAX_IDX}" --time=24:00:00 --mem=30G --cpus-per-task=1 \
      --account="$ACCOUNT" --partition="$PARTITION" \
      --wrap="module load $R_MODULE; cd $PROJECT_ROOT; export SS_DATA_DIR=$DATA_DIR
        LINE=\$(awk -v idx=\$SLURM_ARRAY_TASK_ID '\$1==idx{print}' $MAPPING)
        SN=\$(echo \$LINE | awk '{print \$2}')
        Rscript $EXP_DIR/run_r_baselines.R --type mixed --N $N --sn \$SN --D_type diagonal --only_bam")

    echo "rerun_bam N=$N: gen=$JID_GEN fit=$JID_FIT"
    ALL_FIT_IDS+=("$JID_FIT")
  done

  DEP_DONE="afterany:$(IFS=:; echo "${ALL_FIT_IDS[*]}")"
  sbatch --parsable --dependency=${DEP_DONE} --job-name=ss_bam_cleanup \
    --output="$LOG_DIR/%j_cleanup.out" --error="$LOG_DIR/%j_cleanup.err" \
    --time=00:10:00 --mem=2G --cpus-per-task=1 --account="$ACCOUNT" --partition="$PARTITION" \
    --wrap="rm -rf $EXP_DIR/$DATA_DIR && echo 'Deleted bam rerun data'"
  echo "Monitor: squeue -u \$USER"
  exit 0
fi

# ---------------------------------------------------------------------------
# One language chain. $1 = tag (py|r), $2 = data dir.
# ---------------------------------------------------------------------------
submit_chain() {
  local TAG="$1" DDIR="$2"
  local JID_GEN JID_FIT GEN_PRE FIT_PRE FIT_CMD FIT_TIME FIT_MEM

  # ---- pure chain ----
  if [[ "$TYPE" == "both" || "$TYPE" == "pure" ]]; then
    for N in "${N_LIST_PURE[@]}"; do
      GEN_PRE=""
      [[ "$DISPOSABLE" == "1" ]] && GEN_PRE="rm -f $EXP_DIR/$DDIR/pure_N${N}_*.json"
      JID_GEN=$(sbatch --parsable --job-name=ss_${TAG}_pure_gen_N${N} \
        --output="$LOG_DIR/%j_${TAG}_pure_gen_N${N}.out" --error="$LOG_DIR/%j_${TAG}_pure_gen_N${N}.err" \
        --time=00:30:00 --mem=10G --cpus-per-task=1 --account="$ACCOUNT" --partition="$PARTITION" \
        --wrap="cd $PROJECT_ROOT; export SS_DATA_DIR=$DDIR; $GEN_PRE
          uv run python $EXP_DIR/generate_data.py --type pure --N $N $REPS_ARG")

      if [[ "$TAG" == "py" ]]; then
        FIT_PRE=""; FIT_TIME="12:00:00"; FIT_MEM="20G"
        FIT_CMD="uv run python $EXP_DIR/run_python.py --type pure --N $N $REPS_ARG $RANK_LABEL_ARG"
      else
        FIT_PRE="module load $R_MODULE; "; FIT_TIME="04:00:00"; FIT_MEM="8G"
        FIT_CMD="Rscript $EXP_DIR/run_r_baselines.R --type pure --N $N"
      fi
      JID_FIT=$(sbatch --parsable --dependency=afterok:${JID_GEN} --job-name=ss_${TAG}_pure_fit_N${N} \
        --output="$LOG_DIR/%j_${TAG}_pure_fit_N${N}.out" --error="$LOG_DIR/%j_${TAG}_pure_fit_N${N}.err" \
        --time=$FIT_TIME --mem=$FIT_MEM --cpus-per-task=1 --account="$ACCOUNT" --partition="$PARTITION" \
        --wrap="${FIT_PRE}cd $PROJECT_ROOT; export SS_DATA_DIR=$DDIR
          $FIT_CMD")
      echo "$TAG pure N=$N: gen=$JID_GEN fit=$JID_FIT"
      ALL_FIT_IDS+=("$JID_FIT")
    done
  fi

  # ---- mixed chain ----
  if [[ "$TYPE" == "both" || "$TYPE" == "mixed" ]]; then
    for N in "${N_LIST_MIXED[@]}"; do
      local MAPPING="$LOG_DIR/${TAG}_mixed_mapping_N${N}.txt"; : > "$MAPPING"; local IDX=0
      for PZ in "${PZ_LIST[@]}"; do for SN in "${SN_LIST[@]}"; do for D_TYPE in "${D_TYPE_LIST[@]}"; do
        echo "$IDX $PZ $SN $D_TYPE" >> "$MAPPING"; IDX=$((IDX + 1))
      done; done; done
      local MAX_IDX=$((IDX - 1))

      GEN_PRE=""
      [[ "$DISPOSABLE" == "1" ]] && GEN_PRE="rm -f $EXP_DIR/$DDIR/mixed_N${N}_*.json"
      JID_GEN=$(sbatch --parsable --job-name=ss_${TAG}_mixed_gen_N${N} \
        --output="$LOG_DIR/%j_${TAG}_mixed_gen_N${N}.out" --error="$LOG_DIR/%j_${TAG}_mixed_gen_N${N}.err" \
        --time=01:00:00 --mem=20G --cpus-per-task=1 --account="$ACCOUNT" --partition="$PARTITION" \
        --wrap="cd $PROJECT_ROOT; export SS_DATA_DIR=$DDIR; $GEN_PRE
          uv run python $EXP_DIR/generate_data.py --type mixed --N $N $REPS_ARG")

      FIT_MEM="30G"; [[ "$N" -ge 5000 ]] && FIT_MEM="60G"   # heavier for large N, if ever restored
      local PARSE="LINE=\$(awk -v idx=\$SLURM_ARRAY_TASK_ID '\$1==idx{print}' $MAPPING); PZ=\$(echo \$LINE|awk '{print \$2}'); SN=\$(echo \$LINE|awk '{print \$3}'); D_TYPE=\$(echo \$LINE|awk '{print \$4}')"
      if [[ "$TAG" == "py" ]]; then
        FIT_CMD="$PARSE; uv run python $EXP_DIR/run_python.py --type mixed --N $N --pz \$PZ --sn \$SN --D_type \$D_TYPE $REPS_ARG $RANK_LABEL_ARG"
        JID_FIT=$(sbatch --parsable --dependency=afterok:${JID_GEN} --job-name=ss_${TAG}_mixed_fit_N${N} \
          --output="$LOG_DIR/%A_%a_${TAG}_mixed_fit_N${N}.out" --error="$LOG_DIR/%A_%a_${TAG}_mixed_fit_N${N}.err" \
          --array="0-${MAX_IDX}" --time=24:00:00 --mem=$FIT_MEM --cpus-per-task=1 \
          --account="$ACCOUNT" --partition="$PARTITION" \
          --wrap="cd $PROJECT_ROOT; export SS_DATA_DIR=$DDIR
            $FIT_CMD")
        echo "$TAG mixed N=$N: gen=$JID_GEN fit=$JID_FIT"
        ALL_FIT_IDS+=("$JID_FIT")
      else
        for RM in "${R_METHODS[@]}"; do
          RM_ARG="--only_gamm"; [[ "$RM" == "bam" ]] && RM_ARG="--only_bam"
          FIT_CMD="$PARSE; Rscript $EXP_DIR/run_r_baselines.R --type mixed --N $N --sn \$SN --D_type \$D_TYPE $RM_ARG"
          JID_FIT=$(sbatch --parsable --dependency=afterok:${JID_GEN} --job-name=ss_${TAG}_${RM}_mixed_fit_N${N} \
            --output="$LOG_DIR/%A_%a_${TAG}_${RM}_mixed_fit_N${N}.out" --error="$LOG_DIR/%A_%a_${TAG}_${RM}_mixed_fit_N${N}.err" \
            --array="0-${MAX_IDX}" --time=24:00:00 --mem=$FIT_MEM --cpus-per-task=1 \
            --account="$ACCOUNT" --partition="$PARTITION" \
            --wrap="module load $R_MODULE; cd $PROJECT_ROOT; export SS_DATA_DIR=$DDIR
              $FIT_CMD")
          echo "$TAG $RM mixed N=$N: gen=$JID_GEN fit=$JID_FIT"
          ALL_FIT_IDS+=("$JID_FIT")
        done
      fi
    done
  fi
}

[[ "$LANG" == "both" || "$LANG" == "py" ]] && submit_chain py data_py
[[ "$LANG" == "both" || "$LANG" == "r"  ]] && submit_chain r  data_r

# ---------------------------------------------------------------------------
# Analyze — waits for all fit jobs across both chains.
# ---------------------------------------------------------------------------
DEP_ANALYZE="afterok:$(IFS=:; echo "${ALL_FIT_IDS[*]}")"
JID_ANALYZE=$(sbatch --parsable --dependency=${DEP_ANALYZE} --job-name=ss_analyze \
  --output="$LOG_DIR/%j_analyze.out" --error="$LOG_DIR/%j_analyze.err" \
  --time=00:30:00 --mem=8G --cpus-per-task=1 --account="$ACCOUNT" --partition="$PARTITION" \
  --wrap="cd $PROJECT_ROOT && uv run python $EXP_DIR/analyze_results.py $RANK_LABEL_ARG")

if [[ "$DISPOSABLE" == "1" ]]; then
  sbatch --parsable --dependency=afterok:${JID_ANALYZE} --job-name=ss_final_cleanup \
    --output="$LOG_DIR/%j_cleanup.out" --error="$LOG_DIR/%j_cleanup.err" \
    --time=00:10:00 --mem=2G --cpus-per-task=1 --account="$ACCOUNT" --partition="$PARTITION" \
    --wrap="rm -rf $EXP_DIR/data_py $EXP_DIR/data_r && echo 'Deleted disposable data'"
fi

echo "analyze=$JID_ANALYZE"
echo "Monitor: squeue -u \$USER"
