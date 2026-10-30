#!/bin/bash
# Random-effects scaling experiment (sn=1, pz swept).
# Submits the Python and R chains concurrently (separate data dirs, no race),
# then an analyze job that waits on both.
#
# Usage (from project root):
#   bash slurm/run_array_randeff.sh                 # full run
#   bash slurm/run_array_randeff.sh --reps 3        # pilot: 3 repeats per config
#   bash slurm/run_array_randeff.sh --lang py       # Python chain only
#   bash slurm/run_array_randeff.sh --lang r        # R chain only (gamm + bam, separate jobs)
#
# Within each chain, for each N in sequence:
#   generate(N) -> fit_array(pz × D_type) -> next N
# Data and results are both reused across runs; nothing is deleted.
#
# The R chain submits gamm and bam as SEPARATE, parallel array jobs per N
# rather than one job fitting both sequentially (run_r_baselines.R --only,
# writing its own suffixed CSV) -- bam is markedly slower than gamm at larger
# pz, so sharing one job meant bam's slow reps could starve
# gamm's later cells of wall-clock. Each job's skip-set is seeded from every
# results/r_randeff*.csv on disk, so already-generated results (including the
# pre-split combined file) are never re-fit.
#
LANG="both"
REPS=""
RHO_INIT="moment"
while [[ $# -gt 0 ]]; do
  case $1 in
    --lang)     LANG="$2";     shift 2 ;;
    --reps)     REPS="$2";     shift 2 ;;
    --rho_init) RHO_INIT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
REPS_ARG=""
[[ -n "$REPS" ]] && REPS_ARG="--reps $REPS"
RHO_ARG="--rho_init $RHO_INIT"

N_LIST=(500 1000)
PZ_LIST=(5 8 20)
D_TYPE_LIST=(diagonal unstructured)
# compound_symmetry dropped: bam runs iid only, so CS cells could not put all
# three methods on comparable footing. Add it back here and in _CORR_KEEP in
# experiments/randeff_comparison/analyze_results.py together.
CORR_LIST=(iid)

PARTITION="standard"
ACCOUNT="${SLURM_ACCOUNT:?export SLURM_ACCOUNT=<your SLURM account>}"
R_MODULE="R"

PROJECT_ROOT="$(pwd)"
bash "$PROJECT_ROOT/slurm/check_torchpalmm.sh" || exit 1
EXP_DIR="$PROJECT_ROOT/experiments/randeff_comparison"
LOG_DIR="$PROJECT_ROOT/slurm/logs_randeff"
mkdir -p "$LOG_DIR"

# Build the pz × D_type × corr mapping shared by both chains' fit arrays.
MAPPING="$LOG_DIR/randeff_mapping.txt"
: > "$MAPPING"
IDX=0
for PZ in "${PZ_LIST[@]}"; do
  for D_TYPE in "${D_TYPE_LIST[@]}"; do
    for CORR in "${CORR_LIST[@]}"; do
      echo "$IDX $PZ $D_TYPE $CORR" >> "$MAPPING"
      IDX=$((IDX + 1))
    done
  done
done
MAX_IDX=$((IDX - 1))

# ---------------------------------------------------------------------------
# Python chain.  For each N: generate -> one fit-array job (pz x D_type x corr).
# ---------------------------------------------------------------------------
submit_py_chain() {
  local DDIR="$1"
  local JID_GEN JID_FIT
  local FIT_IDS=()

  for N in "${N_LIST[@]}"; do
    GEN_CMD="export RE_DATA_DIR=$DDIR; uv run python $EXP_DIR/generate_data.py --N $N $REPS_ARG"
    FIT_CMD="export RE_DATA_DIR=$DDIR; LINE=\$(awk -v idx=\$SLURM_ARRAY_TASK_ID '\$1==idx{print}' $MAPPING); PZ=\$(echo \$LINE|awk '{print \$2}'); DT=\$(echo \$LINE|awk '{print \$3}'); CR=\$(echo \$LINE|awk '{print \$4}'); uv run python $EXP_DIR/run_python.py --N $N --pz \$PZ --D_type \$DT --corr \$CR $REPS_ARG $RHO_ARG"

    JID_GEN=$(sbatch --parsable \
      --job-name=re_py_gen_N${N} \
      --output="$LOG_DIR/%j_py_gen_N${N}.out" \
      --error="$LOG_DIR/%j_py_gen_N${N}.err" \
      --time=01:00:00 --mem=20G --cpus-per-task=1 \
      --account="$ACCOUNT" --partition="$PARTITION" \
      --wrap="cd $PROJECT_ROOT; $GEN_CMD")

    JID_FIT=$(sbatch --parsable --dependency=afterok:${JID_GEN} \
      --job-name=re_py_fit_N${N} \
      --output="$LOG_DIR/%A_%a_py_fit_N${N}.out" \
      --error="$LOG_DIR/%A_%a_py_fit_N${N}.err" \
      --array="0-${MAX_IDX}" \
      --time=24:00:00 --mem=30G --cpus-per-task=1 \
      --account="$ACCOUNT" --partition="$PARTITION" \
      --wrap="cd $PROJECT_ROOT
        $FIT_CMD")

    echo "py N=$N: gen=$JID_GEN fit=$JID_FIT"
    FIT_IDS+=("$JID_FIT")
  done
  LAST_CLEANUP=$(IFS=:; echo "${FIT_IDS[*]}")
}

# ---------------------------------------------------------------------------
# R chain.  For each N: generate -> gamm and bam as SEPARATE, parallel
# fit-array jobs (--only'd to one method each, writing their own suffixed csv
# so the two jobs never race on the same output file).
# ---------------------------------------------------------------------------
submit_r_chain() {
  local DDIR="$1"
  local JID_GEN method JID_FIT
  local FIT_IDS=()

  for N in "${N_LIST[@]}"; do
    GEN_CMD="export RE_DATA_DIR=$DDIR; uv run python $EXP_DIR/generate_data.py --N $N $REPS_ARG"
    JID_GEN=$(sbatch --parsable \
      --job-name=re_r_gen_N${N} \
      --output="$LOG_DIR/%j_r_gen_N${N}.out" \
      --error="$LOG_DIR/%j_r_gen_N${N}.err" \
      --time=01:00:00 --mem=20G --cpus-per-task=1 \
      --account="$ACCOUNT" --partition="$PARTITION" \
      --wrap="cd $PROJECT_ROOT; $GEN_CMD")

    for method in gamm bam; do
      FIT_CMD="export RE_DATA_DIR=$DDIR; LINE=\$(awk -v idx=\$SLURM_ARRAY_TASK_ID '\$1==idx{print}' $MAPPING); PZ=\$(echo \$LINE|awk '{print \$2}'); DT=\$(echo \$LINE|awk '{print \$3}'); CR=\$(echo \$LINE|awk '{print \$4}'); Rscript $EXP_DIR/run_r_baselines.R --N $N --pz \$PZ --D_type \$DT --corr \$CR --only $method"

      JID_FIT=$(sbatch --parsable --dependency=afterok:${JID_GEN} \
        --job-name=re_r_${method}_fit_N${N} \
        --output="$LOG_DIR/%A_%a_r_${method}_fit_N${N}.out" \
        --error="$LOG_DIR/%A_%a_r_${method}_fit_N${N}.err" \
        --array="0-${MAX_IDX}" \
        --time=24:00:00 --mem=30G --cpus-per-task=1 \
        --account="$ACCOUNT" --partition="$PARTITION" \
        --wrap="module load $R_MODULE
          cd $PROJECT_ROOT
          $FIT_CMD")

      echo "r $method N=$N: gen=$JID_GEN fit=$JID_FIT"
      FIT_IDS+=("$JID_FIT")
    done
  done
  LAST_CLEANUP=$(IFS=:; echo "${FIT_IDS[*]}")
}

DEPS=()

if [[ "$LANG" == "both" || "$LANG" == "py" ]]; then
  submit_py_chain data_py
  DEPS+=("$LAST_CLEANUP")
fi
if [[ "$LANG" == "both" || "$LANG" == "r" ]]; then
  submit_r_chain data_r
  DEPS+=("$LAST_CLEANUP")
fi

# Analyze waits on whichever chains ran.
DEP_STR=$(IFS=:; echo "${DEPS[*]}")
JID_ANALYZE=$(sbatch --parsable --dependency=afterok:${DEP_STR} \
  --job-name=re_analyze \
  --output="$LOG_DIR/%j_analyze.out" \
  --error="$LOG_DIR/%j_analyze.err" \
  --time=00:30:00 --mem=8G --cpus-per-task=1 \
  --account="$ACCOUNT" --partition="$PARTITION" \
  --wrap="cd $PROJECT_ROOT && uv run python $EXP_DIR/analyze_results.py")

# Data is kept: generate_data.py skips datasets whose JSON already exists.
echo "analyze=$JID_ANALYZE"
echo "Monitor: squeue -u \$USER"
