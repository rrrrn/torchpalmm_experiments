#!/bin/bash
# Submits both simulation_experiments/scaling_py/ pilots (ss_scaling, pz_scaling_ss) --
# python-only, no R baselines, each varying exactly one of {sn, pz} off the shared
# default point (sn=1, px=2, pz=2, N=1000). Each pilot is generate -> fit array
# (value x D_type), chained with --dependency=afterok. No analyze job is submitted
# here -- results are synced to a local machine and analyzed there.
#
# Resume-aware PER VALUE, not just per pilot -- coverage_check.py (in
# simulation_experiments/scaling_py/) loads each value through that pilot's own
# analyze_results.py _load() (local results/ only -- no sibling-experiment
# fallback, best-lr dedup, _KEEP filtering: the exact same data
# scaling_overview.png / scaling_progress.png show) and reports a value
# "covered" only once EVERY D_type has >= TARGET (--reps, default 100)
# converged reps there, from THIS pilot's own fits. This is a rep-count
# check, not a file-existence one: a value with a CSV but only a handful of
# converged reps (e.g. a fit array that hit its walltime partway through)
# reports as missing and gets topped up, rather than being silently treated
# as done forever. generate+fit run only for values still missing after
# that; if every value is covered, nothing is submitted for that pilot.
# Either way, re-running this script after a partial run tops up what's
# missing. generate_data.py/run_python.py's own resume logic still applies
# within a value that DOES get (re)submitted -- already-done (value, D_type,
# repeat_id, optimizer, lr) rows are never regenerated/re-fit.
#
# Usage (from project root):
#   bash slurm/py_scaling_all.sh                 # full (config reps = 100 across both)
#   bash slurm/py_scaling_all.sh --reps 5        # pilot (recommended first)
#   bash slurm/py_scaling_all.sh --only ss_scaling             # subset

REPS=""; ONLY=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --reps) REPS="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
REPS_ARG=""; [[ -n "$REPS" ]] && REPS_ARG="--reps $REPS"
TARGET="${REPS:-100}"   # converged reps required per (value, D_type) to count as covered

PARTITION="standard"; ACCOUNT="${SLURM_ACCOUNT:?export SLURM_ACCOUNT=<your SLURM account>}"
PROJECT_ROOT="$(pwd)"
bash "$PROJECT_ROOT/slurm/check_torchpalmm.sh" || exit 1
PARENT_DIR="$PROJECT_ROOT/simulation_experiments/scaling_py"
LOG_ROOT="$PROJECT_ROOT/slurm/logs_scaling_py"
mkdir -p "$LOG_ROOT"

# Only run a pilot if --only was given and it's not in the list.
_wanted() {
  [[ -z "$ONLY" ]] && return 0
  [[ ",$ONLY," == *",$1,"* ]]
}

# submit_pilot NAME FLAG VALUES_STR D_LIST_STR FIT_TIME FIT_MEM
submit_pilot() {
  local NAME="$1" FLAG="$2" VALUES_STR="$3" D_LIST_STR="$4" FIT_TIME="$5" FIT_MEM="$6"
  _wanted "$NAME" || { echo "[$NAME] skipped (--only)"; return; }

  local EXP_DIR="$PARENT_DIR/$NAME"
  local LOG_DIR="$LOG_ROOT/$NAME"
  mkdir -p "$LOG_DIR"

  read -ra VALUES <<< "$VALUES_STR"
  read -ra D_LIST <<< "$D_LIST_STR"

  # Per-value coverage check (see header comment): a value counts as covered
  # only once EVERY D_type has >= TARGET converged reps there, per
  # coverage_check.py's read of that pilot's own analyze_results.py _load().
  # Fail-safe direction matters here: if the check itself errors (missing
  # file, stale checkout, transient crash), COV_OUT/exit code must NOT be
  # silently read as "nothing missing" -- that would report a pilot as fully
  # covered without ever having checked it. On failure this pilot is skipped
  # entirely (no submission, no false "covered"), not defaulted either way.
  local COV_OUT MISSING=() COVERED=()
  if ! COV_OUT=$(cd "$PROJECT_ROOT" && uv run python "$PARENT_DIR/coverage_check.py" \
       "$NAME" "$FLAG" "$TARGET" --d-types "${D_LIST[@]}" --values "${VALUES[@]}" 2>&1); then
    echo "[$NAME] ERROR: coverage_check.py failed -- skipping this pilot (not submitting, not assuming covered):"
    echo "$COV_OUT" | sed "s/^/[$NAME]   /"
    return 1
  fi
  local got_covered=0 got_missing=0
  while IFS= read -r line; do
    case "$line" in
      COVERED:*) read -ra COVERED <<< "${line#COVERED: }"; got_covered=1 ;;
      MISSING:*) read -ra MISSING <<< "${line#MISSING: }"; got_missing=1 ;;
    esac
  done <<< "$COV_OUT"
  if [[ "$got_covered" -eq 0 || "$got_missing" -eq 0 ]]; then
    echo "[$NAME] ERROR: coverage_check.py returned unparseable output -- skipping this pilot:"
    echo "$COV_OUT" | sed "s/^/[$NAME]   /"
    return 1
  fi
  [[ ${#COVERED[@]} -gt 0 ]] && echo "[$NAME] already covered (>= $TARGET converged reps): ${FLAG}=[${COVERED[*]}]"

  if [[ ${#MISSING[@]} -eq 0 ]]; then
    echo "[$NAME] all values covered -- nothing to submit."
    return
  fi

  # (value x D_type) cell mapping for the fit array -- MISSING values only.
  local MAPPING="$LOG_DIR/cells.txt"; : > "$MAPPING"; local IDX=0
  for V in "${MISSING[@]}"; do for D in "${D_LIST[@]}"; do
    echo "$IDX $V $D" >> "$MAPPING"; IDX=$((IDX + 1))
  done; done
  local MAX_IDX=$((IDX - 1))
  echo "[$NAME] cells: $((MAX_IDX + 1)) (array 0-${MAX_IDX}); ${FLAG}=[${MISSING[*]}]"

  # Stage 1: generate, one call per missing value (serial -> no concurrent config.json write).
  local JID_GEN
  JID_GEN=$(sbatch --parsable --job-name="${NAME}_gen" \
    --output="$LOG_DIR/%j_gen.out" --error="$LOG_DIR/%j_gen.err" \
    --time=02:00:00 --mem=16G --cpus-per-task=1 --account="$ACCOUNT" --partition="$PARTITION" \
    --wrap="cd $PROJECT_ROOT
      for V in ${MISSING[*]}; do
        uv run python $EXP_DIR/generate_data.py --${FLAG} \$V $REPS_ARG
      done")

  # Stage 2: fit array, one (value, D_type) cell per task.
  local PARSE="LINE=\$(awk -v i=\$SLURM_ARRAY_TASK_ID '\$1==i{print}' $MAPPING); V=\$(echo \$LINE|awk '{print \$2}'); D=\$(echo \$LINE|awk '{print \$3}')"
  local JID_FIT
  JID_FIT=$(sbatch --parsable --dependency=afterok:${JID_GEN} --job-name="${NAME}_fit" \
    --output="$LOG_DIR/%A_%a_fit.out" --error="$LOG_DIR/%A_%a_fit.err" \
    --array="0-${MAX_IDX}" --time="$FIT_TIME" --mem="$FIT_MEM" --cpus-per-task=1 \
    --account="$ACCOUNT" --partition="$PARTITION" \
    --wrap="cd $PROJECT_ROOT
      $PARSE
      uv run python $EXP_DIR/run_python.py --${FLAG} \$V --D_type \$D $REPS_ARG")

  echo "[$NAME] gen=$JID_GEN fit=$JID_FIT"
}

# Keep in sync with each simulation_experiments/scaling_py/<pilot>/generate_data.py CONFIG.
submit_pilot "ss_scaling"     "sn" "5 10 20 30"          "diagonal unstructured" "48:00:00" "32G"
submit_pilot "pz_scaling_ss"  "pz" "5 10 20 30"          "diagonal unstructured" "06:00:00" "16G"

echo "Monitor: squeue -u \$USER"
