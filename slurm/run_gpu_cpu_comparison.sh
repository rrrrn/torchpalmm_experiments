#!/bin/bash
# GPU vs CPU comparison at the largest cell of experiments/scaling_py/ss_scaling
# (sn=30). See also slurm/run_gpu_cpu_m.sh, which also covers sn=10.
#
# Chain: generate the extreme data slice (CPU, standard partition)
#     -> fit them on GPU (gpu partition, --gres=gpu:1)
#     -> optional same-node CPU control on the same GPU node, CUDA hidden
#     -> analyze (borrows the CPU rows from scaling_py/).
#
# The CPU numbers this is compared against are NOT re-run here: they are the
# existing scaling_py rows. The --device cpu control job is a small extra that
# measures how much of the gap is node rather than device; skip with --no-control.
#
# Usage (from project root):
#   bash slurm/run_gpu_cpu_comparison.sh                # 20 GPU reps + control
#   bash slurm/run_gpu_cpu_comparison.sh --reps 5       # pilot first
#   bash slurm/run_gpu_cpu_comparison.sh --no-control

REPS=20; CONTROL=1
while [[ $# -gt 0 ]]; do
  case $1 in
    --reps) REPS="$2"; shift 2 ;;
    --no-control) CONTROL=0; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

PARTITION="standard"; GPU_PARTITION="gpu"
ACCOUNT="${SLURM_ACCOUNT:?export SLURM_ACCOUNT=<your SLURM account>}"
PROJECT_ROOT="$(pwd)"
bash "$PROJECT_ROOT/slurm/check_torchpalmm.sh" || exit 1
EXP_DIR="$PROJECT_ROOT/experiments/gpu_cpu_comparison"
PILOT_DIR="$PROJECT_ROOT/experiments/scaling_py"
LOG_DIR="$PROJECT_ROOT/slurm/logs_gpu_cpu"
mkdir -p "$LOG_DIR"

# cell:pilot:generate-flag:value
CELLS=(
  "sn30:ss_scaling:--sn:30"
)

# 1. Data. The scaling_py pilots' data/ already holds the datasets the CPU rows were
#    fit on, so the GPU job reuses those files in place: byte-identical inputs, no
#    reliance on the generator reproducing the same RNG stream on a different node or
#    PyTorch version. A gen job is submitted only for cells with missing reps
#    (generate_data.py skips files that already exist, so it only fills gaps).
GEN_IDS=()
for entry in "${CELLS[@]}"; do
  IFS=":" read -r CELL PILOT FLAG VALUE <<< "$entry"
  PREFIX="${FLAG#--}"
  MISSING=0
  for D in diagonal unstructured; do
    for RID in $(seq 1 "$REPS"); do
      [[ -f "$PILOT_DIR/$PILOT/data/${PREFIX}${VALUE}_D${D}_repeat${RID}.json" ]] || MISSING=$((MISSING+1))
    done
  done
  if [[ "$MISSING" -eq 0 ]]; then
    echo "[gen] $CELL: all $((REPS*2)) datasets present, reusing in place"
    continue
  fi
  echo "[gen] $CELL: $MISSING/$((REPS*2)) datasets missing, submitting generator"
  JID=$(sbatch --parsable --job-name="gen_${CELL}" \
    --time=02:00:00 --mem=32G --cpus-per-task=1 \
    --account="$ACCOUNT" --partition="$PARTITION" \
    --output="$LOG_DIR/gen_${CELL}_%j.out" --error="$LOG_DIR/gen_${CELL}_%j.err" \
    --wrap="cd $PROJECT_ROOT && uv run python $PILOT_DIR/$PILOT/generate_data.py $FLAG $VALUE --reps $REPS")
  echo "[gen] $CELL -> $JID"
  GEN_IDS+=("$JID")
done
GEN_DEP=""
if [[ ${#GEN_IDS[@]} -gt 0 ]]; then
  GEN_DEP="--dependency=afterok:$(IFS=:; echo "${GEN_IDS[*]}")"
fi

# 2. GPU fits. One job for all cells so CUDA context init is paid once and the
#    warm-up fit in run_gpu.py covers the whole run.
GPU_JID=$(sbatch --parsable --job-name="gpu_cmp" \
  $GEN_DEP \
  --time=12:00:00 --mem=64G --cpus-per-task=4 --gres=gpu:1 \
  --account="$ACCOUNT" --partition="$GPU_PARTITION" \
  --output="$LOG_DIR/gpu_%j.out" --error="$LOG_DIR/gpu_%j.err" \
  --wrap="cd $PROJECT_ROOT && nvidia-smi && uv run python $EXP_DIR/run_gpu.py --device gpu --reps $REPS")
echo "[gpu] -> $GPU_JID"

# 3. Same-node CPU control: same GPU partition (so the host CPU is the GPU node's),
#    CUDA hidden so SmoothingSplineMixed's device=None resolves to cpu. Cheap cells
#    only.
DEP="$GPU_JID"
if [[ "$CONTROL" -eq 1 ]]; then
  CTRL_REPS=$(( REPS < 5 ? REPS : 5 ))
  # The control measures the node effect on the sn30 cell.
  # --gres=gpu:1 is required even though the fit is CPU-only: the gpu partition's QOS
  # rejects a job that requests no GRES (QOSMinGRES). CUDA_VISIBLE_DEVICES='' then
  # hides the card so SmoothingSplineMixed's device=None still resolves to cpu.
  for CELL in sn30; do
    CJID=$(sbatch --parsable --job-name="cpuctl_${CELL}" \
      --dependency=afterok:"$GPU_JID" \
      --time=02:00:00 --mem=32G --cpus-per-task=1 --gres=gpu:1 \
      --account="$ACCOUNT" --partition="$GPU_PARTITION" \
      --output="$LOG_DIR/cpuctl_${CELL}_%j.out" --error="$LOG_DIR/cpuctl_${CELL}_%j.err" \
      --wrap="cd $PROJECT_ROOT && CUDA_VISIBLE_DEVICES='' uv run python $EXP_DIR/run_gpu.py --device cpu --cell $CELL --reps $CTRL_REPS")
    if [[ -n "$CJID" ]]; then
      echo "[cpu-control] $CELL -> $CJID"
      DEP="$DEP:$CJID"
    else
      echo "[cpu-control] $CELL -> submit FAILED; continuing without the control"
    fi
  done
fi

# 4. Analyze.
AJID=$(sbatch --parsable --job-name="gpu_cmp_analyze" \
  --dependency=afterok:"$DEP" \
  --time=00:30:00 --mem=8G --cpus-per-task=1 \
  --account="$ACCOUNT" --partition="$PARTITION" \
  --output="$LOG_DIR/analyze_%j.out" --error="$LOG_DIR/analyze_%j.err" \
  --wrap="cd $PROJECT_ROOT && uv run python $EXP_DIR/analyze_results.py")
echo "[analyze] -> $AJID"
