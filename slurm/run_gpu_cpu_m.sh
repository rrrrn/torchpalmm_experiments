#!/bin/bash
# GPU vs CPU comparison over the m (= sn) design: m in {10, 30} x D in {diagonal,
# unstructured}, 100 reps per design cell. Same fit configuration as
# slurm/run_gpu_cpu_comparison.sh (pilot's own run_python.py: L-BFGS, lr=1.0,
# synchronized timer, warm-up fit); only the cells and the rep count change.
#
# Chain: generate missing datasets (CPU, standard partition)
#     -> fit all cells on GPU (gpu partition, --gres=gpu:1)
#     -> same-node CPU control on m=10 (cheap cell), CUDA hidden
#     -> analyze (borrows the CPU rows from scaling_py/ss_scaling/results/).
#
# The CPU rows compared against are NOT re-run: they are scaling_py/ss_scaling's
# existing results for sn=10 and sn=30 (lr=1.0). The control measures the node
# effect only; skip with --no-control.
#
# Usage (from project root):
#   bash slurm/run_gpu_cpu_m.sh                    # 100 reps/cell + control (20 reps)
#   bash slurm/run_gpu_cpu_m.sh --reps 5           # pilot first
#   bash slurm/run_gpu_cpu_m.sh --control-reps 50
#   bash slurm/run_gpu_cpu_m.sh --no-control

REPS=100; CONTROL=1; CTRL_REPS=20
while [[ $# -gt 0 ]]; do
  case $1 in
    --reps) REPS="$2"; shift 2 ;;
    --control-reps) CTRL_REPS="$2"; shift 2 ;;
    --no-control) CONTROL=0; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
[[ "$CTRL_REPS" -gt "$REPS" ]] && CTRL_REPS="$REPS"

PARTITION="standard"; GPU_PARTITION="gpu"
ACCOUNT="${SLURM_ACCOUNT:?export SLURM_ACCOUNT=<your SLURM account>}"
PROJECT_ROOT="$(pwd)"
bash "$PROJECT_ROOT/slurm/check_torchpalmm.sh" || exit 1
EXP_DIR="$PROJECT_ROOT/simulation_experiments/gpu_cpu_comparison"
PILOT="ss_scaling"
PILOT_DIR="$PROJECT_ROOT/simulation_experiments/scaling_py/$PILOT"
LOG_DIR="$PROJECT_ROOT/slurm/logs_gpu_cpu_m"
mkdir -p "$LOG_DIR"

M_LIST=(10 30)
D_LIST=(diagonal unstructured)

# 1. Data. Reuse the scaling_py datasets in place (byte-identical to what the CPU
#    rows were fit on); generate_data.py only fills missing files.
GEN_IDS=()
for M in "${M_LIST[@]}"; do
  MISSING=0
  for D in "${D_LIST[@]}"; do
    for RID in $(seq 1 "$REPS"); do
      [[ -f "$PILOT_DIR/data/sn${M}_D${D}_repeat${RID}.json" ]] || MISSING=$((MISSING+1))
    done
  done
  if [[ "$MISSING" -eq 0 ]]; then
    echo "[gen] m=$M: all $((REPS*2)) datasets present, reusing in place"
    continue
  fi
  echo "[gen] m=$M: $MISSING/$((REPS*2)) datasets missing, submitting generator"
  JID=$(sbatch --parsable --job-name="gen_m${M}" \
    --time=04:00:00 --mem=32G --cpus-per-task=1 \
    --account="$ACCOUNT" --partition="$PARTITION" \
    --output="$LOG_DIR/gen_m${M}_%j.out" --error="$LOG_DIR/gen_m${M}_%j.err" \
    --wrap="cd $PROJECT_ROOT && uv run python $PILOT_DIR/generate_data.py --sn $M --reps $REPS")
  echo "[gen] m=$M -> $JID"
  GEN_IDS+=("$JID")
done
GEN_DEP=""
if [[ ${#GEN_IDS[@]} -gt 0 ]]; then
  GEN_DEP="--dependency=afterok:$(IFS=:; echo "${GEN_IDS[*]}")"
fi

# Borrowed CPU rows: report what is on disk so a short CPU side is visible before
# the GPU hours are spent (rows per file, header excluded; includes every lr).
for M in "${M_LIST[@]}"; do for D in "${D_LIST[@]}"; do
  F="$PILOT_DIR/results/sn${M}_D${D}.csv"
  if [[ -f "$F" ]]; then
    echo "[cpu rows] m=$M $D: $(( $(wc -l < "$F") - 1 )) rows in $F"
  else
    echo "[cpu rows] m=$M $D: MISSING $F (run slurm/py_scaling_all.sh first)"
  fi
done; done

# 2. GPU fits: both cells in one job (one allocation), one run_gpu.py call per cell.
#    --cell selects the cell; run_gpu.py defaults to all cells.
GPU_JID=$(sbatch --parsable --job-name="gpu_cmp_m" \
  $GEN_DEP \
  --time=03:00:00 --mem=64G --cpus-per-task=4 --gres=gpu:1 \
  --account="$ACCOUNT" --partition="$GPU_PARTITION" \
  --output="$LOG_DIR/gpu_%j.out" --error="$LOG_DIR/gpu_%j.err" \
  --wrap="cd $PROJECT_ROOT && nvidia-smi && for C in sn10 sn30; do uv run python $EXP_DIR/run_gpu.py --device gpu --cell \$C --reps $REPS; done")
echo "[gpu] -> $GPU_JID"

# 3. Same-node CPU control on m=10 only (cheap: ~minutes per fit; m=30 is ~20 min per
#    fit on one core). --gres=gpu:1 is required by the gpu partition's QOS even though
#    CUDA_VISIBLE_DEVICES='' hides the card so the fit runs on CPU.
DEP="$GPU_JID"
if [[ "$CONTROL" -eq 1 ]]; then
  CJID=$(sbatch --parsable --job-name="cpuctl_m10" \
    --dependency=afterok:"$GPU_JID" \
    --time=04:00:00 --mem=32G --cpus-per-task=1 --gres=gpu:1 \
    --account="$ACCOUNT" --partition="$GPU_PARTITION" \
    --output="$LOG_DIR/cpuctl_m10_%j.out" --error="$LOG_DIR/cpuctl_m10_%j.err" \
    --wrap="cd $PROJECT_ROOT && CUDA_VISIBLE_DEVICES='' uv run python $EXP_DIR/run_gpu.py --device cpu --cell sn10 --reps $CTRL_REPS")
  if [[ -n "$CJID" ]]; then
    echo "[cpu-control] m=10 ($CTRL_REPS reps) -> $CJID"
    DEP="$DEP:$CJID"
  else
    echo "[cpu-control] m=10 -> submit FAILED; continuing without the control"
  fi
fi

# 4. Analyze.
AJID=$(sbatch --parsable --job-name="gpu_cmp_m_analyze" \
  --dependency=afterok:"$DEP" \
  --time=00:30:00 --mem=8G --cpus-per-task=1 \
  --account="$ACCOUNT" --partition="$PARTITION" \
  --output="$LOG_DIR/analyze_%j.out" --error="$LOG_DIR/analyze_%j.err" \
  --wrap="cd $PROJECT_ROOT && uv run python $EXP_DIR/analyze_results.py")
echo "[analyze] -> $AJID"
