#!/bin/bash
# Historical cluster helper. Submission requires CONFIRM_SLURM_SUBMIT=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$DEFAULT_ROOT}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/runs/semambapp_dnf_paper_noisy}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT_DIR/checkpoints/semambapp_dnf_paper_noisy}"
JOB_NAME="${JOB_NAME:-dnf-paper-scratch-pair}"
PARTITION="${PARTITION:-gpu-debug}"
GPU_TYPE="${GPU_TYPE:-a100}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-48G}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
ARRAY_SPEC="${ARRAY_SPEC:-0-1%1}"

SEED="${SEED:-1234}"
MAX_STEPS="${MAX_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CUT_DURATION="${CUT_DURATION:-1.0}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-8192}"
VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-128}"
LISTENING_SAMPLES="${LISTENING_SAMPLES:-5}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-250 500 750 1000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
TINY_MODEL="${TINY_MODEL:-0}"

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/train/semambapp_shifted_anechoic_online_v1.yaml}"
SPLIT_ROOT="${SPLIT_ROOT:-/scratch/elec/t412-speechcom/Triton - Symptonic/lijie/gap_webdataset_active/splits/hybrid_unise_v1_stream_80_10_10}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT" "$CHECKPOINT_ROOT"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  PAIR_ID="${PAIR_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  sbatch \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --gres="gpu:${GPU_TYPE}:${GPUS}" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --time="$TIME_LIMIT" \
    --array="$ARRAY_SPEC" \
    --output="$LOG_DIR/slurm_%A_%a.out" \
    --error="$LOG_DIR/slurm_%A_%a.err" \
    --export="ALL,PAIR_ID=$PAIR_ID" \
    "$ROOT_DIR/scripts/cluster/slurm_semambapp_dnf_paper_noisy_array.sh"
  exit 0
fi

modes=(nytt dnf_exact)
MODE="${modes[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="${PAIR_ID}__${MODE}__seed${SEED}"
LIVE_LOG="$LOG_DIR/dnf_paper_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${MODE}.log"

cd "$ROOT_DIR"
echo "Job ${SLURM_JOB_ID} array ${SLURM_ARRAY_TASK_ID} mode ${MODE} started at $(date)" | tee -a "$LIVE_LOG"
echo "Run name: $RUN_NAME" | tee -a "$LIVE_LOG"
echo "Shape: ${GPU_TYPE}x${GPUS} batch=$BATCH_SIZE accum=$GRADIENT_ACCUMULATION_STEPS cut=${CUT_DURATION}s steps=$MAX_STEPS" | tee -a "$LIVE_LOG"
echo "Scratch only: init_checkpoint=null resume=null" | tee -a "$LIVE_LOG"

module load "$SOFTWARE_STACK_MODULE"
module load "$COMPILER_MODULE"
eval "$(conda shell.bash hook)"
conda activate semambapp
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "Hostname: $(hostname)" | tee -a "$LIVE_LOG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$LIVE_LOG"
python -c 'import torch; print("torch", torch.__version__); print("cuda_device_count", torch.cuda.device_count()); print("cuda_device_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")' | tee -a "$LIVE_LOG"

args=(
  --mode "$MODE"
  --config "$CONFIG_PATH"
  --split-root "$SPLIT_ROOT"
  --output-root "$OUTPUT_ROOT"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --run-name "$RUN_NAME"
  --seed "$SEED"
  --max-steps "$MAX_STEPS"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --num-workers "$NUM_WORKERS"
  --cut-duration "$CUT_DURATION"
  --samples-per-epoch "$SAMPLES_PER_EPOCH"
  --validation-samples "$VALIDATION_SAMPLES"
  --listening-samples "$LISTENING_SAMPLES"
  --checkpoint-steps $CHECKPOINT_STEPS
  --log-interval "$LOG_INTERVAL"
)

if [[ "$TINY_MODEL" == "1" ]]; then
  args+=(--tiny-model)
fi
python scripts/train_semambapp_dnf_paper_noisy.py "${args[@]}" 2>&1 | tee -a "$LIVE_LOG"
echo "Job ${SLURM_JOB_ID} mode ${MODE} completed at $(date)" | tee -a "$LIVE_LOG"
