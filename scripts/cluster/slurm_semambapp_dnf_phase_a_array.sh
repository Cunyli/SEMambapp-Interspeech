#!/bin/bash
# Historical cluster helper. Submission requires CONFIRM_SLURM_SUBMIT=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$DEFAULT_ROOT}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/runs/semambapp_dnf_phase_a}"
JOB_NAME="${JOB_NAME:-dnf-phase-a-pair}"
PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-64G}"
TIME_LIMIT="${TIME_LIMIT:-05:00:00}"
ARRAY_SPEC="${ARRAY_SPEC:-0-1%2}"

SEED="${SEED:-1234}"
LOSS_VARIANT="${LOSS_VARIANT:-paper_exact}"
MAX_STEPS="${MAX_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CUT_DURATION="${CUT_DURATION:-1.0}"
VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-200}"
LISTENING_SAMPLES="${LISTENING_SAMPLES:-5}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-250 500 1000 2000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

MODEL_CONFIG="${MODEL_CONFIG:-$ROOT_DIR/configs/train/semambapp_shifted_anechoic_online_v1.yaml}"
CONTRACT_PATH="${CONTRACT_PATH:-$ROOT_DIR/configs/train/dnf_phase_ab_v2_contract.json}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-}"
VALID_MANIFEST="${VALID_MANIFEST:-}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

if [[ "$GPUS" != "1" ]]; then
  echo "Phase A is a single-GPU contract; GPUS must equal 1." >&2
  exit 2
fi
case "$ARRAY_SPEC" in
  "0-1%1"|"0-1%2") ;;
  *)
    echo "Phase A ARRAY_SPEC must be 0-1%1 or 0-1%2." >&2
    exit 2
    ;;
esac
case "$LOSS_VARIANT" in
  "paper_exact"|"matched_scale") ;;
  *)
    echo "LOSS_VARIANT must be paper_exact or matched_scale." >&2
    exit 2
    ;;
esac
if [[ -z "$TRAIN_MANIFEST" || -z "$VALID_MANIFEST" ]]; then
  echo "TRAIN_MANIFEST and VALID_MANIFEST are required immutable JSONL paths." >&2
  exit 2
fi

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  PAIR_ID="${PAIR_ID:-phase-a-${LOSS_VARIANT}-$(date -u +%Y%m%dT%H%M%SZ)-seed${SEED}}"
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
    "$ROOT_DIR/scripts/cluster/slurm_semambapp_dnf_phase_a_array.sh"
  exit 0
fi

if [[ -z "${PAIR_ID:-}" ]]; then
  echo "PAIR_ID was not exported into the allocation." >&2
  exit 2
fi
if [[ "${SLURM_ARRAY_TASK_ID:-}" != "0" && "${SLURM_ARRAY_TASK_ID:-}" != "1" ]]; then
  echo "SLURM_ARRAY_TASK_ID must be 0 or 1." >&2
  exit 2
fi

modes=(standard dnf)
MODE="${modes[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="${PAIR_ID}__${MODE}"
PAIR_CONTRACT_DIR="$OUTPUT_ROOT/${PAIR_ID}__pair_contract"
LIVE_LOG="$LOG_DIR/dnf_phase_a_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${MODE}.log"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "event=phase_a_start job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} mode=${MODE} time=$(date -Is)" | tee -a "$LIVE_LOG"
echo "pair_id=$PAIR_ID run_name=$RUN_NAME" | tee -a "$LIVE_LOG"
echo "shape=${GPU_TYPE}x${GPUS} batch=${BATCH_SIZE} accum=${GRADIENT_ACCUMULATION_STEPS} cut=${CUT_DURATION}s steps=${MAX_STEPS}" | tee -a "$LIVE_LOG"
echo "scratch_only=true resume=null init_checkpoint=null gan=false" | tee -a "$LIVE_LOG"
echo "loss_variant=$LOSS_VARIANT" | tee -a "$LIVE_LOG"
echo "contract=$CONTRACT_PATH train_manifest=$TRAIN_MANIFEST valid_manifest=$VALID_MANIFEST" | tee -a "$LIVE_LOG"

module load "$SOFTWARE_STACK_MODULE"
module load "$COMPILER_MODULE"
eval "$(conda shell.bash hook)"
conda activate semambapp
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -c 'import os, torch; print("torch", torch.__version__); print("cuda_available", torch.cuda.is_available()); print("cuda_device_count", torch.cuda.device_count()); print("cuda_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"); print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))' \
  | tee -a "$LIVE_LOG"

args=(
  --mode "$MODE"
  --loss-variant "$LOSS_VARIANT"
  --config "$MODEL_CONFIG"
  --contract "$CONTRACT_PATH"
  --train-manifest "$TRAIN_MANIFEST"
  --valid-manifest "$VALID_MANIFEST"
  --output-root "$OUTPUT_ROOT"
  --pair-contract-dir "$PAIR_CONTRACT_DIR"
  --run-name "$RUN_NAME"
  --seed "$SEED"
  --max-steps "$MAX_STEPS"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --num-workers "$NUM_WORKERS"
  --cut-duration "$CUT_DURATION"
  --validation-samples "$VALIDATION_SAMPLES"
  --listening-samples "$LISTENING_SAMPLES"
  --checkpoint-steps $CHECKPOINT_STEPS
  --log-interval "$LOG_INTERVAL"
)

python scripts/train_semambapp_dnf_phase_a.py "${args[@]}" 2>&1 | tee -a "$LIVE_LOG"
echo "event=phase_a_complete job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} mode=${MODE} time=$(date -Is)" | tee -a "$LIVE_LOG"
