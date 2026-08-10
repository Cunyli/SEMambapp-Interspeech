#!/bin/bash
# Historical cluster helper. Submission requires CONFIRM_SLURM_SUBMIT=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$DEFAULT_ROOT}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
JOB_NAME="${JOB_NAME:-dnf-phase-a-tau-gate}"
PARTITION="${PARTITION:-gpu-debug}"
GPU_TYPE="${GPU_TYPE:-v100}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
MODE="${MODE:-auto}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
CONTROLLED_COMPARISON="${CONTROLLED_COMPARISON:?CONTROLLED_COMPARISON is required}"
PAIR_CSV="${PAIR_CSV:-/scratch/work/lil14/use_simulation_pipeline/outputs/organized/csv/tau_selected_clean_5fold_oof/fold_00/valid_selected_phone_room_aug16k_realrir_pair_manifest.csv}"
CLEAN_CACHE="${CLEAN_CACHE:-/scratch/work/lil14/use_simulation_pipeline/outputs/validation/tau_selected_clean_5fold_oof_refreshed_20260629/clean_cache/selected_phone_room_aug16k_realrir/fold_00_clean_avqi_16k.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/semambapp_dnf_phase_a/tau_gate_${MODEL_NAME}}"
SPEAKERS_PER_GROUP="${SPEAKERS_PER_GROUP:-2}"
SELECTION_SEED="${SELECTION_SEED:-1234}"
WORKERS="${WORKERS:-2}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

if [[ "$GPUS" != "1" ]]; then
  echo "The Phase-A TAU gate is single-GPU; GPUS must equal 1." >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite immutable TAU output: $OUTPUT_DIR" >&2
    exit 2
  fi
  sbatch \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --gres="gpu:${GPU_TYPE}:${GPUS}" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --time="$TIME_LIMIT" \
    --output="$LOG_DIR/slurm_%j.out" \
    --error="$LOG_DIR/slurm_%j.err" \
    --export=ALL \
    "$ROOT_DIR/scripts/cluster/slurm_eval_semambapp_dnf_phase_a_tau_gate.sh"
  exit 0
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite immutable TAU output: $OUTPUT_DIR" >&2
  exit 2
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
LIVE_LOG="$LOG_DIR/dnf_phase_a_tau_gate_${MODEL_NAME}_${SLURM_JOB_ID}.log"
echo "event=phase_a_tau_gate_start job=${SLURM_JOB_ID} time=$(date -Is)" | tee -a "$LIVE_LOG"
echo "model=$MODEL_NAME mode=$MODE checkpoint=$CHECKPOINT" | tee -a "$LIVE_LOG"
echo "controlled_comparison=$CONTROLLED_COMPARISON" | tee -a "$LIVE_LOG"
echo "pair_csv=$PAIR_CSV clean_cache=$CLEAN_CACHE output_dir=$OUTPUT_DIR" | tee -a "$LIVE_LOG"

module load "$SOFTWARE_STACK_MODULE"
module load "$COMPILER_MODULE"
eval "$(conda shell.bash hook)"
conda activate semambapp
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"

python -c 'import os, torch; print("torch", torch.__version__); print("cuda_available", torch.cuda.is_available()); print("cuda_device_count", torch.cuda.device_count()); print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))' \
  | tee -a "$LIVE_LOG"

python scripts/eval_semambapp_dnf_phase_a_tau_gate.py \
  --checkpoint "$CHECKPOINT" \
  --controlled-comparison "$CONTROLLED_COMPARISON" \
  --pair-csv "$PAIR_CSV" \
  --clean-cache "$CLEAN_CACHE" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --mode "$MODE" \
  --speakers-per-group "$SPEAKERS_PER_GROUP" \
  --selection-seed "$SELECTION_SEED" \
  --workers "$WORKERS" \
  2>&1 | tee -a "$LIVE_LOG"

echo "event=phase_a_tau_gate_complete job=${SLURM_JOB_ID} time=$(date -Is)" | tee -a "$LIVE_LOG"
