#!/bin/bash
# Diagnostic only: no generator optimizer step is implemented by the Python job.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$DEFAULT_ROOT}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-$SOURCE_ROOT/scripts/evaluate_avqi_component_backprop.py}"
JOB_NAME="${JOB_NAME:-avqi-component-diagnostic}"
PARTITION="${PARTITION:-gpu-a100-80g}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-48G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_component_backprop_20260811_01}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints/avqi_component_backprop_20260811_01}"
LABEL_BANK="${LABEL_BANK:-$ROOT_DIR/runs/tau_pathology_preservation_eval_phase2_20260809_01/outputs/surrogate/exact_component_label_bank_v1.csv}"
CONFIG="${CONFIG:-$ROOT_DIR/runs/tau_s1_sv_threshold_ablation_20260719_01/configs/s_fidelity_m3_stage0500.yaml}"
CHECKPOINT="${CHECKPOINT:-$ROOT_DIR/runs/tau_s1_sv_threshold_ablation_20260719_01/milestone_checkpoints/S_fidelity_-3/ln_g_00000500.pth}"
EXTERNAL_EXACT_CSV="${EXTERNAL_EXACT_CSV:-$ROOT_DIR/runs/tau_pathology_three_tracks_20260810_01/outputs/intensity_eval/exact_components_all.csv}"
LABEL_BANK_SHA256="${LABEL_BANK_SHA256:?LABEL_BANK_SHA256 is required}"
CHECKPOINT_SHA256="${CHECKPOINT_SHA256:?CHECKPOINT_SHA256 is required}"
EXTERNAL_EXACT_CSV_SHA256="${EXTERNAL_EXACT_CSV_SHA256:?EXTERNAL_EXACT_CSV_SHA256 is required}"

mkdir -p "$LOG_DIR"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" || -e "$CHECKPOINT_DIR" ]]; then
    echo "Refusing to overwrite output or checkpoints: $OUTPUT_DIR $CHECKPOINT_DIR" >&2
    exit 2
  fi
  sbatch \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --gres="gpu:${GPU_TYPE}:1" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --time="$TIME_LIMIT" \
    --output="$LOG_DIR/slurm_%j.out" \
    --error="$LOG_DIR/slurm_%j.err" \
    --export=ALL \
    "$SELF_PATH"
  exit 0
fi

if [[ -e "$OUTPUT_DIR" || -e "$CHECKPOINT_DIR" ]]; then
  echo "Refusing to overwrite output or checkpoints: $OUTPUT_DIR $CHECKPOINT_DIR" >&2
  exit 2
fi

cd "$ROOT_DIR"
module load "$SOFTWARE_STACK_MODULE"
module load "$COMPILER_MODULE"
eval "$(conda shell.bash hook)"
conda activate semambapp
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export PYTHONPATH="$SOURCE_ROOT:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LIVE_LOG="$LOG_DIR/avqi_component_diagnostic_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
python -c 'import os, torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0)); print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))' | tee -a "$LIVE_LOG"

python "$PYTHON_SCRIPT" \
  --label-bank "$LABEL_BANK" \
  --label-bank-sha256 "$LABEL_BANK_SHA256" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --external-exact-csv "$EXTERNAL_EXACT_CSV" \
  --external-exact-csv-sha256 "$EXTERNAL_EXACT_CSV_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --device cuda \
  2>&1 | tee -a "$LIVE_LOG"

echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
