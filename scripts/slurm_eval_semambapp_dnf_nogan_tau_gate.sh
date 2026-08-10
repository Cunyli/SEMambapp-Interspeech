#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/SEMambapp-Interspeech}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
JOB_NAME="${JOB_NAME:-semambapp-dnf-tau-gate}"
PARTITION="${PARTITION:-gpu-debug}"
GPU_TYPE="${GPU_TYPE:-v100}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
MODE="${MODE:-auto}"
MODEL_NAME="${MODEL_NAME:-semambapp-dnf-nogan-gate}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
PAIR_CSV="${PAIR_CSV:-/scratch/work/lil14/use_simulation_pipeline/outputs/organized/csv/tau_selected_clean_5fold_oof/fold_00/valid_selected_phone_room_aug16k_realrir_pair_manifest.csv}"
CLEAN_CACHE="${CLEAN_CACHE:-/scratch/work/lil14/use_simulation_pipeline/outputs/validation/tau_selected_clean_5fold_oof_refreshed_20260629/clean_cache/selected_phone_room_aug16k_realrir/fold_00_clean_avqi_16k.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/semambapp_dnf_nogan/tau_gate_${MODEL_NAME}}"
SPEAKER_LIMIT="${SPEAKER_LIMIT:-2}"
WORKERS="${WORKERS:-2}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

submit_self() {
  local script_path="$ROOT_DIR/scripts/slurm_eval_semambapp_dnf_nogan_tau_gate.sh"
  local sbatch_args=(
    "--job-name=$JOB_NAME"
    "--partition=$PARTITION"
    "--cpus-per-task=$CPUS_PER_TASK"
    "--mem=$MEMORY"
    "--time=$TIME_LIMIT"
    "--output=$LOG_DIR/slurm_%j.out"
    "--error=$LOG_DIR/slurm_%j.err"
  )
  if [[ -n "$GPU_TYPE" ]]; then
    sbatch_args+=("--gres=gpu:${GPU_TYPE}:${GPUS}")
  else
    sbatch_args+=("--gres=gpu:${GPUS}")
  fi
  sbatch "${sbatch_args[@]}" --export=ALL "$script_path"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  submit_self
  exit 0
fi

cd "$ROOT_DIR"
LIVE_LOG="$LOG_DIR/semambapp_dnf_tau_gate_${MODEL_NAME}_${SLURM_JOB_ID}.log"
echo "Live log: $LIVE_LOG"
echo "Job ${SLURM_JOB_ID} started at $(date)" | tee -a "$LIVE_LOG"
echo "Host: $(hostname)" | tee -a "$LIVE_LOG"
echo "Model name: $MODEL_NAME" | tee -a "$LIVE_LOG"
echo "Mode: $MODE" | tee -a "$LIVE_LOG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$LIVE_LOG"
echo "Checkpoint: $CHECKPOINT" | tee -a "$LIVE_LOG"
echo "Pair CSV: $PAIR_CSV" | tee -a "$LIVE_LOG"
echo "Clean cache: $CLEAN_CACHE" | tee -a "$LIVE_LOG"
echo "Output dir: $OUTPUT_DIR" | tee -a "$LIVE_LOG"
echo "Speaker limit: $SPEAKER_LIMIT" | tee -a "$LIVE_LOG"

module load "$SOFTWARE_STACK_MODULE"
module load "$COMPILER_MODULE"
eval "$(conda shell.bash hook)"
conda activate semambapp
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"

python - <<'PY' | tee -a "$LIVE_LOG"
import os
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY

python scripts/eval_semambapp_dnf_nogan_tau_gate.py \
  --checkpoint "$CHECKPOINT" \
  --pair-csv "$PAIR_CSV" \
  --clean-cache "$CLEAN_CACHE" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --mode "$MODE" \
  --speaker-limit "$SPEAKER_LIMIT" \
  --workers "$WORKERS" \
  2>&1 | tee -a "$LIVE_LOG"

echo "Job ${SLURM_JOB_ID} completed at $(date)" | tee -a "$LIVE_LOG"
