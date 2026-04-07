#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/SEMambapp-Interspeech}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-semambapp}"
TRAIN_CMD="${TRAIN_CMD:-python train.py}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

LIVE_LOG="$LOG_DIR/train_${SLURM_JOB_ID}.log"
echo "Live log: $LIVE_LOG"
echo "Job ${SLURM_JOB_ID} started at $(date)" | tee -a "$LIVE_LOG"
echo "Running on host: $(hostname)" | tee -a "$LIVE_LOG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$LIVE_LOG"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV_NAME"
else
  echo "conda not found on PATH" | tee -a "$LIVE_LOG"
  exit 1
fi

echo "Python: $(which python)" | tee -a "$LIVE_LOG"
echo "Command: $TRAIN_CMD" | tee -a "$LIVE_LOG"

$TRAIN_CMD 2>&1 | tee -a "$LIVE_LOG"

echo "Job ${SLURM_JOB_ID} completed at $(date)" | tee -a "$LIVE_LOG"
