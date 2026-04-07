#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/SEMambapp-Interspeech}"
PARTITION="${PARTITION:-gpu}"
GPUS="${GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-48G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
JOB_NAME="${JOB_NAME:-semambapp-train}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-semambapp}"
TRAIN_CMD="${TRAIN_CMD:-python train.py}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"

mkdir -p "$LOG_DIR"

SBATCH_ARGS=(
  "--job-name=$JOB_NAME"
  "--partition=$PARTITION"
  "--cpus-per-task=$CPUS_PER_TASK"
  "--mem=$MEMORY"
  "--time=$TIME_LIMIT"
  "--output=$LOG_DIR/slurm_%j.out"
  "--error=$LOG_DIR/slurm_%j.err"
)

if [[ -n "$GPU_TYPE" ]]; then
  SBATCH_ARGS+=("--gres=gpu:${GPU_TYPE}:${GPUS}")
else
  SBATCH_ARGS+=("--gres=gpu:${GPUS}")
fi

echo "Submitting with:"
printf '  %q\n' sbatch "${SBATCH_ARGS[@]}" \
  --export=ALL,ROOT_DIR="$ROOT_DIR",CONDA_ENV_NAME="$CONDA_ENV_NAME",TRAIN_CMD="$TRAIN_CMD",LOG_DIR="$LOG_DIR" \
  "$ROOT_DIR/scripts/slurm_train.sh"

sbatch "${SBATCH_ARGS[@]}" \
  --export=ALL,ROOT_DIR="$ROOT_DIR",CONDA_ENV_NAME="$CONDA_ENV_NAME",TRAIN_CMD="$TRAIN_CMD",LOG_DIR="$LOG_DIR" \
  "$ROOT_DIR/scripts/slurm_train.sh"
