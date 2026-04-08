#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/SEMambapp-Interspeech}"
PARTITION="${PARTITION:-small}"
CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
MEMORY="${MEMORY:-4G}"
TIME_LIMIT="${TIME_LIMIT:-00:10:00}"
JOB_NAME="${JOB_NAME:-compute-access-check}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-semambapp}"
NOISE_FILE="${NOISE_FILE:-/m/teamwork/t412_symptosonic/databases/DNS/noise/datasets_fullband.noise_fullband.audioset_003/datasets_fullband/noise_fullband/O8KGwB1UNXk.wav}"
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/elec/t412-speechcom/symptonic-r2b}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"

mkdir -p "$LOG_DIR"

SBATCH_ARGS=(
  "--job-name=$JOB_NAME"
  "--partition=$PARTITION"
  "--cpus-per-task=$CPUS_PER_TASK"
  "--mem=$MEMORY"
  "--time=$TIME_LIMIT"
  "--output=$LOG_DIR/access_check_%j.out"
  "--error=$LOG_DIR/access_check_%j.err"
  "--wrap=cd $ROOT_DIR && CONDA_ENV_NAME='$CONDA_ENV_NAME' NOISE_FILE='$NOISE_FILE' SCRATCH_DIR='$SCRATCH_DIR' bash scripts/check_compute_access.sh"
)

echo "Submitting with:"
printf '  %q\n' sbatch "${SBATCH_ARGS[@]}"

sbatch "${SBATCH_ARGS[@]}"
