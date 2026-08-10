#!/bin/bash
# Historical Triton helper. Outside an allocation this script can submit an
# sbatch job, so submission requires CONFIRM_SLURM_SUBMIT=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$DEFAULT_ROOT}"
TASK="${TASK:-${1:-train}}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-semambapp}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
JOB_NAME="${JOB_NAME:-semambapp-$TASK}"
PARTITION="${PARTITION:-gpu-a100-80g}"
GPU_TYPE="${GPU_TYPE:-a100}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-64G}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
DEPENDENCY="${DEPENDENCY:-}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"

export ROOT_DIR TASK CONDA_ENV_NAME LOG_DIR
export WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/runs/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$ROOT_DIR/runs/wandb-cache}"

mkdir -p "$LOG_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"

submit_self() {
  local script_path="$ROOT_DIR/scripts/cluster/slurm.sh"
  local sbatch_args=(
    "--job-name=$JOB_NAME"
    "--partition=$PARTITION"
    "--cpus-per-task=$CPUS_PER_TASK"
    "--mem=$MEMORY"
    "--time=$TIME_LIMIT"
    "--output=$LOG_DIR/slurm_%j.out"
    "--error=$LOG_DIR/slurm_%j.err"
  )

  if [[ -n "$DEPENDENCY" ]]; then
    sbatch_args+=("--dependency=$DEPENDENCY")
  fi

  if [[ -n "$GPU_TYPE" ]]; then
    sbatch_args+=("--gres=gpu:${GPU_TYPE}:${GPUS}")
  else
    sbatch_args+=("--gres=gpu:${GPUS}")
  fi

  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  echo "Submitting $JOB_NAME"
  sbatch "${sbatch_args[@]}" --export=ALL,TASK="$TASK" "$script_path" "$TASK"
}

load_runtime() {
  module load "$SOFTWARE_STACK_MODULE"
  module load "$COMPILER_MODULE"

  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH" | tee -a "$LIVE_LOG"
    exit 1
  fi

  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV_NAME"

  export CC="$(command -v gcc)"
  export CXX="$(command -v g++)"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

best_semambapp_checkpoint() {
  local ckpt_dir="${1:-$ROOT_DIR/checkpoints/train_semambapp_tau_fixed}"
  local ckpt

  ckpt="$(find "$ckpt_dir" -maxdepth 1 -type f -name "best_guarded_ln_g_*.pth" | sort -V | tail -n 1)"
  if [[ -n "$ckpt" ]]; then
    echo "$ckpt"
    return 0
  fi

  ckpt="$(find "$ckpt_dir" -maxdepth 1 -type f -name "best_avqi_gap_ln_g_*.pth" | sort -V | tail -n 1)"
  if [[ -n "$ckpt" ]]; then
    echo "$ckpt"
    return 0
  fi

  find "$ckpt_dir" -maxdepth 1 -type f -name "ln_g_*.pth" | sort -V | tail -n 1
}

write_eval_scps() {
  local pair_csv="$1"
  local out_root="$2"
  python - "$pair_csv" "$out_root" <<'PY'
import csv
import sys
from pathlib import Path

pair_csv = Path(sys.argv[1])
out_root = Path(sys.argv[2])
rows = list(csv.DictReader(pair_csv.open()))

with (out_root / "inf.scp").open("w") as inf, (out_root / "ref.scp").open("w") as ref:
    for row in rows:
        uid = row["uid"]
        enhanced = out_root / "wav" / Path(row["noisy_filepath"]).name
        if not enhanced.is_file():
            raise FileNotFoundError(f"Missing enhanced wav for {uid}: {enhanced}")
        inf.write(f"{uid} {enhanced}\n")
        ref.write(f"{uid} {row['clean_filepath']}\n")

print(f"Wrote inf.scp/ref.scp for {len(rows)} utterances")
PY
}

run_train() {
  CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/train/semambapp_tau_fixed.yaml}"
  USE_SIMULATION_ROOT="${USE_SIMULATION_ROOT:-/scratch/work/lil14/USE_simulation}"
  SEMAMBAPP_EXPERIMENT_ROOT="${SEMAMBAPP_EXPERIMENT_ROOT:-$ROOT_DIR/checkpoints}"
  SEMAMBAPP_PRETRAINED_GENERATOR_CKPT="${SEMAMBAPP_PRETRAINED_GENERATOR_CKPT:-}"
  SEMAMBAPP_PRETRAINED_DISCRIMINATOR_CKPT="${SEMAMBAPP_PRETRAINED_DISCRIMINATOR_CKPT:-}"
  SEMAMBAPP_TAU_FIXED_TRAIN_CSV="${SEMAMBAPP_TAU_FIXED_TRAIN_CSV:-/scratch/work/lil14/data/TAU/simulated/phone_room/train/paired.csv}"
  SEMAMBAPP_TAU_FIXED_VALID_CSV="${SEMAMBAPP_TAU_FIXED_VALID_CSV:-/scratch/work/lil14/data/TAU/simulated/phone_room/valid/paired.csv}"
  export USE_SIMULATION_ROOT SEMAMBAPP_EXPERIMENT_ROOT
  export SEMAMBAPP_PRETRAINED_GENERATOR_CKPT SEMAMBAPP_PRETRAINED_DISCRIMINATOR_CKPT
  export SEMAMBAPP_TAU_FIXED_TRAIN_CSV SEMAMBAPP_TAU_FIXED_VALID_CSV

  test -f "$CONFIG_PATH"
  echo "Config: $CONFIG_PATH" | tee -a "$LIVE_LOG"
  python train.py --config "$CONFIG_PATH" 2>&1 | tee -a "$LIVE_LOG"
}

run_infer() {
  CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/train/semambapp_tau_fixed.yaml}"
  INPUT_DIR="${INPUT_DIR:-${IN_ROOT:-/scratch/work/lil14/data/TAU/simulated/phone_room/test/noisy}}"
  OUTPUT_DIR="${OUTPUT_DIR:-${OUT_ROOT:-/scratch/work/lil14/data/TAU/enhanced/semambapp/phone_room/test}}"
  PAIR_CSV="${PAIR_CSV:-/scratch/work/lil14/data/TAU/simulated/phone_room/test/paired.csv}"
  CKPT="${CKPT:-}"
  CKPT_DIR="${CKPT_DIR:-$ROOT_DIR/checkpoints/train_semambapp_tau_fixed}"
  WAV_DIR="$OUTPUT_DIR/wav"

  test -f "$CONFIG_PATH"
  mkdir -p "$WAV_DIR"

  if [[ -z "$CKPT" ]]; then
    CKPT="$(best_semambapp_checkpoint "$CKPT_DIR")"
  fi
  if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "No checkpoint found. Set CKPT or put ln_g_*.pth under $CKPT_DIR" | tee -a "$LIVE_LOG"
    exit 1
  fi

  echo "Config: $CONFIG_PATH" | tee -a "$LIVE_LOG"
  echo "Checkpoint: $CKPT" | tee -a "$LIVE_LOG"
  echo "Input dir: $INPUT_DIR" | tee -a "$LIVE_LOG"
  echo "Output dir: $OUTPUT_DIR" | tee -a "$LIVE_LOG"

  find "$INPUT_DIR" -type f -name "*.wav" | sort | while read -r in_file; do
    rel="${in_file#$INPUT_DIR/}"
    out_file="$WAV_DIR/$rel"
    mkdir -p "$(dirname "$out_file")"

    if [[ -f "$out_file" ]]; then
      echo "Skip existing: $rel" | tee -a "$LIVE_LOG"
      continue
    fi

    python infer.py \
      --config "$CONFIG_PATH" \
      --checkpoint "$CKPT" \
      --input "$in_file" \
      --output "$out_file" \
      --device cuda \
      2>&1 | tee -a "$LIVE_LOG"
  done

  write_eval_scps "$PAIR_CSV" "$OUTPUT_DIR"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  submit_self
  exit 0
fi

cd "$ROOT_DIR"
LIVE_LOG="$LOG_DIR/${TASK}_${SLURM_JOB_ID}.log"
echo "Live log: $LIVE_LOG"
echo "Job ${SLURM_JOB_ID} started at $(date)" | tee -a "$LIVE_LOG"
echo "Task: $TASK" | tee -a "$LIVE_LOG"
echo "Host: $(hostname)" | tee -a "$LIVE_LOG"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$LIVE_LOG"

load_runtime

case "$TASK" in
  train)
    run_train
    ;;
  infer)
    run_infer
    ;;
  *)
    echo "Unknown task: $TASK" | tee -a "$LIVE_LOG"
    exit 2
    ;;
esac

echo "Job ${SLURM_JOB_ID} completed at $(date)" | tee -a "$LIVE_LOG"
