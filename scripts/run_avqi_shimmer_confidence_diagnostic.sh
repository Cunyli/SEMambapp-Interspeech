#!/bin/bash
# Hash-locked Shimmer pulse-confidence diagnostic. No waveform is optimized.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
PYTHON_SCRIPT="$SOURCE_ROOT/scripts/evaluate_avqi_shimmer_vctk_topology_audit.py"
MODEL_SOURCE="$SOURCE_ROOT/model/avqi_components.py"

LABEL_BANK="${LABEL_BANK:-/scratch/work/lil14/SEMambapp-Interspeech/runs/avqi_component_direct_c_v5_data_20260817_03/outputs/label_bank/vctk_external_exact_components_v4.csv}"
LABEL_BANK_SHA256="${LABEL_BANK_SHA256:-ee6853264500c94ac42691d1144358c4cf306899fe9ab9a308fd690aa0699781}"
EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"
RUN_ROOT="${RUN_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_shimmer_confidence_v8_smoke_20260823_02}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
MAX_SPEAKERS="${MAX_SPEAKERS:-1}"
PARTITION="${PARTITION:-gpu-debug}"
GPU_TYPE="${GPU_TYPE:-v100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-24G}"
TIME_LIMIT="${TIME_LIMIT:-00:15:00}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"
BASE_COMMIT="${BASE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"
MODEL_SHA256="${MODEL_SHA256:?MODEL_SHA256 is required}"
DIAGNOSTIC_SHA256="${DIAGNOSTIC_SHA256:?DIAGNOSTIC_SHA256 is required}"
ALLOW_HASH_LOCKED_SOURCE="${ALLOW_HASH_LOCKED_SOURCE:-0}"

for path in "$PYTHON_SCRIPT" "$MODEL_SOURCE" "$LABEL_BANK" "$EXACT_PYTHON"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing Shimmer confidence input: $path" >&2
    exit 2
  fi
done
if [[ "$(sha256sum "$LABEL_BANK" | awk '{print $1}')" != "$LABEL_BANK_SHA256" ]]; then
  echo "VCTK label-bank hash mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "$MODEL_SOURCE" | awk '{print $1}')" != "$MODEL_SHA256" ]]; then
  echo "Shimmer model-source hash mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "$PYTHON_SCRIPT" | awk '{print $1}')" != "$DIAGNOSTIC_SHA256" ]]; then
  echo "Shimmer diagnostic-source hash mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" && "$ALLOW_HASH_LOCKED_SOURCE" != "1" ]]; then
  echo "Refusing dirty source without ALLOW_HASH_LOCKED_SOURCE=1" >&2
  exit 2
fi
if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$BASE_COMMIT" ]]; then
  echo "Shimmer confidence base commit drifted" >&2
  exit 2
fi
if [[ -e "$RUN_ROOT/outputs" ]]; then
  echo "Refusing to overwrite Shimmer confidence outputs: $RUN_ROOT/outputs" >&2
  exit 2
fi

export ROOT_DIR SOURCE_ROOT PYTHON_SCRIPT MODEL_SOURCE LABEL_BANK
export LABEL_BANK_SHA256 EXACT_PYTHON RUN_ROOT LOG_DIR MAX_SPEAKERS
export PARTITION GPU_TYPE CPUS_PER_TASK MEMORY TIME_LIMIT SOURCE_COMMIT
export BASE_COMMIT MODEL_SHA256 DIAGNOSTIC_SHA256 ALLOW_HASH_LOCKED_SOURCE

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  mkdir -p "$LOG_DIR"
  sbatch \
    --parsable \
    --job-name=avqi-shim-conf \
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

module load triton/2025.1-gcc
module load gcc/13.3.0
eval "$(conda shell.bash hook)"
conda activate semambapp
export PYTHONPATH="$SOURCE_ROOT:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

LIVE_LOG="$LOG_DIR/shimmer_confidence_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID source=$SOURCE_COMMIT base=$BASE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0))' | tee -a "$LIVE_LOG"
python "$PYTHON_SCRIPT" \
  --label-bank "$LABEL_BANK" \
  --label-bank-sha256 "$LABEL_BANK_SHA256" \
  --exact-python "$EXACT_PYTHON" \
  --output-dir "$RUN_ROOT" \
  --max-speakers "$MAX_SPEAKERS" \
  --source-commit "$SOURCE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --device cuda \
  2>&1 | tee -a "$LIVE_LOG"
echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
