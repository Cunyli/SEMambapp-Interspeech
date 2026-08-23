#!/bin/bash
# Hash-locked LTAS-slope authority-relative gate experiment. No training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
PYTHON_SCRIPT="$SOURCE_ROOT/scripts/evaluate_avqi_ltas_slope_gate_alignment.py"
AUTHORITY_HELPER="$SOURCE_ROOT/scripts/evaluate_avqi_ltas_slope_lowpass_authority.py"
MODEL_SOURCE="$SOURCE_ROOT/model/avqi_components.py"

LABEL_BANK="${LABEL_BANK:-/scratch/work/lil14/SEMambapp-Interspeech/runs/avqi_component_direct_c_v5_data_20260817_03/outputs/label_bank/exact_component_label_bank_v4.csv}"
LABEL_BANK_SHA256="${LABEL_BANK_SHA256:-03b8d5e3d0542dbfe60e54723bc89431e8dfd475dcc38284a6058465c5224760}"
PREDICTOR_CHECKPOINT="${PREDICTOR_CHECKPOINT:-/scratch/work/lil14/SEMambapp-Interspeech/checkpoints/avqi_route_c_shimmer_v6_screen_20260821_01/direct_direct_praat_hard_shimmer_pulse_path_v6_estimator.pt}"
PREDICTOR_CHECKPOINT_SHA256="${PREDICTOR_CHECKPOINT_SHA256:-40b819946abdcb8a4b643fe4238d1bb4d31168a3eb2a6d6c786a61752da629bc}"
EXACT_PYTHON="${EXACT_PYTHON:-/scratch/work/lil14/.conda_envs/avqi/bin/python}"
RUN_ROOT="${RUN_ROOT:-$SOURCE_ROOT/runs/avqi_route_c_ltas_slope_gate_alignment_v8_calibration_20260823_01}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
SPLIT="${SPLIT:-surrogate_calibration}"
MAX_CASES="${MAX_CASES:-9}"
PARTITION="${PARTITION:-batch-milan}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-16G}"
TIME_LIMIT="${TIME_LIMIT:-00:15:00}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"
BASE_COMMIT="${BASE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"
MODEL_SHA256="${MODEL_SHA256:?MODEL_SHA256 is required}"
DIAGNOSTIC_SHA256="${DIAGNOSTIC_SHA256:?DIAGNOSTIC_SHA256 is required}"
AUTHORITY_HELPER_SHA256="${AUTHORITY_HELPER_SHA256:?AUTHORITY_HELPER_SHA256 is required}"
ALLOW_HASH_LOCKED_SOURCE="${ALLOW_HASH_LOCKED_SOURCE:-0}"

for path in "$PYTHON_SCRIPT" "$AUTHORITY_HELPER" "$MODEL_SOURCE" \
  "$LABEL_BANK" "$PREDICTOR_CHECKPOINT" "$EXACT_PYTHON"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing LTAS gate-alignment input: $path" >&2
    exit 2
  fi
done
verify_hash() {
  local path="$1"
  local expected="$2"
  local label="$3"
  if [[ "$(sha256sum "$path" | awk '{print $1}')" != "$expected" ]]; then
    echo "$label SHA-256 mismatch" >&2
    exit 2
  fi
}
verify_hash "$LABEL_BANK" "$LABEL_BANK_SHA256" "LTAS label bank"
verify_hash "$PREDICTOR_CHECKPOINT" "$PREDICTOR_CHECKPOINT_SHA256" "LTAS checkpoint"
verify_hash "$MODEL_SOURCE" "$MODEL_SHA256" "LTAS model source"
verify_hash "$PYTHON_SCRIPT" "$DIAGNOSTIC_SHA256" "LTAS diagnostic source"
verify_hash "$AUTHORITY_HELPER" "$AUTHORITY_HELPER_SHA256" "LTAS authority helper"
if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" && "$ALLOW_HASH_LOCKED_SOURCE" != "1" ]]; then
  echo "Refusing dirty source without ALLOW_HASH_LOCKED_SOURCE=1" >&2
  exit 2
fi
if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$BASE_COMMIT" ]]; then
  echo "LTAS gate-alignment base commit drifted" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite LTAS gate-alignment outputs: $OUTPUT_DIR" >&2
  exit 2
fi

export ROOT_DIR SOURCE_ROOT PYTHON_SCRIPT AUTHORITY_HELPER MODEL_SOURCE
export LABEL_BANK LABEL_BANK_SHA256 PREDICTOR_CHECKPOINT
export PREDICTOR_CHECKPOINT_SHA256 EXACT_PYTHON RUN_ROOT OUTPUT_DIR LOG_DIR
export SPLIT MAX_CASES PARTITION CPUS_PER_TASK MEMORY TIME_LIMIT SOURCE_COMMIT
export BASE_COMMIT MODEL_SHA256 DIAGNOSTIC_SHA256 AUTHORITY_HELPER_SHA256
export ALLOW_HASH_LOCKED_SOURCE

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  mkdir -p "$LOG_DIR"
  sbatch \
    --parsable \
    --job-name=avqi-ltas-gate \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
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

LIVE_LOG="$LOG_DIR/ltas_gate_alignment_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID split=$SPLIT source=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
python "$PYTHON_SCRIPT" \
  --label-bank "$LABEL_BANK" \
  --label-bank-sha256 "$LABEL_BANK_SHA256" \
  --predictor-checkpoint "$PREDICTOR_CHECKPOINT" \
  --predictor-checkpoint-sha256 "$PREDICTOR_CHECKPOINT_SHA256" \
  --exact-python "$EXACT_PYTHON" \
  --output-dir "$OUTPUT_DIR" \
  --split "$SPLIT" \
  --max-cases "$MAX_CASES" \
  --source-commit "$SOURCE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --device cpu \
  2>&1 | tee -a "$LIVE_LOG"
echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
