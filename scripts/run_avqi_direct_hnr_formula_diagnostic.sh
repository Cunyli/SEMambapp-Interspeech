#!/bin/bash
# Compare two Route C HNR formulas on non-final data. No waveform is optimized.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR}"
PYTHON_SCRIPT="$SOURCE_ROOT/scripts/evaluate_direct_avqi_hnr_formula.py"

DATA_RUN_ROOT="${DATA_RUN_ROOT:-$ROOT_DIR/runs/avqi_component_direct_c_v5_data_20260817_03}"
LABEL_RECEIPT="${LABEL_RECEIPT:-$DATA_RUN_ROOT/outputs/label_bank/receipt.json}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/avqi_direct_hnr_raw_cc_v3_diagnostic_20260817_01}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/outputs}"
PARTITION="${PARTITION:-gpu-v100-32g}"
GPU_TYPE="${GPU_TYPE:-v100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
SOFTWARE_STACK_MODULE="${SOFTWARE_STACK_MODULE:-triton/2025.1-gcc}"
COMPILER_MODULE="${COMPILER_MODULE:-gcc/13.3.0}"
MINIMUM_RELATIVE_CALIBRATION_IMPROVEMENT="${MINIMUM_RELATIVE_CALIBRATION_IMPROVEMENT:-0.05}"

if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain)" ]]; then
  echo "Refusing to run from a dirty source tree: $SOURCE_ROOT" >&2
  exit 2
fi
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}"

for path in "$PYTHON_SCRIPT" "$LABEL_RECEIPT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing Route C HNR diagnostic source: $path" >&2
    exit 2
  fi
done
if [[ "$(jq -er '.schema_version' "$LABEL_RECEIPT")" != "avqi-component-label-bank-v4" ]]; then
  echo "Unexpected AVQI label receipt schema" >&2
  exit 2
fi

LABEL_BANK="$(jq -er '.internal_label_bank' "$LABEL_RECEIPT")"
LABEL_BANK_SHA256="$(jq -er '.internal_label_bank_sha256' "$LABEL_RECEIPT")"
VCTK_EXTERNAL_LABEL_BANK="$(jq -er '.external_label_bank' "$LABEL_RECEIPT")"
VCTK_EXTERNAL_LABEL_BANK_SHA256="$(jq -er '.external_label_bank_sha256' "$LABEL_RECEIPT")"
for path in "$LABEL_BANK" "$VCTK_EXTERNAL_LABEL_BANK"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing Route C HNR label bank: $path" >&2
    exit 2
  fi
done
if [[ "$(sha256sum "$LABEL_BANK" | awk '{print $1}')" != "$LABEL_BANK_SHA256" ]]; then
  echo "Internal label-bank hash mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "$VCTK_EXTERNAL_LABEL_BANK" | awk '{print $1}')" != "$VCTK_EXTERNAL_LABEL_BANK_SHA256" ]]; then
  echo "VCTK external label-bank hash mismatch" >&2
  exit 2
fi

export ROOT_DIR SOURCE_ROOT PYTHON_SCRIPT DATA_RUN_ROOT LABEL_RECEIPT
export RUN_ROOT LOG_DIR OUTPUT_DIR PARTITION GPU_TYPE CPUS_PER_TASK MEMORY
export TIME_LIMIT SOFTWARE_STACK_MODULE COMPILER_MODULE SOURCE_COMMIT
export LABEL_BANK LABEL_BANK_SHA256 VCTK_EXTERNAL_LABEL_BANK
export VCTK_EXTERNAL_LABEL_BANK_SHA256 MINIMUM_RELATIVE_CALIBRATION_IMPROVEMENT

mkdir -p "$LOG_DIR"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${CONFIRM_SLURM_SUBMIT:-0}" != "1" ]]; then
    echo "Refusing to submit without CONFIRM_SLURM_SUBMIT=1" >&2
    exit 2
  fi
  if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite HNR diagnostic output: $OUTPUT_DIR" >&2
    exit 2
  fi
  sbatch \
    --parsable \
    --job-name=avqi-c-hnr-v3 \
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

if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "Route C HNR source HEAD drifted after submission" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite HNR diagnostic output: $OUTPUT_DIR" >&2
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

LIVE_LOG="$LOG_DIR/avqi_direct_hnr_formula_${SLURM_JOB_ID}.log"
echo "event=start job=$SLURM_JOB_ID commit=$SOURCE_COMMIT time=$(date -Is)" | tee -a "$LIVE_LOG"
python -c 'import os, torch; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0)); print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))' | tee -a "$LIVE_LOG"

python "$PYTHON_SCRIPT" \
  --label-bank "$LABEL_BANK" \
  --label-bank-sha256 "$LABEL_BANK_SHA256" \
  --vctk-external-label-bank "$VCTK_EXTERNAL_LABEL_BANK" \
  --vctk-external-label-bank-sha256 "$VCTK_EXTERNAL_LABEL_BANK_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  --source-commit "$SOURCE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --device cuda \
  --expected-internal-valid-rows 2134 \
  --expected-vctk-valid-rows 192 \
  --expected-train-speakers 197 \
  --expected-calibration-speakers 26 \
  --expected-holdout-speakers 26 \
  --expected-vctk-speakers 12 \
  --minimum-relative-calibration-improvement "$MINIMUM_RELATIVE_CALIBRATION_IMPROVEMENT" \
  2>&1 | tee -a "$LIVE_LOG"

echo "event=complete job=$SLURM_JOB_ID time=$(date -Is)" | tee -a "$LIVE_LOG"
